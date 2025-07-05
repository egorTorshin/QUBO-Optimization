import time
import numpy as np
import sympy as sp
from qdeepsdk import QDeepHybridSolver
from manual_checker import check_minimum

a = np.random.uniform(0.0, 100.0, size=10)
b = np.random.uniform(0.0, 100.0, size=10)
a, b = np.minimum(a, b), np.maximum(a, b)
n = len(a)

# Initialize search bounds & precision
# Global search window for (x,y)
LB_x, UB_x = np.min(a), np.max(b)
LB_y, UB_y = LB_x, UB_x

# Number of bits for representing x,y and slack variables s,t
N, M = 4, 3

# Slack‐variable scaling (s and t will take values 0,1,2,4,...)
sigma_s = sigma_t = 1.0


# QUBO penalty weights
# Heavy weight on interval-length penalty forces the solver to pick shortest
v_interval = 1e4
# Moderate weights for soft interval constraints:
v1, v2 = 10.0, 10.0
# Huge weight to enforce exactly one z_i = 1
v3 = 1e6
# Small weight to keep slack variables from growing unnecessarily
v_slack = 0.01


# Iteration & convergence parameters
max_iter            = 10     # maximum number of multiple‐shooting iterations
tol                 = 1e-3   # convergence tolerance on x
EPS_INTERVAL_MARGIN = 0.1    # margin to shrink the search window each round
prev_x_qp = None            # store previous x* for convergence check


# Build QUBO matrix function
def build_qubo(beta, sigma_x, gamma, sigma_y):
    """
    Construct the QUBO matrix Q and variable ordering for given
    offsets (beta,gamma) and discretization steps (sigma_x,sigma_y).
    """
    # Declare all symbolic bits --
    z = sp.symbols(f'z0:{n}', integer=True)        # selects interval
    d = sp.symbols(f'd0:{N+1}', integer=True)      # bits for x
    g = sp.symbols(f'g0:{N+1}', integer=True)      # bits for y
    e = sp.symbols(f'e0:{M+1}', integer=True)      # bits for slack s
    p = sp.symbols(f'p0:{M+1}', integer=True)      # bits for slack t
    vars_order = list(z) + list(d) + list(g) + list(e) + list(p)

    # Decode real‐valued expressions from bits
    x_expr = beta + sigma_x * sum(2**i * d[i] for i in range(N+1))
    y_expr = gamma + sigma_y * sum(2**i * g[i] for i in range(N+1))
    s_expr = sigma_s * sum(2**i * e[i] for i in range(M+1))
    t_expr = sigma_t * sum(2**i * p[i] for i in range(M+1))

    # Interval endpoints via one‐hot z
    A_expr = sum(a[i] * z[i] for i in range(n))
    B_expr = sum(b[i] * z[i] for i in range(n))
    z_sum  = sum(z)

    # Build the symbolic cost function
    cost = (
        v1 * (x_expr - A_expr - s_expr)**2     # x ≥ A − s
      + v2 * (B_expr - y_expr - t_expr)**2     # y ≤ B + t
      + v3 * (z_sum - 1)**2                    # enforce one‐hot on z
      + v_slack * (s_expr**2 + t_expr**2)      # keep slacks small
      + v_interval * (B_expr - A_expr)**2      # force shortest interval
    )

    # Expand and extract polynomial terms
    P     = sp.Poly(sp.expand(cost), vars_order)
    terms = P.as_dict()

    # Assemble the QUBO matrix Q from the monomial coefficients
    dim = len(vars_order)
    Q   = np.zeros((dim, dim))
    for monom, coef in terms.items():
        idxs = [i for i, pwr in enumerate(monom) if pwr]
        if len(idxs) == 1:
            # diagonal term
            Q[idxs[0], idxs[0]] += float(coef)
        else:
            # off‐diagonal split into two symmetric halves
            i, j = idxs
            Q[i, j] += float(coef) / 2
            Q[j, i] += float(coef) / 2

    return Q, vars_order


start_time = time.perf_counter()

for it in range(max_iter):
    print(f"\n--- iteration {it} ---")

    # Recompute discretization step sizes and offsets
    sigma_x = (UB_x - LB_x) / (2**(N+1) - 1)
    sigma_y = (UB_y - LB_y) / (2**(N+1) - 1)
    beta, gamma = LB_x, LB_y

    # Build and solve the QUBO
    Q, _ = build_qubo(beta, sigma_x, gamma, sigma_y)
    solver = QDeepHybridSolver()
    solver.token = "TOKEN"
    solver.m_budget = 10**10
    solver.num_reads = 5000

    resp = solver.solve(Q)
    sol  = np.array(resp['QdeepHybridSolver']['configuration'])
    energy = resp['QdeepHybridSolver']['energy']
    print(f"QUBO → energy: {energy:.4f}")

    # Decode the one‐hot z vector, pick the chosen interval
    z_sol = sol[:n].round().astype(int)
    if z_sol.sum() != 1:
        print("⚠️ z not one‐hot:", z_sol)
        break
    i_star = int(np.argmax(z_sol))
    lb_int, ub_int = a[i_star], b[i_star]
    print(f"Interval #{i_star}: [{lb_int:.4f}, {ub_int:.4f}]")

    # Refinement: snap x,y to the ends of the chosen interval
    x_qp, y_qp = lb_int, ub_int
    print(f"Refined → x={x_qp:.4f}, y={y_qp:.4f}, length={y_qp-x_qp:.4f}")

    # Convergence check: stop if x_qp barely changes
    if prev_x_qp is not None and abs(x_qp - prev_x_qp) < tol:
        print("✅ Converged.")
        break
    prev_x_qp = x_qp

    # Narrow the search window for the next iteration
    LB_x, UB_x = lb_int, lb_int + EPS_INTERVAL_MARGIN
    LB_y, UB_y = ub_int - EPS_INTERVAL_MARGIN, ub_int

total_time = time.perf_counter() - start_time
print(f"\nTotal execution time: {total_time:.3f} s")


print("\n======= RESULT =======")
print(f"shortest interval = [{a[i_star]:.4f}, {b[i_star]:.4f}]")
print(f"x* = {x_qp:.4f}, y* = {y_qp:.4f}")
print(f"length of interval = {y_qp - x_qp:.4f}")
print(f"one-hot z* = {z_sol}")

print("\n======= MANUAL CHECKER RESULT =======")
print(check_minimum(a, b))
