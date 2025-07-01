import time
import numpy as np
import sympy as sp
from qdeepsdk import QDeepHybridSolver
from manual_checker import check_minimum

# --------------------------------------------------------------------
# Problem data
# --------------------------------------------------------------------
a = np.random.uniform(low=50.0, high=500.0, size=500)
b = np.random.uniform(low=50.0, high=500.0, size=500)


# гарантируем a_i <= b_i
a, b = np.minimum(a, b), np.maximum(a, b)
n = len(a)

# начальные границы поиска
LB_x, UB_x = np.min(a), np.max(b)
LB_y, UB_y = LB_x, UB_x

# параметры битовой дискретизации
N, M = 4, 3

# шаги для slack-переменных
sigma_s = sigma_t = 1.0

# штрафные коэффициенты
v_interval = 1e4      # усиленный квадратичный штраф длины
v1, v2     = 10.0, 10.0
v3         = 1e6
v_slack    = 0.01

# итерационные параметры
max_iter            = 10
tol                 = 1e-3
EPS_INTERVAL_MARGIN = 0.1

prev_x_qp = None

# --------------------------------------------------------------------
# Функция построения QUBO
# --------------------------------------------------------------------
def build_qubo(beta, sigma_x, gamma, sigma_y):
    z = sp.symbols(f'z0:{n}', integer=True)
    d = sp.symbols(f'd0:{N+1}', integer=True)
    g = sp.symbols(f'g0:{N+1}', integer=True)
    e = sp.symbols(f'e0:{M+1}', integer=True)
    p = sp.symbols(f'p0:{M+1}', integer=True)
    vars_order = list(z) + list(d) + list(g) + list(e) + list(p)

    # декодирование x, y и slack
    x_expr = beta + sigma_x * sum(2**i * d[i] for i in range(N+1))
    y_expr = gamma + sigma_y * sum(2**i * g[i] for i in range(N+1))
    s_expr = sigma_s * sum(2**i * e[i] for i in range(M+1))
    t_expr = sigma_t * sum(2**i * p[i] for i in range(M+1))

    A_expr = sum(a[i] * z[i] for i in range(n))
    B_expr = sum(b[i] * z[i] for i in range(n))
    z_sum  = sum(z)

    cost = (
        v1 * (x_expr - A_expr - s_expr)**2
      + v2 * (B_expr - y_expr - t_expr)**2
      + v3 * (z_sum - 1)**2
      + v_slack * (s_expr**2 + t_expr**2)
      + v_interval * (B_expr - A_expr)**2     # усиленный квадратичный штраф длины
    )

    P = sp.Poly(sp.expand(cost), vars_order)
    terms = P.as_dict()

    dim = len(vars_order)
    Q = np.zeros((dim, dim))
    for monom, coef in terms.items():
        idxs = [i for i, pwr in enumerate(monom) if pwr]
        if len(idxs) == 1:
            Q[idxs[0], idxs[0]] += float(coef)
        elif len(idxs) == 2:
            i, j = idxs
            Q[i, j] += float(coef)/2
            Q[j, i] += float(coef)/2

    return Q, vars_order

# --------------------------------------------------------------------
# Основной цикл с таймером
# --------------------------------------------------------------------
start_time = time.perf_counter()

for it in range(max_iter):
    print(f"\n--- iteration {it} ---")

    # пересчитываем дискретизацию
    sigma_x = (UB_x - LB_x) / (2**(N+1) - 1)
    sigma_y = (UB_y - LB_y) / (2**(N+1) - 1)
    beta, gamma = LB_x, LB_y

    # сборка и решение QUBO
    Q, _ = build_qubo(beta, sigma_x, gamma, sigma_y)
    solver = QDeepHybridSolver()
    solver.token     = "akwysie03c"
    solver.m_budget  = 10**10
    solver.num_reads = 5000

    resp = solver.solve(Q)
    sol  = np.array(resp['QdeepHybridSolver']['configuration'])
    energy = resp['QdeepHybridSolver']['energy']
    print(f"QUBO → energy: {energy:.4f}, configuration: {sol.tolist()}")

    # выбираем интервал
    z_sol = sol[:n].round().astype(int)
    if z_sol.sum() != 1:
        print("⚠️ z не one-hot:", z_sol)
        break

    i_star = int(np.argmax(z_sol))
    lb_int, ub_int = a[i_star], b[i_star]
    print(f"Interval #{i_star}: [{lb_int:.4f}, {ub_int:.4f}]")

    # snapping к концам
    x_qp, y_qp = lb_int, ub_int
    print(f"Refined → x={x_qp:.4f}, y={y_qp:.4f}, length={y_qp-x_qp:.4f}")

    # проверка сходимости
    if prev_x_qp is not None and abs(x_qp - prev_x_qp) < tol:
        print("✅ Converged.")
        break
    prev_x_qp = x_qp

    # сужаем окно поиска
    LB_x, UB_x = lb_int, lb_int + EPS_INTERVAL_MARGIN
    LB_y, UB_y = ub_int - EPS_INTERVAL_MARGIN, ub_int

total_time = time.perf_counter() - start_time
print(f"\nTotal execution time: {total_time:.3f} s")

# итоговый отчёт
print("\n======= RESULT =======")
print(f"shortest interval = [{a[i_star]:.4f}, {b[i_star]:.4f}]")
print(f"x* = {x_qp:.4f}, y* = {y_qp:.4f}")
print(f"length of interval = {y_qp - x_qp:.4f}")
print(f"one-hot z* = {z_sol}")

print("\nMANUAL CHECKER RESULT")
print(check_minimum(a, b))