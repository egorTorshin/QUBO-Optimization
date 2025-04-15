import numpy as np
import cvxpy as cp
from qdeepsdk import QDeepHybridSolver
import requests
import sympy as sp

# ----------------------- Входные данные -----------------------
n = 3
a = np.array([1.0, 3.2, 2.7])    # Нижние границы интервалов
b = np.array([2.0, 8.4, 3.0])   # Верхние границы интервалов

sigma_x = 1.0
sigma_y = 1.0
sigma_s = 1.0
sigma_t = 1.0

beta = 0.0
gamma = 0.0
epsilon = 0.0
pi_const = 0.0

N = 4  # Количество бит для x и y (N+1 бит)
M = 3  # Количество бит для слэк-переменных (M+1 бит)

v1 = 10.0    # Штраф за (x - sum(a*z) - s)^2
v2 = 10.0    # Штраф за (y - sum(b*z) + t)^2
v3 = 1e6     # Очень большой штраф для ограничения one-hot: (sum(z)-1)^2

# Общее число переменных: порядок – сначала z, затем d (для x), затем g (для y),
# затем e (для s) и p (для t)
n_vars = n + (N + 1) + (N + 1) + (M + 1) + (M + 1)

# ----------------- Символьное представление переменных -----------------
z_vars = sp.symbols('z0:%d' % n, integer=True)
d_vars = sp.symbols('d0:%d' % (N + 1), integer=True)
g_vars = sp.symbols('g0:%d' % (N + 1), integer=True)
e_vars = sp.symbols('e0:%d' % (M + 1), integer=True)
p_vars = sp.symbols('p0:%d' % (M + 1), integer=True)

vars_order = list(z_vars) + list(d_vars) + list(g_vars) + list(e_vars) + list(p_vars)

# ----------------- Формирование символьных выражений -----------------
x_expr = beta + sigma_x * sp.Add(*[2**i * d_vars[i] for i in range(N + 1)])
y_expr = gamma + sigma_y * sp.Add(*[2**i * g_vars[i] for i in range(N + 1)])
s_expr = epsilon + sigma_s * sp.Add(*[2**i * e_vars[i] for i in range(M + 1)])
t_expr = pi_const + sigma_t * sp.Add(*[2**i * p_vars[i] for i in range(M + 1)])

A_expr = sp.Add(*[a[i] * z_vars[i] for i in range(n)])
B_expr = sp.Add(*[b[i] * z_vars[i] for i in range(n)])
z_sum = sp.Add(*z_vars)

cost_expr = (x_expr - y_expr) + v1 * (x_expr - A_expr - s_expr)**2 + \
            v2 * (y_expr - B_expr + t_expr)**2 + v3 * (z_sum - 1)**2

cost_expanded = sp.expand(cost_expr)
poly_cost = sp.Poly(cost_expanded, vars_order)
terms = poly_cost.as_dict()

# ----------------- Извлечение QUBO-матрицы из символьного выражения -----------------
num_vars = len(vars_order)
Q_matrix = np.zeros((num_vars, num_vars))

for monom, coef in terms.items():
    degree = sum(monom)
    if degree == 0:
        continue
    elif degree == 1:
        # Моном содержит только один член степени 1
        i = monom.index(1)
        Q_matrix[i, i] += float(coef)
    elif degree == 2:
        # Найдём индексы переменных, входящих в моном, и их степени
        indices = [i for i, exp in enumerate(monom) if exp > 0]
        exponents = [exp for exp in monom if exp > 0]
        # Если моном имеет вид x_i^2, то обработаем его как линейный член (x_i^2 = x_i, так как x_i бинарна)
        if len(indices) == 1 and exponents[0] == 2:
            i = indices[0]
            Q_matrix[i, i] += float(coef)
        # Если моном имеет вид x_i * x_j, где i != j
        elif len(indices) == 2 and all(exp == 1 for exp in exponents):
            i, j = indices
            Q_matrix[i, j] += float(coef) / 2.0
            Q_matrix[j, i] += float(coef) / 2.0
        else:
            raise ValueError("Неправильный моном степени 2: " + str(monom))
    else:
        raise ValueError("Обнаружен моном степени > 2: " + str(monom) + ". Проверьте формулировку.")

# ----------------- Решение QUBO с помощью QDeepHybridSolver -----------------
solver = QDeepHybridSolver()
solver.token = "YOUR-API-TOKEN"
solver.m_budget = 10 ** 7
solver.num_reads = 100

try:
    response = solver.solve(Q_matrix)
    results = response['QdeepHybridSolver']
    print("Hybrid Solver Results:", results)
    print()
except ValueError as e:
    print(f"Error: {e}")
except requests.RequestException as e:
    print(f"API Error: {e}")

solution = np.array(results['configuration'])

# Извлечение групп переменных в соответствии с порядком: сначала z, затем d, затем g.
z_sol = solution[0:n].round().astype(int)
d_sol = solution[n: n + (N + 1)].round().astype(int)
g_sol = solution[n + (N + 1): n + 2 * (N + 1)].round().astype(int)

x_qubo = beta + sigma_x * sum((2**i) * d_sol[i] for i in range(len(d_sol)))
y_qubo = gamma + sigma_y * sum((2**i) * g_sol[i] for i in range(len(g_sol)))
print("x (from QUBO) =", x_qubo)
print("y (from QUBO) =", y_qubo)
print()
print("Chosen interval z* =", z_sol)
print()
print("Interval lower bound =", np.dot(a, z_sol))
print("Interval upper bound =", np.dot(b, z_sol))

x_var = cp.Variable()
y_var = cp.Variable()
constraints = [x_var >= np.dot(a, z_sol), y_var <= np.dot(b, z_sol)]
lp_obj = cp.Minimize(x_var - y_var)
lp_prob = cp.Problem(lp_obj, constraints)
lp_prob.solve()

print()
print("Final LP solution:")
print("x* =", x_var.value)
print("y* =", y_var.value)
print("Objective (x* - y*) =", x_var.value - y_var.value)
