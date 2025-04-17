import numpy as np
import cvxpy as cp
import sympy as sp
from qdeepsdk import QDeepHybridSolver

n = 3  # количество интервалов (объектов выбора)
a = np.array([1.0, 3.2, 1.2])  # нижние границы интервалов a_i
b = np.array([10.4, 12.2, 11.2])  # верхние границы интервалов b_i

# Глобальные границы значений x и y
LB_x, UB_x = 1.0, 12.3
LB_y, UB_y = 1.0, 12.3

# Шаги и смещения для slack переменных
sigma_s = 1.0
sigma_t = 1.0
epsilon = 0.0
pi_const = 0.0

# Параметры битовой дискретизации для x, y и slack переменных
N = 4  # количество бит для x и y (N+1 битов)
M = 3  # количество бит для slack переменных (M+1 битов)

# Весовые коэффициенты в целевой функции QUBO
v1, v2, v3 = 10.0, 10.0, 1e6

# Параметры итерационного цикла
max_iter = 10
tol = 1e-3
prev_x_qp = None
prev_interval = None

# Маленький запас вокруг выбранного интервала
EPS_INTERVAL_MARGIN = 0.1


# ----------------------- Построение QUBO матрицы -----------------------
def build_qubo(beta, sigma_x, gamma, sigma_y):
    # Определяем бинарные переменные
    z_vars = sp.symbols('z0:%d' % n, integer=True)  # выбор интервала
    d_vars = sp.symbols('d0:%d' % (N + 1), integer=True)  # биты x
    g_vars = sp.symbols('g0:%d' % (N + 1), integer=True)  # биты y
    e_vars = sp.symbols('e0:%d' % (M + 1), integer=True)  # slack s
    p_vars = sp.symbols('p0:%d' % (M + 1), integer=True)  # slack t
    vars_order = list(z_vars) + list(d_vars) + list(g_vars) + list(e_vars) + list(p_vars)

    # Выражения для x, y, s, t через бинарные переменные и масштаб
    x_expr = beta + sigma_x * sum(2 ** i * d_vars[i] for i in range(N + 1))
    y_expr = gamma + sigma_y * sum(2 ** i * g_vars[i] for i in range(N + 1))
    s_expr = epsilon + sigma_s * sum(2 ** i * e_vars[i] for i in range(M + 1))
    t_expr = pi_const + sigma_t * sum(2 ** i * p_vars[i] for i in range(M + 1))

    # A_expr и B_expr — значения a_i и b_i, выбранные через z-переменные
    A_expr = sum(a[i] * z_vars[i] for i in range(n))
    B_expr = sum(b[i] * z_vars[i] for i in range(n))
    z_sum = sum(z_vars)

    # Целевая функция QUBO: (x - y) + штрафы за ограничения
    cost_expr = (x_expr - y_expr) + \
                v1 * (x_expr - A_expr - s_expr) ** 2 + \
                v2 * (y_expr - B_expr + t_expr) ** 2 + \
                v3 * (z_sum - 1) ** 2  # одноинтервальное условие

    # Преобразуем в полином и извлекаем коэффициенты для QUBO-матрицы
    poly_cost = sp.Poly(sp.expand(cost_expr), vars_order)
    terms = poly_cost.as_dict()

    Q_matrix = np.zeros((len(vars_order), len(vars_order)))
    for monom, coef in terms.items():
        idx = [i for i, p in enumerate(monom) if p > 0]
        if len(idx) == 1:
            Q_matrix[idx[0], idx[0]] += float(coef)
        elif len(idx) == 2:
            i, j = idx
            Q_matrix[i, j] += float(coef) / 2
            Q_matrix[j, i] += float(coef) / 2
    return Q_matrix, vars_order


# Итерации
for it in range(max_iter):
    print(f"\n--- Iteration {it} ---")

    # Обновляем шаг дискретизации x и y
    sigma_x = (UB_x - LB_x) / (2 ** (N + 1) - 1)
    sigma_y = (UB_y - LB_y) / (2 ** (N + 1) - 1)
    beta_val = LB_x
    gamma_val = LB_y

    # Строим QUBO-матрицу
    Q_matrix, vars_order = build_qubo(beta_val, sigma_x, gamma_val, sigma_y)

    # Инициализируем гибридный решатель
    solver = QDeepHybridSolver()
    solver.token = "mtagdfsplb"
    solver.m_budget = 10 ** 8
    solver.num_reads = 100

    # Решаем QUBO-задачу
    response = solver.solve(Q_matrix)
    results = response['QdeepHybridSolver']

    # Извлекаем бинарное решение
    solution = np.array(results['configuration'])
    z_sol = solution[0:n].round().astype(int)  # выбор интервала
    d_sol = solution[n: n + N + 1].round().astype(int)  # x
    g_sol = solution[n + N + 1: n + 2 * (N + 1)].round().astype(int)  # y

    # Декодируем x и y из бинарного представления
    x_qubo = beta_val + sigma_x * sum(2 ** i * d_sol[i] for i in range(len(d_sol)))
    y_qubo = gamma_val + sigma_y * sum(2 ** i * g_sol[i] for i in range(len(g_sol)))

    print(f"QUBO → x: {x_qubo:.4f}, y: {y_qubo:.4f}, z: {z_sol}")

    if np.sum(z_sol) != 1:
        print("Warning: Non one-hot z solution:", z_sol)
        break

    # Получаем выбранный интервал
    i_star = np.argmax(z_sol)
    lb_int = a[i_star]
    ub_int = b[i_star]
    print(f"Chosen interval: [{lb_int}, {ub_int}]")

    # Решаем уточняющую LP-задачу (x - y минимизируется)
    x_var = cp.Variable()
    y_var = cp.Variable()
    constraints = [x_var >= lb_int, y_var <= ub_int]
    lp_obj = cp.Minimize(x_var - y_var)
    lp_prob = cp.Problem(lp_obj, constraints)
    lp_prob.solve()
    x_qp = x_var.value
    y_qp = y_var.value
    print(f"Refined → x: {x_qp:.4f}, y: {y_qp:.4f}")

    # Проверка на сходимость
    if prev_x_qp is not None and abs(x_qp - prev_x_qp) < tol:
        print("Convergence reached.")
        break
    prev_x_qp = x_qp

    # Обновление границ поиска для следующей итерации
    if prev_interval is None or i_star != prev_interval:
        LB_x = a[i_star]
        UB_x = a[i_star] + EPS_INTERVAL_MARGIN
        LB_y = b[i_star] - EPS_INTERVAL_MARGIN
        UB_y = b[i_star]
        print("Interval changed. Updating bounds.")
    else:
        delta_x = 0.1 * (UB_x - LB_x)
        delta_y = 0.1 * (UB_y - LB_y)
        LB_x = max(LB_x, x_qp - delta_x)
        UB_x = min(UB_x, x_qp + delta_x)
        LB_y = max(LB_y, y_qp - delta_y)
        UB_y = min(UB_y, y_qp + delta_y)
        print("Narrowing bounds.")

    prev_interval = i_star

print(f"x* = {x_qp:.4f}")
print(f"y* = {y_qp:.4f}")
print("z* =", z_sol)
