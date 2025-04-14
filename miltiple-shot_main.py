# Incorrect for now

import numpy as np
import cvxpy as cp
from qdeepsdk import QDeepHybridSolver
import requests
import sympy as sp

# ----------------------- Исходные данные -----------------------
n = 3
# Интервальные границы (каждый интервал определяется парой [a[i], b[i]])
a = np.array([1.0, 3.2, 1.2])  # Нижние границы интервалов
b = np.array([10.4, 12.2, 11.2])  # Верхние границы интервалов

# Начальные диапазоны для непрерывных переменных x и y
LB_x = 1.0  # начальное нижнее значение для x
UB_x = 12.3  # начальное верхнее значение для x
LB_y = 1.0  # начальное нижнее значение для y
UB_y = 12.3  # начальное верхнее значение для y

# Параметры для слэк-переменных (фиксированные)
sigma_s = 1.0
sigma_t = 1.0
epsilon = 0.0
pi_const = 0.0

# Количество бит для квантования x и y, а также слэк-переменных
N = 4  # x и y представлены N+1 битами
M = 3  # слэк-переменные представлены M+1 битами

# Штрафные коэффициенты для членов целевой функции
v1 = 10.0  # Штраф за (x - sum(a*z) - s)^2
v2 = 10.0  # Штраф за (y - sum(b*z) + t)^2
v3 = 1e6  # Очень высокий штраф для ограничения one-hot: (sum(z)-1)^2

# Параметры итераций
max_iter = 10
tol = 1e-3
prev_x_qp = None
prev_interval = None  # Для отслеживания изменения выбранного интервала


# ---------------- Функция построения QUBO через символьные вычисления ----------------
def build_qubo(beta, sigma_x, gamma, sigma_y):
    # Определяем символьные бинарные переменные:
    # z: выбор интервала (n штук)
    # d: биты для x (N+1)
    # g: биты для y (N+1)
    # e: биты для слэк s (M+1)
    # p: биты для слэк t (M+1)
    z_vars = sp.symbols('z0:%d' % n, integer=True)
    d_vars = sp.symbols('d0:%d' % (N + 1), integer=True)
    g_vars = sp.symbols('g0:%d' % (N + 1), integer=True)
    e_vars = sp.symbols('e0:%d' % (M + 1), integer=True)
    p_vars = sp.symbols('p0:%d' % (M + 1), integer=True)
    vars_order = list(z_vars) + list(d_vars) + list(g_vars) + list(e_vars) + list(p_vars)

    # Представление x через двоичное разложение: x = beta + sigma_x * Σ(2^i * d_i)
    x_expr = beta + sigma_x * sp.Add(*[2 ** i * d_vars[i] for i in range(N + 1)])
    # Представление y через двоичное разложение: y = gamma + sigma_y * Σ(2^i * g_i)
    y_expr = gamma + sigma_y * sp.Add(*[2 ** i * g_vars[i] for i in range(N + 1)])
    # Слэк-переменная s
    s_expr = epsilon + sigma_s * sp.Add(*[2 ** i * e_vars[i] for i in range(M + 1)])
    # Слэк-переменная t
    t_expr = pi_const + sigma_t * sp.Add(*[2 ** i * p_vars[i] for i in range(M + 1)])

    # Суммы с коэффициентами по интервалам:
    A_expr = sp.Add(*[a[i] * z_vars[i] for i in range(n)])
    B_expr = sp.Add(*[b[i] * z_vars[i] for i in range(n)])
    z_sum = sp.Add(*z_vars)  # Ограничение: сумма z должна равняться 1 (one-hot)

    # Формирование символьного выражения целевой функции QUBO:
    # cost = (x - y) + v1*(x - A_expr - s)^2 + v2*(y - B_expr + t)^2 + v3*(z_sum - 1)^2
    cost_expr = (x_expr - y_expr) + \
                v1 * (x_expr - A_expr - s_expr) ** 2 + \
                v2 * (y_expr - B_expr + t_expr) ** 2 + \
                v3 * (z_sum - 1) ** 2
    cost_expanded = sp.expand(cost_expr)
    poly_cost = sp.Poly(cost_expanded, vars_order)
    terms = poly_cost.as_dict()

    num_vars = len(vars_order)
    Q_matrix = np.zeros((num_vars, num_vars))
    # Распределяем коэффициенты мономов по матрице Q:
    for monom, coef in terms.items():
        degree = sum(monom)
        if degree == 0:
            continue
        elif degree == 1:
            i = monom.index(1)
            Q_matrix[i, i] += float(coef)
        elif degree == 2:
            indices = [i for i, exp in enumerate(monom) if exp > 0]
            exponents = [exp for exp in monom if exp > 0]
            if len(indices) == 1 and exponents[0] == 2:
                i = indices[0]
                Q_matrix[i, i] += float(coef)
            elif len(indices) == 2 and all(exp == 1 for exp in exponents):
                i, j = indices
                Q_matrix[i, j] += float(coef) / 2.0
                Q_matrix[j, i] += float(coef) / 2.0
            else:
                raise ValueError("Неправильный моном степени 2: " + str(monom))
        else:
            raise ValueError("Обнаружен моном степени > 2: " + str(monom))
    return Q_matrix, vars_order


# ---------------- Итерационный цикл ----------------
for it in range(max_iter):
    print(f"\nИтерация {it}:")
    # Обновляем quantization-параметры для x и y с учётом текущих границ
    # Вычисляем sigma так, чтобы при всех единицах двоичного представления получалось UB_x
    sigma_x = (UB_x - LB_x) / (2 ** (N + 1) - 1)
    sigma_y = (UB_y - LB_y) / (2 ** (N + 1) - 1)
    beta_val = LB_x
    gamma_val = LB_y

    # Формируем QUBO-матрицу через символьные вычисления
    Q_matrix, vars_order = build_qubo(beta_val, sigma_x, gamma_val, sigma_y)

    # Решаем QUBO
    solver = QDeepHybridSolver()
    solver.token = "mtagdfsplb"
    solver.m_budget = 10 ** 8
    solver.num_reads = 100
    try:
        response = solver.solve(Q_matrix)
        results = response['QdeepHybridSolver']
        print("QUBO solution:", results)
    except (ValueError, requests.RequestException) as e:
        print("Ошибка при решении QUBO:", e)
        break

    solution = np.array(results['configuration'])
    # Извлекаем переменные согласно порядку: сначала z (n), потом d (N+1), потом g (N+1)
    z_sol = solution[0:n].round().astype(int)
    d_sol = solution[n: n + (N + 1)].round().astype(int)
    g_sol = solution[n + (N + 1): n + 2 * (N + 1)].round().astype(int)

    # Вычисляем значения x и y по двоичным разложениям
    x_qubo = beta_val + sigma_x * sum((2 ** i) * d_sol[i] for i in range(len(d_sol)))
    y_qubo = gamma_val + sigma_y * sum((2 ** i) * g_sol[i] for i in range(len(g_sol)))
    print("x (from QUBO) =", x_qubo)
    print("y (from QUBO) =", y_qubo)
    print("Chosen interval z* =", z_sol)
    lb_int = np.dot(a, z_sol)
    ub_int = np.dot(b, z_sol)
    print("Interval lower bound =", lb_int)
    print("Interval upper bound =", ub_int)

    # Этап уточнения: фиксируем z и решаем LP для обновления x и y
    x_var = cp.Variable()
    y_var = cp.Variable()
    constraints = [x_var >= lb_int, y_var <= ub_int]
    lp_obj = cp.Minimize(x_var - y_var)
    lp_prob = cp.Problem(lp_obj, constraints)
    lp_prob.solve()
    x_qp = x_var.value
    y_qp = y_var.value
    print("QP (LP) solution: x =", x_qp, ", y =", y_qp)

    # Если сходимость достигнута, выходим из цикла
    if prev_x_qp is not None and abs(x_qp - prev_x_qp) < tol:
        print("Достигнута сходимость: значения x не изменились существенно.")
        break
    prev_x_qp = x_qp

    # Обновляем границы quantization.
    # Если выбранный интервал изменился (one-hot z, т.е. индекс, для которого z_sol==1),
    # то устанавливаем новые границы равными границам этого интервала.
    indices = np.where(z_sol == 1)[0]
    if len(indices) != 1:
        print("Ожидалось one-hot решение для z, получено:", z_sol)
        break
    i_star = indices[0]
    if prev_interval is None or i_star != prev_interval:
        # Обновляем границы в соответствии с выбранным интервалом
        LB_x = a[i_star]
        UB_x = a[i_star]
        LB_y = b[i_star]
        UB_y = b[i_star]
        print(f"Обновляем границы: для x [{LB_x}, {UB_x}], для y [{LB_y}, {UB_y}] (смена интервала)")
    else:
        # Если интервал не изменился, можно ужать границы вокруг полученного решения
        delta_x = 0.1 * (UB_x - LB_x)  # пример: уменьшаем диапазон на 10%
        delta_y = 0.1 * (UB_y - LB_y)
        LB_x = max(LB_x, x_qp - delta_x)
        UB_x = min(UB_x, x_qp + delta_x)
        LB_y = max(LB_y, y_qp - delta_y)
        UB_y = min(UB_y, y_qp + delta_y)
        print(f"Ужимаем границы: для x [{LB_x}, {UB_x}], для y [{LB_y}, {UB_y}]")
    prev_interval = i_star

print("\nИтоговое решение:")
print("x* =", x_qp)
print("y* =", y_qp)
print("z* =", z_sol)
