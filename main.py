import numpy as np
import cvxpy as cp
from qdeepsdk import QDeepHybridSolver
import requests

# Количество интервалов и границы интервалов
n = 3
a = np.array([1.0, 3.2, 2.7])    # Нижние границы для каждого интервала
b = np.array([2.0, 12.3, 3.0])   # Верхние границы для каждого интервала

# Масштабирующие параметры для двоичного разложения x, y и дополнительных
# (слэк) переменных s, t
sigma_x = 1.0
sigma_y = 1.0
sigma_s = 1.0
sigma_t = 1.0

# Смещения в двоичных разложениях
beta = 0.0
gamma = 0.0
epsilon = 0.0
pi_const = 0.0

# Количество бит, используемых для представления x и y (N+1 бит)
# и для представления дополнительных переменных (M+1 бит)
N = 4  # для x и y
M = 3  # для переменных слэк

# Штрафные коэффициенты для членов штрафной функции QUBO
v1 = 10.0    # Вес для первого штрафного члена: (x - sum(a*z) - s)^2
v2 = 10.0    # Вес для второго штрафного члена: (y - sum(b*z) + t)^2
v3 = 1000.0  # Вес для ограничения one-hot на z: (sum(z) - 1)^2

# Общее число двоичных переменных:
# - n для вектора выбора интервала z
# - 2*(N+1) для двоичных разложений x (переменные d) и y (переменные g)
# - 2*(M+1) для двоичных разложений дополнительных переменных s (переменные e) и t (переменные p)
n_vars = n + 2 * (N + 1) + 2 * (M + 1)

# Инициализируем матрицу QUBO в виде квадратной матрицы, заполненной нулями
Q = np.zeros((n_vars, n_vars))

# Устанавливаем диагональные элементы для двоичного разложения x (переменные d)
# Переменные d располагаются с индексами от n до n+N
for i in range(N + 1):
    idx = n + i
    Q[idx, idx] += 2**i

# Устанавливаем диагональные элементы для двоичного разложения y (переменные g)
# Переменные g располагаются с индексами от n+(N+1) до n+2*(N+1)
for i in range(N + 1):
    idx = n + (N + 1) + i
    Q[idx, idx] += -2**i  # Обратите внимание на минус, он используется для y

# Вспомогательная функция для добавления квадратичного терма в матрицу QUBO


def add_quadratic(Q, i, j, val):
    Q[i, j] += val
    if i != j:
        Q[j, i] += val  # Обеспечиваем симметричность матрицы


# Определяем индексы для различных групп двоичных переменных:
# d: биты для представления x
d_indices = [n + i for i in range(N + 1)]
# z: переменные выбора интервала
z_indices = [i for i in range(n)]
# e: биты для представления дополнительной переменной s
e_indices = [n + 2 * (N + 1) + i for i in range(M + 1)]

# Коэффициенты, используемые в двоичном разложении:
# Коэффициенты для представления x (2^i)
coeff_d = [2**i for i in range(N + 1)]
# Коэффициенты, связанные с нижней границей a для выбора интервала
coeff_z = [-a_i for a_i in a]
# Коэффициенты для представления переменной s (отрицательные)
coeff_e = [-2**i for i in range(M + 1)]

# Объединяем тип переменной, индексы и коэффициенты для первого штрафного
# члена: (x - sum(a*z) - s)^2
indices2 = [('d', d_indices, coeff_d), ('z', z_indices,
                                        coeff_z), ('e', e_indices, coeff_e)]

# Добавляем квадратичные термы для первого штрафного члена,
# масштабированные коэффициентом v1
for typ, idxs, coeffs in indices2:
    for i_idx, coef_i in zip(idxs, coeffs):
        for j_idx, coef_j in zip(idxs, coeffs):
            add_quadratic(Q, i_idx, j_idx, v1 * coef_i * coef_j)

# Добавляем перекрестные термы между разными группами переменных для
# первого штрафного члена
for g1 in indices2:
    typ1, idxs1, coeffs1 = g1
    for g2 in indices2:
        if g1 is g2:
            continue
        typ2, idxs2, coeffs2 = g2
        for i_idx, coef_i in zip(idxs1, coeffs1):
            for j_idx, coef_j in zip(idxs2, coeffs2):
                if i_idx < j_idx:
                    add_quadratic(Q, i_idx, j_idx, 2 * v1 * coef_i * coef_j)

# Определяем индексы для двоичного разложения y (переменные g) и для
# дополнительной переменной t (переменные p)
# g: биты для представления y (следуют за d)
g_indices = [n + (N + 1) + i for i in range(N + 1)]
# p: биты для представления дополнительной переменной t
p_indices = [n + 2 * (N + 1) + (M + 1) + i for i in range(M + 1)]

# Коэффициенты для двоичного разложения y и переменной t:
# Коэффициенты для представления y (2^i)
coeff_g = [2**i for i in range(N + 1)]
# Коэффициенты, связанные с верхней границей b для выбора интервала
coeff_z2 = [-b_i for b_i in b]
# Коэффициенты для представления t (2^i)
coeff_p = [2**i for i in range(M + 1)]

# Объединяем тип переменной, индексы и коэффициенты для второго штрафного
# члена: (y - sum(b*z) + t)^2
indices3 = [('g', g_indices, coeff_g), ('z', z_indices,
                                        coeff_z2), ('p', p_indices, coeff_p)]

# Добавляем квадратичные термы для второго штрафного члена,
# масштабированные коэффициентом v2
for typ, idxs, coeffs in indices3:
    for i_idx, coef_i in zip(idxs, coeffs):
        for j_idx, coef_j in zip(idxs, coeffs):
            add_quadratic(Q, i_idx, j_idx, v2 * coef_i * coef_j)

# Добавляем перекрестные термы между разными группами переменных для
# второго штрафного члена
for g1 in indices3:
    typ1, idxs1, coeffs1 = g1
    for g2 in indices3:
        if g1 is g2:
            continue
        typ2, idxs2, coeffs2 = g2
        for i_idx, coef_i in zip(idxs1, coeffs1):
            for j_idx, coef_j in zip(idxs2, coeffs2):
                if i_idx < j_idx:
                    add_quadratic(Q, i_idx, j_idx, 2 * v2 * coef_i * coef_j)

# Добавляем штрафные термы для ограничения one-hot на z (выраженного через
# (sum(z) - 1)^2)
for i in z_indices:
    add_quadratic(Q, i, i, v3)
for i in z_indices:
    for j in z_indices:
        if i < j:
            add_quadratic(Q, i, j, 2 * v3)
# Корректируем диагональные элементы для завершения разложения (вычитая
# 2*v3 для каждой переменной z)
for i in z_indices:
    Q[i, i] += -2 * v3

# Регулируем диагональные элементы для переменных d (разложение x) и g
# (разложение y)
for i, coef in zip(d_indices, coeff_d):
    Q[i, i] += coef
for i, coef in zip(g_indices, coeff_g):
    Q[i, i] += -coef

# Инициализируем и настраиваем QDeepHybridSolver
solver = QDeepHybridSolver()
solver.token = "YOUR-API-TOKEN"
solver.m_budget = 10 ** 20  # Устанавливаем очень большое значение бюджета измерений
solver.num_reads = 100      # Количество запусков (reads) для решателя

# Пытаемся решить задачу QUBO с помощью QDeepHybridSolver
try:
    response = solver.solve(Q)
    results = response['QdeepHybridSolver']['configuration']
    print("Hybrid Solver Results:", results)
except ValueError as e:
    print(f"Error: {e}")
except requests.RequestException as e:
    print(f"API Error: {e}")

# Преобразуем полученную конфигурацию в массив numpy
solution = np.array(results)

# Извлекаем подмножества переменных из решения для z, d и g.
z_sol = solution[z_indices].round().astype(int)
d_sol = solution[d_indices].round().astype(int)
g_sol = solution[g_indices].round().astype(int)

# Восстанавливаем непрерывную переменную x из двоичного представления с
# помощью d_sol
x_qubo = beta + sigma_x * sum((2**i) * d_sol[i] for i in range(len(d_sol)))
# Восстанавливаем непрерывную переменную y из двоичного представления с
# помощью g_sol
y_qubo = gamma + sigma_y * sum((2**i) * g_sol[i] for i in range(len(g_sol)))

# Выводим восстановленные значения x и y из решения QUBO
print("x (from QUBO) =", x_qubo)
print("y (from QUBO) =", y_qubo)

# Выводим вектор выбора интервала z (желательно, чтобы он был one-hot)
print("Chosen interval z* =", z_sol)
# Вычисляем и выводим нижнюю границу выбранного интервала как скалярное
# произведение a и z_sol
print("Interval lower bound =", np.dot(a, z_sol))
# Вычисляем и выводим верхнюю границу выбранного интервала как скалярное
# произведение b и z_sol
print("Interval upper bound =", np.dot(b, z_sol))

# Последующий этап уточнения с использованием линейного программирования (LP)
# Определяем непрерывные переменные для LP: x_var и y_var
x_var = cp.Variable()
y_var = cp.Variable()

# Устанавливаем ограничения LP: x_var должно быть не меньше нижней границы,
# а y_var не больше верхней границы выбранного интервала
constraints = [x_var >= np.dot(a, z_sol), y_var <= np.dot(b, z_sol)]
# Определяем цель LP как минимизацию разности (x_var - y_var)
lp_obj = cp.Minimize(x_var - y_var)
lp_prob = cp.Problem(lp_obj, constraints)
lp_prob.solve()

# Выводим итоговое решение LP для x и y
print("Final LP solution:")
print("x* =", x_var.value)
print("y* =", y_var.value)
print("Objective (x* - y*) =", x_var.value - y_var.value)
