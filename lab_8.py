import numpy as np
from helpers.Inaccuracies import calculate_guass_inaccuracies


def check_diagonal_domination(matrix):
    n = len(matrix)

    for i in range(n):
        row_sum_except_diagonal = np.sum(np.abs(matrix[i, 1:i])) + np.sum(np.abs(matrix[i, i+1:n]))

        if abs(matrix[i, i]) <= row_sum_except_diagonal:
            return False

    return True


def simple_iteration_method(coef_matrix, b_column, epsilon=1e-3):
    # Функция принимает на вход:
    # coef_matrix -- матрица коэффициентов СЛАУ
    # b_column -- столбец свободных членов
    # epsilon -- желаемую точность
    # Функция возвращает:
    # x_k_1 -- столбец решений СЛАУ
    # iteration_count -- количество итераций

    # Нахождение обратной матрицы

    if not check_diagonal_domination(coef_matrix):
        raise ValueError("Матрица коэффициентов не имеет диагонального преобладания")

    a_matrix = coef_matrix

    n = len(coef_matrix)
    x_k = np.zeros(n)
    x_k_1 = x_k

    iteration_count = 0

    for j in range(10000):

        x_k_1 = x_k.copy()

        iteration_count += 1

        for i in range(n):
            x_k_1[i] = (b_column[i] - np.sum(a_matrix[i, 0:n] * x_k[0:n])) / a_matrix[i, i] + x_k[i]

        if np.linalg.norm(x_k_1 - x_k, ord=2) < epsilon:
            return x_k_1, iteration_count

        x_k = x_k_1

    return x_k_1, iteration_count


def solve_with_relaxation(coef_matrix, b_column, omega, epsilon=1e-3):
    # Функция принимает на вход:
    # coef_matrix -- матрица коэффициентов СЛАУ
    # b_column -- столбец свободных членов
    # omega -- параметр метода релаксации
    # epsilon -- желаемую точность
    # Функция возвращает:
    # x_k_1 -- столбец решений СЛАУ
    # iteration_count -- количество итераций

    # Нахождение обратной матрицы

    if not check_diagonal_domination(coef_matrix):
        raise ValueError("Матрица коэффициентов не имеет диагонального преобладания")

    a_matrix = coef_matrix

    n = len(coef_matrix)
    x_k = np.zeros(n)
    x_k_1 = x_k

    iteration_count = 0

    for j in range(10000):

        x_k_1 = x_k.copy()

        iteration_count += 1

        for i in range(n):
            a_x_k_1 = omega * np.sum(a_matrix[i, 0:i] * x_k_1[0:i])
            d_x_k = (1 - omega) * x_k[i]
            a_x_k = omega * np.sum(a_matrix[i, i+1:n] * x_k[i+1:n])

            x_k_1[i] = (omega * b_column[i] - a_x_k_1 - a_x_k) / a_matrix[i, i] + d_x_k

        if np.linalg.norm(x_k_1 - x_k, ord=2) < epsilon:
            return x_k_1, iteration_count

        x_k = x_k_1

    return x_k_1, iteration_count


def lab_8(A, b):

    print('Метод простой итерации')
    x_simple, iter_count_simple = simple_iteration_method(A, b)

    print('Вектор решений:', x_simple)
    print('Количество итераций:', iter_count_simple)

    calculate_guass_inaccuracies(A, b, x_simple)

    print('Метод релаксации')
    for omega in [0.01, 0.5, 1, 1.5, 1.99]:
        x, iter_count = solve_with_relaxation(A, b, omega)

        print('Для значения omega:', omega, 'получены результаты')
        print('Вектор решений:', x)
        print('Количество итераций:', iter_count)

        calculate_guass_inaccuracies(A, b, x)
