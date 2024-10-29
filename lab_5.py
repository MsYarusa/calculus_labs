import numpy as np
from helpers.Inaccuracies import calculate_guass_inaccuracies


ERROR_CODE = 'nonsymmetric'


def solve_with_square_root(coef_matrix, b_column):
    # Функция принимает на вход:
    # coef_matrix -- матрица коэффициентов СЛАУ
    # b_column -- столбец свободных членов
    # Функция возвращает:
    # x -- столбец решений СЛАУ

    if not np.array_equal(coef_matrix.T, coef_matrix):
        print("Матрица несимметрична")

        return ERROR_CODE

    coef_matrix = coef_matrix + 0j

    s_matrix = np.zeros(shape=coef_matrix.shape, dtype=complex)

    n = len(coef_matrix)

    s_matrix[0, 0] = coef_matrix[0, 0]**(1/2)
    s_matrix[0, 1:n] = coef_matrix[0, 1:n] / s_matrix[0, 0]

    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                second_summand = - np.sum(s_matrix[:i, i]**2)

                s_matrix[i, j] = (coef_matrix[i, j] + second_summand)**(1/2)

            if j > i:
                second_summand = - np.sum(s_matrix[:i, i] * s_matrix[:i, j])

                s_matrix[i, j] = (coef_matrix[i, j] + second_summand) / s_matrix[i, i]

    print(s_matrix)
    print(np.real(s_matrix.T @ s_matrix))
    print(np.real(coef_matrix))

    y_vector = np.zeros(n, dtype=complex)
    y_vector[0] = b_column[0] / s_matrix[0, 0]

    for i in range(1, n):
        y_second_summand = np.sum([s_matrix[k - 1, i] * y_vector[k - 1] for k in range(1, i + 1)])
        y_vector[i] = (b_column[i] - y_second_summand) / s_matrix[i, i]

    x_vector = np.zeros(n) + 0j
    x_vector[n - 1] = y_vector[n - 1] / s_matrix[n-1, n-1]

    print(y_vector)

    for i in range(2, n + 1):
        x_second_summand = - np.sum([s_matrix[n-i, k] * x_vector[k] for k in range(n - i + 1, n)])
        x_vector[n-i] = (y_vector[n-i] + x_second_summand) / s_matrix[n - i, n - i]

    return np.real(x_vector)


def lab_5(A, b):
    x = solve_with_square_root(A, b)

    if type(x) is not np.ndarray and x == ERROR_CODE:

        return

    if type(x) is not np.ndarray:
        print(
            "Матрица вырождена, точного решения не существует"
        )

        return

    print(
        "Вычисленное решение СЛАУ: ", x
    )

    calculate_guass_inaccuracies(A, b, x)
