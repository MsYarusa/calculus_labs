import numpy as np
from helpers.Inaccuracies import calculate_guass_inaccuracies


def solve_with_lu_decomposition(coef_matrix, b_column):
    # Функция принимает на вход:
    # coef_matrix -- матрица коэффициентов СЛАУ
    # b_column -- столбец свободных членов
    # Функция возвращает:
    # x -- столбец решений СЛАУ

    b_matrix = np.zeros(shape=coef_matrix.shape)
    c_matrix = np.zeros(shape=coef_matrix.shape)

    b_matrix[:, 0] = coef_matrix[:, 0]
    c_matrix[0, :] = coef_matrix[0, :] / b_matrix[0, 0]

    n = len(coef_matrix)

    for i in range(1, n+1):
        for j in range(1, n+1):
            if i >= j > 1:
                b_second_summand = np.sum([b_matrix[i - 1, k - 1]*c_matrix[k - 1, j - 1] for k in range(1, j)])

                b_matrix[i - 1, j - 1] = coef_matrix[i - 1, j - 1] - b_second_summand

            if 1 < i < j:
                c_second_summand = np.sum([b_matrix[i - 1, k - 1]*c_matrix[k - 1, j - 1] for k in range(1, i)])

                c_matrix[i - 1, j - 1] = (coef_matrix[i - 1, j - 1] - c_second_summand) / b_matrix[i - 1, i - 1]

            c_matrix[i - 1, i - 1] = 1

    print(b_matrix)
    print(c_matrix)
    print(b_matrix @ c_matrix)
    print(coef_matrix)

    y_vector = np.zeros(n)
    y_vector[0] = b_column[0] / b_matrix[0, 0]

    for i in range(1, n):
        y_second_summand = np.sum([b_matrix[i, k - 1]*y_vector[k - 1] for k in range(1, i+1)])
        y_vector[i] = (b_column[i] - y_second_summand) / b_matrix[i, i]

    x_vector = np.zeros(n)
    x_vector[n-1] = y_vector[n-1]

    for i in range(2, n+1):
        x_second_summand = np.sum([c_matrix[-i, k - 1] * x_vector[k - 1] for k in range(n-i+1, n+1)])
        x_vector[-i] = y_vector[-i] - x_second_summand

    return x_vector


def lab_4(A, b):
    x = solve_with_lu_decomposition(A, b)

    if type(x) is not np.ndarray:
        print(
            "Матрица вырождена, точного решения не существует"
        )

        return

    print(
        "Вычисленное решение СЛАУ: ", x
    )

    calculate_guass_inaccuracies(A, b, x)