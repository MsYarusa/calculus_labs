import numpy as np
from helpers.Inaccuracies import calculate_guass_inaccuracies


def solve_with_framing(coef_matrix, b_column):
    # Функция принимает на вход:
    # coef_matrix -- матрица коэффициентов СЛАУ
    # b_column -- столбец свободных членов
    # Функция возвращает:
    # x -- столбец решений СЛАУ

    # Нахождение обратной матрицы

    n = len(coef_matrix)
    inverse_of_matrices = [np.zeros(shape=(i, i)) for i in range(1, n+1)]

    inverse_of_matrices[0][0, 0] = 1 / coef_matrix[0, 0]

    for k in range(1, n):
        print('Обратная матрица на шаге', k-1)
        print(inverse_of_matrices[k - 1])

        u_column = - inverse_of_matrices[k-1] @ coef_matrix[:k, k]
        v_row = - coef_matrix[k, :k] @ inverse_of_matrices[k-1]

        print('u, v')
        print(u_column)
        print(v_row)

        alpha_u_summand = coef_matrix[k, :k] @ u_column[:]
        alpha_u = coef_matrix[k, k] + alpha_u_summand

        alpha_v_summand = coef_matrix[:k, k] @ v_row[:]
        alpha_v = coef_matrix[k, k] + alpha_v_summand

        print('Модуль разности alpha_v и alpha_u на шаге', k, 'равен:', abs(alpha_v - alpha_u))

        alpha = (alpha_u + alpha_v) / 2

        print(alpha)

        inverse_of_matrices[k][:k, :k] = inverse_of_matrices[k-1] + np.outer(u_column, v_row) / alpha
        inverse_of_matrices[k][:k, k] = u_column / alpha
        inverse_of_matrices[k][k, :k] = v_row / alpha
        inverse_of_matrices[k][k, k] = 1 / alpha

    inverse_of_matrix = inverse_of_matrices[n-1]

    print('Сравнение обратных матриц полученных нами и встроенным методом')
    print(np.abs(inverse_of_matrix - np.linalg.inv(coef_matrix)))
    print(np.linalg.inv(coef_matrix))

    x_vector = inverse_of_matrix @ b_column

    return x_vector


def lab_6(A, b):
    x = solve_with_framing(A, b)

    if type(x) is not np.ndarray:
        print(
            "Матрица вырождена, точного решения не существует"
        )

        return

    print(
        "Вычисленное решение СЛАУ: ", x
    )

    calculate_guass_inaccuracies(A, b, x)
