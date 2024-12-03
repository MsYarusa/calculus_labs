import numpy as np
from helpers.Inaccuracies import calculate_guass_inaccuracies


def solve_with_reflection(coef_matrix, b_column):
    # Функция принимает на вход:
    # coef_matrix -- матрица коэффициентов СЛАУ
    # b_column -- столбец свободных членов
    # Функция возвращает:
    # x -- столбец решений СЛАУ

    # Нахождение обратной матрицы

    n = len(coef_matrix)
    a_matrix = np.copy(coef_matrix)
    f_vector = np.copy(b_column)

    for k in range(n):
        sigma_k = 1 if a_matrix[k, k] >= 0 else -1

        norms_of_a_k = np.sqrt(np.sum(a_matrix[k:n, k] ** 2))

        p_k = a_matrix[k, k] + sigma_k * norms_of_a_k
        p_l = a_matrix[k+1:n, k]

        p_vector = np.zeros(n)
        p_vector[k] = p_k
        p_vector[k+1:] = p_l

        sigma = np.copy(a_matrix)

        p_matrix = sigma - 2 * np.outer(p_vector, p_vector) / np.sum(p_vector ** 2)

        a_matrix = p_matrix @ a_matrix

        f_vector = p_matrix @ a_matrix

    # print(p_matrix)
    print(a_matrix)
    # print(np.linalg.inv(p_matrix) @ a_matrix)
    print(coef_matrix)
    # print(np.max(abs(np.linalg.inv(p_matrix) @ a_matrix - coef_matrix)))
    print(f_vector)

    x_vector = np.zeros(n)

    for i in range(n-1, -1, -1):
        x_vector[i] = (f_vector[i] - np.sum(a_matrix[i, i+1:] * x_vector[i+1:])) / a_matrix[i, i]

    return x_vector


def lab_7(A, b):
    x = solve_with_reflection(A, b)

    if type(x) is not np.ndarray:
        print(
            "Точного решения не существует"
        )

        return

    print(
        "Вычисленное решение СЛАУ: ", x
    )

    calculate_guass_inaccuracies(A, b, x)
