import numpy as np


def calculate_guass_inaccuracies(coef_matrix, b_column, answers):
    A = coef_matrix
    x = answers
    b = b_column

    delta_b = A @ x - b
    delta_x = np.linalg.solve(coef_matrix, delta_b)

    print(
        "Модуль разности вычисленного решения и решения через библиотеку np: " +
        str(np.linalg.norm(np.linalg.solve(A, b) - x, ord=8))
    )

    print(
        "Невязка: " +
        str(np.linalg.norm(delta_x))
    )


def is_positive(matrix):
    for i in range(len(matrix)):

        if np.linalg.det(matrix[:i + 1, :i + 1]) <= 0:
            return False

    return True


def solve_with_gradient(coef_matrix, b_column, epsilon=1e-15, max_iterations=1000):
    # Функция принимает на вход:
    # coef_matrix -- матрица коэффициентов СЛАУ
    # b_column -- столбец свободных членов
    # epsilon -- желаемую точность (по умолчанию 1e-15)
    # max_iterations -- максимальное количество итераций (по умолчанию 1000)
    # Функция возвращает:
    # x_k_1 -- столбец решений СЛАУ
    # iteration_count -- количество итераций

    # Нахождение обратной матрицы

    if not is_positive(coef_matrix):
        raise Exception('Матрица не положительно определена')

    a_matrix = coef_matrix

    a_squared = a_matrix @ a_matrix.T

    n = len(coef_matrix)
    x_k = np.zeros(n)
    x_k_1 = x_k.copy()

    iteration_count = 0

    for j in range(max_iterations):

        iteration_count += 1

        r_k = a_matrix @ x_k - b_column

        a_a_r = a_squared @ r_k

        mu_k = (r_k @ a_a_r) / (a_a_r @ a_a_r)

        x_k_1 = x_k - mu_k * a_matrix.T @ r_k

        if np.linalg.norm(x_k_1 - x_k, ord=2) < epsilon:
            return x_k_1, iteration_count

        x_k = x_k_1.copy()

    return x_k_1, iteration_count


def lab_9(A, b):

    print('Метод градиентного спуска')
    x_simple, iter_count_simple = solve_with_gradient(A, b)

    print('Вектор решений:', x_simple)
    print('Количество итераций:', iter_count_simple)

    calculate_guass_inaccuracies(A, b, x_simple)


if __name__ == '__main__':
    slay_matrix = np.array([
        [3.82, 1.02, 0.75, 0.81],
        [1.05, 4.53, 0.98, 1.53],
        [0.73, 0.85, 4.71, 0.81],
        [0.88, 0.81, 1.28, 3.50]
    ])

    f_column = np.array([15.655, 22.705, 23.480, 16.110])

    lab_9(slay_matrix, f_column)
