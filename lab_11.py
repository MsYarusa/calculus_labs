import numpy as np
from lab_10 import rotation_method
from helpers.Inaccuracies import calculate_guass_inaccuracies


def is_positive_definite(matrix):
    if not np.allclose(matrix, matrix.T):
        raise ValueError("Матрица не симметричная")

    try:
        for i in range(1, matrix.shape[0] + 1):
            minor = matrix[:i, :i]
            if np.linalg.det(minor) <= 0:
                raise ValueError(f"Определитель минорной матрицы размера {i}x{i} <= 0.")

        return True

    except np.linalg.LinAlgError:
        raise ValueError("Ошибка при вычислении определителей")


def richardson_method(matrix, b_column, n=10, epsilon=1e-15, max_iter=1000):

    is_positive_definite(matrix)

    eigenvalues, _ = rotation_method(matrix)

    print("Собственные значения, найденные методом вращений с преградами:")
    print(eigenvalues)

    max_eigenvalue = np.max(eigenvalues)
    min_eigenvalue = np.min(eigenvalues)

    tau_0 = 2 / (min_eigenvalue + max_eigenvalue)

    eta = min_eigenvalue / max_eigenvalue

    ro_0 = (1 - eta) / (1 + eta)

    size = len(eigenvalues)

    x_k = np.zeros(size)

    x_k_1 = np.zeros(size)

    number_of_iterations = 0

    for i in range(max_iter):

        for k in range(n):
            x_k = x_k_1.copy()

            nu_k = np.cos((2 * k - 1) * np.pi / (2 * n))

            tau_k = tau_0 / (1 + ro_0 * nu_k)

            x_k_1 = (b_column - matrix @ x_k) * tau_k + x_k

            number_of_iterations += 1

        if np.linalg.norm(x_k_1 - x_k) < epsilon:
            break

    return x_k, number_of_iterations


def lab_11(matrix, b_column):
    print('Метод Ричардсона')
    solution, iter_count_simple = richardson_method(matrix, b_column)

    print('Вектор решений:', solution)
    print('Количество итераций:', iter_count_simple)

    calculate_guass_inaccuracies(matrix, b_column, solution)
