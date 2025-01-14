import numpy as np


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def simple_iteration_method(matrix, epsilon=1e-15, max_iter=1000):
    n = matrix.shape[0]
    x_0 = np.ones(n)

    x_k = x_0 / np.linalg.norm(x_0, ord=2)
    eigenvalue = 0

    number_of_iterations = 0

    for k in range(max_iter):
        y_k_1 = matrix @ x_k
        eigenvalue = y_k_1 @ x_k
        x_k_1 = y_k_1 / np.linalg.norm(y_k_1, ord=2)

        number_of_iterations += 1

        if np.linalg.norm(sign(eigenvalue) * x_k_1 - x_k) < epsilon:
            x_k = x_k_1

            break

        x_k = x_k_1

    eigenvector = x_k

    return eigenvalue, eigenvector, number_of_iterations


def use_lu_decomposition(coef_matrix):
    b_matrix = np.zeros(shape=coef_matrix.shape)
    c_matrix = np.zeros(shape=coef_matrix.shape)

    b_matrix[:, 0] = coef_matrix[:, 0]
    c_matrix[0, :] = coef_matrix[0, :] / b_matrix[0, 0]

    n = len(coef_matrix)

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i >= j > 1:
                b_second_summand = np.sum([b_matrix[i - 1, k - 1] * c_matrix[k - 1, j - 1] for k in range(1, j)])

                b_matrix[i - 1, j - 1] = coef_matrix[i - 1, j - 1] - b_second_summand

            if 1 < i < j:
                c_second_summand = np.sum([b_matrix[i - 1, k - 1] * c_matrix[k - 1, j - 1] for k in range(1, i)])

                c_matrix[i - 1, j - 1] = (coef_matrix[i - 1, j - 1] - c_second_summand) / b_matrix[i - 1, i - 1]

            c_matrix[i - 1, i - 1] = 1

    return b_matrix, c_matrix


def solve_using_lu_decomposition(coef_matrix, b_column):
    n = len(b_column)

    b_matrix, c_matrix = use_lu_decomposition(coef_matrix)

    y_vector = np.zeros(n)
    y_vector[0] = b_column[0] / b_matrix[0, 0]

    for i in range(1, n):
        y_second_summand = np.sum([b_matrix[i, k - 1] * y_vector[k - 1] for k in range(1, i + 1)])
        y_vector[i] = (b_column[i] - y_second_summand) / b_matrix[i, i]

    x_vector = np.zeros(n)
    x_vector[n - 1] = y_vector[n - 1]

    for i in range(2, n + 1):
        x_second_summand = np.sum([c_matrix[-i, k - 1] * x_vector[k - 1] for k in range(n - i + 1, n + 1)])
        x_vector[-i] = y_vector[-i] - x_second_summand

    return x_vector


def reversed_iteration_method(matrix, epsilon=1e-15, max_iter=1000):
    n = matrix.shape[0]
    x_0 = np.ones(n)

    x_k = x_0 / np.linalg.norm(x_0, ord=2)

    alpha_k = np.linalg.norm(x_k, ord=np.inf)

    number_of_iterations = 0

    for i in range(max_iter):

        number_of_iterations += 1

        x_k = solve_using_lu_decomposition(matrix, x_k / alpha_k)

        alpha_k_1 = np.linalg.norm(x_k, ord=np.inf)

        if abs(1 / alpha_k_1 - 1 / alpha_k) < epsilon:
            break

        alpha_k = alpha_k_1

    eigenvalue = 1 / alpha_k
    eigenvector = x_k

    return eigenvalue, eigenvector, number_of_iterations


def lab_12(matrix):
    print('Метод простой итерации')
    print()

    eigenvalue, eigenvector, number_of_iterations = simple_iteration_method(matrix)

    print('Максимальное по модулю собственное значение матрицы А:', eigenvalue)

    print('Соответствующий собственный вектор:')
    print(eigenvector)

    print('Количество итераций:', number_of_iterations)

    print()

    print('Проверка:')

    np_eigenvalues, np_eigenvectors = np.linalg.eig(matrix)
    np_max_eig_index = np.argmax(abs(np_eigenvalues))
    np_max_eigenvalue = np_eigenvalues[np_max_eig_index]

    print('Вычислительная погрешность:', np.linalg.norm(matrix @ eigenvector - eigenvalue * eigenvector))

    print('Разница вычисленных значений со значениями полученными методом numpy')
    print('Максимальное по модулю собственное число:', np.linalg.norm(np_max_eigenvalue - eigenvalue))

    print()
    print()


def lab_13(matrix):
    print('Метод обратной итерации')
    print()

    eigenvalue, eigenvector, number_of_iterations = reversed_iteration_method(matrix)

    print('Минимальное по модулю собственное значение матрицы А:', eigenvalue)

    print('Соответствующий собственный вектор:')
    print(eigenvector)

    print('Количество итераций:', number_of_iterations)

    print()

    print('Проверка:')

    np_eigenvalues, np_eigenvectors = np.linalg.eig(matrix)
    np_min_eig_index = np.argmin(abs(np_eigenvalues))
    np_min_eigenvalue = np_eigenvalues[np_min_eig_index]

    print('Вычислительная погрешность:', np.linalg.norm(matrix @ eigenvector - eigenvalue * eigenvector))

    print('Разница вычисленных значений со значениями полученными методом numpy')
    print('Минимальное по модулю собственное число:', np.linalg.norm(np_min_eigenvalue - eigenvalue))
    print()
    print()


if __name__ == '__main__':
    A_1 = np.array([
        [-0.168700, 0.353699, 0.008540, 0.733624],
        [0.353699, 0.056519, -0.723182, -0.076440],
        [0.008540, -0.723182, 0.015938, 0.342333],
        [0.733624, -0.076440, 0.342333, -0.045744]
    ])

    l_1 = np.array([-0.943568, -0.744036, 0.687843, 0.857774])

    A_2 = np.array([
        [2.2, 1, 0.5, 2],
        [1, 1.3, 2, 1],
        [0.5, 2, 0.5, 1.6],
        [2, 1, 1.6, 2]
    ])

    l_2 = np.array([5.652, 1.545, -1.420, 0.2226])

    A_3 = np.array([
        [1, 0.42, 0.54, 0.66],
        [0.42, 1, 0.32, 0.44],
        [0.54, 0.32, 1, 0.22],
        [0.66, 0.44, 0.22, 1]
    ])

    l_3 = np.array([15.655, 22.705, 23.480, 16.110])

    A_4 = np.array([
        [2.0, 1.0],
        [1.0, 2.0]
    ])

    lab_12(A_2)
    lab_13(A_2)
