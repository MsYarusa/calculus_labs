import numpy as np


def find_abs_max_non_diag_indexes(matrix):
    matrix_non_diag = matrix.copy()
    np.fill_diagonal(matrix_non_diag, 0)

    max_index = np.argmax(abs(matrix_non_diag))

    # в случае если матрица нулевая
    if max_index == 0:
        return 0, 1

    return np.unravel_index(max_index, matrix.shape)


def calculate_sigma(max_abs_elem, sigma_level):
    return np.sqrt(abs(max_abs_elem)) * 10**(-sigma_level)


def sign(number):
    if number < 0:
        return -1
    else:
        return 1


def rotation_method(matrix, p=5, max_iter=1000):
    # Проверка, что симметричная невырожденная

    n = len(matrix)

    a = matrix.copy()
    sigma_level = 1
    (i, j) = find_abs_max_non_diag_indexes(a)

    number_of_iterations = 0

    while sigma_level <= p and number_of_iterations < max_iter:
        d = np.sqrt((a[i, i] - a[j, j]) ** 2 + 4 * a[i, j] ** 2)

        diag_elem_diff = abs(a[i, i] - a[j, j]) / d

        c = np.sqrt(1/2 * (1 + diag_elem_diff))

        s_coef = sign(a[i, j] * (a[i, i] - a[j, j]))
        s = s_coef * np.sqrt(1 / 2 * (1 - diag_elem_diff))

        a_1 = a.copy()

        a_1[0:n, i] = c * a[0:n, i] + s * a[0:n, j]
        a_1[i, 0:n] = a_1[0:n, i]

        a_1[0:n, j] = - s * a[0:n, i] + c * a[0:n, j]
        a_1[j, 0:n] = a_1[0:n, j]

        a_1[i, i] = c**2 * a[i, i] + 2 * c * s * a[i, j] + s**2 * a[j, j]
        a_1[j, j] = s**2 * a[i, i] - 2 * c * s * a[i, j] + c**2 * a[j, j]

        a_1[i, j] = a_1[j, i] = 0.0

        a = a_1

        (i, j) = find_abs_max_non_diag_indexes(a)
        sigma = calculate_sigma(a[i, j], sigma_level)

        while (abs(a[i, j]) <= sigma) and (sigma_level <= p):
            sigma_level += 1
            sigma = calculate_sigma(a[i, j], sigma_level)

        number_of_iterations += 1

    eigenvalues = np.diagonal(a)

    return eigenvalues, number_of_iterations


def lab_10(coef_matrix):
    print('Метод вращения с преградами')
    calculated_eigenvalues, number_of_iterations = rotation_method(coef_matrix)

    print(f'Вычисленные за {number_of_iterations} итераций собственные значения:')
    print(calculated_eigenvalues)

    eigenvalues, eigenvectors = np.linalg.eig(coef_matrix)

    print('Разность данных собственных значений и посчитанных:')
    print(abs(np.sort(calculated_eigenvalues) - np.sort(eigenvalues)))


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
        [1.0, 3.0],
        [3.0, 4.0]
    ])

    lab_10(A_1)
