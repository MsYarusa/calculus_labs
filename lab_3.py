import numpy as np
from helpers.Inaccuracies import calculate_guass_inaccuracies


def solve_with_optimal_exception(coef_matrix, b_column):
    # Функция принимает на вход:
    # coef_matrix -- матрица коэффициентов СЛАУ
    # b_column -- столбец свободных членов
    # Функция возвращает:
    # x -- столбец решений СЛАУ

    # Добавление столбца свободных членов к матрице коэффициентов
    matrix = np.hstack((coef_matrix, np.reshape(b_column, (len(b_column), 1))))

    computed_matrix = []  # стек для хранения удаленных из системы строк
    max_indexes_order = []  # стек для хранения порядка индексов максимальных столбцов
    answers = np.zeros(len(matrix))

    while len(matrix) > 0:
        # Нахождение максимального по модулю элемента и его индексы
        max_index = np.argmax(np.abs(matrix[:, :-1]))
        (max_row, max_col) = np.unravel_index(max_index, matrix[:, :-1].shape)
        max_elem = matrix[max_row][max_col]

        if max_elem == 0:
            return

        # Сохранение его индексов
        max_indexes_order.append(max_col)

        print(np.array(matrix))
        print('/')
        print(np.array(computed_matrix))
        print()

        # Сохранение строки с максимальным элементом и удаление ее из основной матрицы
        row_with_max = matrix[max_row] / max_elem
        matrix = np.delete(matrix, max_row, axis=0)

        for i in range(len(matrix)):
            matrix[i] = matrix[i] - row_with_max * matrix[i][max_col]

        for i in range(len(computed_matrix)):
            computed_matrix[i] = computed_matrix[i] - row_with_max * computed_matrix[i][max_col]

        computed_matrix.append(row_with_max)

    # Находим столбец свободных членов
    b = np.array(computed_matrix)[:, -1]
    print(np.array(computed_matrix))

    # Сортируем ответы
    for i in range(len(max_indexes_order)):
        answers[max_indexes_order[i]] = b[i]

    return answers


def lab_3(A, b):
    x = solve_with_optimal_exception(A, b)

    if type(x) is not np.ndarray:
        print(
            "Матрица вырождена, точного решения не существует"
        )

        return

    print(
        "Вычисленное решение СЛАУ: ", x
    )

    calculate_guass_inaccuracies(A, b, x)
