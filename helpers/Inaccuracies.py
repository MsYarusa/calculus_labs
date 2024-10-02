import numpy as np


def calculate_guass_inaccuracies(coef_matrix, b_column, answers):
    A = coef_matrix
    x = answers
    b = b_column

    delta_b = A @ x - b
    delta_x = np.linalg.solve(coef_matrix, delta_b)

    print(
        "Модуль разности вычисленного решения и решения через библиотеку np: " +
        str(np.linalg.norm(np.linalg.solve(A, b) - x))
    )

    print(
        "Оценка абсолютной погрешности решения СЛАУ (https://studfile.net/preview/4426799/page:9/): " +
        str(np.linalg.norm(delta_x))
    )
