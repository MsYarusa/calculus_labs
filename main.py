import numpy as np
from lab_2 import lab_2
from lab_3 import lab_3
from lab_4 import lab_4
from lab_5 import lab_5
from lab_6 import lab_6
from lab_7 import lab_7
from lab_8 import lab_8

# Конифигурация вывода
np.set_printoptions(linewidth=320)

# Данные
A = np.array([
    [0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
    [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
    [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
    [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
    [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
    [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
    [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105],
])
b = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])

A_symmetric_positive = np.array([
    [1, 3, -2, 0, -2],
    [3, 4, -5, 1, -3],
    [-2, -5, 3, -2, 2],
    [0, 1, -2, 5, 3],
    [-2, -3, 2, 3, 4]
])

b_symmetric_positive = np.array([0.5, 5.4, 5.0, 7.5, 3.3])

A_symmetric_nonpositive = np.array([
    [2.2, 4, -3, 1.5, 0.6, 2, 0.7],
    [4, 3.2, 1.5, -0.7, -0.8, 3, 1],
    [-3, 1.5, 1.8, 0.9, 3, 2, 2],
    [1.5, -0.7, 0.9, 2.2, 4, 3, 1],
    [0.6, -0.8, 3, 4, 3.2, 0.6, 0.7],
    [2, 3, 2, 3, 0.6, 2.2, 4],
    [0.7, 1, 2, 1, 0.7, 4, 3.2]
], dtype=complex)

b_symmetric_nonpositive = np.array([3.2, 4.3, -0.1, 3.5, 5.3, 9.0, 3.7])

A_0 = np.array([[0, 1], [0, 0]])
b_0 = np.array([0, 1])

A_lab_6 = np.array([
    [1.00, 0.42, 0.54, 0.66],
    [0.42, 1.00, 0.32, 0.44],
    [0.54, 0.32, 1.00, 0.22],
    [0.66, 0.44, 0.22, 1.00]
])

b_lab_6 = np.array([1, 1, 1, 1])

A_lab_7 = np.array([
    [6.03, 13, -17],
    [13, 29.03, -38],
    [-17, -38, 50.03]
])

b_lab_7 = np.array([2.0909, 4.1509, -5.1191])

A_lab_8 = np.array([
    [10.9, 1.2, 2.1, 0.9],
    [1.2, 11.2, 1.5, 2.5],
    [2.1, 1.5, 9.8, 1.3],
    [0.9, 2.5, 1.3, 12.1]
])

b_lab_8 = np.array([-7.0, 5.3, 10.3, 24.6])

A_lab_8_1 = np.array([
    [3.82, 1.02, 0.75, 0.81],
    [1.05, 4.53, 0.98, 1.53],
    [0.73, 0.85, 4.71, 0.81],
    [0.88, 0.81, 1.28, 3.50]
])
b_lab_8_1 = np.array([15.655, 22.705, 23.480, 16.110])

# Тело программы
if __name__ == '__main__':
    # print("Схема Гаусса с выбором главного элемента")
    # lab_2(A, b)
    # print()
    #
    # print("Метод оптимального исключения")
    # lab_3(A, b)
    # print()

    # print("LU - разложение")
    # lab_4(A, b)
    # print()

    # print("Метод квадратного корня")
    # lab_5(A_symmetric_nonpositive, b_symmetric_nonpositive)
    # print()

    # print("Метод окаймления")
    # lab_6(A, b)
    # print()

    # print("Метод отражений")
    # lab_7(A_lab_7, b_lab_7)
    # print()

    lab_8(A_lab_8, b_lab_8)
    print()



