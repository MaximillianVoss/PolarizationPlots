import numpy as np

def transition_matrix(L: int):
    """
    Формируем матрицу переходов D(L) для орбитального числа L.
    Элементы считаются по формулам Клебша-Гордана для соседних Lz.
    """
    size = 2 * L + 1
    D = np.zeros((size, size))
    Lz_values = np.arange(-L, L+1)

    for i, m in enumerate(Lz_values):  # строка
        for j, n in enumerate(Lz_values):  # столбец
            if n == m + 1:  # переход Lz -> Lz+1
                D[i, j] = np.sqrt((L - m) * (L + m + 1)) / (2 * L + 1)
            elif n == m - 1:  # переход Lz -> Lz-1
                D[i, j] = np.sqrt((L + m) * (L - m + 1)) / (2 * L + 1)
            elif n == m:  # диагональные элементы (сохранение Lz)
                D[i, j] = m / (L + 0.5)  # примерная форма, корректируется под вашу формулу

    # Проверка на обратимость
    det = np.linalg.det(D)
    if abs(det) < 1e-12:
        print("⚠️ Внимание: матрица D(L) вырожденная, обратная не существует.")
        D_inv = None
    else:
        D_inv = np.linalg.inv(D)

    return D, D_inv
