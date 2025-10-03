import math
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np


Matrix2x2 = np.ndarray


def _safe_sqrt(value: float) -> float:
    """Извлечение корня с отсечкой отрицательных значений от ошибок округления."""
    if value < 0:
        if value >= -1e-12:
            return 0.0
        raise ValueError(f"Невозможно извлечь корень из отрицательного значения {value}")
    return math.sqrt(value)


def transition_matrices(L_source: int, L_target: Optional[int] = None
                        ) -> Tuple[Dict[int, Matrix2x2], Dict[int, Optional[Matrix2x2]]]:
    """
    Формирует набор 2x2 матриц переходов для фиксированного орбитального числа L_source.

    Параметры
    ---------
    L_source : int
        Орбитальное число исходного уровня (обозначено как L_s в методичке).
    L_target : int | None, optional
        Орбитальное число второго уровня. Если не указано, используется L_source,
        что соответствует запросу заказчика для проверки алгоритма.

    Результат
    ---------
    matrices : dict[int, np.ndarray]
        Словарь {L_n: D}, где D — матрица переходов для заданного магнитного
        числа L_n (L_z). Значения L_n пробегают диапазон [-L_target, L_target].
    inverses : dict[int, np.ndarray | None]
        Словарь обратных матриц. Если матрица вырождена, значение — None.
    """

    if L_source < 0:
        raise ValueError("L_source должен быть неотрицательным")

    if L_target is None:
        L_target = L_source

    if L_target < 0:
        raise ValueError("L_target должен быть неотрицательным")

    denom = 2 * L_source + 1
    if denom == 0:
        raise ValueError("Недопустимое значение L_source = -0.5")

    matrices: Dict[int, Matrix2x2] = OrderedDict()
    inverses: Dict[int, Optional[Matrix2x2]] = OrderedDict()

    for L_n in range(-L_target, L_target + 1):
        num_top_left = L_source + L_n + 1
        num_top_right = L_source - L_n + 1
        num_bottom_left = L_source - L_n
        num_bottom_right = L_source + L_n

        D = np.array(
            [
                [_safe_sqrt(num_top_left / denom), - _safe_sqrt(num_top_right / denom)],
                [_safe_sqrt(num_bottom_left / denom), _safe_sqrt(num_bottom_right / denom)],
            ],
            dtype=float,
        )

        matrices[L_n] = D

        det = np.linalg.det(D)
        if abs(det) < 1e-12:
            inverses[L_n] = None
        else:
            inverses[L_n] = np.linalg.inv(D)

    return matrices, inverses


# Обратная совместимость со старым API
def transition_matrix(L: int):  # pragma: no cover - удобный шорткат для GUI
    """Оставлено для совместимости: возвращает первый набор матриц."""
    matrices, inverses = transition_matrices(L)
    # Старый код ожидал одну матрицу, поэтому возвращаем первую по L_n
    first_key = next(iter(matrices))
    return matrices[first_key], inverses[first_key]
