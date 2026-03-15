# -*- coding: utf-8 -*-
import math
from collections import OrderedDict
from typing import Optional

import numpy as np


Matrix2x2 = np.ndarray


def _safe_sqrt(value: float) -> float:
    if value < 0:
        if value >= -1e-12:
            return 0.0
        raise ValueError(f"Невозможно извлечь корень из отрицательного значения {value}")
    return math.sqrt(value)


def build_transition_matrices(
    source_orbital_l: int,
    target_orbital_l: Optional[int] = None,
) -> tuple[dict[int, Matrix2x2], dict[int, Matrix2x2 | None]]:
    """
    Формирует набор 2x2 матриц переходов для фиксированного орбитального числа.
    """
    if source_orbital_l < 0:
        raise ValueError("source_orbital_l должен быть неотрицательным")

    if target_orbital_l is None:
        target_orbital_l = source_orbital_l
    if target_orbital_l < 0:
        raise ValueError("target_orbital_l должен быть неотрицательным")

    denom = 2 * source_orbital_l + 1
    if denom == 0:
        raise ValueError("Недопустимое значение source_orbital_l = -0.5")

    matrices: dict[int, Matrix2x2] = OrderedDict()
    inverses: dict[int, Matrix2x2 | None] = OrderedDict()

    for magnetic_lz in range(-target_orbital_l, target_orbital_l + 1):
        # num_top_left = source_orbital_l + magnetic_lz + 1
        # num_top_right = source_orbital_l - magnetic_lz + 1
        # num_bottom_left = source_orbital_l - magnetic_lz
        # num_bottom_right = source_orbital_l + magnetic_lz

        num_top_left = source_orbital_l + magnetic_lz + 1
        num_top_right = source_orbital_l - magnetic_lz
        num_bottom_left = source_orbital_l - magnetic_lz
        num_bottom_right = source_orbital_l + magnetic_lz + 1

        matrix = np.array(
            [
                [_safe_sqrt(num_top_left / denom), -_safe_sqrt(num_top_right / denom)],
                [_safe_sqrt(num_bottom_left / denom), _safe_sqrt(num_bottom_right / denom)],
            ],
            dtype=float,
        )
        matrices[magnetic_lz] = matrix

        determinant = np.linalg.det(matrix)
        inverses[magnetic_lz] = None if abs(determinant) < 1e-12 else np.linalg.inv(matrix)

    return matrices, inverses


def build_first_transition_matrix(orbital_l: int) -> tuple[Matrix2x2, Matrix2x2 | None]:
    matrices, inverses = build_transition_matrices(source_orbital_l=orbital_l)
    first_key = next(iter(matrices))
    return matrices[first_key], inverses[first_key]


def transition_matrices(L_source: int, L_target: Optional[int] = None):
    return build_transition_matrices(L_source, L_target)


def transition_matrix(L: int):
    return build_first_transition_matrix(L)


__all__ = [
    "Matrix2x2",
    "build_transition_matrices",
    "build_first_transition_matrix",
    "transition_matrices",
    "transition_matrix",
]
