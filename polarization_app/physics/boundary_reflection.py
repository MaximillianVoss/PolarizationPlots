# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_COS_ALPHA_EPS = 1e-9


@dataclass(frozen=True)
class BoundaryReflectionCurves:
    energies_eV: np.ndarray
    reflection_coefficient: np.ndarray
    reflection_probability_estimate: np.ndarray
    transmission_angle_deg: np.ndarray
    wavevector_ratio: np.ndarray


@dataclass(frozen=True)
class BoundaryPointResult:
    energy_eV: float
    work_function_eV: float
    incidence_angle_deg: float
    reflection_coefficient: float
    reflection_probability_estimate: float
    transmission_angle_deg: float | None
    wavevector_ratio: float | None
    regime: str


def compute_boundary_reflection_curves(
    energies_eV: np.ndarray,
    *,
    work_function_eV: float,
    incidence_angle_deg: float,
) -> BoundaryReflectionCurves:
    energies = np.asarray(energies_eV, dtype=float)
    _validate_inputs(energies, work_function_eV, incidence_angle_deg)

    alpha_rad = np.radians(float(incidence_angle_deg))
    cos_alpha = float(np.cos(alpha_rad))
    sin_alpha_sq = float(np.sin(alpha_rad) ** 2)

    reflection = np.ones_like(energies, dtype=float)
    reflection_probability = np.ones_like(energies, dtype=float)
    transmission_angle_deg = np.full_like(energies, np.nan, dtype=float)
    wavevector_ratio = np.full_like(energies, np.nan, dtype=float)

    if cos_alpha <= _COS_ALPHA_EPS:
        return BoundaryReflectionCurves(
            energies_eV=energies,
            reflection_coefficient=reflection,
            reflection_probability_estimate=reflection_probability,
            transmission_angle_deg=transmission_angle_deg,
            wavevector_ratio=wavevector_ratio,
        )

    above_barrier_mask = energies > float(work_function_eV)
    if not np.any(above_barrier_mask):
        return BoundaryReflectionCurves(
            energies_eV=energies,
            reflection_coefficient=reflection,
            reflection_probability_estimate=reflection_probability,
            transmission_angle_deg=transmission_angle_deg,
            wavevector_ratio=wavevector_ratio,
        )

    energies_above = energies[above_barrier_mask]
    k_ratio = np.sqrt((energies_above - float(work_function_eV)) / energies_above)
    wavevector_ratio[above_barrier_mask] = k_ratio

    sin_beta_sq = sin_alpha_sq / np.maximum(k_ratio ** 2, 1e-18)
    transmission_mask_local = sin_beta_sq <= 1.0
    if np.any(transmission_mask_local):
        transmission_indices = np.flatnonzero(above_barrier_mask)[transmission_mask_local]
        sin_beta = np.sqrt(np.clip(sin_beta_sq[transmission_mask_local], 0.0, 1.0))
        beta_rad = np.arcsin(sin_beta)
        cos_beta = np.cos(beta_rad)
        term = k_ratio[transmission_mask_local] * (cos_beta / cos_alpha)
        reflection_values = (1.0 - term) / (1.0 + term)

        reflection[transmission_indices] = reflection_values
        reflection_probability[transmission_indices] = np.clip(reflection_values ** 2, 0.0, 1.0)
        transmission_angle_deg[transmission_indices] = np.degrees(beta_rad)

    return BoundaryReflectionCurves(
        energies_eV=energies,
        reflection_coefficient=reflection,
        reflection_probability_estimate=reflection_probability,
        transmission_angle_deg=transmission_angle_deg,
        wavevector_ratio=wavevector_ratio,
    )


def compute_boundary_point(
    energy_eV: float,
    *,
    work_function_eV: float,
    incidence_angle_deg: float,
) -> BoundaryPointResult:
    curves = compute_boundary_reflection_curves(
        np.array([float(energy_eV)], dtype=float),
        work_function_eV=work_function_eV,
        incidence_angle_deg=incidence_angle_deg,
    )

    beta_value = float(curves.transmission_angle_deg[0]) if np.isfinite(curves.transmission_angle_deg[0]) else None
    k_ratio_value = float(curves.wavevector_ratio[0]) if np.isfinite(curves.wavevector_ratio[0]) else None
    if energy_eV <= work_function_eV:
        regime = "Энергии недостаточно для прохождения через границу (E <= A)."
    elif beta_value is None:
        regime = "Полное отражение: угол после прохождения для этих параметров не реализуется."
    else:
        regime = "Частичное прохождение через границу."

    return BoundaryPointResult(
        energy_eV=float(energy_eV),
        work_function_eV=float(work_function_eV),
        incidence_angle_deg=float(incidence_angle_deg),
        reflection_coefficient=float(curves.reflection_coefficient[0]),
        reflection_probability_estimate=float(curves.reflection_probability_estimate[0]),
        transmission_angle_deg=beta_value,
        wavevector_ratio=k_ratio_value,
        regime=regime,
    )


def _validate_inputs(energies_eV: np.ndarray, work_function_eV: float, incidence_angle_deg: float) -> None:
    if energies_eV.ndim != 1 or len(energies_eV) == 0:
        raise ValueError("Сетка энергий должна быть непустым одномерным массивом.")
    if np.any(~np.isfinite(energies_eV)) or np.any(energies_eV <= 0.0):
        raise ValueError("Все энергии должны быть конечными и положительными.")
    if not np.isfinite(work_function_eV) or float(work_function_eV) < 0.0:
        raise ValueError("Работа выхода A должна быть конечной и неотрицательной.")
    if not np.isfinite(incidence_angle_deg) or float(incidence_angle_deg) < 0.0 or float(incidence_angle_deg) >= 90.0:
        raise ValueError("Угол падения α должен лежать в диапазоне [0, 90).")


__all__ = [
    "BoundaryReflectionCurves",
    "BoundaryPointResult",
    "compute_boundary_reflection_curves",
    "compute_boundary_point",
]
