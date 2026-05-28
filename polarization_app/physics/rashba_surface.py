# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from polarization_app.physics.phase_integrals import BOHR_TO_ANGSTROM


HARTREE_EV = 27.211386245988


@dataclass(frozen=True)
class RashbaSurfaceRequest:
    energy_min_eV: float = 10.0
    energy_max_eV: float = 1000.0
    point_count: int = 240
    layer_thickness_ang: float = 1.0
    rashba_alpha_au: float = 0.05
    emission_angle_deg: float = 45.0
    surface_potential_eV: float = 5.0
    ver_up_to_down: np.ndarray | float = 0.0
    ver_down_to_up: np.ndarray | float = 0.0


@dataclass(frozen=True)
class RashbaSurfaceResult:
    request: RashbaSurfaceRequest
    frame: pd.DataFrame


def compute_rashba_surface(request: RashbaSurfaceRequest) -> RashbaSurfaceResult:
    _validate_request(request)
    energies_eV = np.linspace(request.energy_min_eV, request.energy_max_eV, int(request.point_count), dtype=float)
    frame = compute_rashba_surface_frame(
        energies_eV=energies_eV,
        layer_thickness_ang=request.layer_thickness_ang,
        rashba_alpha_au=request.rashba_alpha_au,
        emission_angle_deg=request.emission_angle_deg,
        surface_potential_eV=request.surface_potential_eV,
        ver_up_to_down=request.ver_up_to_down,
        ver_down_to_up=request.ver_down_to_up,
    )
    return RashbaSurfaceResult(request=request, frame=frame)


def compute_rashba_surface_frame(
    *,
    energies_eV: np.ndarray,
    layer_thickness_ang: float,
    rashba_alpha_au: float,
    emission_angle_deg: float,
    surface_potential_eV: float,
    ver_up_to_down: np.ndarray | float = 0.0,
    ver_down_to_up: np.ndarray | float = 0.0,
) -> pd.DataFrame:
    energies_eV = np.asarray(energies_eV, dtype=float)
    if energies_eV.ndim != 1 or len(energies_eV) == 0:
        raise ValueError("Нужен непустой одномерный массив энергий.")
    if np.any(~np.isfinite(energies_eV)) or np.any(energies_eV <= 0.0):
        raise ValueError("Энергии должны быть положительными.")
    if not np.isfinite(layer_thickness_ang) or layer_thickness_ang <= 0.0:
        raise ValueError("Толщина слоя d должна быть положительной.")
    if not np.isfinite(rashba_alpha_au):
        raise ValueError("Коэффициент Рашбы α должен быть конечным.")
    if not np.isfinite(emission_angle_deg) or not (0.0 <= emission_angle_deg < 90.0):
        raise ValueError("Угол θ должен быть в диапазоне 0 <= θ < 90°.")
    if not np.isfinite(surface_potential_eV) or surface_potential_eV < 0.0:
        raise ValueError("Потенциал поверхности U не может быть отрицательным.")

    energies_au = energies_eV / HARTREE_EV
    surface_potential_au = float(surface_potential_eV) / HARTREE_EV
    thickness_bohr = float(layer_thickness_ang) / BOHR_TO_ANGSTROM
    theta_rad = np.deg2rad(float(emission_angle_deg))

    kx = np.sqrt(2.0 * energies_au) * np.sin(theta_rad)
    ky_sq = 2.0 * energies_au * (np.cos(theta_rad) ** 2)
    ky_prime_sq_up = ky_sq + 2.0 * float(rashba_alpha_au) * kx
    ky_prime_sq_down = ky_sq - 2.0 * float(rashba_alpha_au) * kx

    reflection_up = _reflection_probability(ky_sq, ky_prime_sq_up, thickness_bohr)
    reflection_down = _reflection_probability(ky_sq, ky_prime_sq_down, thickness_bohr)
    transmission_factor = np.sqrt(np.clip((energies_au - surface_potential_au) / energies_au, 0.0, None))
    transmission_up = np.clip(transmission_factor * (1.0 - reflection_up), 0.0, 1.0)
    transmission_down = np.clip(transmission_factor * (1.0 - reflection_down), 0.0, 1.0)

    ver_up_to_down_arr = _probability_array(ver_up_to_down, len(energies_eV), "Ver(+→-)")
    ver_down_to_up_arr = _probability_array(ver_down_to_up, len(energies_eV), "Ver(-→+)")
    t_plus_sq = transmission_up * (1.0 + ver_down_to_up_arr - ver_up_to_down_arr)
    t_minus_sq = transmission_down * (1.0 + ver_up_to_down_arr - ver_down_to_up_arr)
    denominator = t_plus_sq + t_minus_sq
    polarization = np.divide(
        t_plus_sq - t_minus_sq,
        denominator,
        out=np.zeros_like(denominator),
        where=np.abs(denominator) > 1e-15,
    )

    return pd.DataFrame(
        {
            "energy_eV": energies_eV,
            "energy_au": energies_au,
            "kx_au": kx,
            "ky_sq_au": ky_sq,
            "ky_prime_sq_up_au": ky_prime_sq_up,
            "ky_prime_sq_down_au": ky_prime_sq_down,
            "reflection_up": reflection_up,
            "reflection_down": reflection_down,
            "transmission_up": transmission_up,
            "transmission_down": transmission_down,
            "ver_up_to_down": ver_up_to_down_arr,
            "ver_down_to_up": ver_down_to_up_arr,
            "t_plus_sq": np.clip(t_plus_sq, 0.0, None),
            "t_minus_sq": np.clip(t_minus_sq, 0.0, None),
            "polarization": np.clip(polarization, -1.0, 1.0),
        }
    )


def _reflection_probability(ky_sq: np.ndarray, ky_prime_sq: np.ndarray, thickness_bohr: float) -> np.ndarray:
    ky_sq = np.asarray(ky_sq, dtype=float)
    ky_prime_sq = np.asarray(ky_prime_sq, dtype=float)
    reflection = np.ones_like(ky_sq, dtype=float)
    propagating = (ky_sq > 0.0) & (ky_prime_sq > 0.0)
    if not np.any(propagating):
        return reflection

    ky_prime = np.sqrt(ky_prime_sq[propagating])
    argument = ky_prime * float(thickness_bohr)
    sin_sq = np.sin(argument) ** 2
    cos_sq = np.cos(argument) ** 2
    delta = ky_prime_sq[propagating] - ky_sq[propagating]
    summed = ky_prime_sq[propagating] + ky_sq[propagating]
    numerator = (delta ** 2) * sin_sq
    denominator = (
        4.0 * ky_sq[propagating] * ky_prime_sq[propagating] * cos_sq
        + (summed ** 2) * sin_sq
    )
    reflection[propagating] = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=np.abs(denominator) > 1e-15,
    )
    return np.clip(reflection, 0.0, 1.0)


def _probability_array(values: np.ndarray | float, length: int, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.full(length, float(arr), dtype=float)
    if arr.shape != (length,):
        raise ValueError(f"{label} должен быть скаляром или массивом длины {length}.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{label} содержит нечисловые значения.")
    return np.clip(arr, 0.0, 1.0)


def _validate_request(request: RashbaSurfaceRequest) -> None:
    if int(request.point_count) < 2:
        raise ValueError("Для графика нужно минимум 2 точки.")
    if request.energy_min_eV <= 0.0:
        raise ValueError("Emin должен быть положительным.")
    if request.energy_max_eV <= request.energy_min_eV:
        raise ValueError("Emax должен быть больше Emin.")


__all__ = [
    "HARTREE_EV",
    "RashbaSurfaceRequest",
    "RashbaSurfaceResult",
    "compute_rashba_surface",
    "compute_rashba_surface_frame",
]
