# -*- coding: utf-8 -*-
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd

from polarization_app.physics.spin_transport import compute_atom_probabilities
from polarization_app.physics.trajectory_phase import (
    ELECTRON_MASS_AMU,
    DEFAULT_THOMAS_FERMI_B_BOHR,
    compute_atom_trajectory_phase,
)


TRAJECTORY_SWEEP_ENERGY = "energy"
TRAJECTORY_SWEEP_IMPACT = "impact_parameter"
TRAJECTORY_SWEEP_ANGLE_STEP = "angle_step"
TrajectorySweepMode = Literal["energy", "impact_parameter", "angle_step"]

TRAJECTORY_SWEEP_LABELS = {
    TRAJECTORY_SWEEP_ENERGY: "Энергия E",
    TRAJECTORY_SWEEP_IMPACT: "Прицельное расстояние r_п",
    TRAJECTORY_SWEEP_ANGLE_STEP: "Шаг угла dθ",
}
TRAJECTORY_SWEEP_BY_LABEL = {label: key for key, label in TRAJECTORY_SWEEP_LABELS.items()}
TRAJECTORY_AXIS_LABELS = {
    TRAJECTORY_SWEEP_ENERGY: "E, эВ",
    TRAJECTORY_SWEEP_IMPACT: "r_п, Å",
    TRAJECTORY_SWEEP_ANGLE_STEP: "dθ, °",
}


@dataclass(frozen=True)
class TrajectorySweepRequest:
    sweep_mode: TrajectorySweepMode
    point_count: int
    atomic_number: float
    mass_amu: float = ELECTRON_MASS_AMU
    energy_eV: float = 100.0
    energy_min_eV: float = 10.0
    energy_max_eV: float = 1000.0
    impact_parameter_ang: float = 0.8
    impact_min_ang: float = 0.3
    impact_max_ang: float = 2.0
    r0_ang: float = 10.0
    angle_step_deg: float = 1.0
    angle_step_min_deg: float = 0.1
    angle_step_max_deg: float = 5.0
    b_bohr: float = DEFAULT_THOMAS_FERMI_B_BOHR
    orbital_l: int = 1
    magnetic_m: int = 0
    random_m: bool = False
    min_steps: int = 30
    max_refinements: int = 6
    parallel_workers: int = 1


@dataclass(frozen=True)
class TrajectorySweepResult:
    request: TrajectorySweepRequest
    frame: pd.DataFrame
    elapsed_ms: float
    magnetic_m_chain: tuple[int, ...]


def execute_trajectory_sweep(
    request: TrajectorySweepRequest,
    rng: np.random.Generator | None = None,
) -> TrajectorySweepResult:
    _validate_sweep_request(request)
    rng = rng or np.random.default_rng()
    values = _build_sweep_values(request)
    magnetic_values = _build_magnetic_chain(request, len(values), rng)

    tasks = [(float(value), int(magnetic_m)) for value, magnetic_m in zip(values, magnetic_values)]
    started = perf_counter()
    workers = max(1, int(request.parallel_workers))
    if workers == 1 or len(tasks) <= 1:
        rows = [_compute_sweep_row(request, value, magnetic_m) for value, magnetic_m in tasks]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(tasks)), thread_name_prefix="trajectory-sweep") as executor:
            rows = list(executor.map(lambda item: _compute_sweep_row(request, item[0], item[1]), tasks))

    elapsed_ms = (perf_counter() - started) * 1000.0
    return TrajectorySweepResult(
        request=request,
        frame=pd.DataFrame(rows),
        elapsed_ms=elapsed_ms,
        magnetic_m_chain=tuple(int(value) for value in magnetic_values),
    )


def trajectory_export_metadata(result: TrajectorySweepResult) -> dict[str, object]:
    request = result.request
    return {
        "sweep_mode": request.sweep_mode,
        "sweep_label": TRAJECTORY_SWEEP_LABELS[request.sweep_mode],
        "point_count": request.point_count,
        "atomic_number": request.atomic_number,
        "mass_amu": request.mass_amu,
        "energy_eV": request.energy_eV,
        "energy_min_eV": request.energy_min_eV,
        "energy_max_eV": request.energy_max_eV,
        "impact_parameter_ang": request.impact_parameter_ang,
        "impact_min_ang": request.impact_min_ang,
        "impact_max_ang": request.impact_max_ang,
        "r0_ang": request.r0_ang,
        "angle_step_deg": request.angle_step_deg,
        "angle_step_min_deg": request.angle_step_min_deg,
        "angle_step_max_deg": request.angle_step_max_deg,
        "b_bohr": request.b_bohr,
        "orbital_l": request.orbital_l,
        "magnetic_m": request.magnetic_m,
        "random_m": request.random_m,
        "min_steps": request.min_steps,
        "max_refinements": request.max_refinements,
        "parallel_workers": request.parallel_workers,
        "elapsed_ms": result.elapsed_ms,
        "magnetic_m_chain": result.magnetic_m_chain,
    }


def _compute_sweep_row(
    request: TrajectorySweepRequest,
    value: float,
    magnetic_m: int,
) -> dict[str, object]:
    energy_eV, impact_parameter_ang, angle_step_deg = _resolve_point_inputs(request, float(value))
    point_started = perf_counter()
    try:
        trajectory = compute_atom_trajectory_phase(
            energy_eV=energy_eV,
            mass_amu=request.mass_amu,
            atomic_number=request.atomic_number,
            impact_parameter_ang=impact_parameter_ang,
            r0_ang=request.r0_ang,
            angle_step_rad=float(np.deg2rad(angle_step_deg)),
            orbital_l=request.orbital_l,
            b_bohr=request.b_bohr,
            min_steps=request.min_steps,
            max_refinements=request.max_refinements,
        )
    except Exception as ex:
        return _failed_sweep_row(
            request=request,
            value=value,
            magnetic_m=magnetic_m,
            energy_eV=energy_eV,
            impact_parameter_ang=impact_parameter_ang,
            angle_step_deg=angle_step_deg,
            error=ex,
            runtime_ms=(perf_counter() - point_started) * 1000.0,
        )

    runtime_ms = (perf_counter() - point_started) * 1000.0
    p1, p2 = compute_atom_probabilities(
        np.asarray([trajectory.phase_rad], dtype=float),
        orbital_l=request.orbital_l,
        magnetic_lz=magnetic_m,
    )
    return {
        "sweep_parameter": request.sweep_mode,
        "sweep_value": float(value),
        "energy_eV": trajectory.energy_eV,
        "mass_amu": trajectory.mass_amu,
        "speed_m_per_s": trajectory.speed_mps,
        "speed_au": trajectory.speed_au,
        "atomic_number": trajectory.atomic_number,
        "impact_parameter_ang": trajectory.impact_parameter_ang,
        "r0_ang": trajectory.r0_ang,
        "b_bohr": trajectory.b_bohr,
        "angle_step_deg": float(np.rad2deg(trajectory.angle_step_rad)),
        "r_min_ang": trajectory.r_min_ang,
        "theta_rad": trajectory.theta_rad,
        "theta_deg": float(np.rad2deg(trajectory.theta_rad)),
        "trajectory_phi_rad": trajectory.trajectory_angle_rad,
        "trajectory_phi_deg": float(np.rad2deg(trajectory.trajectory_angle_rad)),
        "phase_rad": trajectory.phase_rad,
        "steps": trajectory.steps,
        "dt_initial_au": trajectory.dt_initial_au,
        "dt_final_au": trajectory.dt_final_au,
        "refinements": trajectory.refinements,
        "converged": trajectory.converged,
        "status": trajectory.status,
        "orbital_l": int(request.orbital_l),
        "magnetic_m": int(magnetic_m),
        "p_no_flip_initial_up": float(p1[0]),
        "p_no_flip_initial_down": float(p2[0]),
        "p_flip_initial_up": float(1.0 - p1[0]),
        "p_flip_initial_down": float(1.0 - p2[0]),
        "runtime_ms": runtime_ms,
    }


def _failed_sweep_row(
    *,
    request: TrajectorySweepRequest,
    value: float,
    magnetic_m: int,
    energy_eV: float,
    impact_parameter_ang: float,
    angle_step_deg: float,
    error: Exception,
    runtime_ms: float,
) -> dict[str, object]:
    return {
        "sweep_parameter": request.sweep_mode,
        "sweep_value": float(value),
        "energy_eV": float(energy_eV),
        "mass_amu": float(request.mass_amu),
        "speed_m_per_s": np.nan,
        "speed_au": np.nan,
        "atomic_number": float(request.atomic_number),
        "impact_parameter_ang": float(impact_parameter_ang),
        "r0_ang": float(request.r0_ang),
        "b_bohr": float(request.b_bohr),
        "angle_step_deg": float(angle_step_deg),
        "r_min_ang": np.nan,
        "theta_rad": np.nan,
        "theta_deg": np.nan,
        "trajectory_phi_rad": np.nan,
        "trajectory_phi_deg": np.nan,
        "phase_rad": np.nan,
        "steps": np.nan,
        "dt_initial_au": np.nan,
        "dt_final_au": np.nan,
        "refinements": np.nan,
        "converged": False,
        "status": _format_point_error(error, request),
        "orbital_l": int(request.orbital_l),
        "magnetic_m": int(magnetic_m),
        "p_no_flip_initial_up": np.nan,
        "p_no_flip_initial_down": np.nan,
        "p_flip_initial_up": np.nan,
        "p_flip_initial_down": np.nan,
        "runtime_ms": float(runtime_ms),
    }


def _format_point_error(error: Exception, request: TrajectorySweepRequest) -> str:
    message = str(error)
    if "max_steps" in message or "dθ" in message:
        if request.sweep_mode == TRAJECTORY_SWEEP_ANGLE_STEP:
            return (
                f"{message} Подсказка: увеличьте нижнюю границу «dθ min (°)» "
                "или верхнюю границу «dθ max (°)» для диапазона шага."
            )
        if request.sweep_mode == TRAJECTORY_SWEEP_IMPACT:
            return (
                f"{message} Подсказка: поднимите «r_п min (Å)» до 0.25-0.3 Å "
                "или увеличьте «dθ фикс. (°)»."
            )
        return (
            f"{message} Подсказка: увеличьте ползунок «dθ фикс. (°)» "
            "например до 2-5°, либо поднимите «r_п min (Å)», если сбой возникает на малых r_п."
        )
    if "r0" in message or "r_п" in message:
        return f"{message} Подсказка: проверьте «r0 (Å)» и диапазон «r_п min/max (Å)»."
    return message


def _build_sweep_values(request: TrajectorySweepRequest) -> np.ndarray:
    if request.sweep_mode == TRAJECTORY_SWEEP_ENERGY:
        return np.linspace(request.energy_min_eV, request.energy_max_eV, int(request.point_count), dtype=float)
    if request.sweep_mode == TRAJECTORY_SWEEP_IMPACT:
        return np.linspace(request.impact_min_ang, request.impact_max_ang, int(request.point_count), dtype=float)
    if request.sweep_mode == TRAJECTORY_SWEEP_ANGLE_STEP:
        return np.linspace(request.angle_step_min_deg, request.angle_step_max_deg, int(request.point_count), dtype=float)
    raise ValueError(f"Неизвестный режим траекторного расчёта: {request.sweep_mode}")


def _resolve_point_inputs(request: TrajectorySweepRequest, value: float) -> tuple[float, float, float]:
    if request.sweep_mode == TRAJECTORY_SWEEP_ENERGY:
        return value, request.impact_parameter_ang, request.angle_step_deg
    if request.sweep_mode == TRAJECTORY_SWEEP_IMPACT:
        return request.energy_eV, value, request.angle_step_deg
    if request.sweep_mode == TRAJECTORY_SWEEP_ANGLE_STEP:
        return request.energy_eV, request.impact_parameter_ang, value
    raise ValueError(f"Неизвестный режим траекторного расчёта: {request.sweep_mode}")


def _build_magnetic_chain(
    request: TrajectorySweepRequest,
    point_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    orbital_l = int(request.orbital_l)
    if request.random_m:
        return rng.integers(-orbital_l, orbital_l + 1, size=point_count, endpoint=False, dtype=int)
    return np.full(point_count, int(request.magnetic_m), dtype=int)


def _validate_sweep_request(request: TrajectorySweepRequest) -> None:
    if int(request.point_count) < 1:
        raise ValueError("Для траекторного расчёта нужна хотя бы одна точка.")
    if int(request.parallel_workers) < 1:
        raise ValueError("parallel_workers должен быть положительным.")
    if int(request.orbital_l) < 0:
        raise ValueError("L должен быть неотрицательным.")
    if not request.random_m and abs(int(request.magnetic_m)) > int(request.orbital_l):
        raise ValueError("Для ручного M требуется -L <= M <= L.")
    if request.sweep_mode == TRAJECTORY_SWEEP_ENERGY and request.energy_max_eV <= request.energy_min_eV:
        raise ValueError("Для энергии требуется Emin < Emax.")
    if request.sweep_mode == TRAJECTORY_SWEEP_IMPACT and request.impact_max_ang <= request.impact_min_ang:
        raise ValueError("Для r_п требуется min < max.")
    if request.sweep_mode == TRAJECTORY_SWEEP_ANGLE_STEP and request.angle_step_max_deg <= request.angle_step_min_deg:
        raise ValueError("Для dθ требуется min < max.")


__all__ = [
    "TRAJECTORY_SWEEP_ENERGY",
    "TRAJECTORY_SWEEP_IMPACT",
    "TRAJECTORY_SWEEP_ANGLE_STEP",
    "TRAJECTORY_SWEEP_LABELS",
    "TRAJECTORY_SWEEP_BY_LABEL",
    "TRAJECTORY_AXIS_LABELS",
    "TrajectorySweepMode",
    "TrajectorySweepRequest",
    "TrajectorySweepResult",
    "execute_trajectory_sweep",
    "trajectory_export_metadata",
]
