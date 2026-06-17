# -*- coding: utf-8 -*-
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd

from polarization_app.physics.spin_transport import compute_atom_probabilities
from polarization_app.physics.compute_backend import cpu_worker_count
from polarization_app.physics.trajectory_phase import (
    ELECTRON_MASS_AMU,
    DEFAULT_THOMAS_FERMI_B_BOHR,
    RADIAL_BASE_PANEL_LIMIT,
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
DEFAULT_TRAJECTORY_MIN_STEPS = 100
DEFAULT_PRECISE_TRAJECTORY_MIN_STEPS = 300
DEFAULT_TRAJECTORY_MAX_PHASE_STEP_RAD = 0.05
DEFAULT_PRECISE_TRAJECTORY_MAX_PHASE_STEP_RAD = 0.02
DEFAULT_TRAJECTORY_CONVERGENCE_PHASE_TOLERANCE_RAD = 0.03
DEFAULT_TRAJECTORY_CONVERGENCE_PROBABILITY_TOLERANCE = 0.03


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
    orbital_l: int = 1
    magnetic_m: int = 0
    random_m: bool = False
    min_steps: int = DEFAULT_TRAJECTORY_MIN_STEPS
    max_refinements: int = 6
    precise_mode: bool = False
    convergence_check: bool = False
    max_phase_step_rad: float = DEFAULT_TRAJECTORY_MAX_PHASE_STEP_RAD
    convergence_phase_tolerance_rad: float = DEFAULT_TRAJECTORY_CONVERGENCE_PHASE_TOLERANCE_RAD
    convergence_probability_tolerance: float = DEFAULT_TRAJECTORY_CONVERGENCE_PROBABILITY_TOLERANCE
    parallel_workers: int = cpu_worker_count()


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
        "b_bohr": DEFAULT_THOMAS_FERMI_B_BOHR,
        "orbital_l": request.orbital_l,
        "magnetic_m": request.magnetic_m,
        "random_m": request.random_m,
        "min_steps": request.min_steps,
        "radial_base_panel_limit": RADIAL_BASE_PANEL_LIMIT,
        "max_refinements": request.max_refinements,
        "precise_mode": request.precise_mode,
        "convergence_check": request.convergence_check,
        "max_phase_step_rad": request.max_phase_step_rad,
        "convergence_phase_tolerance_rad": request.convergence_phase_tolerance_rad,
        "convergence_probability_tolerance": request.convergence_probability_tolerance,
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
            min_steps=_effective_min_steps(request),
            max_refinements=request.max_refinements,
            max_phase_step_rad=_effective_max_phase_step_rad(request),
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
    p1_no_flip, p2_no_flip, p_up_flip, p_down_flip = _spin_probabilities_for_phase(
        trajectory.phase_rad,
        orbital_l=request.orbital_l,
        magnetic_m=magnetic_m,
    )
    row = {
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
        "p_no_flip_initial_up": p1_no_flip,
        "p_no_flip_initial_down": p2_no_flip,
        "p_flip_initial_up": p_up_flip,
        "p_flip_initial_down": p_down_flip,
        "convergence_checked": False,
        "convergence_unstable": False,
        "convergence_phase_error_rad": 0.0,
        "convergence_probability_error": 0.0,
        "phase_rad_dtheta_half": np.nan,
        "phase_rad_dtheta_quarter": np.nan,
        "p_flip_initial_up_dtheta_half": np.nan,
        "p_flip_initial_up_dtheta_quarter": np.nan,
        "p_flip_initial_down_dtheta_half": np.nan,
        "p_flip_initial_down_dtheta_quarter": np.nan,
        "runtime_ms": runtime_ms,
    }
    if request.convergence_check:
        row.update(
            _compute_convergence_diagnostics(
                request=request,
                base_phase_rad=trajectory.phase_rad,
                base_p_up_flip=p_up_flip,
                base_p_down_flip=p_down_flip,
                energy_eV=energy_eV,
                impact_parameter_ang=impact_parameter_ang,
                angle_step_deg=angle_step_deg,
                magnetic_m=magnetic_m,
            )
        )
    return row


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
        "b_bohr": DEFAULT_THOMAS_FERMI_B_BOHR,
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
        "convergence_checked": bool(request.convergence_check),
        "convergence_unstable": False,
        "convergence_phase_error_rad": np.nan,
        "convergence_probability_error": np.nan,
        "phase_rad_dtheta_half": np.nan,
        "phase_rad_dtheta_quarter": np.nan,
        "p_flip_initial_up_dtheta_half": np.nan,
        "p_flip_initial_up_dtheta_quarter": np.nan,
        "p_flip_initial_down_dtheta_half": np.nan,
        "p_flip_initial_down_dtheta_quarter": np.nan,
        "runtime_ms": float(runtime_ms),
    }


def _effective_min_steps(request: TrajectorySweepRequest) -> int:
    minimum = DEFAULT_PRECISE_TRAJECTORY_MIN_STEPS if request.precise_mode else DEFAULT_TRAJECTORY_MIN_STEPS
    return max(int(request.min_steps), minimum)


def _effective_max_phase_step_rad(request: TrajectorySweepRequest) -> float:
    limit = float(request.max_phase_step_rad)
    if request.precise_mode:
        limit = min(limit, DEFAULT_PRECISE_TRAJECTORY_MAX_PHASE_STEP_RAD)
    return limit


def _spin_probabilities_for_phase(
    phase_rad: float,
    *,
    orbital_l: int,
    magnetic_m: int,
) -> tuple[float, float, float, float]:
    p1, p2 = compute_atom_probabilities(
        np.asarray([phase_rad], dtype=float),
        orbital_l=orbital_l,
        magnetic_lz=magnetic_m,
    )
    p1_no_flip = float(p1[0])
    p2_no_flip = float(p2[0])
    return p1_no_flip, p2_no_flip, float(1.0 - p1_no_flip), float(1.0 - p2_no_flip)


def _compute_convergence_diagnostics(
    *,
    request: TrajectorySweepRequest,
    base_phase_rad: float,
    base_p_up_flip: float,
    base_p_down_flip: float,
    energy_eV: float,
    impact_parameter_ang: float,
    angle_step_deg: float,
    magnetic_m: int,
) -> dict[str, object]:
    phases = [float(base_phase_rad)]
    up_flip = [float(base_p_up_flip)]
    down_flip = [float(base_p_down_flip)]
    for divisor in (2.0, 4.0):
        refined = compute_atom_trajectory_phase(
            energy_eV=energy_eV,
            mass_amu=request.mass_amu,
            atomic_number=request.atomic_number,
            impact_parameter_ang=impact_parameter_ang,
            r0_ang=request.r0_ang,
            angle_step_rad=float(np.deg2rad(angle_step_deg / divisor)),
            orbital_l=request.orbital_l,
            min_steps=_effective_min_steps(request),
            max_refinements=request.max_refinements,
            max_phase_step_rad=_effective_max_phase_step_rad(request),
        )
        _, _, refined_up_flip, refined_down_flip = _spin_probabilities_for_phase(
            refined.phase_rad,
            orbital_l=request.orbital_l,
            magnetic_m=magnetic_m,
        )
        phases.append(float(refined.phase_rad))
        up_flip.append(refined_up_flip)
        down_flip.append(refined_down_flip)

    phase_error = max(abs(phases[0] - phases[1]), abs(phases[1] - phases[2]))
    probability_error = max(
        abs(up_flip[0] - up_flip[1]),
        abs(up_flip[1] - up_flip[2]),
        abs(down_flip[0] - down_flip[1]),
        abs(down_flip[1] - down_flip[2]),
    )
    unstable = (
        phase_error > float(request.convergence_phase_tolerance_rad)
        or probability_error > float(request.convergence_probability_tolerance)
    )
    return {
        "convergence_checked": True,
        "convergence_unstable": bool(unstable),
        "convergence_phase_error_rad": float(phase_error),
        "convergence_probability_error": float(probability_error),
        "phase_rad_dtheta_half": phases[1],
        "phase_rad_dtheta_quarter": phases[2],
        "p_flip_initial_up_dtheta_half": up_flip[1],
        "p_flip_initial_up_dtheta_quarter": up_flip[2],
        "p_flip_initial_down_dtheta_half": down_flip[1],
        "p_flip_initial_down_dtheta_quarter": down_flip[2],
    }


def _format_point_error(error: Exception, request: TrajectorySweepRequest) -> str:
    message = str(error)
    if "max_steps" in message or "dθ" in message:
        if request.sweep_mode == TRAJECTORY_SWEEP_ANGLE_STEP:
            return (
                f"{message} Подсказка: проверьте малые r_п и точный режим; "
                "dθ теперь влияет на базовую сетку, но не должен использоваться как способ скрыть ошибку."
            )
        if request.sweep_mode == TRAJECTORY_SWEEP_IMPACT:
            return (
                f"{message} Подсказка: поднимите «r_п min (Å)» до 0.25-0.3 Å "
                "или уменьшите требуемую точность интегрирования."
            )
        return (
            f"{message} Подсказка: если сбой возникает на малых r_п, поднимите «r_п min (Å)» "
            "или уменьшите требуемую точность интегрирования."
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
    if float(request.max_phase_step_rad) <= 0.0:
        raise ValueError("max_phase_step_rad должен быть положительным.")
    if float(request.convergence_phase_tolerance_rad) <= 0.0:
        raise ValueError("convergence_phase_tolerance_rad должен быть положительным.")
    if float(request.convergence_probability_tolerance) <= 0.0:
        raise ValueError("convergence_probability_tolerance должен быть положительным.")
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
    "DEFAULT_TRAJECTORY_MIN_STEPS",
    "DEFAULT_PRECISE_TRAJECTORY_MIN_STEPS",
    "DEFAULT_TRAJECTORY_MAX_PHASE_STEP_RAD",
    "DEFAULT_PRECISE_TRAJECTORY_MAX_PHASE_STEP_RAD",
    "DEFAULT_TRAJECTORY_CONVERGENCE_PHASE_TOLERANCE_RAD",
    "DEFAULT_TRAJECTORY_CONVERGENCE_PROBABILITY_TOLERANCE",
    "TrajectorySweepMode",
    "TrajectorySweepRequest",
    "TrajectorySweepResult",
    "execute_trajectory_sweep",
    "trajectory_export_metadata",
]
