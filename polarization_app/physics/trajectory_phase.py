# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    from scipy.optimize import brentq
except ImportError:  # pragma: no cover - exercised only in lean runtime environments
    brentq = None

from polarization_app.physics.phase_integrals import (
    BOHR_TO_ANGSTROM,
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    INVERSE_FINE_STRUCTURE,
    LIGHT_SPEED,
    ChiFunction,
    scalar_spline_thomas_fermi_chi,
    scalar_spline_thomas_fermi_chi_derivative,
    spline_thomas_fermi_chi,
    spline_thomas_fermi_chi_derivative,
)


ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
ELECTRON_MASS_AMU = ELECTRON_MASS / ATOMIC_MASS_UNIT_KG
ATOMIC_SPEED_MPS = LIGHT_SPEED / INVERSE_FINE_STRUCTURE
DEFAULT_THOMAS_FERMI_B_BOHR = 0.885
DEFAULT_SPIN_ORBIT_C1 = 1.0 / (4.0 * INVERSE_FINE_STRUCTURE * INVERSE_FINE_STRUCTURE)


@dataclass(frozen=True)
class AtomTrajectoryResult:
    energy_eV: float
    mass_amu: float
    atomic_number: float
    impact_parameter_ang: float
    r0_ang: float
    b_bohr: float
    angle_step_rad: float
    speed_mps: float
    speed_au: float
    r_min_ang: float
    theta_rad: float
    trajectory_angle_rad: float
    phase_rad: float
    steps: int
    dt_initial_au: float
    dt_final_au: float
    refinements: int
    converged: bool
    status: str


def energy_eV_to_speed_mps_for_mass(energy_eV: float | np.ndarray, mass_amu: float) -> np.ndarray:
    mass_kg = float(mass_amu) * ATOMIC_MASS_UNIT_KG
    if not np.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError("Масса в а.е.м должна быть положительной.")
    energy_joule = np.asarray(energy_eV, dtype=float) * ELECTRON_CHARGE
    if np.any(~np.isfinite(energy_joule)) or np.any(energy_joule <= 0.0):
        raise ValueError("Энергия должна быть положительной.")
    return np.sqrt(2.0 * energy_joule / mass_kg)


def speed_mps_to_atomic_units(speed_mps: float | np.ndarray) -> np.ndarray:
    speed_au = np.asarray(speed_mps, dtype=float) / ATOMIC_SPEED_MPS
    if np.any(~np.isfinite(speed_au)) or np.any(speed_au <= 0.0):
        raise ValueError("Скорость должна быть положительной.")
    return speed_au


def thomas_fermi_potential_au(
    r_bohr: float | np.ndarray,
    atomic_number: float,
    b_bohr: float = DEFAULT_THOMAS_FERMI_B_BOHR,
    chi: ChiFunction = spline_thomas_fermi_chi,
) -> np.ndarray:
    r = np.asarray(r_bohr, dtype=float)
    if np.any(r <= 0.0):
        raise ValueError("Расстояние r должно быть положительным.")
    z = float(atomic_number)
    x = (z ** (1.0 / 3.0)) * r / float(b_bohr)
    return -z * chi(x) / r


def thomas_fermi_potential_derivative_au(
    r_bohr: float | np.ndarray,
    atomic_number: float,
    b_bohr: float = DEFAULT_THOMAS_FERMI_B_BOHR,
    chi: ChiFunction = spline_thomas_fermi_chi,
    chi_derivative: ChiFunction = spline_thomas_fermi_chi_derivative,
) -> np.ndarray:
    r = np.asarray(r_bohr, dtype=float)
    if np.any(r <= 0.0):
        raise ValueError("Расстояние r должно быть положительным.")
    z = float(atomic_number)
    b = float(b_bohr)
    x = (z ** (1.0 / 3.0)) * r / b
    return z * chi(x) / (r * r) - (z ** (4.0 / 3.0)) * chi_derivative(x) / (r * b)


def compute_atom_trajectory_phase(
    *,
    energy_eV: float,
    mass_amu: float = ELECTRON_MASS_AMU,
    atomic_number: float,
    impact_parameter_ang: float,
    r0_ang: float,
    angle_step_rad: float,
    b_bohr: float = DEFAULT_THOMAS_FERMI_B_BOHR,
    min_steps: int = 30,
    max_refinements: int = 6,
    max_steps: int = 200_000,
    chi: ChiFunction = spline_thomas_fermi_chi,
    chi_derivative: ChiFunction = spline_thomas_fermi_chi_derivative,
    spin_orbit_c1: float = DEFAULT_SPIN_ORBIT_C1,
) -> AtomTrajectoryResult:
    _validate_inputs(
        atomic_number=atomic_number,
        impact_parameter_ang=impact_parameter_ang,
        r0_ang=r0_ang,
        angle_step_rad=angle_step_rad,
        b_bohr=b_bohr,
        min_steps=min_steps,
        max_refinements=max_refinements,
        max_steps=max_steps,
    )

    speed_mps = float(energy_eV_to_speed_mps_for_mass(float(energy_eV), mass_amu))
    speed_au = float(speed_mps_to_atomic_units(speed_mps))
    impact_bohr = float(impact_parameter_ang) / BOHR_TO_ANGSTROM
    r0_bohr = float(r0_ang) / BOHR_TO_ANGSTROM
    r_min_bohr = find_minimum_approach_bohr(
        atomic_number=atomic_number,
        impact_parameter_bohr=impact_bohr,
        r0_bohr=r0_bohr,
        speed_au=speed_au,
        b_bohr=b_bohr,
        chi=chi,
    )
    u0_au = _potential_scalar(r0_bohr, atomic_number, b_bohr=b_bohr, chi=chi)
    dt_initial_au = float(angle_step_rad) * (r_min_bohr * r_min_bohr) / (impact_bohr * speed_au)

    last_result: AtomTrajectoryResult | None = None
    for refinements in range(int(max_refinements) + 1):
        dt_au = dt_initial_au / (10.0 ** refinements)
        result = _integrate_half_trajectory(
            energy_eV=float(energy_eV),
            mass_amu=float(mass_amu),
            atomic_number=float(atomic_number),
            impact_parameter_ang=float(impact_parameter_ang),
            r0_ang=float(r0_ang),
            b_bohr=float(b_bohr),
            angle_step_rad=float(angle_step_rad),
            speed_mps=speed_mps,
            speed_au=speed_au,
            impact_bohr=impact_bohr,
            r0_bohr=r0_bohr,
            u0_au=u0_au,
            r_min_bohr=r_min_bohr,
            dt_initial_au=dt_initial_au,
            dt_au=dt_au,
            refinements=refinements,
            min_steps=int(min_steps),
            max_steps=int(max_steps),
            atomic_number_for_potential=float(atomic_number),
            chi=chi,
            chi_derivative=chi_derivative,
            spin_orbit_c1=float(spin_orbit_c1),
        )
        last_result = result
        if result.steps >= int(min_steps):
            return result

    if last_result is None:
        raise RuntimeError("Не удалось выполнить траекторный расчёт.")
    return AtomTrajectoryResult(
        **{
            **last_result.__dict__,
            "converged": False,
            "status": f"steps < {int(min_steps)} после {int(max_refinements)} уточнений dt",
        }
    )


def find_minimum_approach_bohr(
    *,
    atomic_number: float,
    impact_parameter_bohr: float,
    r0_bohr: float,
    speed_au: float,
    b_bohr: float = DEFAULT_THOMAS_FERMI_B_BOHR,
    chi: ChiFunction = spline_thomas_fermi_chi,
) -> float:
    if impact_parameter_bohr <= 0.0 or r0_bohr <= impact_parameter_bohr:
        raise ValueError("Нужно выполнить 0 < r_п < r0.")

    u0_au = _potential_scalar(r0_bohr, atomic_number, b_bohr=b_bohr, chi=chi)

    def equation(r_bohr: float) -> float:
        return _radial_speed_squared(
            r_bohr,
            atomic_number=atomic_number,
            impact_parameter_bohr=impact_parameter_bohr,
            r0_bohr=r0_bohr,
            u0_au=u0_au,
            speed_au=speed_au,
            b_bohr=b_bohr,
            chi=chi,
        )

    upper = r0_bohr
    upper_value = equation(upper)
    if not np.isfinite(upper_value) or upper_value <= 0.0:
        raise ValueError("В r0 радиальная скорость не положительна. Проверьте r0 и r_п.")

    lower = max(min(impact_parameter_bohr, r0_bohr) * 1e-8, 1e-10)
    radii = np.geomspace(lower, upper, 256)
    previous_r = float(radii[0])
    previous_value = equation(previous_r)
    for current_r in radii[1:]:
        current_r = float(current_r)
        current_value = equation(current_r)
        if not np.isfinite(previous_value):
            previous_r, previous_value = current_r, current_value
            continue
        if np.isfinite(current_value) and previous_value * current_value <= 0.0:
            return _solve_bracketed_root(equation, previous_r, current_r)
        previous_r, previous_value = current_r, current_value

    raise RuntimeError("Не удалось найти r_min: нет смены знака у уравнения сближения.")


def _solve_bracketed_root(function: Callable[[float], float], lower: float, upper: float) -> float:
    if brentq is not None:
        return float(brentq(function, lower, upper, xtol=1e-12, rtol=1e-12, maxiter=100))

    low = float(lower)
    high = float(upper)
    f_low = float(function(low))
    f_high = float(function(high))
    if f_low == 0.0:
        return low
    if f_high == 0.0:
        return high
    if f_low * f_high > 0.0:
        raise RuntimeError("Для бисекции нужен интервал со сменой знака.")

    for _ in range(160):
        mid = 0.5 * (low + high)
        f_mid = float(function(mid))
        if abs(f_mid) < 1e-12 or abs(high - low) < 1e-12:
            return mid
        if f_low * f_mid <= 0.0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return 0.5 * (low + high)


def _integrate_half_trajectory(
    *,
    energy_eV: float,
    mass_amu: float,
    atomic_number: float,
    impact_parameter_ang: float,
    r0_ang: float,
    b_bohr: float,
    angle_step_rad: float,
    speed_mps: float,
    speed_au: float,
    impact_bohr: float,
    r0_bohr: float,
    u0_au: float,
    r_min_bohr: float,
    dt_initial_au: float,
    dt_au: float,
    refinements: int,
    min_steps: int,
    max_steps: int,
    atomic_number_for_potential: float,
    chi: ChiFunction,
    chi_derivative: ChiFunction,
    spin_orbit_c1: float,
) -> AtomTrajectoryResult:
    r_bohr = float(r0_bohr)
    theta_half = 0.0
    phase_half = 0.0
    steps = 0
    dt_used_au = float(dt_au)

    for _ in range(max_steps):
        radial_speed = _radial_speed(
            r_bohr,
            atomic_number=atomic_number_for_potential,
            impact_parameter_bohr=impact_bohr,
            r0_bohr=r0_bohr,
            u0_au=u0_au,
            speed_au=speed_au,
            b_bohr=b_bohr,
            chi=chi,
        )
        angular_rate = impact_bohr * speed_au / (r_bohr * r_bohr)
        phase_rate = 0.5 * spin_orbit_c1 * float(
            _potential_derivative_scalar(
                r_bohr,
                atomic_number_for_potential,
                b_bohr=b_bohr,
                chi=chi,
                chi_derivative=chi_derivative,
            )
        )
        dr_bohr = radial_speed * dt_used_au
        if r_bohr - dr_bohr <= r_min_bohr:
            dr_final = max(r_bohr - r_min_bohr, 0.0)
            dt_final_au = dr_final / radial_speed if radial_speed > 0.0 else 0.0
            theta_half += angular_rate * dt_final_au
            phase_half += phase_rate * dt_final_au
            steps += 1
            break

        theta_half += angular_rate * dt_used_au
        phase_half += phase_rate * dt_used_au
        r_bohr -= dr_bohr
        steps += 1
    else:
        raise RuntimeError(f"Траекторный цикл превысил max_steps={max_steps}. Увеличьте dθ или лимит шагов.")

    theta_rad = 2.0 * theta_half
    phase_rad = 2.0 * phase_half
    alpha_rad = float(np.arcsin(np.clip(impact_bohr / r0_bohr, -1.0, 1.0)))
    trajectory_angle_rad = 2.0 * alpha_rad + theta_rad - np.pi
    converged = steps >= min_steps
    status = "ok" if converged else f"steps < {min_steps}: dt будет уменьшен в 10 раз"

    return AtomTrajectoryResult(
        energy_eV=energy_eV,
        mass_amu=mass_amu,
        atomic_number=atomic_number,
        impact_parameter_ang=impact_parameter_ang,
        r0_ang=r0_ang,
        b_bohr=b_bohr,
        angle_step_rad=angle_step_rad,
        speed_mps=speed_mps,
        speed_au=speed_au,
        r_min_ang=r_min_bohr * BOHR_TO_ANGSTROM,
        theta_rad=theta_rad,
        trajectory_angle_rad=trajectory_angle_rad,
        phase_rad=phase_rad,
        steps=steps,
        dt_initial_au=dt_initial_au,
        dt_final_au=dt_used_au,
        refinements=refinements,
        converged=converged,
        status=status,
    )


def _radial_speed(
    r_bohr: float,
    *,
    atomic_number: float,
    impact_parameter_bohr: float,
    r0_bohr: float,
    u0_au: float,
    speed_au: float,
    b_bohr: float,
    chi: ChiFunction,
) -> float:
    value = _radial_speed_squared(
        r_bohr,
        atomic_number=atomic_number,
            impact_parameter_bohr=impact_parameter_bohr,
            r0_bohr=r0_bohr,
            u0_au=u0_au,
            speed_au=speed_au,
        b_bohr=b_bohr,
        chi=chi,
    )
    if value < 0.0 and abs(value) < 1e-12:
        value = 0.0
    if not np.isfinite(value) or value < 0.0:
        raise RuntimeError("Подкоренное выражение для dr стало отрицательным.")
    return float(np.sqrt(value))


def _radial_speed_squared(
    r_bohr: float,
    *,
    atomic_number: float,
    impact_parameter_bohr: float,
    r0_bohr: float,
    u0_au: float,
    speed_au: float,
    b_bohr: float,
    chi: ChiFunction,
) -> float:
    ur = _potential_scalar(r_bohr, atomic_number, b_bohr=b_bohr, chi=chi)
    centrifugal = (impact_parameter_bohr * impact_parameter_bohr * speed_au * speed_au) / (r_bohr * r_bohr)
    return float(speed_au * speed_au + 2.0 * (u0_au - ur) - centrifugal)


def _potential_scalar(
    r_bohr: float,
    atomic_number: float,
    *,
    b_bohr: float,
    chi: ChiFunction,
) -> float:
    if r_bohr <= 0.0:
        raise ValueError("Расстояние r должно быть положительным.")
    z = float(atomic_number)
    x = (z ** (1.0 / 3.0)) * float(r_bohr) / float(b_bohr)
    return -z * _chi_scalar(chi, x) / float(r_bohr)


def _potential_derivative_scalar(
    r_bohr: float,
    atomic_number: float,
    *,
    b_bohr: float,
    chi: ChiFunction,
    chi_derivative: ChiFunction,
) -> float:
    if r_bohr <= 0.0:
        raise ValueError("Расстояние r должно быть положительным.")
    z = float(atomic_number)
    b = float(b_bohr)
    r = float(r_bohr)
    x = (z ** (1.0 / 3.0)) * r / b
    return z * _chi_scalar(chi, x) / (r * r) - (z ** (4.0 / 3.0)) * _chi_scalar(chi_derivative, x) / (r * b)


def _chi_scalar(chi: ChiFunction, x: float) -> float:
    if chi is spline_thomas_fermi_chi:
        return scalar_spline_thomas_fermi_chi(x)
    if chi is spline_thomas_fermi_chi_derivative:
        return scalar_spline_thomas_fermi_chi_derivative(x)
    value = chi(np.asarray([x], dtype=float))
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def _validate_inputs(
    *,
    atomic_number: float,
    impact_parameter_ang: float,
    r0_ang: float,
    angle_step_rad: float,
    b_bohr: float,
    min_steps: int,
    max_refinements: int,
    max_steps: int,
) -> None:
    if not np.isfinite(atomic_number) or atomic_number <= 0.0:
        raise ValueError("Z должен быть положительным.")
    if not np.isfinite(impact_parameter_ang) or impact_parameter_ang <= 0.0:
        raise ValueError("Прицельное расстояние r_п должно быть положительным.")
    if not np.isfinite(r0_ang) or r0_ang <= impact_parameter_ang:
        raise ValueError("r0 должно быть больше r_п.")
    if not np.isfinite(angle_step_rad) or angle_step_rad <= 0.0:
        raise ValueError("dθ должно быть положительным.")
    if not np.isfinite(b_bohr) or b_bohr <= 0.0:
        raise ValueError("b должно быть положительным.")
    if min_steps <= 0:
        raise ValueError("min_steps должен быть положительным.")
    if max_refinements < 0:
        raise ValueError("max_refinements не может быть отрицательным.")
    if max_steps <= 0:
        raise ValueError("max_steps должен быть положительным.")


__all__ = [
    "ATOMIC_MASS_UNIT_KG",
    "ELECTRON_MASS_AMU",
    "ATOMIC_SPEED_MPS",
    "DEFAULT_THOMAS_FERMI_B_BOHR",
    "DEFAULT_SPIN_ORBIT_C1",
    "AtomTrajectoryResult",
    "energy_eV_to_speed_mps_for_mass",
    "speed_mps_to_atomic_units",
    "thomas_fermi_potential_au",
    "thomas_fermi_potential_derivative_au",
    "find_minimum_approach_bohr",
    "compute_atom_trajectory_phase",
]
