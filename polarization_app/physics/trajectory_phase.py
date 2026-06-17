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
DEFAULT_THOMAS_FERMI_B_BOHR = 0.5 * ((3.0 * np.pi / 4.0) ** (2.0 / 3.0))
DEFAULT_SPIN_ORBIT_C1 = 1.0 / (4.0 * INVERSE_FINE_STRUCTURE * INVERSE_FINE_STRUCTURE)
RADIAL_ROOT_REFINE_THRESHOLD = 2e-3
RADIAL_ROOT_REFINE_SAMPLES = 96
RADIAL_TURNING_SPEED_RELATIVE_THRESHOLD = 1e-3
RADIAL_DOMAIN_REPAIR_SAMPLES = 256
RADIAL_DOMAIN_REPAIR_ATTEMPTS = 8
RADIAL_BASE_PANEL_LIMIT = 1000


@dataclass(frozen=True)
class AtomTrajectoryResult:
    energy_eV: float
    mass_amu: float
    orbital_l: int
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


def mass_amu_to_electron_masses(mass_amu: float) -> float:
    mass_electron_units = float(mass_amu) / ELECTRON_MASS_AMU
    if not np.isfinite(mass_electron_units) or mass_electron_units <= 0.0:
        raise ValueError("Масса в а.е.м должна быть положительной.")
    return mass_electron_units


def speed_mps_to_atomic_units(speed_mps: float | np.ndarray) -> np.ndarray:
    speed_au = np.asarray(speed_mps, dtype=float) / ATOMIC_SPEED_MPS
    if np.any(~np.isfinite(speed_au)) or np.any(speed_au <= 0.0):
        raise ValueError("Скорость должна быть положительной.")
    return speed_au


def thomas_fermi_potential_au(
    r_bohr: float | np.ndarray,
    atomic_number: float,
    chi: ChiFunction = spline_thomas_fermi_chi,
) -> np.ndarray:
    r = np.asarray(r_bohr, dtype=float)
    if np.any(r <= 0.0):
        raise ValueError("Расстояние r должно быть положительным.")
    z = float(atomic_number)
    x = (z ** (1.0 / 3.0)) * r / DEFAULT_THOMAS_FERMI_B_BOHR
    return -z * chi(x) / r


def thomas_fermi_potential_derivative_au(
    r_bohr: float | np.ndarray,
    atomic_number: float,
    chi: ChiFunction = spline_thomas_fermi_chi,
    chi_derivative: ChiFunction = spline_thomas_fermi_chi_derivative,
) -> np.ndarray:
    r = np.asarray(r_bohr, dtype=float)
    if np.any(r <= 0.0):
        raise ValueError("Расстояние r должно быть положительным.")
    z = float(atomic_number)
    b = DEFAULT_THOMAS_FERMI_B_BOHR
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
    orbital_l: int = 0,
    min_steps: int = 30,
    max_refinements: int = 6,
    max_steps: int = 200_000,
    max_phase_step_rad: float | None = None,
    chi: ChiFunction = spline_thomas_fermi_chi,
    chi_derivative: ChiFunction = spline_thomas_fermi_chi_derivative,
    spin_orbit_c1: float = DEFAULT_SPIN_ORBIT_C1,
) -> AtomTrajectoryResult:
    _validate_inputs(
        atomic_number=atomic_number,
        impact_parameter_ang=impact_parameter_ang,
        r0_ang=r0_ang,
        angle_step_rad=angle_step_rad,
        min_steps=min_steps,
        max_refinements=max_refinements,
        max_steps=max_steps,
        orbital_l=orbital_l,
        max_phase_step_rad=max_phase_step_rad,
    )

    speed_mps = float(energy_eV_to_speed_mps_for_mass(float(energy_eV), mass_amu))
    speed_au = float(speed_mps_to_atomic_units(speed_mps))
    mass_electron_units = mass_amu_to_electron_masses(float(mass_amu))
    impact_bohr = float(impact_parameter_ang) / BOHR_TO_ANGSTROM
    r0_bohr = float(r0_ang) / BOHR_TO_ANGSTROM
    r_min_bohr = find_minimum_approach_bohr(
        atomic_number=atomic_number,
        impact_parameter_bohr=impact_bohr,
        r0_bohr=r0_bohr,
        speed_au=speed_au,
        mass_electron_units=mass_electron_units,
        chi=chi,
    )
    u0_au = _potential_scalar(r0_bohr, atomic_number, chi=chi)
    initial_radial_speed_squared = _radial_speed_squared(
        r0_bohr,
        atomic_number=atomic_number,
        impact_parameter_bohr=impact_bohr,
        r0_bohr=r0_bohr,
        u0_au=u0_au,
        mass_electron_units=mass_electron_units,
        speed_au=speed_au,
        chi=chi,
    )
    turning_value_threshold = initial_radial_speed_squared * (RADIAL_TURNING_SPEED_RELATIVE_THRESHOLD ** 2)
    dt_initial_au = float(angle_step_rad) * (r_min_bohr * r_min_bohr) / (impact_bohr * speed_au)

    return _integrate_half_trajectory(
        energy_eV=float(energy_eV),
        mass_amu=float(mass_amu),
        atomic_number=float(atomic_number),
        impact_parameter_ang=float(impact_parameter_ang),
        r0_ang=float(r0_ang),
        angle_step_rad=float(angle_step_rad),
        speed_mps=speed_mps,
        speed_au=speed_au,
        impact_bohr=impact_bohr,
        r0_bohr=r0_bohr,
        u0_au=u0_au,
        mass_electron_units=mass_electron_units,
        r_min_bohr=r_min_bohr,
        dt_initial_au=dt_initial_au,
        turning_value_threshold=turning_value_threshold,
        min_steps=int(min_steps),
        max_refinements=int(max_refinements),
        max_steps=int(max_steps),
        max_phase_step_rad=max_phase_step_rad,
        orbital_l=int(orbital_l),
        atomic_number_for_potential=float(atomic_number),
        chi=chi,
        chi_derivative=chi_derivative,
        spin_orbit_c1=float(spin_orbit_c1),
    )


def find_minimum_approach_bohr(
    *,
    atomic_number: float,
    impact_parameter_bohr: float,
    r0_bohr: float,
    speed_au: float,
    mass_electron_units: float = 1.0,
    chi: ChiFunction = spline_thomas_fermi_chi,
) -> float:
    if impact_parameter_bohr <= 0.0 or r0_bohr <= impact_parameter_bohr:
        raise ValueError("Нужно выполнить 0 < r_п < r0.")

    u0_au = _potential_scalar(r0_bohr, atomic_number, chi=chi)

    def equation(r_bohr: float) -> float:
        return _radial_speed_squared(
            r_bohr,
            atomic_number=atomic_number,
            impact_parameter_bohr=impact_parameter_bohr,
            r0_bohr=r0_bohr,
            u0_au=u0_au,
            mass_electron_units=mass_electron_units,
            speed_au=speed_au,
            chi=chi,
        )

    upper = r0_bohr
    upper_value = equation(upper)
    if not np.isfinite(upper_value) or upper_value <= 0.0:
        raise ValueError("В r0 радиальная скорость не положительна. Проверьте r0 и r_п.")

    turning_value_threshold = upper_value * (RADIAL_TURNING_SPEED_RELATIVE_THRESHOLD ** 2)
    lower = max(min(impact_parameter_bohr, r0_bohr) * 1e-8, 1e-10)
    radii = np.geomspace(lower, upper, 768)
    previous_r = upper
    previous_value = upper_value
    best_near_turning_r = upper
    best_near_turning_value = upper_value
    for current_r in reversed(radii[:-1]):
        current_r = float(current_r)
        current_value = equation(current_r)
        if np.isfinite(current_value) and current_value >= 0.0 and current_value < best_near_turning_value:
            best_near_turning_r = current_r
            best_near_turning_value = current_value
        if not np.isfinite(previous_value):
            previous_r, previous_value = current_r, current_value
            continue
        root = _find_outermost_root_in_interval(
            equation,
            inner_r=current_r,
            outer_r=previous_r,
            inner_value=current_value,
            outer_value=previous_value,
            turning_value_threshold=turning_value_threshold,
        )
        if root is not None:
            return root
        previous_r, previous_value = current_r, current_value

    if best_near_turning_value <= turning_value_threshold:
        return float(best_near_turning_r)
    raise RuntimeError("Не удалось найти r_min: нет смены знака у уравнения сближения.")


def _find_outermost_root_in_interval(
    function: Callable[[float], float],
    *,
    inner_r: float,
    outer_r: float,
    inner_value: float,
    outer_value: float,
    turning_value_threshold: float,
) -> float | None:
    if not (np.isfinite(inner_value) and np.isfinite(outer_value)):
        return None
    if inner_value * outer_value <= 0.0:
        return _solve_bracketed_root(function, inner_r, outer_r)
    if min(abs(inner_value), abs(outer_value)) > RADIAL_ROOT_REFINE_THRESHOLD:
        return None

    sample_radii = np.linspace(float(outer_r), float(inner_r), RADIAL_ROOT_REFINE_SAMPLES)
    previous_r = float(sample_radii[0])
    previous_value = float(outer_value)
    best_r = previous_r
    best_value = previous_value if previous_value >= 0.0 else float("inf")
    for current_r in sample_radii[1:]:
        current_r = float(current_r)
        current_value = float(function(current_r))
        if np.isfinite(current_value) and current_value >= 0.0 and current_value < best_value:
            best_r = current_r
            best_value = current_value
        if not (np.isfinite(previous_value) and np.isfinite(current_value)):
            previous_r, previous_value = current_r, current_value
            continue
        if previous_value * current_value <= 0.0:
            return _solve_bracketed_root(function, current_r, previous_r)
        previous_r, previous_value = current_r, current_value
    if best_value <= turning_value_threshold:
        return float(best_r)
    return None


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


def _repair_radial_integration_lower_bound(
    *,
    r_min_bohr: float,
    r0_bohr: float,
    atomic_number: float,
    impact_parameter_bohr: float,
    u0_au: float,
    mass_electron_units: float,
    speed_au: float,
    turning_value_threshold: float,
    chi: ChiFunction,
) -> float:
    lower = float(r_min_bohr)
    upper = float(r0_bohr)
    tolerance = max(1e-12, float(turning_value_threshold))
    if lower <= 0.0 or lower >= upper:
        return lower

    def equation(r_bohr: float) -> float:
        return _radial_speed_squared(
            float(r_bohr),
            atomic_number=atomic_number,
            impact_parameter_bohr=impact_parameter_bohr,
            r0_bohr=upper,
            u0_au=u0_au,
            mass_electron_units=mass_electron_units,
            speed_au=speed_au,
            chi=chi,
        )

    for _ in range(RADIAL_DOMAIN_REPAIR_ATTEMPTS):
        y_values = np.linspace(0.0, 1.0, RADIAL_DOMAIN_REPAIR_SAMPLES, dtype=float)
        sample_radii = lower + (upper - lower) * y_values * y_values
        sample_values = np.array([equation(radius) for radius in sample_radii], dtype=float)
        if np.any(~np.isfinite(sample_values)):
            raise RuntimeError("Подкоренное выражение для dr стало нечисловым.")

        negative_indices = np.flatnonzero(sample_values < -tolerance)
        if negative_indices.size == 0:
            return lower

        last_negative_index = int(negative_indices[-1])
        outer_indices = np.flatnonzero(sample_values[last_negative_index + 1 :] >= 0.0)
        if outer_indices.size == 0:
            raise RuntimeError("Подкоренное выражение для dr осталось отрицательным до r0.")

        outer_index = last_negative_index + 1 + int(outer_indices[0])
        repaired = _solve_bracketed_root(
            equation,
            float(sample_radii[last_negative_index]),
            float(sample_radii[outer_index]),
        )
        guard = max(abs(repaired) * 1e-12, abs(upper - repaired) * 1e-12, 1e-14)
        lower = min(upper, float(repaired) + guard)

    raise RuntimeError("Не удалось вывести r_min из области отрицательного подкоренного выражения.")


def _base_quadrature_panel_count(*, angular_step_rad: float, min_steps: int) -> int:
    angle_driven_panels = int(np.ceil(np.pi / max(float(angular_step_rad), 1e-12)))
    capped_angle_panels = min(angle_driven_panels, RADIAL_BASE_PANEL_LIMIT)
    return max(int(min_steps), capped_angle_panels)


def _integrate_half_trajectory(
    *,
    energy_eV: float,
    mass_amu: float,
    atomic_number: float,
    impact_parameter_ang: float,
    r0_ang: float,
    angle_step_rad: float,
    speed_mps: float,
    speed_au: float,
    impact_bohr: float,
    r0_bohr: float,
    u0_au: float,
    mass_electron_units: float,
    r_min_bohr: float,
    dt_initial_au: float,
    turning_value_threshold: float,
    min_steps: int,
    max_refinements: int,
    max_steps: int,
    max_phase_step_rad: float | None,
    orbital_l: int,
    atomic_number_for_potential: float,
    chi: ChiFunction,
    chi_derivative: ChiFunction,
    spin_orbit_c1: float,
) -> AtomTrajectoryResult:
    effective_r_min_bohr = _repair_radial_integration_lower_bound(
        r_min_bohr=float(r_min_bohr),
        r0_bohr=float(r0_bohr),
        atomic_number=float(atomic_number_for_potential),
        impact_parameter_bohr=float(impact_bohr),
        u0_au=float(u0_au),
        mass_electron_units=float(mass_electron_units),
        speed_au=float(speed_au),
        turning_value_threshold=float(turning_value_threshold),
        chi=chi,
    )
    angular_step_rad = float(angle_step_rad)
    angular_denominator = impact_bohr * speed_au

    def radial_equation(r_bohr: float) -> float:
        return _radial_speed_squared(
            float(r_bohr),
            atomic_number=float(atomic_number_for_potential),
            impact_parameter_bohr=float(impact_bohr),
            r0_bohr=float(r0_bohr),
            u0_au=float(u0_au),
            mass_electron_units=float(mass_electron_units),
            speed_au=float(speed_au),
            chi=chi,
        )

    def integrate_by_radius(panel_count: int) -> tuple[float, float, bool]:
        nonlocal effective_r_min_bohr
        repaired_domain = False
        panels = int(panel_count)
        z = float(atomic_number_for_potential)
        b = DEFAULT_THOMAS_FERMI_B_BOHR
        tolerance = max(1e-12, float(turning_value_threshold))
        for _ in range(RADIAL_DOMAIN_REPAIR_ATTEMPTS):
            y = (np.arange(panels, dtype=float) + 0.5) / panels
            radial_span = r0_bohr - effective_r_min_bohr
            r_values = effective_r_min_bohr + radial_span * y * y
            dr_dy = 2.0 * radial_span * y
            x_values = (z ** (1.0 / 3.0)) * r_values / b
            chi_values = np.asarray(chi(x_values), dtype=float)
            chi_derivative_values = np.asarray(chi_derivative(x_values), dtype=float)
            potential_values = -z * chi_values / r_values
            radial_speed_squared = (
                speed_au * speed_au
                + 2.0 * (u0_au - potential_values) / float(mass_electron_units)
                - (impact_bohr * impact_bohr * speed_au * speed_au) / (r_values * r_values)
            )
            if np.any(~np.isfinite(radial_speed_squared)):
                raise RuntimeError("Подкоренное выражение для dr стало нечисловым.")

            small_negative = (radial_speed_squared < 0.0) & (np.abs(radial_speed_squared) <= tolerance)
            radial_speed_squared = np.where(small_negative, 0.0, radial_speed_squared)
            negative_indices = np.flatnonzero(radial_speed_squared < 0.0)
            if negative_indices.size:
                last_negative_index = int(negative_indices[-1])
                outer_indices = np.flatnonzero(radial_speed_squared[last_negative_index + 1 :] >= 0.0)
                if outer_indices.size:
                    outer_radius = float(r_values[last_negative_index + 1 + int(outer_indices[0])])
                elif radial_equation(r0_bohr) >= 0.0:
                    outer_radius = float(r0_bohr)
                else:
                    raise RuntimeError("Подкоренное выражение для dr осталось отрицательным до r0.")

                repaired = _solve_bracketed_root(
                    radial_equation,
                    float(r_values[last_negative_index]),
                    outer_radius,
                )
                guard = max(abs(repaired) * 1e-12, abs(r0_bohr - repaired) * 1e-12, 1e-14)
                new_lower = min(float(r0_bohr), float(repaired) + guard)
                if new_lower <= effective_r_min_bohr:
                    new_lower = min(float(r0_bohr), np.nextafter(effective_r_min_bohr, float(r0_bohr)))
                effective_r_min_bohr = new_lower
                repaired_domain = True
                continue

            radial_speed = np.sqrt(np.maximum(radial_speed_squared, 1e-300))
            theta_integrand = impact_bohr * speed_au * dr_dy / (r_values * r_values * radial_speed)
            orbital_factor = 2 * int(orbital_l) + 1
            phase_rate = 0.5 * float(spin_orbit_c1) * orbital_factor * (
                z * chi_values / (r_values * r_values * r_values)
                - (z ** (4.0 / 3.0)) * chi_derivative_values / (r_values * r_values * b)
            )
            phase_integrand = phase_rate * dr_dy / radial_speed
            return float(np.sum(theta_integrand) / panels), float(np.sum(phase_integrand) / panels), repaired_domain

        raise RuntimeError("Не удалось вывести r_min из области отрицательного подкоренного выражения.")

    if r0_bohr <= effective_r_min_bohr:
        theta_half = 0.0
        phase_half = 0.0
        steps = 1
        grid_refinements = 0
        quadrature_converged = True
    else:
        base_steps = _base_quadrature_panel_count(
            angular_step_rad=angular_step_rad,
            min_steps=int(min_steps),
        )

        previous_theta_half: float | None = None
        previous_phase_half: float | None = None
        theta_half = 0.0
        phase_half = 0.0
        steps = base_steps
        grid_refinements = 0
        quadrature_converged = int(max_refinements) <= 0
        phase_tolerance = min(2e-3, float(max_phase_step_rad) * 0.05) if max_phase_step_rad is not None else 2e-3
        theta_tolerance = 1e-3
        for refinement_index in range(int(max_refinements) + 1):
            steps = min(max(base_steps * (2 ** refinement_index), 1), int(max_steps))
            grid_refinements = refinement_index
            theta_half, phase_half, repaired_domain = integrate_by_radius(steps)
            if repaired_domain:
                previous_theta_half = None
                previous_phase_half = None
            if previous_theta_half is not None and previous_phase_half is not None:
                if (
                    abs(theta_half - previous_theta_half) <= theta_tolerance
                    and abs(phase_half - previous_phase_half) <= phase_tolerance
                ):
                    quadrature_converged = True
                    break
            previous_theta_half = theta_half
            previous_phase_half = phase_half
            if steps >= int(max_steps):
                break

    theta_rad = 2.0 * theta_half
    phase_rad = 2.0 * phase_half
    alpha_rad = float(np.arcsin(np.clip(impact_bohr / r0_bohr, -1.0, 1.0)))
    trajectory_angle_rad = 2.0 * alpha_rad + theta_rad - np.pi
    has_minimum_panels = steps >= min_steps
    converged = bool(has_minimum_panels and quadrature_converged)
    if converged:
        status = "ok"
    elif not quadrature_converged:
        status = (
            f"квадратура не сошлась после {grid_refinements} уточнений сетки; "
            f"steps={steps}, max_steps={max_steps}"
        )
    else:
        status = f"steps < {min_steps}: сетка интегрирования будет уточнена"

    return AtomTrajectoryResult(
        energy_eV=energy_eV,
        mass_amu=mass_amu,
        orbital_l=orbital_l,
        atomic_number=atomic_number,
        impact_parameter_ang=impact_parameter_ang,
        r0_ang=r0_ang,
        b_bohr=DEFAULT_THOMAS_FERMI_B_BOHR,
        angle_step_rad=angle_step_rad,
        speed_mps=speed_mps,
        speed_au=speed_au,
        r_min_ang=effective_r_min_bohr * BOHR_TO_ANGSTROM,
        theta_rad=theta_rad,
        trajectory_angle_rad=trajectory_angle_rad,
        phase_rad=phase_rad,
        steps=steps,
        dt_initial_au=dt_initial_au,
        dt_final_au=theta_half * effective_r_min_bohr * effective_r_min_bohr / max(angular_denominator * max(steps, 1), 1e-300),
        refinements=grid_refinements,
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
    mass_electron_units: float,
    speed_au: float,
    chi: ChiFunction,
) -> float:
    value = _radial_speed_squared(
        r_bohr,
        atomic_number=atomic_number,
        impact_parameter_bohr=impact_parameter_bohr,
        r0_bohr=r0_bohr,
        u0_au=u0_au,
        mass_electron_units=mass_electron_units,
        speed_au=speed_au,
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
    mass_electron_units: float,
    speed_au: float,
    chi: ChiFunction,
) -> float:
    ur = _potential_scalar(r_bohr, atomic_number, chi=chi)
    centrifugal = (impact_parameter_bohr * impact_parameter_bohr * speed_au * speed_au) / (r_bohr * r_bohr)
    return float(speed_au * speed_au + 2.0 * (u0_au - ur) / float(mass_electron_units) - centrifugal)


def _potential_scalar(
    r_bohr: float,
    atomic_number: float,
    *,
    chi: ChiFunction,
) -> float:
    if r_bohr <= 0.0:
        raise ValueError("Расстояние r должно быть положительным.")
    z = float(atomic_number)
    x = (z ** (1.0 / 3.0)) * float(r_bohr) / DEFAULT_THOMAS_FERMI_B_BOHR
    return -z * _chi_scalar(chi, x) / float(r_bohr)


def _potential_derivative_scalar(
    r_bohr: float,
    atomic_number: float,
    *,
    chi: ChiFunction,
    chi_derivative: ChiFunction,
) -> float:
    if r_bohr <= 0.0:
        raise ValueError("Расстояние r должно быть положительным.")
    z = float(atomic_number)
    b = DEFAULT_THOMAS_FERMI_B_BOHR
    r = float(r_bohr)
    x = (z ** (1.0 / 3.0)) * r / b
    return z * _chi_scalar(chi, x) / (r * r) - (z ** (4.0 / 3.0)) * _chi_scalar(chi_derivative, x) / (r * b)


def _trajectory_phase_rate_scalar(
    r_bohr: float,
    atomic_number: float,
    *,
    chi: ChiFunction,
    chi_derivative: ChiFunction,
    spin_orbit_c1: float,
    orbital_l: int,
) -> float:
    if r_bohr <= 0.0:
        raise ValueError("Расстояние r должно быть положительным.")
    z = float(atomic_number)
    b = DEFAULT_THOMAS_FERMI_B_BOHR
    r = float(r_bohr)
    x = (z ** (1.0 / 3.0)) * r / b
    orbital_factor = 2 * int(orbital_l) + 1
    return 0.5 * float(spin_orbit_c1) * orbital_factor * (
        z * _chi_scalar(chi, x) / (r * r * r)
        - (z ** (4.0 / 3.0)) * _chi_scalar(chi_derivative, x) / (r * r * b)
    )


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
    min_steps: int,
    max_refinements: int,
    max_steps: int,
    orbital_l: int,
    max_phase_step_rad: float | None,
) -> None:
    if not np.isfinite(atomic_number) or atomic_number <= 0.0:
        raise ValueError("Z должен быть положительным.")
    if not np.isfinite(impact_parameter_ang) or impact_parameter_ang <= 0.0:
        raise ValueError("Прицельное расстояние r_п должно быть положительным.")
    if not np.isfinite(r0_ang) or r0_ang <= impact_parameter_ang:
        raise ValueError("r0 должно быть больше r_п.")
    if not np.isfinite(angle_step_rad) or angle_step_rad <= 0.0:
        raise ValueError("dθ должно быть положительным.")
    if min_steps <= 0:
        raise ValueError("min_steps должен быть положительным.")
    if max_refinements < 0:
        raise ValueError("max_refinements не может быть отрицательным.")
    if max_steps <= 0:
        raise ValueError("max_steps должен быть положительным.")
    if int(orbital_l) < 0:
        raise ValueError("L должен быть неотрицательным.")
    if max_phase_step_rad is not None and (
        not np.isfinite(max_phase_step_rad) or float(max_phase_step_rad) <= 0.0
    ):
        raise ValueError("max_phase_step_rad должен быть положительным.")


__all__ = [
    "ATOMIC_MASS_UNIT_KG",
    "ELECTRON_MASS_AMU",
    "ATOMIC_SPEED_MPS",
    "DEFAULT_THOMAS_FERMI_B_BOHR",
    "DEFAULT_SPIN_ORBIT_C1",
    "RADIAL_BASE_PANEL_LIMIT",
    "AtomTrajectoryResult",
    "energy_eV_to_speed_mps_for_mass",
    "mass_amu_to_electron_masses",
    "speed_mps_to_atomic_units",
    "thomas_fermi_potential_au",
    "thomas_fermi_potential_derivative_au",
    "find_minimum_approach_bohr",
    "compute_atom_trajectory_phase",
]
