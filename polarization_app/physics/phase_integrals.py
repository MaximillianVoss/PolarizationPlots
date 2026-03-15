# -*- coding: utf-8 -*-
import logging
import os
from datetime import datetime
from typing import Callable, Literal

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

ELECTRON_CHARGE = 1.602176634e-19
ELECTRON_MASS = 9.1093837015e-31
BOHR_TO_ANGSTROM = 0.52917721092
LIGHT_SPEED = 299792458.0
INVERSE_FINE_STRUCTURE = 137.04

_THOMAS_FERMI_X = np.array([
    0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20,
    0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
    2.2, 2.4, 2.6, 2.8, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
], dtype=float)
_THOMAS_FERMI_Y = np.array([
    1.000, 0.972, 0.947, 0.924, 0.902, 0.882, 0.863, 0.845, 0.828, 0.812, 0.797,
    0.721, 0.667, 0.621, 0.580, 0.544, 0.512, 0.482, 0.454, 0.374, 0.333, 0.298, 0.268, 0.243,
    0.221, 0.202, 0.185, 0.170, 0.157, 0.105, 0.0788, 0.0594, 0.0366, 0.0243, 0.0123, 0.0088,
    0.0035, 0.0022, 0.00063, 0.00039,
], dtype=float)

ChiFunction = Callable[[np.ndarray, dict[str, float] | None], np.ndarray]


def energy_to_speed_mps(energy_eV: np.ndarray) -> np.ndarray:
    energy_joule = np.asarray(energy_eV, dtype=float) * ELECTRON_CHARGE
    return np.sqrt(2.0 * energy_joule / ELECTRON_MASS)


def speed_to_energy_eV(speed_mps: np.ndarray) -> np.ndarray:
    energy_joule = 0.5 * ELECTRON_MASS * np.asarray(speed_mps, dtype=float) ** 2
    return energy_joule / ELECTRON_CHARGE


def interpolate_thomas_fermi_chi(x: np.ndarray, params: dict[str, float] | None = None) -> np.ndarray:
    del params
    x = np.asarray(x, dtype=float)
    return np.interp(x, _THOMAS_FERMI_X, _THOMAS_FERMI_Y, left=_THOMAS_FERMI_Y[0], right=_THOMAS_FERMI_Y[-1])


def exponential_chi(x: np.ndarray, params: dict[str, float] | None = None) -> np.ndarray:
    del params
    x = np.asarray(x, dtype=float)
    return np.exp(-x)


def _speed_mps_to_atomic_units(speed_mps: float | np.ndarray) -> np.ndarray:
    speed_au = (np.asarray(speed_mps, dtype=float) / LIGHT_SPEED) * INVERSE_FINE_STRUCTURE
    if np.any(~np.isfinite(speed_au)) or np.any(speed_au <= 0.0):
        raise ValueError(f"Некорректная скорость: {speed_mps}")
    return speed_au


def _evaluate_chi_values(
    chi: ChiFunction,
    x_values: np.ndarray,
    chi_params: dict[str, float] | None,
) -> np.ndarray:
    values = chi(x_values, chi_params) if chi_params is not None else chi(x_values)
    return np.clip(np.asarray(values, dtype=float), 0.0, None)


def _compute_phase_geometry_coefficients(
    *,
    a_ang: float,
    Z: float,
    b_ang: float,
    c1: float,
    c2: float,
    dr_ang: float,
    r_max_ang: float,
    chi: ChiFunction,
    chi_params: dict[str, float] | None = None,
    i3_mode: str = "sum_avg",
) -> tuple[float, float, float, float]:
    a = float(a_ang) / BOHR_TO_ANGSTROM
    b = float(b_ang) / BOHR_TO_ANGSTROM
    dr = float(dr_ang) / BOHR_TO_ANGSTROM
    r_max = float(r_max_ang) / BOHR_TO_ANGSTROM

    eps_a = 1e-6
    if not np.isfinite(a) or a <= eps_a:
        raise ValueError(f"a слишком мал/некорректен: a_ang={a_ang}, a={a} a0")
    if not np.isfinite(b) or b <= 0.0:
        raise ValueError(f"b некорректен: b_ang={b_ang}, b={b} a0")
    if not np.isfinite(dr) or dr <= 0.0:
        raise ValueError(f"dr некорректен: dr_ang={dr_ang}, dr={dr} a0")
    if not np.isfinite(r_max) or r_max <= a:
        raise ValueError(f"r_max должен быть > a: r_max_ang={r_max_ang}, r_max={r_max} a0, a={a} a0")

    z13 = Z ** (1.0 / 3.0)
    prefactor = 1.0 / (a ** 5)

    r_grid = np.linspace(a, r_max, int(max(2, np.ceil((r_max - a) / dr) + 1)))
    term = r_grid ** 2 - a ** 2
    chi_r = _evaluate_chi_values(chi, z13 * r_grid / b, chi_params)
    chi_r_32 = np.power(chi_r, 1.5)

    f1 = term / np.power(r_grid, 2.5) * chi_r_32
    f2 = term / np.power(r_grid, 4.0) * chi_r
    i1_integral = float(np.trapezoid(f1, r_grid))
    i2_integral = float(np.trapezoid(f2, r_grid))

    if i3_mode == "trapz":
        r_shift = np.minimum(r_grid + dr, r_max)
        chi_shift = _evaluate_chi_values(chi, z13 * r_shift / b, chi_params)
        f3 = term / np.power(r_grid, 3.0) * (chi_shift - chi_r)
        i3_integral = float(np.trapezoid(f3, r_grid))
    elif i3_mode == "sum_avg":
        r_values = np.arange(a, (r_max - dr) + dr * 1e-9, dr, dtype=float)
        if r_values.size == 0:
            i3_integral = 0.0
        else:
            chi0 = _evaluate_chi_values(chi, z13 * r_values / b, chi_params)
            chi1 = _evaluate_chi_values(chi, z13 * (r_values + dr) / b, chi_params)
            i3_terms = (r_values ** 2 - a ** 2) / np.power(r_values, 3.0) * (chi1 - chi0)
            i3_integral = float(i3_terms.mean())
    else:
        raise ValueError(f"Неизвестный i3_mode: {i3_mode}")

    i1_coefficient = (-(2.0 * c1 * c2)) * (Z ** 1.5) * prefactor * i1_integral
    i2_coefficient = (-(6.0 * c1 * Z)) * prefactor * i2_integral
    i3_coefficient = ((6.0 * c1 * Z * b) / (Z ** (1.0 / 3.0))) * prefactor * i3_integral
    total_phase_coefficient = i1_coefficient + i2_coefficient + i3_coefficient
    return i1_coefficient, i2_coefficient, i3_coefficient, total_phase_coefficient


def compute_phase_integral_components(
    speed_mps: float,
    a_ang: float,
    Z: float,
    b_ang: float,
    c1: float,
    c2: float,
    dr_ang: float,
    r_max_ang: float,
    chi: ChiFunction,
    chi_params: dict[str, float] | None = None,
    i3_mode: str = "sum_avg",
) -> tuple[float, float, float, float]:
    """
    Считает I1, I2, I3 и суммарную фазу.
    """
    speed_au = float(_speed_mps_to_atomic_units(float(speed_mps)))
    coefficients = np.asarray(
        _compute_phase_geometry_coefficients(
            a_ang=a_ang,
            Z=Z,
            b_ang=b_ang,
            c1=c1,
            c2=c2,
            dr_ang=dr_ang,
            r_max_ang=r_max_ang,
            chi=chi,
            chi_params=chi_params,
            i3_mode=i3_mode,
        ),
        dtype=float,
    )
    scaled_components = coefficients / speed_au
    return tuple(float(value) for value in scaled_components)


def compute_phase_grid_for_atoms(
    Emin_eV: float,
    Emax_eV: float,
    N: int,
    *,
    a_list_ang: list[float],
    Z: float,
    b_ang: float,
    c1: float,
    c2: float,
    dr_ang: float,
    r_max_ang: float,
    chi: ChiFunction = interpolate_thomas_fermi_chi,
    chi_params: dict[str, float] | None = None,
    i3_mode: Literal["trapz", "sum_avg"] = "sum_avg",
    dump_atom_phi_csv: bool = True,
    max_atoms_dump: int = 200,
) -> pd.DataFrame:
    _validate_atom_phase_inputs(Emin_eV, Emax_eV, a_list_ang)
    grid, atom_phase_matrix, impact_parameters = _build_atom_phase_grid(
        Emin_eV=Emin_eV,
        Emax_eV=Emax_eV,
        N=N,
        a_list_ang=a_list_ang,
        Z=Z,
        b_ang=b_ang,
        c1=c1,
        c2=c2,
        dr_ang=dr_ang,
        r_max_ang=r_max_ang,
        chi=chi,
        chi_params=chi_params,
        i3_mode=i3_mode,
        max_atoms_dump=max_atoms_dump,
        include_atom_phase_matrix=dump_atom_phi_csv,
    )
    if dump_atom_phi_csv and atom_phase_matrix is not None:
        _dump_per_atom_phase_csv(
            energy_grid=grid["E_eV"].to_numpy(dtype=float),
            impact_parameters=impact_parameters,
            atom_phase_matrix=atom_phase_matrix,
        )
    return grid


def compute_phase_grid_for_atoms_with_matrix(
    Emin_eV: float,
    Emax_eV: float,
    N: int,
    *,
    a_list_ang: list[float],
    Z: float,
    b_ang: float,
    c1: float,
    c2: float,
    dr_ang: float,
    r_max_ang: float,
    chi: ChiFunction = interpolate_thomas_fermi_chi,
    chi_params: dict[str, float] | None = None,
    i3_mode: str = "sum_avg",
    dump_atom_phi_csv: bool = True,
    max_atoms_dump: int = 200,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    _validate_atom_phase_inputs(Emin_eV, Emax_eV, a_list_ang)
    grid, atom_phase_matrix, impact_parameters = _build_atom_phase_grid(
        Emin_eV=Emin_eV,
        Emax_eV=Emax_eV,
        N=N,
        a_list_ang=a_list_ang,
        Z=Z,
        b_ang=b_ang,
        c1=c1,
        c2=c2,
        dr_ang=dr_ang,
        r_max_ang=r_max_ang,
        chi=chi,
        chi_params=chi_params,
        i3_mode=i3_mode,
        max_atoms_dump=max_atoms_dump,
        include_atom_phase_matrix=True,
    )
    if dump_atom_phi_csv and atom_phase_matrix is not None:
        _dump_per_atom_phase_csv(
            energy_grid=grid["E_eV"].to_numpy(dtype=float),
            impact_parameters=impact_parameters,
            atom_phase_matrix=atom_phase_matrix,
        )
    return grid, atom_phase_matrix, impact_parameters


def compute_single_atom_phase_grid(
    Emin_eV: float,
    Emax_eV: float,
    N: int,
    *,
    Z: float,
    a_ang: float,
    b_ang: float,
    c1: float,
    c2: float,
    dr_ang: float,
    r_max_ang: float,
    chi: ChiFunction = interpolate_thomas_fermi_chi,
    chi_params: dict[str, float] | None = None,
    i3_mode: Literal["trapz", "sum_avg"] = "sum_avg",
) -> pd.DataFrame:
    if Emin_eV <= 0 or Emax_eV <= 0 or Emax_eV <= Emin_eV:
        raise ValueError("Требуется 0 < Emin < Emax.")

    energy_grid = np.logspace(np.log10(Emin_eV), np.log10(Emax_eV), int(N))
    speed_grid = energy_to_speed_mps(energy_grid)
    speed_au = _speed_mps_to_atomic_units(speed_grid)
    coefficients = np.asarray(
        _compute_phase_geometry_coefficients(
            a_ang=a_ang,
            Z=Z,
            b_ang=b_ang,
            c1=c1,
            c2=c2,
            dr_ang=dr_ang,
            r_max_ang=r_max_ang,
            chi=chi,
            chi_params=chi_params,
            i3_mode=i3_mode,
        ),
        dtype=float,
    )
    component_matrix = coefficients[np.newaxis, :] / speed_au[:, np.newaxis]
    i1_values = component_matrix[:, 0]
    i2_values = component_matrix[:, 1]
    i3_values = component_matrix[:, 2]
    total_phase_values = component_matrix[:, 3]

    logger.info("compute_single_atom_phase_grid | Emin=%.3g eV, Emax=%.3g eV, N=%d, i3_mode=%s", Emin_eV, Emax_eV, N, i3_mode)
    phase_array = np.asarray(total_phase_values, dtype=float)
    logger.info(
        "compute_single_atom_phase_grid | Phi stats: min=%.6g, max=%.6g, mean=%.6g",
        float(phase_array.min()),
        float(phase_array.max()),
        float(phase_array.mean()),
    )

    _dump_single_atom_phase_csv(
        energy_grid=energy_grid,
        speed_grid=speed_grid,
        i1_values=np.asarray(i1_values, dtype=float),
        i2_values=np.asarray(i2_values, dtype=float),
        i3_values=np.asarray(i3_values, dtype=float),
        total_phase_values=phase_array,
    )

    return pd.DataFrame(
        {
            "E_eV": energy_grid,
            "V_m_per_s": speed_grid,
            "I1": np.asarray(i1_values, dtype=float),
            "I2": np.asarray(i2_values, dtype=float),
            "I3": np.asarray(i3_values, dtype=float),
            "Phi": phase_array,
        }
    )


def _validate_atom_phase_inputs(Emin_eV: float, Emax_eV: float, a_list_ang: list[float]) -> None:
    if Emin_eV <= 0 or Emax_eV <= 0 or Emax_eV <= Emin_eV:
        raise ValueError("Требуется 0 < Emin < Emax.")
    if not a_list_ang:
        raise ValueError("Список a_list_ang пуст: нет атомов для суммирования.")
    if any(a <= 0 for a in a_list_ang):
        raise ValueError("В a_list_ang найдены некорректные значения (<=0).")


def _build_atom_phase_grid(
    *,
    Emin_eV: float,
    Emax_eV: float,
    N: int,
    a_list_ang: list[float],
    Z: float,
    b_ang: float,
    c1: float,
    c2: float,
    dr_ang: float,
    r_max_ang: float,
    chi: ChiFunction,
    chi_params: dict[str, float] | None,
    i3_mode: str,
    max_atoms_dump: int,
    include_atom_phase_matrix: bool,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray]:
    energy_grid = np.logspace(np.log10(Emin_eV), np.log10(Emax_eV), int(N))
    speed_grid = energy_to_speed_mps(energy_grid)
    speed_au = _speed_mps_to_atomic_units(speed_grid)

    impact_parameters = np.asarray(a_list_ang, dtype=float)
    impact_parameters = np.sort(impact_parameters)[:int(max_atoms_dump)]
    coefficient_matrix = np.asarray(
        [
            _compute_phase_geometry_coefficients(
                a_ang=float(impact_parameter),
                Z=Z,
                b_ang=b_ang,
                c1=c1,
                c2=c2,
                dr_ang=dr_ang,
                r_max_ang=r_max_ang,
                chi=chi,
                chi_params=chi_params,
                i3_mode=i3_mode,
            )
            for impact_parameter in impact_parameters
        ],
        dtype=float,
    )
    summed_coefficients = coefficient_matrix.sum(axis=0)
    inv_speed_au = 1.0 / speed_au

    i1_values = summed_coefficients[0] * inv_speed_au
    i2_values = summed_coefficients[1] * inv_speed_au
    i3_values = summed_coefficients[2] * inv_speed_au
    total_phase_values = summed_coefficients[3] * inv_speed_au

    atom_phase_matrix = None
    if include_atom_phase_matrix:
        atom_phase_matrix = coefficient_matrix[:, 3][np.newaxis, :] * inv_speed_au[:, np.newaxis]

    phase_array = np.asarray(total_phase_values, dtype=float)
    logger.info(
        "compute_phase_grid_for_atoms | Emin=%.3g eV, Emax=%.3g eV, N=%d, atoms=%d, i3_mode=%s",
        Emin_eV,
        Emax_eV,
        int(N),
        int(len(impact_parameters)),
        i3_mode,
    )
    logger.info(
        "compute_phase_grid_for_atoms | Phi stats: min=%.6g, max=%.6g, mean=%.6g",
        float(phase_array.min()),
        float(phase_array.max()),
        float(phase_array.mean()),
    )

    return (
        pd.DataFrame(
            {
                "E_eV": energy_grid,
                "V_m_per_s": speed_grid,
                "I1": np.asarray(i1_values, dtype=float),
                "I2": np.asarray(i2_values, dtype=float),
                "I3": np.asarray(i3_values, dtype=float),
                "Phi": phase_array,
            }
        ),
        atom_phase_matrix,
        impact_parameters,
    )


def _dump_single_atom_phase_csv(
    *,
    energy_grid: np.ndarray,
    speed_grid: np.ndarray,
    i1_values: np.ndarray,
    i2_values: np.ndarray,
    i3_values: np.ndarray,
    total_phase_values: np.ndarray,
) -> None:
    try:
        os.makedirs("data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join("data", f"phi_dump_{timestamp}.csv")
        pd.DataFrame(
            {
                "E_eV": energy_grid,
                "V_m_per_s": speed_grid,
                "I1": i1_values,
                "I2": i2_values,
                "I3": i3_values,
                "Phi": total_phase_values,
            }
        ).to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(
            "compute_single_atom_phase_grid | Phi saved to CSV: %s (%d rows)",
            os.path.abspath(csv_path),
            len(energy_grid),
        )
        _prune_generated_csv_files("data", "phi_dump_*.csv", keep_latest=10)
    except Exception:
        logger.exception("compute_single_atom_phase_grid | failed to save Phi CSV")


def _dump_per_atom_phase_csv(
    *,
    energy_grid: np.ndarray,
    impact_parameters: np.ndarray,
    atom_phase_matrix: np.ndarray,
) -> None:
    try:
        os.makedirs("data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        wide_columns = ["E_eV"] + [f"a_{index}_A={impact_parameters[index]:.6g}" for index in range(len(impact_parameters))]
        wide_frame = pd.DataFrame(np.column_stack([energy_grid, atom_phase_matrix]), columns=wide_columns)
        wide_path = os.path.join("data", f"phi_atoms_matrix_{timestamp}.csv")
        wide_frame.to_csv(wide_path, index=False, encoding="utf-8")
        logger.info("compute_phase_grid_for_atoms | saved per-atom Phi matrix: %s", os.path.abspath(wide_path))

        long_frame = pd.DataFrame(
            {
                "E_eV": np.repeat(energy_grid, len(impact_parameters)),
                "atom_idx": np.tile(np.arange(len(impact_parameters)), len(energy_grid)),
                "a_ang": np.tile(impact_parameters, len(energy_grid)),
                "Phi_atom": atom_phase_matrix.reshape(-1),
            }
        )
        long_path = os.path.join("data", f"phi_atoms_long_{timestamp}.csv")
        long_frame.to_csv(long_path, index=False, encoding="utf-8")
        logger.info("compute_phase_grid_for_atoms | saved per-atom Phi long: %s", os.path.abspath(long_path))
        _prune_generated_csv_files("data", "phi_atoms_matrix_*.csv", keep_latest=10)
        _prune_generated_csv_files("data", "phi_atoms_long_*.csv", keep_latest=10)
    except Exception:
        logger.exception("compute_phase_grid_for_atoms | failed to save per-atom Phi CSV")


def _prune_generated_csv_files(data_dir: str, pattern: str, keep_latest: int) -> None:
    if keep_latest < 0:
        return

    try:
        entries = sorted(
            (
                os.path.join(data_dir, name)
                for name in os.listdir(data_dir)
                if _matches_glob(name, pattern)
            ),
            key=os.path.getmtime,
            reverse=True,
        )
        for path in entries[keep_latest:]:
            os.remove(path)
    except Exception:
        logger.exception("phase_integrals | failed to prune generated files for pattern %s", pattern)


def _matches_glob(filename: str, pattern: str) -> bool:
    import fnmatch

    return fnmatch.fnmatch(filename, pattern)


energy_eV_to_speed_mps = energy_to_speed_mps
speed_mps_to_energy_eV = speed_to_energy_eV
chi_table_interp = interpolate_thomas_fermi_chi
chi_default = exponential_chi
compute_I_components = compute_phase_integral_components
compute_grid_atoms = compute_phase_grid_for_atoms
compute_grid_atoms_with_phi_matrix = compute_phase_grid_for_atoms_with_matrix
compute_grid = compute_single_atom_phase_grid


__all__ = [
    "ChiFunction",
    "energy_to_speed_mps",
    "speed_to_energy_eV",
    "interpolate_thomas_fermi_chi",
    "exponential_chi",
    "compute_phase_integral_components",
    "compute_phase_grid_for_atoms",
    "compute_phase_grid_for_atoms_with_matrix",
    "compute_single_atom_phase_grid",
    "energy_eV_to_speed_mps",
    "speed_mps_to_energy_eV",
    "chi_table_interp",
    "chi_default",
    "compute_I_components",
    "compute_grid_atoms",
    "compute_grid_atoms_with_phi_matrix",
    "compute_grid",
]
