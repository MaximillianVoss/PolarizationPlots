# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from polarization_app.domain.transitions import build_transition_matrices
from polarization_app.physics.phase_integrals import (
    ChiFunction,
    compute_phase_grid_for_atoms,
    compute_phase_grid_for_atoms_with_matrix,
    interpolate_thomas_fermi_chi,
)
from polarization_app.physics.spin_transport import (
    compute_spin_observables,
    compute_spin_observables_for_chain,
)


FORMULA_LEGACY = "legacy_formula_2_1"
FORMULA_NEW = "new_formula"

FORMULA_LABELS = {
    FORMULA_LEGACY: "Модель (2.1): общий Lz для всех атомов",
    FORMULA_NEW: "Обобщённая модель: случайный Lz по атомам",
}
FORMULA_BY_LABEL = {label: key for key, label in FORMULA_LABELS.items()}
FORMULA_HINTS = {
    FORMULA_LEGACY: "Упрощённый случай из отчёта: на всей цепочке используется один и тот же Lz.",
    FORMULA_NEW: "Для каждого атома выбирается собственный Lz и перемножается вероятностная матрица атома.",
}


@dataclass(frozen=True)
class PhaseGridRequest:
    Emin_eV: float
    Emax_eV: float
    N: int
    a_list_ang: list[float]
    Z: float
    b_ang: float
    c1: float
    c2: float
    dr_ang: float
    r_max_ang: float
    chi: ChiFunction = interpolate_thomas_fermi_chi
    chi_params: dict[str, float] | None = None
    i3_mode: Literal["trapz", "sum_avg"] = "sum_avg"
    dump_atom_phi_csv: bool = False
    max_atoms_dump: int = 200


@dataclass(frozen=True)
class FormulaResult:
    grid: pd.DataFrame
    spin_curves: dict[str, np.ndarray]
    orbital_l: int
    formula_variant: str
    fixed_lz: int | None = None
    lz_chain: tuple[int, ...] = ()
    phi_atoms: np.ndarray | None = None
    impact_parameters: np.ndarray | None = None


def select_fixed_lz_for_legacy_formula(orbital_l: int, matrices: dict[int, np.ndarray]) -> int:
    if not matrices:
        raise ValueError("Не удалось построить набор матриц перехода.")
    if orbital_l in matrices:
        return int(orbital_l)
    return int(max(matrices))


def sample_random_lz_chain(
    matrices: dict[int, np.ndarray],
    atom_count: int,
    rng: np.random.Generator | None = None,
) -> tuple[list[np.ndarray], tuple[int, ...]]:
    if atom_count < 0:
        raise ValueError("atom_count должен быть неотрицательным.")
    if not matrices and atom_count:
        raise ValueError("Для новой формулы требуется непустой набор матриц перехода.")

    if atom_count == 0:
        return [], ()

    rng = rng or np.random.default_rng()
    keys = np.asarray(list(matrices.keys()), dtype=int)
    sampled = tuple(int(value) for value in rng.choice(keys, size=atom_count, replace=True))
    return [matrices[lz] for lz in sampled], sampled


def execute_formula_variant(
    *,
    formula_variant: str,
    orbital_l: int,
    phase_request: PhaseGridRequest,
    rng: np.random.Generator | None = None,
) -> FormulaResult:
    matrices, _ = build_transition_matrices(source_orbital_l=orbital_l)

    if formula_variant == FORMULA_LEGACY:
        fixed_lz = select_fixed_lz_for_legacy_formula(orbital_l, matrices)
        grid = compute_phase_grid_for_atoms(
            Emin_eV=phase_request.Emin_eV,
            Emax_eV=phase_request.Emax_eV,
            N=phase_request.N,
            a_list_ang=phase_request.a_list_ang,
            Z=phase_request.Z,
            b_ang=phase_request.b_ang,
            c1=phase_request.c1,
            c2=phase_request.c2,
            dr_ang=phase_request.dr_ang,
            r_max_ang=phase_request.r_max_ang,
            chi=phase_request.chi,
            chi_params=phase_request.chi_params,
            i3_mode=phase_request.i3_mode,
            dump_atom_phi_csv=phase_request.dump_atom_phi_csv,
            max_atoms_dump=phase_request.max_atoms_dump,
        )
        spin_curves = compute_spin_observables(
            grid["E_eV"].to_numpy(dtype=float),
            grid["Phi"].to_numpy(dtype=float),
            matrices[fixed_lz],
            orbital_l=orbital_l,
            magnetic_lz=fixed_lz,
        )
        return FormulaResult(
            grid=grid,
            spin_curves=spin_curves,
            orbital_l=orbital_l,
            formula_variant=formula_variant,
            fixed_lz=fixed_lz,
        )

    if formula_variant == FORMULA_NEW:
        grid, phi_atoms, impact_parameters = compute_phase_grid_for_atoms_with_matrix(
            Emin_eV=phase_request.Emin_eV,
            Emax_eV=phase_request.Emax_eV,
            N=phase_request.N,
            a_list_ang=phase_request.a_list_ang,
            Z=phase_request.Z,
            b_ang=phase_request.b_ang,
            c1=phase_request.c1,
            c2=phase_request.c2,
            dr_ang=phase_request.dr_ang,
            r_max_ang=phase_request.r_max_ang,
            chi=phase_request.chi,
            chi_params=phase_request.chi_params,
            i3_mode=phase_request.i3_mode,
            dump_atom_phi_csv=phase_request.dump_atom_phi_csv,
            max_atoms_dump=phase_request.max_atoms_dump,
        )
        _transition_chain, lz_chain = sample_random_lz_chain(
            matrices=matrices,
            atom_count=int(phi_atoms.shape[1]),
            rng=rng,
        )
        spin_curves = compute_spin_observables_for_chain(
            grid["E_eV"].to_numpy(dtype=float),
            phi_atoms,
            orbital_l=orbital_l,
            magnetic_lz_chain=lz_chain,
        )
        return FormulaResult(
            grid=grid,
            spin_curves=spin_curves,
            orbital_l=orbital_l,
            formula_variant=formula_variant,
            lz_chain=lz_chain,
            phi_atoms=phi_atoms,
            impact_parameters=impact_parameters,
        )

    raise ValueError(f"Неизвестный вариант формулы: {formula_variant}")


PhaseGridParams = PhaseGridRequest
FormulaRunResult = FormulaResult
legacy_lz_for_orbital = select_fixed_lz_for_legacy_formula
build_random_lz_chain = sample_random_lz_chain
run_formula_variant = execute_formula_variant


__all__ = [
    "FORMULA_LEGACY",
    "FORMULA_NEW",
    "FORMULA_LABELS",
    "FORMULA_BY_LABEL",
    "FORMULA_HINTS",
    "PhaseGridRequest",
    "FormulaResult",
    "select_fixed_lz_for_legacy_formula",
    "sample_random_lz_chain",
    "execute_formula_variant",
    "PhaseGridParams",
    "FormulaRunResult",
    "legacy_lz_for_orbital",
    "build_random_lz_chain",
    "run_formula_variant",
]
