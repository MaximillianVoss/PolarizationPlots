# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Callable, Dict, Literal

import numpy as np
import pandas as pd

from polarization_part2 import (
    chi_table_interp,
    compute_grid_atoms,
    compute_grid_atoms_with_phi_matrix,
    spin_amplitudes_both,
    spin_amplitudes_both_chain,
)
from transitions import transition_matrices


FORMULA_LEGACY = "legacy_formula_2_1"
FORMULA_NEW = "new_formula"

FORMULA_LABELS = {
    FORMULA_LEGACY: "Старая формула (2.1)",
    FORMULA_NEW: "Новая формула",
}
FORMULA_BY_LABEL = {label: key for key, label in FORMULA_LABELS.items()}


@dataclass(frozen=True)
class PhaseGridParams:
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
    chi: Callable[[np.ndarray, Dict[str, float] | None], np.ndarray] = chi_table_interp
    chi_params: Dict[str, float] | None = None
    i3_mode: Literal["trapz", "sum_avg"] = "sum_avg"
    dump_atom_phi_csv: bool = True
    max_atoms_dump: int = 200


@dataclass(frozen=True)
class FormulaRunResult:
    grid: pd.DataFrame
    spin_curves: dict[str, np.ndarray]
    orbital_l: int
    formula_variant: str
    fixed_lz: int | None = None
    lz_chain: tuple[int, ...] = ()
    phi_atoms: np.ndarray | None = None
    impact_parameters: np.ndarray | None = None


def legacy_lz_for_orbital(orbital_l: int, matrices: Dict[int, np.ndarray]) -> int:
    if not matrices:
        raise ValueError("Не удалось построить набор матриц перехода.")
    if orbital_l in matrices:
        return int(orbital_l)
    return int(max(matrices))


def build_random_lz_chain(
    matrices: Dict[int, np.ndarray],
    atom_count: int,
    rng: np.random.Generator | None = None,
) -> tuple[list[np.ndarray], tuple[int, ...]]:
    if atom_count < 0:
        raise ValueError("atom_count должен быть неотрицательным.")
    if not matrices and atom_count:
        raise ValueError("Для новой формулы требуется непустой набор матриц перехода.")

    rng = rng or np.random.default_rng()
    keys = np.asarray(list(matrices.keys()), dtype=int)
    if atom_count == 0:
        return [], ()

    chosen = tuple(int(value) for value in rng.choice(keys, size=atom_count, replace=True))
    return [matrices[lz] for lz in chosen], chosen


def run_formula_variant(
    *,
    formula_variant: str,
    orbital_l: int,
    phase_params: PhaseGridParams,
    rng: np.random.Generator | None = None,
) -> FormulaRunResult:
    matrices, _ = transition_matrices(L_source=orbital_l)

    if formula_variant == FORMULA_LEGACY:
        fixed_lz = legacy_lz_for_orbital(orbital_l, matrices)
        grid = compute_grid_atoms(
            Emin_eV=phase_params.Emin_eV,
            Emax_eV=phase_params.Emax_eV,
            N=phase_params.N,
            a_list_ang=phase_params.a_list_ang,
            Z=phase_params.Z,
            b_ang=phase_params.b_ang,
            c1=phase_params.c1,
            c2=phase_params.c2,
            dr_ang=phase_params.dr_ang,
            r_max_ang=phase_params.r_max_ang,
            chi=phase_params.chi,
            chi_params=phase_params.chi_params,
            i3_mode=phase_params.i3_mode,
            dump_atom_phi_csv=phase_params.dump_atom_phi_csv,
            max_atoms_dump=phase_params.max_atoms_dump,
        )
        spin_curves = spin_amplitudes_both(
            grid["E_eV"].to_numpy(dtype=float),
            grid["Phi"].to_numpy(dtype=float),
            matrices[fixed_lz],
            L_atom=orbital_l,
        )
        return FormulaRunResult(
            grid=grid,
            spin_curves=spin_curves,
            orbital_l=orbital_l,
            formula_variant=formula_variant,
            fixed_lz=fixed_lz,
        )

    if formula_variant == FORMULA_NEW:
        grid, phi_atoms, impact_parameters = compute_grid_atoms_with_phi_matrix(
            Emin_eV=phase_params.Emin_eV,
            Emax_eV=phase_params.Emax_eV,
            N=phase_params.N,
            a_list_ang=phase_params.a_list_ang,
            Z=phase_params.Z,
            b_ang=phase_params.b_ang,
            c1=phase_params.c1,
            c2=phase_params.c2,
            dr_ang=phase_params.dr_ang,
            r_max_ang=phase_params.r_max_ang,
            chi=phase_params.chi,
            chi_params=phase_params.chi_params,
            i3_mode=phase_params.i3_mode,
            dump_atom_phi_csv=phase_params.dump_atom_phi_csv,
            max_atoms_dump=phase_params.max_atoms_dump,
        )
        d_chain, lz_chain = build_random_lz_chain(
            matrices=matrices,
            atom_count=int(phi_atoms.shape[1]),
            rng=rng,
        )
        spin_curves = spin_amplitudes_both_chain(
            grid["E_eV"].to_numpy(dtype=float),
            phi_atoms,
            d_chain,
            L_atom=orbital_l,
        )
        return FormulaRunResult(
            grid=grid,
            spin_curves=spin_curves,
            orbital_l=orbital_l,
            formula_variant=formula_variant,
            lz_chain=lz_chain,
            phi_atoms=phi_atoms,
            impact_parameters=impact_parameters,
        )

    raise ValueError(f"Неизвестный вариант формулы: {formula_variant}")
