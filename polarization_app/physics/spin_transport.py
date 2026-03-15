# -*- coding: utf-8 -*-
import numpy as np


def compute_spin_observables(
    energies_eV: np.ndarray,
    phase_values: np.ndarray,
    transition_matrix: np.ndarray,
    orbital_l: int = 0,
) -> dict[str, np.ndarray]:
    """
    Вероятности и спиновые комбинации для одной матрицы перехода.
    """
    del energies_eV

    transition_matrix = np.asarray(transition_matrix, dtype=complex)
    transition_inverse = np.linalg.inv(transition_matrix)
    phase_values = np.asarray(phase_values, dtype=float)
    phase_matrix = _phase_matrix_batch(phase_values, int(orbital_l))
    transition_steps = transition_inverse[np.newaxis, :, :] @ phase_matrix @ transition_matrix[np.newaxis, :, :]

    ket_up = np.array([1.0 + 0j, 0.0 + 0j], dtype=complex)
    ket_down = np.array([0.0 + 0j, 1.0 + 0j], dtype=complex)
    amplitudes_up = np.einsum("eij,j->ei", transition_steps, ket_up)
    amplitudes_down = np.einsum("eij,j->ei", transition_steps, ket_down)

    p_up_from_up = np.abs(amplitudes_up[:, 0]) ** 2
    p_down_from_up = np.abs(amplitudes_up[:, 1]) ** 2
    p_up_from_down = np.abs(amplitudes_down[:, 0]) ** 2
    p_down_from_down = np.abs(amplitudes_down[:, 1]) ** 2

    return {
        "sum_check_up": p_up_from_up + p_down_from_up,
        "spin_mean_up": p_up_from_up - p_down_from_up,
        "sum_check_dn": p_up_from_down + p_down_from_down,
        "spin_mean_dn": p_up_from_down - p_down_from_down,
    }


def compute_spin_observables_for_chain(
    energies_eV: np.ndarray,
    phase_matrix_by_atom: np.ndarray,
    transition_chain: list[np.ndarray],
    orbital_l,
) -> dict[str, np.ndarray]:
    """
    Вероятности и спиновые комбинации для цепочки матриц перехода по атомам.
    """
    energies_eV = np.asarray(energies_eV, dtype=float)
    phase_matrix_by_atom = np.asarray(phase_matrix_by_atom, dtype=float)

    ket_up = np.array([1.0 + 0j, 0.0 + 0j], dtype=complex)
    ket_down = np.array([0.0 + 0j, 1.0 + 0j], dtype=complex)

    if phase_matrix_by_atom.ndim != 2:
        raise ValueError("phase_matrix_by_atom должен быть матрицей формы [энергия, атом].")
    if phase_matrix_by_atom.shape[1] != len(transition_chain):
        raise ValueError("Число матриц перехода должно совпадать с числом атомов.")

    if np.isscalar(orbital_l):
        orbital_chain = None
        fixed_orbital_l = int(orbital_l)
    else:
        orbital_chain = [int(value) for value in orbital_l]
        fixed_orbital_l = None
        if len(orbital_chain) != len(transition_chain):
            raise ValueError("Список orbital_l по атомам должен совпадать по длине с transition_chain.")

    inverse_chain = [np.linalg.inv(np.asarray(matrix, dtype=complex)) for matrix in transition_chain]
    complex_chain = [np.asarray(matrix, dtype=complex) for matrix in transition_chain]
    psi_up = np.tile(ket_up, (len(energies_eV), 1))
    psi_down = np.tile(ket_down, (len(energies_eV), 1))

    for atom_index, phase_values in enumerate(phase_matrix_by_atom.T):
        current_l = fixed_orbital_l if orbital_chain is None else orbital_chain[atom_index]
        matrix = complex_chain[atom_index]
        matrix_inverse = inverse_chain[atom_index]
        phase_matrix = _phase_matrix_batch(phase_values, current_l)
        transition_steps = matrix_inverse[np.newaxis, :, :] @ phase_matrix @ matrix[np.newaxis, :, :]
        psi_up = np.einsum("eij,ej->ei", transition_steps, psi_up)
        psi_down = np.einsum("eij,ej->ei", transition_steps, psi_down)

    p_up_from_up = np.abs(psi_up[:, 0]) ** 2
    p_down_from_up = np.abs(psi_up[:, 1]) ** 2
    p_up_from_down = np.abs(psi_down[:, 0]) ** 2
    p_down_from_down = np.abs(psi_down[:, 1]) ** 2

    return {
        "sum_check_up": p_up_from_up + p_down_from_up,
        "spin_mean_up": p_up_from_up - p_down_from_up,
        "sum_check_dn": p_up_from_down + p_down_from_down,
        "spin_mean_dn": p_up_from_down - p_down_from_down,
    }


def _phase_matrix(phase_value: float, orbital_l: int) -> np.ndarray:
    return np.array(
        [
            [np.exp(1j * orbital_l * phase_value), 0.0 + 0j],
            [0.0 + 0j, np.exp(-1j * (orbital_l + 1) * phase_value)],
        ],
        dtype=complex,
    )


def _phase_matrix_batch(phase_values: np.ndarray, orbital_l: int) -> np.ndarray:
    phase_values = np.asarray(phase_values, dtype=float)
    matrix = np.zeros((len(phase_values), 2, 2), dtype=complex)
    matrix[:, 0, 0] = np.exp(1j * orbital_l * phase_values)
    matrix[:, 1, 1] = np.exp(-1j * (orbital_l + 1) * phase_values)
    return matrix


spin_amplitudes_both = compute_spin_observables
spin_amplitudes_both_chain = compute_spin_observables_for_chain


__all__ = [
    "compute_spin_observables",
    "compute_spin_observables_for_chain",
    "spin_amplitudes_both",
    "spin_amplitudes_both_chain",
]
