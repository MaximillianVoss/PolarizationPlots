# -*- coding: utf-8 -*-
import numpy as np

from polarization_app.physics.compute_backend import cupy_if_available


CUDA_PHASE_ELEMENT_THRESHOLD = 100_000


def compute_spin_observables(
    energies_eV: np.ndarray,
    phase_values: np.ndarray,
    transition_matrix: np.ndarray | None = None,
    orbital_l: int = 0,
    magnetic_lz: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Вероятности и спиновые комбинации для одной эффективной фазы.
    """
    del energies_eV, transition_matrix

    orbital_l = int(orbital_l)
    magnetic_lz = orbital_l if magnetic_lz is None else int(magnetic_lz)
    phase_values = np.asarray(phase_values, dtype=float)
    transition_steps = probability_transition_matrix_batch(phase_values, orbital_l, magnetic_lz)
    return _observables_from_transition_batch(transition_steps)


def compute_spin_observables_for_chain(
    energies_eV: np.ndarray,
    phase_matrix_by_atom: np.ndarray,
    transition_chain: list[np.ndarray] | tuple[int, ...] | None = None,
    orbital_l=0,
    magnetic_lz_chain: list[int] | tuple[int, ...] | None = None,
    use_cuda: bool | None = None,
) -> dict[str, np.ndarray]:
    """
    Вероятности и спиновые комбинации для цепочки атомов.
    """
    energies_eV = np.asarray(energies_eV, dtype=float)
    phase_matrix_by_atom = np.asarray(phase_matrix_by_atom, dtype=float)

    if phase_matrix_by_atom.ndim != 2:
        raise ValueError("phase_matrix_by_atom должен быть матрицей формы [энергия, атом].")
    atom_count = phase_matrix_by_atom.shape[1]
    if len(energies_eV) != phase_matrix_by_atom.shape[0]:
        raise ValueError("Число энергий должно совпадать с числом строк phase_matrix_by_atom.")

    orbital_chain = _resolve_orbital_chain(orbital_l, atom_count)
    lz_chain = _resolve_lz_chain(magnetic_lz_chain, transition_chain, orbital_chain)

    cp = None
    if use_cuda is not False:
        threshold = 0 if use_cuda is True else CUDA_PHASE_ELEMENT_THRESHOLD
        cp = cupy_if_available(min_element_count=threshold, element_count=phase_matrix_by_atom.size)
    xp = cp if cp is not None else np

    phase_matrix = xp.asarray(phase_matrix_by_atom, dtype=float)
    state_up_0 = xp.ones(len(energies_eV), dtype=float)
    state_up_1 = xp.zeros(len(energies_eV), dtype=float)
    state_down_0 = xp.zeros(len(energies_eV), dtype=float)
    state_down_1 = xp.ones(len(energies_eV), dtype=float)

    for atom_index in range(atom_count):
        p1, p2 = _atom_probabilities_xp(
            phase_matrix[:, atom_index],
            orbital_chain[atom_index],
            lz_chain[atom_index],
            xp,
        )
        state_up_0, state_up_1 = (
            p1 * state_up_0 + (1.0 - p2) * state_up_1,
            (1.0 - p1) * state_up_0 + p2 * state_up_1,
        )
        state_down_0, state_down_1 = (
            p1 * state_down_0 + (1.0 - p2) * state_down_1,
            (1.0 - p1) * state_down_0 + p2 * state_down_1,
        )

    result = {
        "sum_check_up": state_up_0 + state_up_1,
        "spin_mean_up": state_up_0 - state_up_1,
        "sum_check_dn": state_down_0 + state_down_1,
        "spin_mean_dn": state_down_0 - state_down_1,
    }
    if cp is not None:  # pragma: no cover - requires local CUDA runtime
        return {key: cp.asnumpy(value) for key, value in result.items()}
    return {key: np.asarray(value, dtype=float) for key, value in result.items()}


def _atom_probabilities_xp(phase_values, orbital_l: int, magnetic_lz: int, xp):
    orbital_l = int(orbital_l)
    magnetic_lz = int(magnetic_lz)
    phase_angle = (orbital_l + 0.5) * phase_values
    denominator = 2 * orbital_l + 1
    p1_coeff = (2 * magnetic_lz + 1) / denominator
    p2_coeff = (2 * magnetic_lz - 1) / denominator
    cos_squared = xp.cos(phase_angle) ** 2
    sin_squared = xp.sin(phase_angle) ** 2
    return (
        cos_squared + (p1_coeff ** 2) * sin_squared,
        cos_squared + (p2_coeff ** 2) * sin_squared,
    )


def compute_atom_probabilities(
    phase_values: np.ndarray,
    orbital_l: int,
    magnetic_lz: int,
) -> tuple[np.ndarray, np.ndarray]:
    phase_angle = (int(orbital_l) + 0.5) * np.asarray(phase_values, dtype=float)
    denominator = 2 * int(orbital_l) + 1
    p1_coeff = (2 * int(magnetic_lz) + 1) / denominator
    p2_coeff = (2 * int(magnetic_lz) - 1) / denominator
    cos_squared = np.cos(phase_angle) ** 2
    sin_squared = np.sin(phase_angle) ** 2
    p1 = cos_squared + (p1_coeff ** 2) * sin_squared
    p2 = cos_squared + (p2_coeff ** 2) * sin_squared
    return p1, p2


def probability_transition_matrix_batch(
    phase_values: np.ndarray,
    orbital_l: int,
    magnetic_lz: int,
) -> np.ndarray:
    p1, p2 = compute_atom_probabilities(phase_values, orbital_l, magnetic_lz)
    matrix = np.zeros((len(p1), 2, 2), dtype=float)
    matrix[:, 0, 0] = p1
    matrix[:, 0, 1] = 1.0 - p2
    matrix[:, 1, 0] = 1.0 - p1
    matrix[:, 1, 1] = p2
    return matrix


def _resolve_orbital_chain(orbital_l, atom_count: int) -> list[int]:
    if np.isscalar(orbital_l):
        return [int(orbital_l)] * atom_count

    orbital_chain = [int(value) for value in orbital_l]
    if len(orbital_chain) != atom_count:
        raise ValueError("Список orbital_l по атомам должен совпадать с числом атомов.")
    return orbital_chain


def _resolve_lz_chain(
    magnetic_lz_chain: list[int] | tuple[int, ...] | None,
    transition_chain: list[np.ndarray] | tuple[int, ...] | None,
    orbital_chain: list[int],
) -> list[int]:
    atom_count = len(orbital_chain)
    if magnetic_lz_chain is not None:
        lz_chain = [int(value) for value in magnetic_lz_chain]
    elif transition_chain is not None and all(np.isscalar(value) for value in transition_chain):
        lz_chain = [int(value) for value in transition_chain]
    else:
        lz_chain = list(orbital_chain)

    if len(lz_chain) != atom_count:
        raise ValueError("Список Lz по атомам должен совпадать с числом атомов.")
    return lz_chain


def _observables_from_transition_batch(transition_steps: np.ndarray) -> dict[str, np.ndarray]:
    state_up = np.einsum("eij,j->ei", transition_steps, np.array([1.0, 0.0], dtype=float))
    state_down = np.einsum("eij,j->ei", transition_steps, np.array([0.0, 1.0], dtype=float))
    return _observables_from_states(state_up, state_down)


def _observables_from_states(state_up: np.ndarray, state_down: np.ndarray) -> dict[str, np.ndarray]:
    p_up_from_up = state_up[:, 0]
    p_down_from_up = state_up[:, 1]
    p_up_from_down = state_down[:, 0]
    p_down_from_down = state_down[:, 1]

    return {
        "sum_check_up": p_up_from_up + p_down_from_up,
        "spin_mean_up": p_up_from_up - p_down_from_up,
        "sum_check_dn": p_up_from_down + p_down_from_down,
        "spin_mean_dn": p_up_from_down - p_down_from_down,
    }


spin_amplitudes_both = compute_spin_observables
spin_amplitudes_both_chain = compute_spin_observables_for_chain


__all__ = [
    "compute_spin_observables",
    "compute_spin_observables_for_chain",
    "compute_atom_probabilities",
    "probability_transition_matrix_batch",
    "spin_amplitudes_both",
    "spin_amplitudes_both_chain",
]
