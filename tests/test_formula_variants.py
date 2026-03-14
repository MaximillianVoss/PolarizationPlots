import unittest

import numpy as np

from formula_variants import build_random_lz_chain
from polarization_part2 import (
    chi_default,
    compute_grid_atoms,
    compute_grid_atoms_with_phi_matrix,
    spin_amplitudes_both,
    spin_amplitudes_both_chain,
)
from transitions import transition_matrices


class PolarizationPart2TestCase(unittest.TestCase):
    def test_compute_grid_atoms_with_phi_matrix_matches_total_phi(self):
        grid_kwargs = {
            "Emin_eV": 10.0,
            "Emax_eV": 20.0,
            "N": 3,
            "a_list_ang": [1.0, 0.5],
            "Z": 29.0,
            "b_ang": 0.53,
            "c1": 1.0,
            "c2": 1.0,
            "dr_ang": 0.05,
            "r_max_ang": 5.0,
            "chi": chi_default,
            "i3_mode": "sum_avg",
            "dump_atom_phi_csv": False,
            "max_atoms_dump": 10,
        }

        grid = compute_grid_atoms(**grid_kwargs)
        grid_with_matrix, phi_atoms, a_arr = compute_grid_atoms_with_phi_matrix(**grid_kwargs)

        np.testing.assert_allclose(grid["Phi"].to_numpy(dtype=float), grid_with_matrix["Phi"].to_numpy(dtype=float))
        np.testing.assert_allclose(phi_atoms.sum(axis=1), grid_with_matrix["Phi"].to_numpy(dtype=float))
        np.testing.assert_allclose(a_arr, np.array([0.5, 1.0]))

    def test_spin_amplitudes_both_preserve_spin_for_identity_transition(self):
        result = spin_amplitudes_both(
            E=np.array([1.0, 2.0]),
            Phi=np.array([0.0, 0.7]),
            D=np.eye(2),
            L_atom=3,
        )

        np.testing.assert_allclose(result["sum_check_up"], np.ones(2))
        np.testing.assert_allclose(result["spin_mean_up"], np.ones(2))
        np.testing.assert_allclose(result["sum_check_dn"], np.ones(2))
        np.testing.assert_allclose(result["spin_mean_dn"], -np.ones(2))

    def test_spin_amplitudes_both_chain_preserve_spin_for_identity_transition(self):
        result = spin_amplitudes_both_chain(
            E=np.array([1.0, 2.0]),
            Phi_atoms=np.array([[0.1, 0.2], [0.3, 0.4]]),
            D_chain=[np.eye(2), np.eye(2)],
            L_atom=2,
        )

        np.testing.assert_allclose(result["sum_check_up"], np.ones(2))
        np.testing.assert_allclose(result["spin_mean_up"], np.ones(2))
        np.testing.assert_allclose(result["sum_check_dn"], np.ones(2))
        np.testing.assert_allclose(result["spin_mean_dn"], -np.ones(2))


class FormulaVariantsTestCase(unittest.TestCase):
    def test_random_lz_chain_is_deterministic_with_seed(self):
        matrices, _ = transition_matrices(2)
        d_chain, lz_chain = build_random_lz_chain(
            matrices,
            atom_count=6,
            rng=np.random.default_rng(123),
        )

        self.assertEqual(lz_chain, (-2, 1, 0, -2, 2, -1))
        self.assertEqual(len(d_chain), len(lz_chain))
        for lz, matrix in zip(lz_chain, d_chain):
            np.testing.assert_allclose(matrix, matrices[lz])


if __name__ == "__main__":
    unittest.main()
