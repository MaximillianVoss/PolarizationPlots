import unittest

import numpy as np

from polarization_app.application.formulas import sample_random_lz_chain
from polarization_app.physics.phase_integrals import (
    exponential_chi,
    compute_phase_grid_for_atoms,
    compute_phase_grid_for_atoms_with_matrix,
)
from polarization_app.physics.spin_transport import (
    compute_atom_probabilities,
    compute_spin_observables,
    compute_spin_observables_for_chain,
)
from polarization_app.domain.transitions import build_transition_matrices


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
            "chi": exponential_chi,
            "i3_mode": "sum_avg",
            "dump_atom_phi_csv": False,
            "max_atoms_dump": 10,
        }

        grid = compute_phase_grid_for_atoms(**grid_kwargs)
        grid_with_matrix, phi_atoms, a_arr = compute_phase_grid_for_atoms_with_matrix(**grid_kwargs)

        np.testing.assert_allclose(grid["Phi"].to_numpy(dtype=float), grid_with_matrix["Phi"].to_numpy(dtype=float))
        np.testing.assert_allclose(phi_atoms.sum(axis=1), grid_with_matrix["Phi"].to_numpy(dtype=float))
        np.testing.assert_allclose(a_arr, np.array([0.5, 1.0]))

    def test_atom_probabilities_match_corrected_formula(self):
        phase_values = np.array([np.pi / 3.0])
        p1, p2 = compute_atom_probabilities(phase_values, orbital_l=1, magnetic_lz=0)

        np.testing.assert_allclose(p1, np.array([1.0 / 9.0]), atol=1e-12)
        np.testing.assert_allclose(p2, np.array([1.0 / 9.0]), atol=1e-12)

    def test_spin_observables_use_probability_matrix(self):
        result = compute_spin_observables(
            energies_eV=np.array([1.0, 2.0]),
            phase_values=np.array([0.0, np.pi / 3.0]),
            transition_matrix=np.eye(2),
            orbital_l=1,
            magnetic_lz=0,
        )

        np.testing.assert_allclose(result["sum_check_up"], np.ones(2))
        np.testing.assert_allclose(result["sum_check_dn"], np.ones(2))
        np.testing.assert_allclose(result["spin_mean_up"], np.array([1.0, -7.0 / 9.0]), atol=1e-12)
        np.testing.assert_allclose(result["spin_mean_dn"], np.array([-1.0, 7.0 / 9.0]), atol=1e-12)

    def test_spin_observables_for_chain_preserve_probability_sum(self):
        result = compute_spin_observables_for_chain(
            energies_eV=np.array([1.0, 2.0]),
            phase_matrix_by_atom=np.array([[0.1, 0.2], [0.3, 0.4]]),
            magnetic_lz_chain=[0, 1],
            orbital_l=2,
        )

        np.testing.assert_allclose(result["sum_check_up"], np.ones(2))
        np.testing.assert_allclose(result["sum_check_dn"], np.ones(2))
        self.assertTrue(np.all(np.abs(result["spin_mean_up"]) <= 1.0 + 1e-12))
        self.assertTrue(np.all(np.abs(result["spin_mean_dn"]) <= 1.0 + 1e-12))


class FormulaVariantsTestCase(unittest.TestCase):
    def test_random_lz_chain_is_deterministic_with_seed(self):
        matrices, _ = build_transition_matrices(2)
        d_chain, lz_chain = sample_random_lz_chain(
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
