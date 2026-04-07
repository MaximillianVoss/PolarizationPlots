import unittest

import numpy as np

from polarization_app.domain.transitions import build_transition_matrices


class TransitionMatricesTestCase(unittest.TestCase):
    def test_l_equals_one_matrices(self):
        matrices, inverses = build_transition_matrices(1)

        expected_keys = [-1, 0, 1]
        self.assertEqual(list(matrices.keys()), expected_keys)

        expected_matrices = {
            -1: np.array([[np.sqrt(1.0 / 3.0), -np.sqrt(2.0 / 3.0)], [np.sqrt(2.0 / 3.0), np.sqrt(1.0 / 3.0)]]),
            0: np.array([[np.sqrt(2.0 / 3.0), -np.sqrt(1.0 / 3.0)], [np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0)]]),
            1: np.array([[1.0, 0.0], [0.0, 1.0]]),
        }

        for lz, expected in expected_matrices.items():
            np.testing.assert_allclose(matrices[lz], expected, atol=1e-8)

        for lz in expected_keys:
            inv = inverses[lz]
            self.assertIsNotNone(inv)
            np.testing.assert_allclose(inv @ matrices[lz], np.eye(2), atol=1e-8)
            np.testing.assert_allclose(matrices[lz] @ inv, np.eye(2), atol=1e-8)


if __name__ == "__main__":
    unittest.main()
