import unittest

import numpy as np

from transitions import transition_matrices


class TransitionMatricesTestCase(unittest.TestCase):
    def test_l_equals_one_matrices(self):
        matrices, inverses = transition_matrices(1)

        expected_keys = [-1, 0, 1]
        self.assertEqual(list(matrices.keys()), expected_keys)

        expected_matrices = {
            -1: np.array([[0.57735027, 1.0], [0.81649658, 0.0]]),
            0: np.array([[0.81649658, 0.81649658], [0.57735027, 0.57735027]]),
            1: np.array([[1.0, 0.57735027], [0.0, 0.81649658]]),
        }

        for ln, expected in expected_matrices.items():
            np.testing.assert_allclose(matrices[ln], expected, atol=1e-8)

        # Ln = 0 даёт вырожденную матрицу
        self.assertIsNone(inverses[0])

        # Для остальных Ln обратная матрица существует и соответствует единичной при умножении
        for ln in (-1, 1):
            inv = inverses[ln]
            self.assertIsNotNone(inv)
            np.testing.assert_allclose(inv @ matrices[ln], np.eye(2), atol=1e-8)
            np.testing.assert_allclose(matrices[ln] @ inv, np.eye(2), atol=1e-8)


if __name__ == "__main__":
    unittest.main()
