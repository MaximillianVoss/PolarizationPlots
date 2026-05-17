import unittest

import numpy as np

from polarization_app.physics.boundary_reflection import (
    compute_boundary_point,
    compute_boundary_reflection_curves,
)


class BoundaryReflectionTestCase(unittest.TestCase):
    def test_curve_values_stay_in_expected_ranges(self):
        curves = compute_boundary_reflection_curves(
            np.linspace(10.0, 500.0, 60),
            work_function_eV=5.0,
            incidence_angle_deg=45.0,
        )

        self.assertEqual(len(curves.energies_eV), 60)
        self.assertTrue(np.all(curves.reflection_coefficient >= 0.0))
        self.assertTrue(np.all(curves.reflection_coefficient <= 1.0 + 1e-12))
        self.assertTrue(np.all(curves.reflection_probability_estimate >= 0.0))
        self.assertTrue(np.all(curves.reflection_probability_estimate <= 1.0 + 1e-12))
        finite_beta = curves.transmission_angle_deg[np.isfinite(curves.transmission_angle_deg)]
        self.assertTrue(np.all(finite_beta >= 0.0))
        self.assertTrue(np.all(finite_beta <= 90.0 + 1e-12))

    def test_energy_below_barrier_gives_full_reflection(self):
        point = compute_boundary_point(
            4.0,
            work_function_eV=5.0,
            incidence_angle_deg=25.0,
        )

        self.assertAlmostEqual(point.reflection_coefficient, 1.0)
        self.assertAlmostEqual(point.reflection_probability_estimate, 1.0)
        self.assertIsNone(point.transmission_angle_deg)
        self.assertIsNone(point.wavevector_ratio)
        self.assertIn("E <= A", point.regime)

    def test_large_angle_can_produce_total_reflection_even_above_barrier(self):
        point = compute_boundary_point(
            12.0,
            work_function_eV=5.0,
            incidence_angle_deg=70.0,
        )

        self.assertAlmostEqual(point.reflection_coefficient, 1.0)
        self.assertAlmostEqual(point.reflection_probability_estimate, 1.0)
        self.assertIsNone(point.transmission_angle_deg)
        self.assertIsNotNone(point.wavevector_ratio)
        self.assertIn("Полное отражение", point.regime)


if __name__ == "__main__":
    unittest.main()
