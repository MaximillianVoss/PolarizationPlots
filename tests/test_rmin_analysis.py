import unittest

import pandas as pd

from polarization_app.application.rmin_analysis import (
    compute_rmin_analysis_metrics,
    format_rmin_analysis_report,
    thomas_fermi_radius_ang_from_frame,
)
from polarization_app.physics.phase_integrals import BOHR_TO_ANGSTROM
from polarization_app.physics.trajectory_phase import DEFAULT_THOMAS_FERMI_B_BOHR


class RminAnalysisTestCase(unittest.TestCase):
    def test_metrics_ignore_failed_points_and_mark_thomas_fermi_region(self):
        frame = pd.DataFrame(
            {
                "atomic_number": [82.0, 82.0, 82.0, 82.0],
                "r_min_ang": [0.05, 0.10, 0.15, 0.20],
                "p_flip_initial_up": [0.2, 0.55, 0.90, 0.99],
                "p_flip_initial_down": [0.1, 0.45, 0.70, 0.98],
                "converged": [True, True, True, False],
                "convergence_unstable": [False, False, True, False],
            }
        )

        metrics = compute_rmin_analysis_metrics(frame)
        expected_r_tf = DEFAULT_THOMAS_FERMI_B_BOHR * BOHR_TO_ANGSTROM / (82.0 ** (1.0 / 3.0))

        self.assertEqual(metrics.total_count, 4)
        self.assertEqual(metrics.successful_count, 3)
        self.assertEqual(metrics.failed_count, 1)
        self.assertEqual(metrics.unstable_count, 1)
        self.assertAlmostEqual(metrics.r_tf_ang, expected_r_tf)
        self.assertEqual(metrics.inside_tf_count, 2)
        self.assertAlmostEqual(metrics.p_max, 0.90)
        self.assertAlmostEqual(metrics.p_max_rmin_ang, 0.15)
        self.assertAlmostEqual(metrics.p_over_half_min_ang, 0.10)
        self.assertAlmostEqual(metrics.p_over_half_max_ang, 0.15)
        self.assertFalse(metrics.convergence_ok)

    def test_report_contains_record_ready_summary(self):
        frame = pd.DataFrame(
            {
                "atomic_number": [29.0, 29.0],
                "r_min_ang": [0.08, 0.16],
                "p_flip_initial_up": [0.4, 0.7],
                "p_flip_initial_down": [0.3, 0.6],
                "converged": [True, True],
            }
        )

        metrics = compute_rmin_analysis_metrics(frame)
        report = format_rmin_analysis_report(metrics)

        self.assertEqual(thomas_fermi_radius_ang_from_frame(frame), metrics.r_tf_ang)
        self.assertIn("r_TF", report)
        self.assertIn("Вывод для записки", report)
        self.assertIn("Pmax", report)


if __name__ == "__main__":
    unittest.main()
