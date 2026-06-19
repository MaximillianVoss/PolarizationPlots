import unittest

from polarization_app.application.trajectory import TRAJECTORY_SWEEP_IMPACT, TrajectorySweepRequest
from polarization_app.qt import rmin_window


class QtRminPrototypeTestCase(unittest.TestCase):
    def test_default_request_uses_existing_trajectory_model(self):
        request = rmin_window.build_default_request()

        self.assertIsInstance(request, TrajectorySweepRequest)
        self.assertEqual(request.sweep_mode, TRAJECTORY_SWEEP_IMPACT)
        self.assertGreaterEqual(request.point_count, 300)
        self.assertTrue(request.convergence_check)
        self.assertTrue(request.precise_mode)

    def test_qss_contains_core_shell_styles(self):
        qss = rmin_window.build_qss()

        self.assertIn("QMainWindow", qss)
        self.assertIn("QToolBar", qss)
        self.assertIn("primaryButton", qss)
        self.assertIn("QTableWidget", qss)

    def test_pyside6_missing_message_is_actionable_without_dependency(self):
        if rmin_window.is_pyside6_available():
            self.skipTest("PySide6 is installed in this environment.")

        message = rmin_window.pyside6_missing_message()

        self.assertIn("PySide6", message)
        self.assertIn("requirements-pyside6.txt", message)
        with self.assertRaises(RuntimeError):
            rmin_window.RminAnalysisWindow()


if __name__ == "__main__":
    unittest.main()
