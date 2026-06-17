import unittest

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from polarization_app.gui.plotting import draw_trajectory_probability_by_rmin, draw_trajectory_sweep_plots


class TrajectoryPlottingTestCase(unittest.TestCase):
    def test_trajectory_legends_explain_phi_angles(self):
        frame = pd.DataFrame(
            {
                "energy_eV": [100.0, 200.0],
                "phase_rad": [0.1, 0.2],
                "p_flip_initial_up": [0.12, 0.18],
                "p_flip_initial_down": [0.08, 0.14],
                "theta_deg": [10.0, 12.0],
                "trajectory_phi_deg": [1.0, 1.5],
                "r_min_ang": [0.3, 0.4],
                "steps": [50, 60],
            }
        )
        fig = Figure()
        phase_axis = fig.add_subplot(311)
        angle_axis = fig.add_subplot(312)
        diagnostic_axis = fig.add_subplot(313)

        draw_trajectory_sweep_plots(
            phase_axis,
            angle_axis,
            diagnostic_axis,
            frame,
            x_column="energy_eV",
            x_label="Энергия, эВ",
        )

        spin_flip_labels = phase_axis.get_legend_handles_labels()[1]
        angle_labels = angle_axis.get_legend_handles_labels()[1]
        diagnostic_labels = [text.get_text() for text in diagnostic_axis.get_legend().get_texts()]

        self.assertEqual(phase_axis.get_title(), "Вероятность изменения спина после СОВ")
        self.assertEqual(diagnostic_axis.get_title(), "r_min и панели интегрирования")
        self.assertIn("начальный ↑: ↑→↓", spin_flip_labels)
        self.assertIn("начальный ↓: ↓→↑", spin_flip_labels)
        self.assertIn("φ, угол траектории после взаимодействия", angle_labels)
        self.assertIn("steps, панели квадратуры", diagnostic_labels)

    def test_unstable_convergence_points_are_highlighted(self):
        frame = pd.DataFrame(
            {
                "energy_eV": [100.0, 200.0, 300.0],
                "phase_rad": [0.1, 0.2, 0.3],
                "p_flip_initial_up": [0.12, 0.18, 0.2],
                "p_flip_initial_down": [0.08, 0.14, 0.16],
                "theta_deg": [10.0, 12.0, 13.0],
                "trajectory_phi_deg": [1.0, 1.5, 2.0],
                "r_min_ang": [0.3, 0.4, 0.5],
                "steps": [100, 120, 130],
                "convergence_unstable": [False, True, False],
            }
        )
        fig = Figure()
        phase_axis = fig.add_subplot(311)
        angle_axis = fig.add_subplot(312)
        diagnostic_axis = fig.add_subplot(313)

        draw_trajectory_sweep_plots(
            phase_axis,
            angle_axis,
            diagnostic_axis,
            frame,
            x_column="energy_eV",
            x_label="Энергия, эВ",
        )

        self.assertGreater(len(phase_axis.patches), 0)
        self.assertIn("неустойчиво по dθ", phase_axis.get_legend_handles_labels()[1])
        self.assertTrue(any("Неустойчиво по dθ" in text.get_text() for text in diagnostic_axis.texts))

    def test_non_converged_points_break_result_curves(self):
        frame = pd.DataFrame(
            {
                "energy_eV": [100.0, 200.0, 300.0],
                "phase_rad": [0.1, 0.2, 0.3],
                "p_flip_initial_up": [0.12, 0.99, 0.2],
                "p_flip_initial_down": [0.08, 0.98, 0.16],
                "theta_deg": [10.0, 99.0, 13.0],
                "trajectory_phi_deg": [1.0, 99.0, 2.0],
                "r_min_ang": [0.3, 9.9, 0.5],
                "steps": [100, 6400, 130],
                "converged": [True, False, True],
            }
        )
        fig = Figure()
        phase_axis = fig.add_subplot(311)
        angle_axis = fig.add_subplot(312)
        diagnostic_axis = fig.add_subplot(313)

        draw_trajectory_sweep_plots(
            phase_axis,
            angle_axis,
            diagnostic_axis,
            frame,
            x_column="energy_eV",
            x_label="Энергия, эВ",
        )

        self.assertTrue(np.isnan(phase_axis.lines[0].get_ydata()[1]))
        self.assertTrue(np.isnan(phase_axis.lines[1].get_ydata()[1]))
        self.assertTrue(np.isnan(angle_axis.lines[0].get_ydata()[1]))
        self.assertTrue(np.isnan(angle_axis.lines[1].get_ydata()[1]))
        self.assertTrue(np.isnan(diagnostic_axis.lines[0].get_ydata()[1]))
        self.assertTrue(any("Ошибок в точках" in text.get_text() for text in diagnostic_axis.texts))

    def test_probability_by_rmin_sorts_valid_points_and_marks_screening_radius(self):
        frame = pd.DataFrame(
            {
                "sweep_value": [300.0, 100.0, 200.0],
                "atomic_number": [29.0, 29.0, 29.0],
                "p_flip_initial_up": [0.3, 0.1, 0.2],
                "p_flip_initial_down": [0.25, 0.08, 0.15],
                "theta_deg": [13.0, 10.0, 12.0],
                "trajectory_phi_deg": [2.0, 1.0, 1.5],
                "r_min_ang": [0.5, 0.3, 0.4],
                "steps": [130, 100, 120],
                "converged": [True, True, True],
            }
        )
        fig = Figure()
        phase_axis = fig.add_subplot(311)
        angle_axis = fig.add_subplot(312)
        diagnostic_axis = fig.add_subplot(313)

        draw_trajectory_probability_by_rmin(
            phase_axis,
            angle_axis,
            diagnostic_axis,
            frame,
            sweep_x_label="E, эВ",
        )

        np.testing.assert_allclose(phase_axis.lines[1].get_xdata(), [0.3, 0.4, 0.5])
        np.testing.assert_allclose(phase_axis.lines[1].get_ydata(), [0.1, 0.2, 0.3])
        self.assertEqual(phase_axis.get_title(), "Вероятность изменения спина от минимального сближения")
        self.assertEqual(phase_axis.get_xlabel(), "r_min, Å")
        self.assertEqual(angle_axis.get_ylabel(), "E, эВ")
        labels = phase_axis.get_legend_handles_labels()[1]
        self.assertTrue(any(label.startswith("r_TF=b·Z^(-1/3)=") for label in labels))


if __name__ == "__main__":
    unittest.main()
