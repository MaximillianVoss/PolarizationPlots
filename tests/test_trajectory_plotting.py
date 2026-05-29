import unittest

import pandas as pd
from matplotlib.figure import Figure

from polarization_app.gui.plotting import draw_trajectory_sweep_plots


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
        self.assertEqual(diagnostic_axis.get_title(), "r_min и внутренние шаги интегрирования")
        self.assertIn("начальный ↑: ↑→↓", spin_flip_labels)
        self.assertIn("начальный ↓: ↓→↑", spin_flip_labels)
        self.assertIn("φ, угол траектории после взаимодействия", angle_labels)
        self.assertIn("steps, внутренние шаги интегрирования", diagnostic_labels)


if __name__ == "__main__":
    unittest.main()
