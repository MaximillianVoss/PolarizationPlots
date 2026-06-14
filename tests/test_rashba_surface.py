import unittest

import numpy as np
from matplotlib.figure import Figure

from polarization_app.gui.plotting import draw_rashba_surface_plots
from polarization_app.physics.rashba_surface import RashbaSurfaceRequest, compute_rashba_surface


class RashbaSurfaceTestCase(unittest.TestCase):
    def test_zero_rashba_has_equal_transmission_and_zero_polarization(self):
        result = compute_rashba_surface(
            RashbaSurfaceRequest(
                energy_min_eV=100.0,
                energy_max_eV=200.0,
                point_count=5,
                rashba_alpha_au=0.0,
                surface_potential_eV=5.0,
                ver_up_to_down=0.0,
                ver_down_to_up=0.0,
            )
        )
        frame = result.frame

        np.testing.assert_allclose(frame["transmission_up"], frame["transmission_down"], atol=1e-12)
        np.testing.assert_allclose(frame["t_plus_sq"], frame["t_minus_sq"], atol=1e-12)
        np.testing.assert_allclose(frame["t_plus_sq"], frame["transmission_up"] * 0.5, atol=1e-12)
        np.testing.assert_allclose(frame["t_minus_sq"], frame["transmission_down"] * 0.5, atol=1e-12)
        np.testing.assert_allclose(frame["polarization"], np.zeros(len(frame)), atol=1e-12)

    def test_volume_flip_probabilities_are_applied_to_exit_channels(self):
        result = compute_rashba_surface(
            RashbaSurfaceRequest(
                energy_min_eV=100.0,
                energy_max_eV=200.0,
                point_count=3,
                rashba_alpha_au=0.0,
                surface_potential_eV=1.0,
                ver_up_to_down=np.array([0.25, 0.25, 0.25]),
                ver_down_to_up=np.array([0.0, 0.0, 0.0]),
            )
        )
        frame = result.frame

        np.testing.assert_allclose(frame["t_plus_sq"], frame["transmission_up"] * 0.375, atol=1e-12)
        np.testing.assert_allclose(frame["t_minus_sq"], frame["transmission_down"] * 0.625, atol=1e-12)
        self.assertTrue(np.all(frame["polarization"].to_numpy(dtype=float) < 0.0))

    def test_volume_flip_probabilities_keep_exit_channels_bounded(self):
        result = compute_rashba_surface(
            RashbaSurfaceRequest(
                energy_min_eV=100.0,
                energy_max_eV=200.0,
                point_count=3,
                rashba_alpha_au=0.02,
                surface_potential_eV=1.0,
                ver_up_to_down=np.array([0.0, 0.5, 1.0]),
                ver_down_to_up=np.array([1.0, 0.5, 0.0]),
            )
        )
        frame = result.frame

        self.assertTrue(np.all(frame["t_plus_sq"].to_numpy(dtype=float) >= 0.0))
        self.assertTrue(np.all(frame["t_plus_sq"].to_numpy(dtype=float) <= 1.0 + 1e-12))
        self.assertTrue(np.all(frame["t_minus_sq"].to_numpy(dtype=float) >= 0.0))
        self.assertTrue(np.all(frame["t_minus_sq"].to_numpy(dtype=float) <= 1.0 + 1e-12))

    def test_rashba_plot_titles_and_legends_are_named(self):
        result = compute_rashba_surface(
            RashbaSurfaceRequest(
                energy_min_eV=100.0,
                energy_max_eV=200.0,
                point_count=3,
                rashba_alpha_au=0.02,
                surface_potential_eV=1.0,
            )
        )
        fig = Figure()
        transmission_axis = fig.add_subplot(211)
        polarization_axis = fig.add_subplot(212)

        draw_rashba_surface_plots(transmission_axis, polarization_axis, result.frame)

        transmission_labels = transmission_axis.get_legend_handles_labels()[1]
        polarization_labels = polarization_axis.get_legend_handles_labels()[1]
        self.assertEqual(transmission_axis.get_title(), "Вероятности прохождения через поверхность с Рашбой")
        self.assertIn("t_+^2, вышел ↑", transmission_labels)
        self.assertIn("t_-^2, вышел ↓", transmission_labels)
        self.assertIn("P=(t_+^2-t_-^2)/(t_+^2+t_-^2)", polarization_labels)


if __name__ == "__main__":
    unittest.main()
