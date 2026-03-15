import unittest

import numpy as np
from matplotlib.figure import Figure

from polarization_app.application.geometry import GeometryContext, collect_atom_selection
from polarization_app.gui.plotting import build_geometry_preview_data, draw_geometry_preview, zoom_3d_axis


class GeometryPreviewTestCase(unittest.TestCase):
    def test_build_geometry_preview_data_contains_expected_core_fields(self):
        geometry = GeometryContext(
            lattice_constant_ang=4.75,
            bohr_radius_ang=0.53,
            alpha_deg=35.0,
            beta_deg=20.0,
            lattice_radius=4,
            source_layer=1,
            orbital_l=1,
        )
        atom_selection = collect_atom_selection(geometry, max_atoms=12)

        preview = build_geometry_preview_data(geometry, atom_selection, max_lattice_points=300, max_selected_points=10)

        self.assertLessEqual(len(preview.lattice_points), 300)
        self.assertLessEqual(len(preview.selected_points), 10)
        np.testing.assert_allclose(preview.origin, np.array([0.0, 0.0, geometry.source_layer * geometry.lattice_constant_ang]))
        self.assertEqual(preview.source_depth, geometry.source_depth)
        self.assertAlmostEqual(preview.surface_z_ang, 0.0)
        self.assertAlmostEqual(np.linalg.norm(preview.direction), 1.0)
        self.assertGreater(np.linalg.norm(preview.trajectory_end - preview.origin), 0.0)
        if len(preview.source_layer_points):
            np.testing.assert_allclose(preview.source_layer_points[:, 2], np.full(len(preview.source_layer_points), preview.origin[2]))

    def test_draw_geometry_preview_renders_titles_and_axes(self):
        geometry = GeometryContext(
            lattice_constant_ang=4.75,
            bohr_radius_ang=0.53,
            alpha_deg=25.0,
            beta_deg=-30.0,
            lattice_radius=3,
            source_layer=0,
            orbital_l=1,
        )
        atom_selection = collect_atom_selection(geometry, max_atoms=8)
        preview = build_geometry_preview_data(geometry, atom_selection, max_lattice_points=150, max_selected_points=8)

        figure = Figure(figsize=(6.0, 4.5), dpi=100)
        grid = figure.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0])
        axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
        axis_xz = figure.add_subplot(grid[0, 1])
        axis_xy = figure.add_subplot(grid[1, 1])

        draw_geometry_preview(axis_3d, axis_xz, axis_xy, preview)

        self.assertEqual(axis_3d.get_title(), "3D схема решётки и траектории")
        self.assertEqual(axis_xz.get_title(), "Проекция XZ: полярный угол и глубина")
        self.assertEqual(axis_xy.get_title(), "Проекция XY: азимутальный угол")
        self.assertEqual(axis_xz.get_xlabel(), "x, Å")
        self.assertEqual(axis_xz.get_ylabel(), "глубина, Å")
        self.assertEqual(axis_xy.get_ylabel(), "y, Å")

    def test_preview_trajectory_goes_toward_surface_for_small_alpha(self):
        geometry = GeometryContext(
            lattice_constant_ang=4.75,
            bohr_radius_ang=0.53,
            alpha_deg=0.0,
            beta_deg=0.0,
            lattice_radius=3,
            source_layer=2,
            orbital_l=1,
        )
        atom_selection = collect_atom_selection(geometry, max_atoms=8)

        preview = build_geometry_preview_data(geometry, atom_selection, max_lattice_points=150, max_selected_points=8)

        self.assertLess(preview.trajectory_end[2], preview.origin[2])

    def test_zoom_3d_axis_changes_visible_span(self):
        figure = Figure(figsize=(4.0, 3.0), dpi=100)
        axis_3d = figure.add_subplot(111, projection="3d")
        axis_3d.set_xlim(-10.0, 10.0)
        axis_3d.set_ylim(-8.0, 12.0)
        axis_3d.set_zlim(15.0, 0.0)

        before = (
            abs(axis_3d.get_xlim()[1] - axis_3d.get_xlim()[0]),
            abs(axis_3d.get_ylim()[1] - axis_3d.get_ylim()[0]),
            abs(axis_3d.get_zlim()[1] - axis_3d.get_zlim()[0]),
        )

        zoom_3d_axis(axis_3d, 0.85)

        after = (
            abs(axis_3d.get_xlim()[1] - axis_3d.get_xlim()[0]),
            abs(axis_3d.get_ylim()[1] - axis_3d.get_ylim()[0]),
            abs(axis_3d.get_zlim()[1] - axis_3d.get_zlim()[0]),
        )

        self.assertLess(after[0], before[0])
        self.assertLess(after[1], before[1])
        self.assertLess(after[2], before[2])


if __name__ == "__main__":
    unittest.main()
