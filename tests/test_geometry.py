import unittest
import math

from polarization_app.application.geometry import GeometryContext, collect_atom_selection
from polarization_app.domain.lattice import (
    direction_from_spherical_angles,
    estimate_lattice_radius,
    estimate_lattice_radius_details,
    estimate_lattice_search_region,
)


class GeometryContextTestCase(unittest.TestCase):
    def test_context_converts_angles_and_interaction_radius(self):
        geometry = GeometryContext(
            lattice_constant_ang=4.75,
            bohr_radius_ang=0.53,
            alpha_deg=30.0,
            beta_deg=-45.0,
            lattice_radius=3,
            source_layer=1,
            orbital_l=2,
        )

        self.assertAlmostEqual(geometry.interaction_radius_ang, 2.65)
        self.assertAlmostEqual(geometry.alpha_rad, 0.5235987755982988)
        self.assertAlmostEqual(geometry.beta_rad, -0.7853981633974483)
        self.assertEqual(geometry.source_depth, 2)
        self.assertAlmostEqual(geometry.source_z_ang, 4.75)
        self.assertAlmostEqual(geometry.surface_z_ang, 0.0)

    def test_collect_atom_selection_limits_impact_parameters(self):
        geometry = GeometryContext(
            lattice_constant_ang=4.75,
            bohr_radius_ang=0.53,
            alpha_deg=30.0,
            beta_deg=60.0,
            lattice_radius=4,
            source_layer=0,
            orbital_l=1,
        )

        selection = collect_atom_selection(geometry, max_atoms=5)
        self.assertLessEqual(len(selection.selected_atoms), 5)
        self.assertTrue(all(value > 0 for value in selection.impact_parameters_ang))

    def test_zero_polar_angle_points_toward_surface(self):
        direction = direction_from_spherical_angles(alpha_rad=0.0, beta_rad=0.0)

        self.assertAlmostEqual(direction[0], 0.0)
        self.assertAlmostEqual(direction[1], 0.0)
        self.assertLess(direction[2], 0.0)

    def test_estimate_lattice_radius_returns_positive_value(self):
        radius = estimate_lattice_radius(
            lattice_constant_ang=4.75,
            bohr_radius_ang=0.53,
            alpha_rad=0.5,
            beta_rad=1.0,
            source_layer=2,
        )
        self.assertGreaterEqual(radius, 1)

    def test_estimate_lattice_radius_grows_with_larger_tilt(self):
        small_tilt = estimate_lattice_radius(
            lattice_constant_ang=4.75,
            bohr_radius_ang=0.53,
            alpha_rad=math.radians(10.0),
            beta_rad=0.0,
            source_layer=1,
        )
        large_tilt = estimate_lattice_radius(
            lattice_constant_ang=4.75,
            bohr_radius_ang=0.53,
            alpha_rad=math.radians(75.0),
            beta_rad=0.0,
            source_layer=1,
        )
        self.assertGreaterEqual(large_tilt, small_tilt)

    def test_estimate_lattice_radius_details_report_limit_cap(self):
        details = estimate_lattice_radius_details(
            lattice_constant_ang=4.75,
            bohr_radius_ang=0.53,
            alpha_rad=math.radians(88.0),
            beta_rad=0.0,
            source_layer=0,
            max_atoms=125,
        )
        self.assertTrue(details.capped_by_max_atoms)
        self.assertGreater(details.required_radius, details.radius)

    def test_search_region_shifts_along_z_with_source_layer(self):
        shallow = estimate_lattice_search_region(
            lattice_constant_ang=4.75,
            bohr_radius_ang=0.53,
            alpha_rad=math.radians(35.0),
            beta_rad=math.radians(20.0),
            source_layer=0,
        ).region
        deeper = estimate_lattice_search_region(
            lattice_constant_ang=4.75,
            bohr_radius_ang=0.53,
            alpha_rad=math.radians(35.0),
            beta_rad=math.radians(20.0),
            source_layer=4,
        ).region
        self.assertEqual(shallow.x_radius, deeper.x_radius)
        self.assertEqual(shallow.y_radius, deeper.y_radius)
        self.assertEqual(shallow.z_min_layer, 0)
        self.assertEqual(deeper.z_min_layer, 0)
        self.assertGreater(deeper.z_max_layer, shallow.z_max_layer)


if __name__ == "__main__":
    unittest.main()
