import unittest

import numpy as np

from polarization_app.application.trajectory import (
    TRAJECTORY_SWEEP_ENERGY,
    TRAJECTORY_SWEEP_IMPACT,
    TrajectorySweepRequest,
    execute_trajectory_sweep,
)
from polarization_app.physics.phase_integrals import (
    spline_thomas_fermi_chi,
    spline_thomas_fermi_chi_derivative,
)
from polarization_app.physics.trajectory_phase import (
    DEFAULT_THOMAS_FERMI_B_BOHR,
    ELECTRON_MASS_AMU,
    _trajectory_phase_rate_scalar,
    compute_atom_trajectory_phase,
    energy_eV_to_speed_mps_for_mass,
    mass_amu_to_electron_masses,
)


class TrajectoryPhaseTestCase(unittest.TestCase):
    def test_thomas_fermi_b_matches_documented_constant(self):
        expected = 0.5 * ((3.0 * np.pi / 4.0) ** (2.0 / 3.0))

        self.assertAlmostEqual(DEFAULT_THOMAS_FERMI_B_BOHR, expected)

    def test_thomas_fermi_spline_and_derivative_are_finite(self):
        x_values = np.array([0.0, 0.1, 1.0, 10.0])

        chi_values = spline_thomas_fermi_chi(x_values)
        derivative_values = spline_thomas_fermi_chi_derivative(x_values)

        self.assertTrue(np.all(np.isfinite(chi_values)))
        self.assertTrue(np.all(np.isfinite(derivative_values)))
        self.assertTrue(np.all(chi_values >= 0.0))

    def test_energy_to_speed_uses_mass_in_amu(self):
        electron_speed = float(energy_eV_to_speed_mps_for_mass(100.0, ELECTRON_MASS_AMU))
        heavier_speed = float(energy_eV_to_speed_mps_for_mass(100.0, ELECTRON_MASS_AMU * 4.0))

        self.assertAlmostEqual(electron_speed / 2.0, heavier_speed, delta=electron_speed * 1e-12)

    def test_one_amu_mass_is_scaled_in_electron_masses(self):
        self.assertAlmostEqual(mass_amu_to_electron_masses(1.0), 1.0 / ELECTRON_MASS_AMU)

    def test_atom_trajectory_accepts_one_amu_mass_without_max_steps(self):
        result = compute_atom_trajectory_phase(
            energy_eV=100.0,
            mass_amu=1.0,
            atomic_number=29.0,
            impact_parameter_ang=0.8,
            r0_ang=10.0,
            angle_step_rad=np.deg2rad(3.0),
            orbital_l=1,
            min_steps=30,
        )

        self.assertTrue(result.converged)
        self.assertEqual(result.status, "ok")
        self.assertLess(result.steps, 1000)
        self.assertTrue(np.isfinite(result.phase_rad))

    def test_small_impact_uses_adaptive_dt_without_huge_step_count(self):
        result = compute_atom_trajectory_phase(
            energy_eV=100.0,
            mass_amu=ELECTRON_MASS_AMU,
            atomic_number=60.5,
            impact_parameter_ang=0.3,
            r0_ang=3.5,
            angle_step_rad=np.deg2rad(3.0),
            orbital_l=5,
            min_steps=30,
        )

        self.assertTrue(result.converged)
        self.assertLess(result.steps, 1000)

    def test_minimum_approach_uses_outer_turning_point(self):
        result = compute_atom_trajectory_phase(
            energy_eV=100.0,
            mass_amu=ELECTRON_MASS_AMU,
            atomic_number=60.5,
            impact_parameter_ang=0.7,
            r0_ang=3.5,
            angle_step_rad=np.deg2rad(3.0),
            orbital_l=5,
            min_steps=30,
        )

        self.assertTrue(result.converged)
        self.assertGreater(result.r_min_ang, 0.14)
        self.assertLess(result.steps, 1000)

    def test_atom_trajectory_returns_expected_diagnostics(self):
        result = compute_atom_trajectory_phase(
            energy_eV=100.0,
            mass_amu=ELECTRON_MASS_AMU,
            atomic_number=29.0,
            impact_parameter_ang=0.8,
            r0_ang=10.0,
            angle_step_rad=np.deg2rad(3.0),
            orbital_l=1,
            min_steps=30,
        )

        self.assertGreater(result.r_min_ang, 0.0)
        self.assertLess(result.r_min_ang, result.r0_ang)
        self.assertGreaterEqual(result.steps, 30)
        self.assertTrue(result.converged)
        self.assertTrue(np.isfinite(result.phase_rad))
        self.assertTrue(np.isfinite(result.trajectory_angle_rad))

    def test_dt_is_refined_until_minimum_step_count(self):
        result = compute_atom_trajectory_phase(
            energy_eV=100.0,
            mass_amu=ELECTRON_MASS_AMU,
            atomic_number=29.0,
            impact_parameter_ang=0.8,
            r0_ang=10.0,
            angle_step_rad=np.deg2rad(3.0),
            orbital_l=1,
            min_steps=1000,
            max_refinements=2,
        )

        self.assertGreaterEqual(result.steps, 1000)
        self.assertGreaterEqual(result.refinements, 1)
        self.assertTrue(result.converged)

    def test_corrected_phase_rate_uses_orbital_factor_and_r_cubed(self):
        one = lambda x: np.ones_like(x, dtype=float)
        zero = lambda x: np.zeros_like(x, dtype=float)

        rate = _trajectory_phase_rate_scalar(
            2.0,
            3.0,
            chi=one,
            chi_derivative=zero,
            spin_orbit_c1=0.25,
            orbital_l=2,
        )

        self.assertAlmostEqual(rate, 0.5 * 0.25 * (2 * 2 + 1) * 3.0 / (2.0 ** 3))


class TrajectorySweepTestCase(unittest.TestCase):
    def test_energy_sweep_builds_exportable_frame(self):
        result = execute_trajectory_sweep(
            TrajectorySweepRequest(
                sweep_mode=TRAJECTORY_SWEEP_ENERGY,
                point_count=2,
                atomic_number=29.0,
                energy_min_eV=100.0,
                energy_max_eV=120.0,
                impact_parameter_ang=0.8,
                r0_ang=10.0,
                angle_step_deg=3.0,
            ),
            rng=np.random.default_rng(123),
        )

        self.assertEqual(len(result.frame), 2)
        self.assertIn("phase_rad", result.frame.columns)
        self.assertIn("p_no_flip_initial_up", result.frame.columns)
        self.assertTrue(result.frame["converged"].all())

    def test_impact_sweep_changes_impact_parameter(self):
        result = execute_trajectory_sweep(
            TrajectorySweepRequest(
                sweep_mode=TRAJECTORY_SWEEP_IMPACT,
                point_count=3,
                atomic_number=29.0,
                energy_eV=100.0,
                impact_min_ang=0.5,
                impact_max_ang=0.9,
                r0_ang=10.0,
                angle_step_deg=3.0,
            )
        )

        np.testing.assert_allclose(result.frame["impact_parameter_ang"].to_numpy(), np.array([0.5, 0.7, 0.9]))

    def test_failed_sweep_point_keeps_remaining_results(self):
        result = execute_trajectory_sweep(
            TrajectorySweepRequest(
                sweep_mode=TRAJECTORY_SWEEP_IMPACT,
                point_count=2,
                atomic_number=29.0,
                energy_eV=100.0,
                impact_min_ang=0.8,
                impact_max_ang=11.0,
                r0_ang=10.0,
                angle_step_deg=3.0,
            )
        )

        self.assertEqual(len(result.frame), 2)
        self.assertTrue(bool(result.frame.iloc[0]["converged"]))
        self.assertFalse(bool(result.frame.iloc[1]["converged"]))
        self.assertIn("r0", result.frame.iloc[1]["status"])

    def test_parallel_sweep_matches_sequential_values(self):
        request_kwargs = {
            "sweep_mode": TRAJECTORY_SWEEP_ENERGY,
            "point_count": 3,
            "atomic_number": 29.0,
            "energy_min_eV": 100.0,
            "energy_max_eV": 140.0,
            "impact_parameter_ang": 0.8,
            "r0_ang": 10.0,
            "angle_step_deg": 3.0,
        }

        sequential = execute_trajectory_sweep(TrajectorySweepRequest(**request_kwargs, parallel_workers=1))
        parallel = execute_trajectory_sweep(TrajectorySweepRequest(**request_kwargs, parallel_workers=2))

        np.testing.assert_allclose(
            sequential.frame[["phase_rad", "theta_deg", "trajectory_phi_deg", "r_min_ang"]].to_numpy(dtype=float),
            parallel.frame[["phase_rad", "theta_deg", "trajectory_phi_deg", "r_min_ang"]].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
