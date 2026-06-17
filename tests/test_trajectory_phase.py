import unittest
from unittest.mock import patch

import numpy as np

import polarization_app.physics.trajectory_phase as trajectory_phase
from polarization_app.application.trajectory import (
    DEFAULT_PRECISE_TRAJECTORY_MAX_PHASE_STEP_RAD,
    DEFAULT_PRECISE_TRAJECTORY_MIN_STEPS,
    DEFAULT_TRAJECTORY_MIN_STEPS,
    TRAJECTORY_SWEEP_ANGLE_STEP,
    TRAJECTORY_SWEEP_ENERGY,
    TRAJECTORY_SWEEP_IMPACT,
    TrajectorySweepRequest,
    execute_trajectory_sweep,
)
from polarization_app.physics.phase_integrals import (
    BOHR_TO_ANGSTROM,
    spline_thomas_fermi_chi,
    spline_thomas_fermi_chi_derivative,
)
from polarization_app.physics.trajectory_phase import (
    DEFAULT_THOMAS_FERMI_B_BOHR,
    ELECTRON_MASS_AMU,
    RADIAL_BASE_PANEL_LIMIT,
    _base_quadrature_panel_count,
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

    def test_thomas_fermi_spline_is_smooth_at_table_nodes(self):
        x0 = 1.2
        eps = 1e-4

        center = float(spline_thomas_fermi_chi(np.array([x0]))[0])
        left = float(spline_thomas_fermi_chi(np.array([x0 - eps]))[0])
        right = float(spline_thomas_fermi_chi(np.array([x0 + eps]))[0])
        left_slope = (center - left) / eps
        right_slope = (right - center) / eps
        derivative = float(spline_thomas_fermi_chi_derivative(np.array([x0]))[0])

        self.assertAlmostEqual(left_slope, right_slope, delta=1e-3)
        self.assertAlmostEqual(derivative, 0.5 * (left_slope + right_slope), delta=1e-3)

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

    def test_near_critical_outer_turning_point_is_not_missed(self):
        result = compute_atom_trajectory_phase(
            energy_eV=152.329210,
            mass_amu=1.0,
            atomic_number=80.0,
            impact_parameter_ang=0.621,
            r0_ang=2.17,
            angle_step_rad=np.deg2rad(3.0),
            orbital_l=6,
            min_steps=30,
            max_refinements=6,
        )

        self.assertTrue(result.converged)
        self.assertEqual(result.status, "ok")
        self.assertGreater(result.r_min_ang, 0.13)
        self.assertLess(result.steps, 200)

    def test_negative_radial_domain_repairs_lower_bound(self):
        params = {
            "energy_eV": 152.329210,
            "mass_amu": 1.0,
            "atomic_number": 80.0,
            "impact_parameter_ang": 0.621,
            "r0_ang": 2.17,
            "angle_step_rad": np.deg2rad(3.0),
            "orbital_l": 6,
            "min_steps": 30,
            "max_refinements": 6,
        }
        speed_mps = float(
            trajectory_phase.energy_eV_to_speed_mps_for_mass(params["energy_eV"], params["mass_amu"])
        )
        speed_au = float(trajectory_phase.speed_mps_to_atomic_units(speed_mps))
        mass_electron_units = trajectory_phase.mass_amu_to_electron_masses(params["mass_amu"])
        impact_bohr = params["impact_parameter_ang"] / BOHR_TO_ANGSTROM
        r0_bohr = params["r0_ang"] / BOHR_TO_ANGSTROM
        true_r_min_bohr = trajectory_phase.find_minimum_approach_bohr(
            atomic_number=params["atomic_number"],
            impact_parameter_bohr=impact_bohr,
            r0_bohr=r0_bohr,
            speed_au=speed_au,
            mass_electron_units=mass_electron_units,
            chi=trajectory_phase.spline_thomas_fermi_chi,
        )
        bad_r_min_bohr = true_r_min_bohr * 0.8

        with patch.object(trajectory_phase, "find_minimum_approach_bohr", return_value=bad_r_min_bohr):
            result = trajectory_phase.compute_atom_trajectory_phase(**params)

        self.assertTrue(result.converged)
        self.assertEqual(result.status, "ok")
        self.assertGreater(result.r_min_ang, bad_r_min_bohr * BOHR_TO_ANGSTROM)
        self.assertAlmostEqual(result.r_min_ang, true_r_min_bohr * BOHR_TO_ANGSTROM, delta=1e-6)

    def test_low_energy_large_impact_finishes_near_turning_point(self):
        result = compute_atom_trajectory_phase(
            energy_eV=10.0,
            mass_amu=1.0,
            atomic_number=80.0,
            impact_parameter_ang=2.15,
            r0_ang=2.17,
            angle_step_rad=np.deg2rad(3.0),
            orbital_l=6,
            min_steps=30,
            max_refinements=6,
        )

        self.assertTrue(result.converged)
        self.assertEqual(result.status, "ok")
        self.assertGreater(result.r_min_ang, 2.0)
        self.assertLess(result.r_min_ang, result.r0_ang)

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

    def test_quadrature_uses_requested_minimum_step_count(self):
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
        self.assertTrue(result.converged)

    def test_tiny_angle_step_is_capped_for_base_quadrature_grid(self):
        tiny_angle_panels = _base_quadrature_panel_count(
            angular_step_rad=np.deg2rad(0.01),
            min_steps=30,
        )
        explicit_minimum_panels = _base_quadrature_panel_count(
            angular_step_rad=np.deg2rad(0.01),
            min_steps=RADIAL_BASE_PANEL_LIMIT + 250,
        )

        self.assertEqual(tiny_angle_panels, RADIAL_BASE_PANEL_LIMIT)
        self.assertEqual(explicit_minimum_panels, RADIAL_BASE_PANEL_LIMIT + 250)

    def test_capped_tiny_angle_keeps_uncapped_accuracy_after_refinement(self):
        params = {
            "energy_eV": 386.0,
            "mass_amu": 1.0,
            "atomic_number": 81.0,
            "impact_parameter_ang": 0.2,
            "r0_ang": 3.0,
            "angle_step_rad": np.deg2rad(0.1),
            "orbital_l": 4,
            "min_steps": 100,
            "max_refinements": 6,
            "max_phase_step_rad": 0.05,
        }
        original_limit = trajectory_phase.RADIAL_BASE_PANEL_LIMIT
        try:
            trajectory_phase.RADIAL_BASE_PANEL_LIMIT = 1000
            capped = trajectory_phase.compute_atom_trajectory_phase(**params)
            trajectory_phase.RADIAL_BASE_PANEL_LIMIT = 1_000_000
            uncapped = trajectory_phase.compute_atom_trajectory_phase(**params)
        finally:
            trajectory_phase.RADIAL_BASE_PANEL_LIMIT = original_limit

        self.assertTrue(capped.converged)
        self.assertTrue(uncapped.converged)
        self.assertLess(capped.steps, uncapped.steps)
        self.assertAlmostEqual(capped.phase_rad, uncapped.phase_rad, delta=1e-4)
        self.assertAlmostEqual(capped.theta_rad, uncapped.theta_rad, delta=1e-4)

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
    def test_precise_mode_uses_stricter_step_requirements(self):
        result = execute_trajectory_sweep(
            TrajectorySweepRequest(
                sweep_mode=TRAJECTORY_SWEEP_IMPACT,
                point_count=1,
                atomic_number=80.0,
                mass_amu=1.0,
                energy_eV=3000.0,
                impact_min_ang=0.18,
                impact_max_ang=0.2,
                r0_ang=3.0,
                angle_step_deg=3.0,
                orbital_l=5,
                magnetic_m=4,
                precise_mode=True,
                max_phase_step_rad=DEFAULT_PRECISE_TRAJECTORY_MAX_PHASE_STEP_RAD,
                parallel_workers=1,
            )
        )

        self.assertTrue(result.request.precise_mode)
        self.assertGreaterEqual(int(result.frame.iloc[0]["steps"]), DEFAULT_PRECISE_TRAJECTORY_MIN_STEPS)

    def test_convergence_check_adds_dtheta_diagnostics(self):
        result = execute_trajectory_sweep(
            TrajectorySweepRequest(
                sweep_mode=TRAJECTORY_SWEEP_IMPACT,
                point_count=2,
                atomic_number=29.0,
                energy_eV=200.0,
                impact_min_ang=0.7,
                impact_max_ang=0.8,
                r0_ang=10.0,
                angle_step_deg=2.0,
                orbital_l=1,
                magnetic_m=0,
                convergence_check=True,
                parallel_workers=1,
            )
        )
        frame = result.frame

        self.assertTrue(frame["convergence_checked"].all())
        self.assertIn("phase_rad_dtheta_half", frame.columns)
        self.assertIn("phase_rad_dtheta_quarter", frame.columns)
        self.assertTrue(np.isfinite(frame["convergence_phase_error_rad"].to_numpy(dtype=float)).all())
        self.assertTrue(np.isfinite(frame["convergence_probability_error"].to_numpy(dtype=float)).all())

    def test_angle_step_sweep_is_stable_for_high_z_case(self):
        result = execute_trajectory_sweep(
            TrajectorySweepRequest(
                sweep_mode=TRAJECTORY_SWEEP_ANGLE_STEP,
                point_count=25,
                atomic_number=81.0,
                mass_amu=1.0,
                energy_eV=386.0,
                impact_parameter_ang=0.2,
                r0_ang=3.0,
                angle_step_min_deg=0.1,
                angle_step_max_deg=5.0,
                orbital_l=4,
                magnetic_m=2,
                parallel_workers=2,
            )
        )
        frame = result.frame

        self.assertTrue(frame["converged"].all())
        self.assertLess(
            float(np.ptp(frame["p_flip_initial_up"].to_numpy(dtype=float))),
            0.005,
        )
        self.assertLess(
            float(np.ptp(frame["p_flip_initial_down"].to_numpy(dtype=float))),
            0.006,
        )

    def test_default_min_steps_avoids_coarse_impact_jump(self):
        result = execute_trajectory_sweep(
            TrajectorySweepRequest(
                sweep_mode=TRAJECTORY_SWEEP_IMPACT,
                point_count=13,
                atomic_number=80.0,
                mass_amu=1.0,
                energy_eV=3000.0,
                impact_min_ang=0.166,
                impact_max_ang=0.171,
                r0_ang=3.0,
                angle_step_deg=3.0,
                orbital_l=5,
                magnetic_m=4,
                parallel_workers=2,
            )
        )

        frame = result.frame
        max_angle_jump = np.max(np.abs(np.diff(frame["theta_deg"].to_numpy(dtype=float))))
        self.assertEqual(result.request.min_steps, DEFAULT_TRAJECTORY_MIN_STEPS)
        self.assertTrue(frame["converged"].all())
        self.assertGreaterEqual(int(frame["steps"].min()), DEFAULT_TRAJECTORY_MIN_STEPS)
        self.assertLess(max_angle_jump, 5.0)

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
