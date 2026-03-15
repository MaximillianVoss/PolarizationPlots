import unittest

import numpy as np

from polarization_app.physics.phase_integrals import (
    compute_phase_integral_components,
    compute_single_atom_phase_grid,
    energy_to_speed_mps,
    exponential_chi,
)


class PhaseIntegralsTestCase(unittest.TestCase):
    def test_single_atom_grid_matches_pointwise_component_calculation(self):
        grid = compute_single_atom_phase_grid(
            Emin_eV=10.0,
            Emax_eV=50.0,
            N=4,
            Z=29.0,
            a_ang=0.8,
            b_ang=0.53,
            c1=1.0,
            c2=1.0,
            dr_ang=0.05,
            r_max_ang=6.0,
            chi=exponential_chi,
            i3_mode="sum_avg",
        )

        expected = np.asarray(
            [
                compute_phase_integral_components(
                    speed_mps=float(speed),
                    Z=29.0,
                    a_ang=0.8,
                    b_ang=0.53,
                    c1=1.0,
                    c2=1.0,
                    dr_ang=0.05,
                    r_max_ang=6.0,
                    chi=exponential_chi,
                    i3_mode="sum_avg",
                )
                for speed in energy_to_speed_mps(grid["E_eV"].to_numpy(dtype=float))
            ],
            dtype=float,
        )

        np.testing.assert_allclose(
            grid[["I1", "I2", "I3", "Phi"]].to_numpy(dtype=float),
            expected,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_phase_components_scale_inverse_to_speed(self):
        component_fast = np.asarray(
            compute_phase_integral_components(
                speed_mps=2.0e6,
                Z=29.0,
                a_ang=1.1,
                b_ang=0.53,
                c1=1.0,
                c2=1.0,
                dr_ang=0.05,
                r_max_ang=6.0,
                chi=exponential_chi,
                i3_mode="trapz",
            ),
            dtype=float,
        )
        component_slow = np.asarray(
            compute_phase_integral_components(
                speed_mps=1.0e6,
                Z=29.0,
                a_ang=1.1,
                b_ang=0.53,
                c1=1.0,
                c2=1.0,
                dr_ang=0.05,
                r_max_ang=6.0,
                chi=exponential_chi,
                i3_mode="trapz",
            ),
            dtype=float,
        )

        np.testing.assert_allclose(component_fast * 2.0, component_slow, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
