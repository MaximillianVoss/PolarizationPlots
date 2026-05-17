import unittest

from polarization_app.application.formulas import FORMULA_LABELS, FORMULA_NEW
from polarization_app.gui.app import App


class AppBindingProbe(App):
    def __init__(self):
        self.left_direct_calls = 0
        self.left_scheduled_calls: list[int] = []
        self.right_scheduled_calls: list[int] = []
        super().__init__()
        self.withdraw()
        self.left_direct_calls = 0
        self.left_scheduled_calls.clear()
        self.right_scheduled_calls.clear()

    def update_output_left(self) -> None:
        self.left_direct_calls += 1

    def _schedule_left_update(self, delay_ms: int = 0) -> None:
        self.left_scheduled_calls.append(delay_ms)

    def _schedule_right_update(self, delay_ms: int = 0) -> None:
        self.right_scheduled_calls.append(delay_ms)


class AppBindingsTestCase(unittest.TestCase):
    def setUp(self):
        self.app = AppBindingProbe()

    def tearDown(self):
        if self.app.winfo_exists():
            self.app._on_close()

    def test_geometry_controls_trigger_left_and_right_auto_recalc(self):
        actions = [
            ("a", lambda: self.app.a.set(self.app.a.get() + 0.2)),
            ("R_bohr", lambda: self.app.R_bohr.set(self.app.R_bohr.get() + 0.1)),
            ("alpha", lambda: self.app.alpha_deg.set(self.app.alpha_deg.get() + 1.0)),
            ("beta", lambda: self.app.beta_deg.set(self.app.beta_deg.get() - 1.0)),
            ("n", lambda: self.app.lattice_radius.set(self.app.lattice_radius.get() + 1)),
            ("d", lambda: self.app.d_layer.set(self.app.d_layer.get() + 1)),
            ("auto_n", lambda: self.app.auto_n.set(not self.app.auto_n.get())),
        ]

        for label, action in actions:
            with self.subTest(control=label):
                left_before = len(self.app.left_scheduled_calls)
                right_before = len(self.app.right_scheduled_calls)
                action()
                self.assertEqual(len(self.app.left_scheduled_calls), left_before + 1)
                self.assertEqual(len(self.app.right_scheduled_calls), right_before + 1)

    def test_orbital_l_updates_left_immediately_and_right_when_auto_enabled(self):
        left_before = self.app.left_direct_calls
        right_before = len(self.app.right_scheduled_calls)

        self.app.orbital_l.set(self.app.orbital_l.get() + 1)

        self.assertEqual(self.app.left_direct_calls, left_before + 1)
        self.assertEqual(len(self.app.right_scheduled_calls), right_before + 1)

    def test_calculation_controls_trigger_right_auto_recalc(self):
        actions = [
            ("Z", lambda: self.app.Z.set(self.app.Z.get() + 1.0)),
            ("b", lambda: self.app.b.set(self.app.b.get() + 0.05)),
            ("c1", lambda: self.app.c1.set(self.app.c1.get() + 0.1)),
            ("c2", lambda: self.app.c2.set(self.app.c2.get() + 0.1)),
            ("dr", lambda: self.app.dr.set(self.app.dr.get() + 0.001)),
            ("rmax", lambda: self.app.rmax.set(self.app.rmax.get() + 1.0)),
            ("Emin", lambda: self.app.Emin.set(self.app.Emin.get() + 1.0)),
            ("Emax", lambda: self.app.Emax.set(self.app.Emax.get() + 1000.0)),
            ("Npts", lambda: self.app.Npts.set(self.app.Npts.get() + 1)),
            ("formula", lambda: self.app.formula_variant_label.set(FORMULA_LABELS[FORMULA_NEW])),
            ("chi", lambda: self.app.use_table_chi.set(not self.app.use_table_chi.get())),
            ("i3", lambda: self.app.i3_mode_sum.set(not self.app.i3_mode_sum.get())),
        ]

        for label, action in actions:
            with self.subTest(control=label):
                left_before = len(self.app.left_scheduled_calls)
                right_before = len(self.app.right_scheduled_calls)
                action()
                self.assertEqual(len(self.app.left_scheduled_calls), left_before)
                self.assertEqual(len(self.app.right_scheduled_calls), right_before + 1)

    def test_boundary_controls_do_not_trigger_main_recalc(self):
        actions = [
            ("boundary_alpha", lambda: self.app.boundary_alpha_deg.set(self.app.boundary_alpha_deg.get() + 1.0)),
            ("boundary_A", lambda: self.app.boundary_work_function.set(self.app.boundary_work_function.get() + 0.2)),
            ("boundary_E", lambda: self.app.boundary_energy_point.set(self.app.boundary_energy_point.get() + 5.0)),
        ]

        for label, action in actions:
            with self.subTest(control=label):
                left_before = len(self.app.left_scheduled_calls)
                right_before = len(self.app.right_scheduled_calls)
                action()
                self.assertEqual(len(self.app.left_scheduled_calls), left_before)
                self.assertEqual(len(self.app.right_scheduled_calls), right_before)

    def test_auto_disabled_blocks_auto_recalc_for_non_orbital_controls(self):
        self.app.auto.set(False)
        self.app.left_scheduled_calls.clear()
        self.app.right_scheduled_calls.clear()
        left_direct_before = self.app.left_direct_calls

        self.app.a.set(self.app.a.get() + 0.2)
        self.app.Z.set(self.app.Z.get() + 1.0)

        self.assertEqual(self.app.left_direct_calls, left_direct_before)
        self.assertEqual(self.app.left_scheduled_calls, [])
        self.assertEqual(self.app.right_scheduled_calls, [])

    def test_orbital_l_still_refreshes_left_when_auto_disabled(self):
        self.app.auto.set(False)
        self.app.left_scheduled_calls.clear()
        self.app.right_scheduled_calls.clear()
        left_before = self.app.left_direct_calls

        self.app.orbital_l.set(self.app.orbital_l.get() + 1)

        self.assertEqual(self.app.left_direct_calls, left_before + 1)
        self.assertEqual(self.app.left_scheduled_calls, [])
        self.assertEqual(self.app.right_scheduled_calls, [])

    def test_user_depth_is_translated_to_zero_based_source_layer(self):
        self.app.d_layer.set(4)

        geometry = self.app._current_geometry()

        self.assertEqual(geometry.source_depth, 4)
        self.assertEqual(geometry.source_layer, 3)


if __name__ == "__main__":
    unittest.main()
