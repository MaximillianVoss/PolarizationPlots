import unittest
import tkinter as tk
from types import SimpleNamespace

import numpy as np
import pandas as pd
from matplotlib.colors import to_hex

from polarization_app.application.formulas import FORMULA_LABELS, FORMULA_NEW
from polarization_app.application.trajectory import TRAJECTORY_SWEEP_ANGLE_STEP, TRAJECTORY_SWEEP_ENERGY, TRAJECTORY_SWEEP_LABELS
from polarization_app.physics.compute_backend import cpu_worker_count
from polarization_app.gui.app import APP_ICON_ICO, APP_ICON_PNG, App, RASHBA_SOURCE_TRAJECTORY
from polarization_app.gui.theme import DARK_THEME, THEMES
from polarization_app.physics.phase_integrals import DEFAULT_PHASE_C1, DEFAULT_PHASE_C2
from polarization_app.physics.trajectory_phase import DEFAULT_THOMAS_FERMI_B_BOHR


class AppBindingProbe(App):
    def __init__(self):
        self.left_direct_calls = 0
        self.left_scheduled_calls: list[int] = []
        self.right_scheduled_calls: list[int] = []
        self.trajectory_scheduled_calls: list[int] = []
        self.trajectory_start_calls = 0
        self.boundary_update_calls = 0
        self.rashba_update_calls = 0
        super().__init__()
        self.withdraw()
        self.left_direct_calls = 0
        self.left_scheduled_calls.clear()
        self.right_scheduled_calls.clear()
        self.trajectory_scheduled_calls.clear()
        self.trajectory_start_calls = 0
        self.boundary_update_calls = 0
        self.rashba_update_calls = 0

    def update_output_left(self) -> None:
        self.left_direct_calls += 1

    def _schedule_left_update(self, delay_ms: int = 0) -> None:
        self.left_scheduled_calls.append(delay_ms)

    def _schedule_right_update(self, delay_ms: int = 0) -> None:
        self.right_scheduled_calls.append(delay_ms)

    def _schedule_trajectory_update(self, delay_ms: int = 0) -> None:
        self.trajectory_scheduled_calls.append(delay_ms)

    def _start_trajectory_update(self, request) -> None:
        self.trajectory_start_calls += 1

    def _update_boundary_utility(self) -> None:
        self.boundary_update_calls += 1

    def _update_rashba_surface(self) -> None:
        self.rashba_update_calls += 1


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
                self.assertGreaterEqual(self.app.boundary_update_calls, 1)

    def test_auto_flag_is_shared_with_trajectory_tab(self):
        self.assertIs(self.app.trajectory_auto, self.app.auto)

        self.app.auto.set(False)

        self.assertFalse(self.app.trajectory_auto.get())

    def test_settings_tab_is_present(self):
        tab_texts = [self.app.notebook.tab(tab_id, "text") for tab_id in self.app.notebook.tabs()]

        self.assertIn("Настройки", tab_texts)

    def test_rmin_analysis_tab_is_present(self):
        tab_texts = [self.app.notebook.tab(tab_id, "text") for tab_id in self.app.notebook.tabs()]

        self.assertIn("Анализ r_min", tab_texts)
        self.assertIsNotNone(self.app.rmin_analysis_output)
        self.assertIsNotNone(self.app.rmin_analysis_canvas)

    def test_shell_navigation_uses_icons_and_exposes_settings_tab(self):
        self.assertIn("Настройки", self.app._nav_buttons)

        for tab_text, button in self.app._nav_buttons.items():
            with self.subTest(tab=tab_text):
                self.assertTrue(str(button.cget("image")))
                self.assertEqual(str(button.cget("compound")), "left")

        self.app._select_notebook_tab_by_text("Анализ r_min")

        self.assertEqual(self.app._nav_buttons["Анализ r_min"].cget("style"), "NavActive.TButton")
        self.assertEqual(self.app._toolbar_primary_button.cget("text"), "Построить график")

    def test_shell_toolbar_actions_use_icons(self):
        toolbar_widgets = (
            self.app._toolbar_primary_button,
            self.app._toolbar_copy_button,
            self.app._toolbar_png_button,
            self.app._toolbar_xlsx_button,
            self.app._toolbar_status_label,
        )

        for widget in toolbar_widgets:
            with self.subTest(widget=widget):
                self.assertIsNotNone(widget)
                self.assertTrue(str(widget.cget("image")))
                self.assertEqual(str(widget.cget("compound")), "left")

        self.app.status_text.set("Ошибка тестового расчёта")
        self.app._sync_toolbar_status()

        self.assertEqual(self.app._toolbar_status_icon_name, "status_error")
        self.assertEqual(self.app._toolbar_status_label.cget("text"), "Ошибка")

    def test_rmin_analysis_updates_from_latest_trajectory_payload(self):
        self.app._latest_trajectory_payload = SimpleNamespace(
            frame=pd.DataFrame(
                {
                    "atomic_number": [29.0, 29.0, 29.0],
                    "r_min_ang": [0.10, 0.20, 0.30],
                    "p_flip_initial_up": [0.2, 0.8, 0.3],
                    "p_flip_initial_down": [0.1, 0.6, 0.25],
                    "steps": [100, 120, 110],
                    "converged": [True, True, True],
                    "convergence_unstable": [False, False, False],
                }
            )
        )

        self.app._update_rmin_analysis()

        self.assertIn("Pmax", self.app.rmin_analysis_metrics_text.get())
        self.assertIn("r_TF", self.app.rmin_analysis_output.get("1.0", tk.END))
        self.assertEqual(
            self.app.ax_rmin_analysis_probability.get_title(),
            "P(изменение спина) от минимального расстояния сближения",
        )

    def test_application_icon_assets_are_loaded(self):
        self.assertTrue(APP_ICON_ICO.exists())
        self.assertTrue(APP_ICON_PNG.exists())
        self.assertIsNotNone(self.app._window_icon_image)
        self.assertEqual(int(self.app._window_icon_image.width()), 256)
        self.assertEqual(int(self.app._window_icon_image.height()), 256)

    def test_theme_switch_updates_widgets_and_figures(self):
        self.app.theme_name.set(DARK_THEME)
        self.app.update_idletasks()
        theme = THEMES[DARK_THEME]

        self.assertEqual(self.app.geometry_output.cget("background"), theme.surface)
        self.assertEqual(self.app.geometry_output.cget("foreground"), theme.text)
        self.assertEqual(self.app._style.lookup("TLabel", "foreground"), theme.text)
        self.assertEqual(to_hex(self.app.fig.get_facecolor()), theme.background)
        self.assertIn("Контраст", self.app.theme_status_text.get())

    def test_trajectory_auto_schedules_only_trajectory_recalc(self):
        self.app.trajectory_auto.set(True)
        self.app.trajectory_scheduled_calls.clear()
        left_before = len(self.app.left_scheduled_calls)
        right_before = len(self.app.right_scheduled_calls)

        self.app.trajectory_energy.set(self.app.trajectory_energy.get() + 10.0)

        self.assertEqual(len(self.app.left_scheduled_calls), left_before)
        self.assertEqual(len(self.app.right_scheduled_calls), right_before)
        self.assertEqual(self.app.trajectory_scheduled_calls, [450])

    def test_trajectory_controls_do_not_trigger_main_recalc(self):
        actions = [
            ("trajectory_Z", lambda: self.app.trajectory_Z.set(self.app.trajectory_Z.get() + 1.0)),
            ("trajectory_mass", lambda: self.app.trajectory_mass_amu.set(self.app.trajectory_mass_amu.get() + 0.001)),
            ("trajectory_E", lambda: self.app.trajectory_energy.set(self.app.trajectory_energy.get() + 10.0)),
            ("trajectory_rp", lambda: self.app.trajectory_impact.set(self.app.trajectory_impact.get() + 0.1)),
            ("trajectory_dtheta", lambda: self.app.trajectory_angle_step_deg.set(self.app.trajectory_angle_step_deg.get() + 0.1)),
        ]

        for label, action in actions:
            with self.subTest(control=label):
                left_before = len(self.app.left_scheduled_calls)
                right_before = len(self.app.right_scheduled_calls)
                action()
                self.assertEqual(len(self.app.left_scheduled_calls), left_before)
                self.assertEqual(len(self.app.right_scheduled_calls), right_before)

    def test_rashba_controls_refresh_only_rashba_tab(self):
        left_before = len(self.app.left_scheduled_calls)
        right_before = len(self.app.right_scheduled_calls)
        trajectory_before = len(self.app.trajectory_scheduled_calls)

        self.app.rashba_alpha.set(self.app.rashba_alpha.get() + 0.01)

        self.assertEqual(len(self.app.left_scheduled_calls), left_before)
        self.assertEqual(len(self.app.right_scheduled_calls), right_before)
        self.assertEqual(len(self.app.trajectory_scheduled_calls), trajectory_before)
        self.assertEqual(self.app.rashba_update_calls, 1)

    def test_auto_disabled_blocks_boundary_trajectory_and_rashba_recalc(self):
        self.app.auto.set(False)
        self.app.boundary_update_calls = 0
        self.app.trajectory_scheduled_calls.clear()
        self.app.rashba_update_calls = 0

        self.app.boundary_alpha_deg.set(self.app.boundary_alpha_deg.get() + 1.0)
        self.app.trajectory_energy.set(self.app.trajectory_energy.get() + 10.0)
        self.app.rashba_alpha.set(self.app.rashba_alpha.get() + 0.01)

        self.assertEqual(self.app.boundary_update_calls, 0)
        self.assertEqual(self.app.trajectory_scheduled_calls, [])
        self.assertEqual(self.app.rashba_update_calls, 0)

    def test_rashba_trajectory_source_ignores_non_converged_points(self):
        self.app.rashba_source_label.set(RASHBA_SOURCE_TRAJECTORY)
        self.app._latest_trajectory_payload = SimpleNamespace(
            frame=pd.DataFrame(
                {
                    "energy_eV": [100.0, 200.0, 300.0],
                    "p_flip_initial_up": [0.1, 0.99, 0.3],
                    "p_flip_initial_down": [0.05, 0.98, 0.25],
                    "converged": [True, False, True],
                }
            )
        )

        up, down = self.app._rashba_volume_flip_probabilities(np.asarray([100.0, 200.0, 300.0]))

        np.testing.assert_allclose(up, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(down, [0.05, 0.15, 0.25])

    def test_fixed_angle_step_is_disabled_when_sweeping_angle_step(self):
        control_widgets = self.app._slider_controls["angle_step"]["widgets"]

        self.app.trajectory_sweep_label.set(TRAJECTORY_SWEEP_LABELS[TRAJECTORY_SWEEP_ANGLE_STEP])

        self.assertTrue(all("disabled" in widget.state() for widget in control_widgets))

        self.app.trajectory_sweep_label.set(TRAJECTORY_SWEEP_LABELS[TRAJECTORY_SWEEP_ENERGY])

        self.assertTrue(all("disabled" not in widget.state() for widget in control_widgets))

    def test_trajectory_point_count_is_labeled_as_graph_points(self):
        labels = [getattr(widget, "_slider_label", "") for widget in self.app._slider_value_entries]

        self.assertIn("N точек графика", labels)

    def test_thomas_fermi_b_is_constant_not_user_control(self):
        labels = [getattr(widget, "_slider_label", "") for widget in self.app._slider_value_entries]
        request = self.app._current_trajectory_request()

        self.assertNotIn("b Thomas-Fermi (a0)", labels)
        self.assertFalse(hasattr(request, "b_bohr"))
        self.assertAlmostEqual(DEFAULT_THOMAS_FERMI_B_BOHR, 0.5 * ((3.0 * 3.141592653589793 / 4.0) ** (2.0 / 3.0)))

    def test_phase_c1_c2_are_constants_not_user_controls(self):
        labels = [getattr(widget, "_slider_label", "") for widget in self.app._slider_value_entries]
        request = self.app._phase_grid_request([1.0])

        self.assertNotIn("c1", labels)
        self.assertNotIn("c2", labels)
        self.assertAlmostEqual(DEFAULT_PHASE_C1, 0.00001331)
        self.assertAlmostEqual(DEFAULT_PHASE_C2, 1.2004)
        self.assertAlmostEqual(request.c1, DEFAULT_PHASE_C1)
        self.assertAlmostEqual(request.c2, DEFAULT_PHASE_C2)

    def test_cpu_parallel_defaults_use_available_cores(self):
        self.assertEqual(self.app.trajectory_parallel_workers.get(), cpu_worker_count())
        self.assertEqual(self.app._current_trajectory_request().parallel_workers, cpu_worker_count())
        self.assertEqual(self.app._phase_grid_request([1.0]).parallel_workers, cpu_worker_count())

    def test_slider_value_entry_updates_synced_variable(self):
        entry = next(
            widget for widget in self.app._slider_value_entries
            if getattr(widget, "_slider_label", "") == "Z (заряд ядра)"
        )
        self.app.right_scheduled_calls.clear()

        entry.delete(0, tk.END)
        entry.insert(0, "31,5")
        entry._commit_value()

        self.assertAlmostEqual(self.app.Z.get(), 31.5)
        self.assertEqual(entry.get(), "31.5")
        self.assertEqual(len(self.app.right_scheduled_calls), 1)

        self.app.Z.set(29.0)

        self.assertEqual(entry.get(), "29")

    def test_trajectory_tab_has_scrollable_controls(self):
        self.assertGreaterEqual(len(self.app._scrollable_control_canvases), 1)

    def test_trajectory_controls_have_hints_and_tooltips(self):
        tooltip_texts = [getattr(widget, "_tooltip_text", "") for widget in self.app._tooltip_targets]

        self.assertTrue(any("Вертикальная прокрутка панели параметров" in text for text in tooltip_texts))
        self.assertTrue(any("Заряд ядра атома" in text for text in tooltip_texts))
        self.assertTrue(any("Масса частицы" in text for text in tooltip_texts))
        self.assertTrue(any("Прицельное расстояние r_п" in text for text in tooltip_texts))
        self.assertFalse(any("Параметр удара" in text for text in tooltip_texts))
        self.assertTrue(any("Угловой шаг интегрирования" in text for text in tooltip_texts))

    def test_trajectory_invalid_parameter_is_shown_near_control(self):
        self.app.trajectory_orbital_l.set(1)
        self.app.trajectory_magnetic_m.set(3)

        error_text = self.app._trajectory_error_labels["magnetic_m"].cget("text")

        self.assertIn("-L <= M <= L", error_text)

    def test_trajectory_invalid_parameters_block_calculation_start(self):
        self.app.trajectory_Emax.set(50.0)
        self.app._update_trajectory_utility()

        error_text = self.app._trajectory_error_labels["energy_max"].cget("text")

        self.assertIn("Emax должен быть больше Emin", error_text)
        self.assertEqual(self.app.trajectory_start_calls, 0)

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
