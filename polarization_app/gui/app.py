# -*- coding: utf-8 -*-
import logging
import os
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from polarization_app.application.formulas import (
    FORMULA_BY_LABEL,
    FORMULA_HINTS,
    FORMULA_LABELS,
    FORMULA_LEGACY,
    FormulaResult,
    PhaseGridRequest,
    execute_formula_variant,
)
from polarization_app.application.geometry import GeometryContext, collect_atom_selection
from polarization_app.application.spectrum_export import export_spectrum_bundle
from polarization_app.application.trajectory import (
    TRAJECTORY_AXIS_LABELS,
    TRAJECTORY_SWEEP_BY_LABEL,
    TRAJECTORY_SWEEP_ANGLE_STEP,
    TRAJECTORY_SWEEP_ENERGY,
    TRAJECTORY_SWEEP_IMPACT,
    TRAJECTORY_SWEEP_LABELS,
    TrajectorySweepRequest,
    TrajectorySweepResult,
    execute_trajectory_sweep,
    trajectory_export_metadata,
)
from polarization_app.application.trajectory_export import export_trajectory_bundle
from polarization_app.domain.lattice import LatticeSearchRegion, estimate_lattice_search_region
from polarization_app.domain.transitions import build_transition_matrices
from polarization_app.gui.plotting import (
    build_geometry_preview_data,
    capture_view_limits,
    draw_boundary_utility_plots,
    draw_geometry_preview,
    draw_rashba_surface_plots,
    draw_spin_plots,
    draw_trajectory_sweep_plots,
    restore_view_limits,
    zoom_axis,
    zoom_3d_axis,
    zoom_axis_around_point,
)
from polarization_app.physics.boundary_reflection import compute_boundary_point, compute_boundary_reflection_curves
from polarization_app.physics.phase_integrals import exponential_chi, interpolate_thomas_fermi_chi
from polarization_app.physics.rashba_surface import RashbaSurfaceRequest, RashbaSurfaceResult, compute_rashba_surface
from polarization_app.physics.trajectory_phase import DEFAULT_THOMAS_FERMI_B_BOHR, ELECTRON_MASS_AMU


logger = logging.getLogger(__name__)
CONTROL_PANEL_WIDTH = 360
CONTROL_WRAP_LENGTH = 300
VALIDATION_ERROR_COLOR = "#b00020"
RASHBA_SOURCE_ZERO = "Без объёмного переворота (Ver=0)"
RASHBA_SOURCE_SPECTRUM = "Из вкладки «Спектры и формулы»"
RASHBA_SOURCE_TRAJECTORY = "Из вкладки «Траекторный расчёт»"
RASHBA_SOURCE_LABELS = (
    RASHBA_SOURCE_ZERO,
    RASHBA_SOURCE_SPECTRUM,
    RASHBA_SOURCE_TRAJECTORY,
)


def configure_logging() -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return
    logging.basicConfig(
        filename="log.txt",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class PlotComputationRequest:
    geometry: GeometryContext
    all_atom_count: int
    formula_variant: str
    formula_label: str
    phase_request: PhaseGridRequest
    use_table_chi: bool
    i3_mode_sum: bool


@dataclass(frozen=True)
class PlotComputationResult:
    request: PlotComputationRequest
    result: FormulaResult


class Tooltip:
    def __init__(self, widget, text: str, delay_ms: int = 450) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id: str | None = None
        self._window: tk.Toplevel | None = None

        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self) -> None:
        if self._window is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        window = tk.Toplevel(self.widget)
        window.wm_overrideredirect(True)
        window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            window,
            text=self.text,
            justify="left",
            background="#ffffe8",
            relief="solid",
            borderwidth=1,
            padx=6,
            pady=4,
            wraplength=320,
        )
        label.pack()
        self._window = window

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._window is not None:
            self._window.destroy()
            self._window = None


def run_plot_calculation(request: PlotComputationRequest) -> PlotComputationResult:
    result = execute_formula_variant(
        formula_variant=request.formula_variant,
        orbital_l=request.geometry.orbital_l,
        phase_request=request.phase_request,
    )
    return PlotComputationResult(request=request, result=result)


class App(tk.Tk):
    def __init__(self):
        configure_logging()
        super().__init__()
        self.title("Графики поляризации электрона")
        self.geometry("1180x860")
        self.minsize(980, 720)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._create_variables()
        self._build_layout()
        self._bind_variable_handlers()
        self._update_trajectory_control_states()

        self._recompute_lattice_radius()
        self._update_formula_hint()
        self.update_output_left()
        self.after(0, self.update_output_right)
        self.after(0, self._update_boundary_utility)
        self.after(0, self._update_rashba_surface)
        self._set_text_output(
            self.trajectory_output,
            "[Траекторный расчёт]\nНастройте параметры и нажмите «Рассчитать». "
            "Если нужен пересчёт при движении ползунков, включите автопересчёт слева.\n",
        )
        self._update_trajectory_validation_hints()

    def _create_variables(self) -> None:
        self.a = tk.DoubleVar(value=4.75)
        self.R_bohr = tk.DoubleVar(value=0.53)
        self.alpha_deg = tk.DoubleVar(value=30.0)
        self.beta_deg = tk.DoubleVar(value=60.0)
        self.lattice_radius = tk.IntVar(value=3)
        self.d_layer = tk.IntVar(value=1)
        self.orbital_l = tk.IntVar(value=1)
        self.auto_n = tk.BooleanVar(value=True)

        self.Z = tk.DoubleVar(value=29.0)
        self.b = tk.DoubleVar(value=0.53)
        self.c1 = tk.DoubleVar(value=1.0)
        self.c2 = tk.DoubleVar(value=1.0)
        self.dr = tk.DoubleVar(value=0.01)
        self.rmax = tk.DoubleVar(value=15.0)
        self.Emin = tk.DoubleVar(value=10.0)
        self.Emax = tk.DoubleVar(value=1.0e5)
        self.Npts = tk.IntVar(value=160)
        self.auto = tk.BooleanVar(value=True)
        self.formula_variant_label = tk.StringVar(value=FORMULA_LABELS[FORMULA_LEGACY])
        self.use_table_chi = tk.BooleanVar(value=True)
        self.i3_mode_sum = tk.BooleanVar(value=True)
        self.boundary_alpha_deg = tk.DoubleVar(value=45.0)
        self.boundary_work_function = tk.DoubleVar(value=5.0)
        self.boundary_energy_point = tk.DoubleVar(value=100.0)
        self.boundary_Emin = tk.DoubleVar(value=10.0)
        self.boundary_Emax = tk.DoubleVar(value=500.0)
        self.boundary_Npts = tk.IntVar(value=240)
        self.trajectory_Z = tk.DoubleVar(value=29.0)
        self.trajectory_mass_amu = tk.DoubleVar(value=ELECTRON_MASS_AMU)
        self.trajectory_energy = tk.DoubleVar(value=100.0)
        self.trajectory_Emin = tk.DoubleVar(value=100.0)
        self.trajectory_Emax = tk.DoubleVar(value=1000.0)
        self.trajectory_impact = tk.DoubleVar(value=0.8)
        self.trajectory_impact_min = tk.DoubleVar(value=0.3)
        self.trajectory_impact_max = tk.DoubleVar(value=2.0)
        self.trajectory_r0 = tk.DoubleVar(value=10.0)
        self.trajectory_angle_step_deg = tk.DoubleVar(value=3.0)
        self.trajectory_angle_step_min_deg = tk.DoubleVar(value=0.1)
        self.trajectory_angle_step_max_deg = tk.DoubleVar(value=5.0)
        self.trajectory_b_bohr = tk.DoubleVar(value=DEFAULT_THOMAS_FERMI_B_BOHR)
        self.trajectory_Npts = tk.IntVar(value=25)
        self.trajectory_orbital_l = tk.IntVar(value=1)
        self.trajectory_magnetic_m = tk.IntVar(value=0)
        self.trajectory_random_m = tk.BooleanVar(value=False)
        self.trajectory_sweep_label = tk.StringVar(value=TRAJECTORY_SWEEP_LABELS[TRAJECTORY_SWEEP_ENERGY])
        self.trajectory_auto = tk.BooleanVar(value=False)
        self.trajectory_parallel_workers = tk.IntVar(value=max(1, min(2, os.cpu_count() or 1)))
        self.rashba_Emin = tk.DoubleVar(value=10.0)
        self.rashba_Emax = tk.DoubleVar(value=1000.0)
        self.rashba_Npts = tk.IntVar(value=240)
        self.rashba_layer_thickness = tk.DoubleVar(value=1.0)
        self.rashba_alpha = tk.DoubleVar(value=0.05)
        self.rashba_theta_deg = tk.DoubleVar(value=45.0)
        self.rashba_surface_potential = tk.DoubleVar(value=5.0)
        self.rashba_source_label = tk.StringVar(value=RASHBA_SOURCE_ZERO)
        self.status_text = tk.StringVar(value="Готово.")
        self.formula_hint_text = tk.StringVar(value="")

        self.geometry_output: tk.Text | None = None
        self.spectrum_output: tk.Text | None = None
        self.boundary_output: tk.Text | None = None
        self.trajectory_output: tk.Text | None = None
        self.rashba_output: tk.Text | None = None
        self.output: tk.Text | None = None
        self.n_auto_label: ttk.Label | None = None
        self.ax_sum = None
        self.ax_spin = None
        self.ax_geometry_3d = None
        self.ax_geometry_xz = None
        self.ax_geometry_xy = None
        self.ax_boundary_reflection = None
        self.ax_boundary_angle = None
        self.ax_trajectory_phase = None
        self.ax_trajectory_angle = None
        self.ax_trajectory_diagnostics = None
        self.ax_rashba_transmission = None
        self.ax_rashba_polarization = None
        self.canvas: FigureCanvasTkAgg | None = None
        self.fig: Figure | None = None
        self.geometry_canvas: FigureCanvasTkAgg | None = None
        self.geometry_fig: Figure | None = None
        self.boundary_canvas: FigureCanvasTkAgg | None = None
        self.boundary_fig: Figure | None = None
        self.trajectory_canvas: FigureCanvasTkAgg | None = None
        self.trajectory_fig: Figure | None = None
        self.rashba_canvas: FigureCanvasTkAgg | None = None
        self.rashba_fig: Figure | None = None
        self._default_view_limits = None
        self._boundary_view_limits = None
        self._trajectory_view_limits = None
        self._rashba_view_limits = None
        self._scheduled_left_after: str | None = None
        self._scheduled_right_after: str | None = None
        self._scheduled_trajectory_after: str | None = None
        self._running_future: Future | None = None
        self._running_trajectory_future: Future | None = None
        self._queued_request: PlotComputationRequest | None = None
        self._queued_trajectory_request: TrajectorySweepRequest | None = None
        self._latest_plot_payload: PlotComputationResult | None = None
        self._latest_trajectory_payload: TrajectorySweepResult | None = None
        self._latest_rashba_payload: RashbaSurfaceResult | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="polarization")
        self._trajectory_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trajectory-ui")
        self._scrollable_control_canvases: list[tk.Canvas] = []
        self._active_scroll_canvas: tk.Canvas | None = None
        self._tooltip_targets: list[tk.Widget] = []
        self._slider_value_entries: list[ttk.Entry] = []
        self._slider_controls: dict[str, dict[str, object]] = {}
        self._trajectory_error_labels: dict[str, ttk.Label] = {}
        self._closing = False
        self._geometry_change_in_progress = False

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew")

        geometry_tab = ttk.Frame(notebook)
        spectrum_tab = ttk.Frame(notebook)
        boundary_tab = ttk.Frame(notebook)
        trajectory_tab = ttk.Frame(notebook)
        rashba_tab = ttk.Frame(notebook)
        notebook.add(geometry_tab, text="Геометрия и переходы")
        notebook.add(spectrum_tab, text="Спектры и формулы")
        notebook.add(boundary_tab, text="Граница раздела")
        notebook.add(trajectory_tab, text="Траекторный расчёт")
        notebook.add(rashba_tab, text="Рашба-поверхность")

        self._build_geometry_tab(geometry_tab)
        self._build_spectrum_tab(spectrum_tab)
        self._build_boundary_tab(boundary_tab)
        self._build_trajectory_tab(trajectory_tab)
        self._build_rashba_tab(rashba_tab)
        self.bind_all("<MouseWheel>", self._on_controls_mousewheel)
        self.bind_all("<Button-4>", self._on_controls_mousewheel)
        self.bind_all("<Button-5>", self._on_controls_mousewheel)

        ttk.Label(self, textvariable=self.status_text, anchor="w", padding=(8, 4)).grid(row=1, column=0, sticky="ew")

    def _build_geometry_tab(self, panel) -> None:
        panel.columnconfigure(0, weight=0, minsize=CONTROL_PANEL_WIDTH)
        panel.columnconfigure(1, weight=1)
        panel.rowconfigure(0, weight=1)

        controls = ttk.Frame(panel, padding=(6, 6, 6, 6), width=CONTROL_PANEL_WIDTH)
        controls.grid(row=0, column=0, sticky="nsw")
        controls.columnconfigure(0, weight=1)
        self._build_geometry_section(controls, row=0)

        body = ttk.Panedwindow(panel, orient=tk.VERTICAL)
        body.grid(row=0, column=1, sticky="nsew", padx=(0, 6), pady=6)

        preview_panel = ttk.Frame(body, padding=6)
        output_panel = ttk.Frame(body, padding=6)
        body.add(preview_panel, weight=4)
        body.add(output_panel, weight=2)

        self._build_geometry_preview_area(preview_panel)
        self.geometry_output = self._build_text_output_panel(
            output_panel,
            title="Атомы в рабочей области и матрицы перехода",
        )
        self.output = self.geometry_output

    def _build_spectrum_tab(self, panel) -> None:
        panel.columnconfigure(0, weight=0, minsize=CONTROL_PANEL_WIDTH)
        panel.columnconfigure(1, weight=1)
        panel.rowconfigure(0, weight=1)

        controls = ttk.Frame(panel, padding=(6, 6, 6, 6), width=CONTROL_PANEL_WIDTH)
        controls.grid(row=0, column=0, sticky="nsw")
        controls.columnconfigure(0, weight=1)
        self._build_interaction_section(controls, row=0)
        self._build_calculation_section(controls, row=1)

        body = ttk.Panedwindow(panel, orient=tk.VERTICAL)
        body.grid(row=0, column=1, sticky="nsew", padx=(0, 6), pady=6)

        plot_panel = ttk.Frame(body, padding=6)
        output_panel = ttk.Frame(body, padding=6)
        body.add(plot_panel, weight=4)
        body.add(output_panel, weight=1)

        self._build_spin_plot_area(plot_panel)
        self.spectrum_output = self._build_text_output_panel(
            output_panel,
            title="Сводка расчёта спектра",
            height=8,
        )

    def _build_boundary_tab(self, panel) -> None:
        panel.columnconfigure(0, weight=0, minsize=CONTROL_PANEL_WIDTH)
        panel.columnconfigure(1, weight=1)
        panel.rowconfigure(0, weight=1)

        controls = ttk.Frame(panel, padding=(6, 6, 6, 6), width=CONTROL_PANEL_WIDTH)
        controls.grid(row=0, column=0, sticky="nsw")
        controls.columnconfigure(0, weight=1)
        self._build_boundary_section(controls, row=0)

        body = ttk.Panedwindow(panel, orient=tk.VERTICAL)
        body.grid(row=0, column=1, sticky="nsew", padx=(0, 6), pady=6)

        plot_panel = ttk.Frame(body, padding=6)
        output_panel = ttk.Frame(body, padding=6)
        body.add(plot_panel, weight=4)
        body.add(output_panel, weight=2)

        self._build_boundary_plot_area(plot_panel)
        self.boundary_output = self._build_text_output_panel(
            output_panel,
            title="Сводка по выбранной энергии",
            height=10,
        )

    def _build_trajectory_tab(self, panel) -> None:
        panel.columnconfigure(0, weight=0, minsize=CONTROL_PANEL_WIDTH)
        panel.columnconfigure(1, weight=1)
        panel.rowconfigure(0, weight=1)

        controls = self._build_scrollable_control_panel(panel)
        self._build_trajectory_section(controls, row=0)

        body = ttk.Panedwindow(panel, orient=tk.VERTICAL)
        body.grid(row=0, column=1, sticky="nsew", padx=(0, 6), pady=6)

        plot_panel = ttk.Frame(body, padding=6)
        output_panel = ttk.Frame(body, padding=6)
        body.add(plot_panel, weight=4)
        body.add(output_panel, weight=2)

        self._build_trajectory_plot_area(plot_panel)
        self.trajectory_output = self._build_text_output_panel(
            output_panel,
            title="Сводка траекторного расчёта",
            height=12,
        )

    def _build_rashba_tab(self, panel) -> None:
        panel.columnconfigure(0, weight=0, minsize=CONTROL_PANEL_WIDTH)
        panel.columnconfigure(1, weight=1)
        panel.rowconfigure(0, weight=1)

        controls = self._build_scrollable_control_panel(panel, tooltip_text="Вертикальная прокрутка панели параметров Рашбы.")
        self._build_rashba_section(controls, row=0)

        body = ttk.Panedwindow(panel, orient=tk.VERTICAL)
        body.grid(row=0, column=1, sticky="nsew", padx=(0, 6), pady=6)

        plot_panel = ttk.Frame(body, padding=6)
        output_panel = ttk.Frame(body, padding=6)
        body.add(plot_panel, weight=4)
        body.add(output_panel, weight=2)

        self._build_rashba_plot_area(plot_panel)
        self.rashba_output = self._build_text_output_panel(
            output_panel,
            title="Сводка прохождения через поверхность",
            height=10,
        )

    def _build_geometry_section(self, parent, row: int) -> None:
        section = ttk.LabelFrame(parent, text="Геометрия кристалла и матрицы перехода", padding=10)
        section.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        section.columnconfigure(0, weight=1)

        current_row = 0
        self._make_slider(section, "Постоянная решётки a (Å)", self.a, 1, 10, current_row, description="Расстояние между узлами решётки"); current_row += 1
        self._make_slider(section, "Радиус Бора R_bohr (Å)", self.R_bohr, 0.1, 2.0, current_row, description="Радиус взаимодействия (×5 для поиска атомов)"); current_row += 1
        self._make_slider(section, "Полярный угол вылета α (°)", self.alpha_deg, -90, 90, current_row, description="Угол вылета относительно внешней нормали к поверхности"); current_row += 1
        self._make_slider(section, "Азимутальный угол β (°)", self.beta_deg, -90, 90, current_row, description="Разворот направления вылета вокруг нормали к поверхности"); current_row += 1
        self._make_slider(section, "Ручной размер области n", self.lattice_radius, 1, 20, current_row, description="Используется как половина куба поиска, если автообласть выключена", resolution=1); current_row += 1
        self._make_slider(
            section,
            "Глубина источника d",
            self.d_layer,
            1,
            21,
            current_row,
            description="Номер слоя от поверхности: d=1 соответствует поверхностному слою z=0",
            resolution=1,
        ); current_row += 1
        self._make_slider(section, "Орбитальное число L", self.orbital_l, 1, 10, current_row, description="L задаёт фазовую матрицу, Lz выбирает D", resolution=1); current_row += 1

        ttk.Checkbutton(section, text="Автоподбор n по углам и глубине", variable=self.auto_n).grid(
            row=current_row, column=0, sticky="w", pady=(4, 0)
        )
        current_row += 1

        self.n_auto_label = ttk.Label(section, text="")
        self.n_auto_label.grid(row=current_row, column=0, sticky="w", pady=(2, 0))
        current_row += 1

        ttk.Button(section, text="Обновить атомы и матрицы", command=self.update_output_left).grid(
            row=current_row, column=0, sticky="w", pady=(6, 0)
        )

    def _build_interaction_section(self, parent, row: int) -> None:
        section = ttk.LabelFrame(parent, text="Параметры взаимодействия и интегрирования", padding=10)
        section.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        section.columnconfigure(0, weight=1)

        current_row = 0
        self._make_slider(section, "Z (заряд ядра)", self.Z, 1, 92, current_row); current_row += 1
        self._make_slider(section, "b (Å)", self.b, 0.1, 2.0, current_row); current_row += 1
        self._make_slider(section, "c1", self.c1, 0.1, 3.0, current_row); current_row += 1
        self._make_slider(section, "c2", self.c2, 0.1, 3.0, current_row); current_row += 1
        self._make_slider(section, "dr (Å)", self.dr, 0.001, 0.1, current_row); current_row += 1
        self._make_slider(section, "r_max (Å)", self.rmax, 5.0, 40.0, current_row); current_row += 1

    def _build_calculation_section(self, parent, row: int) -> None:
        section = ttk.LabelFrame(parent, text="Расчёт фаз и графиков", padding=10)
        section.grid(row=row, column=0, sticky="ew")
        section.columnconfigure(1, weight=1)

        current_row = 0
        self._make_slider(section, "Emin (эВ)", self.Emin, 1.0, 1000.0, current_row); current_row += 1
        self._make_slider(section, "Emax (эВ)", self.Emax, 1000.0, 200000.0, current_row); current_row += 1
        self._make_slider(section, "N точек", self.Npts, 20, 600, current_row, resolution=1); current_row += 1

        ttk.Label(section, text="Расчётная схема").grid(row=current_row, column=0, sticky="w", pady=(6, 0))
        formula_box = ttk.Combobox(
            section,
            textvariable=self.formula_variant_label,
            values=list(FORMULA_LABELS.values()),
            state="readonly",
            width=30,
        )
        formula_box.grid(row=current_row, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        current_row += 1

        ttk.Label(section, textvariable=self.formula_hint_text, foreground="#555", wraplength=CONTROL_WRAP_LENGTH, justify="left").grid(
            row=current_row, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        current_row += 1

        ttk.Checkbutton(section, text="Автопересчёт после изменения параметров", variable=self.auto).grid(
            row=current_row, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        current_row += 1

        ttk.Checkbutton(
            section,
            text="χ(x): табличная интерполяция Thomas-Fermi",
            variable=self.use_table_chi,
        ).grid(row=current_row, column=0, columnspan=2, sticky="w", pady=(2, 0))
        current_row += 1
        ttk.Checkbutton(
            section,
            text="I3 как усреднённая сумма с шагом dr",
            variable=self.i3_mode_sum,
        ).grid(row=current_row, column=0, columnspan=2, sticky="w", pady=(2, 0))
        current_row += 1

        actions = ttk.Frame(section)
        actions.grid(row=current_row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(actions, text="Построить графики", command=self.update_output_right).pack(side="left")
        ttk.Button(actions, text="Экспорт JSON/XML/XLSX", command=self._export_spectrum_data).pack(side="left", padx=(8, 0))
        ttk.Label(
            actions,
            text="Колесо мыши масштабирует график.",
            foreground="#555",
            wraplength=150,
            justify="left",
        ).pack(side="left", padx=(10, 0))

    def _build_boundary_section(self, parent, row: int) -> None:
        section = ttk.LabelFrame(parent, text="Мини-утилита отражения от границы", padding=10)
        section.grid(row=row, column=0, sticky="ew")
        section.columnconfigure(0, weight=1)

        current_row = 0
        ttk.Label(
            section,
            text=(
                "Утилита не влияет на основной расчёт поляризации. "
                "Здесь α означает угол падения к нормали границы, а β - угол после прохождения через границу."
            ),
            foreground="#555",
            wraplength=CONTROL_WRAP_LENGTH,
            justify="left",
        ).grid(row=current_row, column=0, sticky="w", pady=(0, 6))
        current_row += 1

        self._make_slider(
            section,
            "Угол падения α (°)",
            self.boundary_alpha_deg,
            0.0,
            89.0,
            current_row,
            description="Задаётся относительно нормали к границе раздела",
        ); current_row += 1
        self._make_slider(
            section,
            "Работа выхода A (эВ)",
            self.boundary_work_function,
            0.0,
            15.0,
            current_row,
            description="Скачок потенциальной энергии на границе",
        ); current_row += 1
        self._make_slider(
            section,
            "Выбранная энергия E (эВ)",
            self.boundary_energy_point,
            1.0,
            1000.0,
            current_row,
            description="Для этой точки отдельно выводятся β, k'/k и R",
        ); current_row += 1
        self._make_slider(
            section,
            "Emin для графика (эВ)",
            self.boundary_Emin,
            1.0,
            1000.0,
            current_row,
        ); current_row += 1
        self._make_slider(
            section,
            "Emax для графика (эВ)",
            self.boundary_Emax,
            10.0,
            2000.0,
            current_row,
        ); current_row += 1
        self._make_slider(
            section,
            "N точек графика",
            self.boundary_Npts,
            40,
            800,
            current_row,
            resolution=1,
        ); current_row += 1

        ttk.Button(section, text="Обновить утилиту", command=self._update_boundary_utility).grid(
            row=current_row, column=0, sticky="w", pady=(8, 0)
        )

    def _build_trajectory_section(self, parent, row: int) -> None:
        section = ttk.LabelFrame(parent, text="Траекторный расчёт одного атома", padding=10)
        section.grid(row=row, column=0, sticky="ew")
        section.columnconfigure(1, weight=1)

        current_row = 0
        ttk.Label(
            section,
            text=(
                "Расчёт из алгоритма 2.0: r_min, θ, φ, фаза ϕ и вероятности после СОВ. "
                "Текущие спектры при этом не изменяются."
            ),
            foreground="#555",
            wraplength=CONTROL_WRAP_LENGTH,
            justify="left",
        ).grid(row=current_row, column=0, columnspan=2, sticky="w", pady=(0, 6))
        current_row += 1

        sweep_label = ttk.Label(section, text="Что меняем")
        sweep_label.grid(row=current_row, column=0, sticky="w", pady=(6, 0))
        self._attach_tooltip(
            sweep_label,
            "Выбирает параметр для оси X. Остальные параметры берутся из фиксированных ползунков.",
        )
        sweep_box = ttk.Combobox(
            section,
            textvariable=self.trajectory_sweep_label,
            values=list(TRAJECTORY_SWEEP_LABELS.values()),
            state="readonly",
            width=30,
        )
        sweep_box.grid(row=current_row, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        self._attach_tooltip(
            sweep_box,
            "Выбирает параметр для оси X. Остальные параметры берутся из фиксированных ползунков.",
        )
        current_row += 1

        actions = ttk.Frame(section)
        actions.grid(row=current_row, column=0, columnspan=2, sticky="ew", pady=(8, 6))
        ttk.Button(actions, text="Рассчитать", command=self._update_trajectory_utility).pack(side="left")
        ttk.Button(actions, text="Экспорт JSON/XML/XLSX", command=self._export_trajectory_data).pack(side="left", padx=(8, 0))
        current_row += 1

        auto_check = ttk.Checkbutton(
            section,
            text="Автопересчёт после изменения параметров",
            variable=self.trajectory_auto,
        )
        auto_check.grid(row=current_row, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self._attach_tooltip(auto_check, "Если включено, расчёт запускается автоматически после изменения параметров.")
        current_row += 1

        self._make_slider(
            section,
            "Потоки расчёта",
            self.trajectory_parallel_workers,
            1,
            max(1, min(8, os.cpu_count() or 1)),
            current_row,
            description="Сколько потоков использовать для параллельного расчёта точек диапазона",
            resolution=1,
            validation_key="parallel_workers",
        ); current_row += 1

        ttk.Separator(section, orient="horizontal").grid(row=current_row, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        current_row += 1

        self._make_slider(
            section,
            "Z",
            self.trajectory_Z,
            1,
            92,
            current_row,
            description="Заряд ядра атома в потенциале Томаса-Ферми",
            validation_key="atomic_number",
        ); current_row += 1
        self._make_slider(
            section,
            "Масса (а.е.м)",
            self.trajectory_mass_amu,
            0.0001,
            5.0,
            current_row,
            description="Масса частицы в атомных единицах массы; электрон по умолчанию 0.000549 а.е.м",
            resolution=0.0001,
            validation_key="mass_amu",
        ); current_row += 1
        self._make_slider(
            section,
            "E фикс. (эВ)",
            self.trajectory_energy,
            1.0,
            5000.0,
            current_row,
            description="Энергия одной расчётной точки; используется, когда ось X не энергия",
            validation_key="energy",
        ); current_row += 1
        self._make_slider(
            section,
            "Emin (эВ)",
            self.trajectory_Emin,
            1.0,
            5000.0,
            current_row,
            description="Нижняя граница диапазона энергии при sweep по E",
            validation_key="energy_min",
        ); current_row += 1
        self._make_slider(
            section,
            "Emax (эВ)",
            self.trajectory_Emax,
            10.0,
            10000.0,
            current_row,
            description="Верхняя граница диапазона энергии при sweep по E",
            validation_key="energy_max",
        ); current_row += 1
        self._make_slider(
            section,
            "r_п фикс. (Å)",
            self.trajectory_impact,
            0.05,
            5.0,
            current_row,
            description="Параметр удара r_п; используется, когда ось X не r_п",
            validation_key="impact_fixed",
        ); current_row += 1
        self._make_slider(
            section,
            "r_п min (Å)",
            self.trajectory_impact_min,
            0.05,
            5.0,
            current_row,
            description="Нижняя граница параметра удара для sweep по r_п; при max_steps поднимите до 0.25-0.3 Å",
            validation_key="impact_min",
        ); current_row += 1
        self._make_slider(
            section,
            "r_п max (Å)",
            self.trajectory_impact_max,
            0.1,
            8.0,
            current_row,
            description="Верхняя граница параметра удара для sweep по r_п",
            validation_key="impact_max",
        ); current_row += 1
        self._make_slider(
            section,
            "r0 (Å)",
            self.trajectory_r0,
            1.0,
            40.0,
            current_row,
            description="Начальное расстояние интегрирования; должно быть больше r_п и r_min",
            validation_key="r0",
        ); current_row += 1
        self._make_slider(
            section,
            "dθ фикс. (°)",
            self.trajectory_angle_step_deg,
            0.1,
            5.0,
            current_row,
            description="Угловой шаг интегрирования; если видите max_steps, увеличьте этот ползунок до 2-5°",
            validation_key="angle_step",
        ); current_row += 1
        self._make_slider(
            section,
            "dθ min (°)",
            self.trajectory_angle_step_min_deg,
            0.1,
            5.0,
            current_row,
            description="Нижняя граница шага dθ для sweep по точности интегрирования",
            validation_key="angle_step_min",
        ); current_row += 1
        self._make_slider(
            section,
            "dθ max (°)",
            self.trajectory_angle_step_max_deg,
            0.1,
            5.0,
            current_row,
            description="Верхняя граница шага dθ для sweep по точности интегрирования",
            validation_key="angle_step_max",
        ); current_row += 1
        self._make_slider(
            section,
            "b Thomas-Fermi (a0)",
            self.trajectory_b_bohr,
            0.2,
            2.0,
            current_row,
            description="Масштаб экранирования b в потенциале U(r), в атомных радиусах a0",
            validation_key="b_bohr",
        ); current_row += 1
        self._make_slider(
            section,
            "N точек графика",
            self.trajectory_Npts,
            1,
            300,
            current_row,
            description="Количество расчётных точек на выбранной оси X",
            resolution=1,
            validation_key="point_count",
        ); current_row += 1
        self._make_slider(
            section,
            "L",
            self.trajectory_orbital_l,
            0,
            10,
            current_row,
            description="Орбитальное квантовое число для матрицы переходов",
            resolution=1,
            validation_key="orbital_l",
        ); current_row += 1
        self._make_slider(
            section,
            "M",
            self.trajectory_magnetic_m,
            -10,
            10,
            current_row,
            description="Магнитное квантовое число; физически должно лежать в пределах -L..L",
            resolution=1,
            validation_key="magnetic_m",
        ); current_row += 1

        random_m_check = ttk.Checkbutton(
            section,
            text="Случайный M для каждой точки",
            variable=self.trajectory_random_m,
        )
        random_m_check.grid(row=current_row, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self._attach_tooltip(
            random_m_check,
            "Если включено, для каждой точки M выбирается случайно из допустимого диапазона.",
        )
        current_row += 1

    def _build_rashba_section(self, parent, row: int) -> None:
        section = ttk.LabelFrame(parent, text="Прохождение через поверхность с Рашбой", padding=10)
        section.grid(row=row, column=0, sticky="ew")
        section.columnconfigure(1, weight=1)

        current_row = 0
        ttk.Label(
            section,
            text=(
                "Расчёт по RASBA_ALG: kx, ky, k'_y, R, T и итоговая поляризация после поверхности. "
                "В базовом режиме Ver(+→-) и Ver(-→+) равны нулю."
            ),
            foreground="#555",
            wraplength=CONTROL_WRAP_LENGTH,
            justify="left",
        ).grid(row=current_row, column=0, columnspan=2, sticky="w", pady=(0, 6))
        current_row += 1

        source_label = ttk.Label(section, text="Источник Ver")
        source_label.grid(row=current_row, column=0, sticky="w", pady=(6, 0))
        self._attach_tooltip(
            source_label,
            "Откуда брать вероятности переворота спина до поверхности. Для текущего случая оставьте Ver=0.",
        )
        source_box = ttk.Combobox(
            section,
            textvariable=self.rashba_source_label,
            values=list(RASHBA_SOURCE_LABELS),
            state="readonly",
            width=30,
        )
        source_box.grid(row=current_row, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        self._attach_tooltip(
            source_box,
            "Ver(+→-) и Ver(-→+) можно взять из уже рассчитанного спектра или траекторной вкладки.",
        )
        current_row += 1

        ttk.Button(section, text="Рассчитать", command=self._update_rashba_surface).grid(
            row=current_row, column=0, sticky="w", pady=(8, 6)
        )
        current_row += 1

        self._make_slider(
            section,
            "Emin (эВ)",
            self.rashba_Emin,
            1.0,
            1000.0,
            current_row,
            description="Нижняя граница энергетического диапазона на оси X",
        ); current_row += 1
        self._make_slider(
            section,
            "Emax (эВ)",
            self.rashba_Emax,
            10.0,
            10000.0,
            current_row,
            description="Верхняя граница энергетического диапазона на оси X",
        ); current_row += 1
        self._make_slider(
            section,
            "N точек",
            self.rashba_Npts,
            20,
            1000,
            current_row,
            description="Количество точек энергетической сетки",
            resolution=1,
        ); current_row += 1
        self._make_slider(
            section,
            "d слоя (Å)",
            self.rashba_layer_thickness,
            0.1,
            10.0,
            current_row,
            description="Толщина слоя Рашбы; в расчёте переводится из Å в атомные единицы длины",
        ); current_row += 1
        self._make_slider(
            section,
            "α Рашбы (а.е.)",
            self.rashba_alpha,
            0.0,
            1.0,
            current_row,
            description="Коэффициент Рашбы в атомных единицах; α=0 отключает спиновое расщепление на поверхности",
            resolution=0.001,
        ); current_row += 1
        self._make_slider(
            section,
            "θ вылета (°)",
            self.rashba_theta_deg,
            0.0,
            89.0,
            current_row,
            description="Угол вылета относительно нормали к поверхности: 0° вдоль нормали, 90° вдоль поверхности",
        ); current_row += 1
        self._make_slider(
            section,
            "U поверхности (эВ)",
            self.rashba_surface_potential,
            0.0,
            20.0,
            current_row,
            description="Поверхностный потенциальный барьер; при E <= U прохождение обнуляется",
        ); current_row += 1

    def _build_spin_plot_area(self, panel) -> None:
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        self._build_zoom_toolbar(panel, row=0)

        self.fig = Figure(figsize=(7.4, 6.0), dpi=100)
        self.ax_sum = self.fig.add_subplot(211)
        self.ax_spin = self.fig.add_subplot(212)
        for axis in (self.ax_sum, self.ax_spin):
            axis.grid(True, which="both")

        self.canvas = FigureCanvasTkAgg(self.fig, master=panel)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self.canvas.mpl_connect("scroll_event", self._on_plot_scroll)

    def _build_boundary_plot_area(self, panel) -> None:
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        zoom_bar = ttk.Frame(panel)
        zoom_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(zoom_bar, text="Колесо мыши масштабирует графики утилиты относительно курсора.").pack(side="left")
        ttk.Button(zoom_bar, text="Сбросить масштаб", command=self._reset_boundary_zoom).pack(side="right")

        self.boundary_fig = Figure(figsize=(7.4, 6.0), dpi=100)
        self.ax_boundary_reflection = self.boundary_fig.add_subplot(211)
        self.ax_boundary_angle = self.boundary_fig.add_subplot(212)
        for axis in (self.ax_boundary_reflection, self.ax_boundary_angle):
            axis.grid(True, which="both")

        self.boundary_canvas = FigureCanvasTkAgg(self.boundary_fig, master=panel)
        self.boundary_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self.boundary_canvas.mpl_connect("scroll_event", self._on_boundary_plot_scroll)

    def _build_trajectory_plot_area(self, panel) -> None:
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        zoom_bar = ttk.Frame(panel)
        zoom_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(zoom_bar, text="Колесо мыши масштабирует графики траекторного расчёта.").pack(side="left")
        ttk.Button(zoom_bar, text="Сбросить масштаб", command=self._reset_trajectory_zoom).pack(side="right")

        self.trajectory_fig = Figure(figsize=(7.4, 6.4), dpi=100)
        self.ax_trajectory_phase = self.trajectory_fig.add_subplot(311)
        self.ax_trajectory_angle = self.trajectory_fig.add_subplot(312)
        self.ax_trajectory_diagnostics = self.trajectory_fig.add_subplot(313)
        for axis in (self.ax_trajectory_phase, self.ax_trajectory_angle, self.ax_trajectory_diagnostics):
            axis.grid(True, which="both")

        self.trajectory_canvas = FigureCanvasTkAgg(self.trajectory_fig, master=panel)
        self.trajectory_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self.trajectory_canvas.mpl_connect("scroll_event", self._on_trajectory_plot_scroll)

    def _build_rashba_plot_area(self, panel) -> None:
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        zoom_bar = ttk.Frame(panel)
        zoom_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(zoom_bar, text="Колесо мыши масштабирует графики прохождения через поверхность.").pack(side="left")
        ttk.Button(zoom_bar, text="Сбросить масштаб", command=self._reset_rashba_zoom).pack(side="right")

        self.rashba_fig = Figure(figsize=(7.4, 6.0), dpi=100)
        self.ax_rashba_transmission = self.rashba_fig.add_subplot(211)
        self.ax_rashba_polarization = self.rashba_fig.add_subplot(212)
        for axis in (self.ax_rashba_transmission, self.ax_rashba_polarization):
            axis.grid(True, which="both")

        self.rashba_canvas = FigureCanvasTkAgg(self.rashba_fig, master=panel)
        self.rashba_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self.rashba_canvas.mpl_connect("scroll_event", self._on_rashba_plot_scroll)

    def _build_geometry_preview_area(self, panel) -> None:
        panel.columnconfigure(0, weight=1)
        panel.columnconfigure(1, weight=0)
        panel.rowconfigure(1, weight=1)

        ttk.Label(
            panel,
            text=(
                "Схема показывает рабочую область решётки, поверхность z=0, траекторию электрона и атомы,"
                " которые реально попадают в выбор и расчёт. В авто-режиме область может стать параллелепипедом."
            ),
            foreground="#555",
            wraplength=760,
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        self.geometry_fig = Figure(figsize=(7.6, 5.8), dpi=100)
        grid = self.geometry_fig.add_gridspec(2, 2, width_ratios=[1.75, 1.0], height_ratios=[1.0, 1.0])
        self.ax_geometry_3d = self.geometry_fig.add_subplot(grid[:, 0], projection="3d")
        self.ax_geometry_xz = self.geometry_fig.add_subplot(grid[0, 1])
        self.ax_geometry_xy = self.geometry_fig.add_subplot(grid[1, 1])

        self.geometry_canvas = FigureCanvasTkAgg(self.geometry_fig, master=panel)
        self.geometry_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self.geometry_canvas.mpl_connect("scroll_event", self._on_geometry_scroll)

        legend_panel = ttk.LabelFrame(panel, text="Обозначения", padding=10)
        legend_panel.grid(row=1, column=1, sticky="ns", padx=(10, 0))
        self._build_geometry_legend(legend_panel)

    def _build_text_output_panel(self, panel, *, title: str, height: int = 14) -> tk.Text:
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)
        ttk.Label(panel, text=title).grid(row=0, column=0, sticky="w")

        text_frame = ttk.Frame(panel)
        text_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        output = tk.Text(text_frame, font=("Consolas", 10), wrap="word", height=height)
        output.grid(row=0, column=0, sticky="nsew")
        text_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=output.yview)
        text_scrollbar.grid(row=0, column=1, sticky="ns")
        output.configure(yscrollcommand=text_scrollbar.set)
        return output

    def _build_scrollable_control_panel(self, parent, *, tooltip_text: str = "Вертикальная прокрутка панели параметров траекторного расчёта.") -> ttk.Frame:
        container = ttk.Frame(parent, padding=(6, 6, 6, 6), width=CONTROL_PANEL_WIDTH)
        container.grid(row=0, column=0, sticky="nsw")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        canvas = tk.Canvas(container, width=CONTROL_PANEL_WIDTH - 20, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)
        self._attach_tooltip(scrollbar, tooltip_text)

        content = ttk.Frame(canvas)
        content.columnconfigure(0, weight=1)
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")

        def update_scroll_region(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def update_content_width(event):
            canvas.itemconfigure(window_id, width=event.width)

        content.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", update_content_width)
        canvas.bind("<Enter>", lambda _event: self._active_scroll_canvas_set(canvas))
        canvas.bind("<Leave>", lambda _event: self._active_scroll_canvas_set(None))
        content.bind("<Enter>", lambda _event: self._active_scroll_canvas_set(canvas))
        content.bind("<Leave>", lambda _event: self._active_scroll_canvas_set(None))

        self._scrollable_control_canvases.append(canvas)
        return content

    def _build_zoom_toolbar(self, panel, row: int) -> None:
        zoom_bar = ttk.Frame(panel)
        zoom_bar.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(zoom_bar, text="Колесо мыши масштабирует график относительно курсора.").pack(side="left")
        ttk.Button(zoom_bar, text="Сбросить масштаб", command=self._reset_zoom).pack(side="right")

    def _build_geometry_legend(self, panel) -> None:
        items = [
            ("#b8b8b8", "Узлы решётки внутри рабочей области"),
            ("#7a7a7a", "Поверхность кристалла z=0"),
            ("#3a6ee8", "Слой на глубине d от поверхности"),
            ("#f28e2b", "Атомы в зоне взаимодействия"),
            ("#cf2f2f", "Атомы, попавшие в расчёт"),
            ("#0f7c2b", "Старт и траектория электрона"),
        ]
        for row, (color, text) in enumerate(items):
            swatch = tk.Label(panel, background=color, width=2, relief="solid", borderwidth=1)
            swatch.grid(row=row, column=0, sticky="nw", padx=(0, 8), pady=2)
            ttk.Label(panel, text=text, wraplength=210, justify="left").grid(row=row, column=1, sticky="w", pady=2)

        ttk.Separator(panel, orient="horizontal").grid(row=len(items), column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(
            panel,
            text=(
                "В авто-режиме область становится прямоугольной: отдельно по x, y и слоям z."
                " Глубина d считается от поверхности, начиная с 1."
                " Вне этого параллелепипеда атомы не учитываются."
            ),
            foreground="#555",
            wraplength=230,
            justify="left",
        ).grid(row=len(items) + 1, column=0, columnspan=2, sticky="w")

    def _bind_variable_handlers(self) -> None:
        for variable in (self.a, self.R_bohr, self.alpha_deg, self.beta_deg, self.lattice_radius, self.d_layer):
            variable.trace_add("write", lambda *_: self._on_geometry_inputs_changed())

        for variable in (self.Z, self.b, self.c1, self.c2, self.dr, self.rmax, self.Emin, self.Emax, self.Npts):
            variable.trace_add("write", lambda *_: self._on_calculation_inputs_changed())

        self.auto_n.trace_add("write", lambda *_: self._on_geometry_inputs_changed())
        self.auto.trace_add("write", lambda *_: self._on_auto_toggle())
        self.use_table_chi.trace_add("write", lambda *_: self._on_calculation_inputs_changed())
        self.i3_mode_sum.trace_add("write", lambda *_: self._on_calculation_inputs_changed())
        self.formula_variant_label.trace_add("write", lambda *_: self._on_formula_variant_changed())
        self.orbital_l.trace_add("write", lambda *_: self._on_orbital_l_changed())

        for variable in (
            self.boundary_alpha_deg,
            self.boundary_work_function,
            self.boundary_energy_point,
            self.boundary_Emin,
            self.boundary_Emax,
            self.boundary_Npts,
        ):
            variable.trace_add("write", lambda *_: self._update_boundary_utility())

        for variable in (
            self.trajectory_Z,
            self.trajectory_mass_amu,
            self.trajectory_energy,
            self.trajectory_Emin,
            self.trajectory_Emax,
            self.trajectory_impact,
            self.trajectory_impact_min,
            self.trajectory_impact_max,
            self.trajectory_r0,
            self.trajectory_angle_step_deg,
            self.trajectory_angle_step_min_deg,
            self.trajectory_angle_step_max_deg,
            self.trajectory_b_bohr,
            self.trajectory_Npts,
            self.trajectory_orbital_l,
            self.trajectory_magnetic_m,
            self.trajectory_random_m,
            self.trajectory_sweep_label,
            self.trajectory_parallel_workers,
        ):
            variable.trace_add("write", lambda *_: self._on_trajectory_inputs_changed())

        self.trajectory_auto.trace_add("write", lambda *_: self._on_trajectory_auto_toggle())

        for variable in (
            self.rashba_Emin,
            self.rashba_Emax,
            self.rashba_Npts,
            self.rashba_layer_thickness,
            self.rashba_alpha,
            self.rashba_theta_deg,
            self.rashba_surface_potential,
            self.rashba_source_label,
        ):
            variable.trace_add("write", lambda *_: self._update_rashba_surface())

    def _attach_tooltip(self, widget, text: str):
        if text:
            setattr(widget, "_tooltip_text", text)
            setattr(widget, "_tooltip", Tooltip(widget, text))
            self._tooltip_targets.append(widget)
        return widget

    def _make_slider(
        self,
        parent,
        label,
        variable,
        min_value,
        max_value,
        row,
        description="",
        resolution=0.01,
        validation_key: str | None = None,
    ):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        frame.columnconfigure(0, weight=1)

        label_frame = ttk.Frame(frame)
        label_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        label_widget = ttk.Label(label_frame, text=label)
        label_widget.grid(row=0, column=0, sticky="w")
        self._attach_tooltip(label_widget, description)
        control_widgets = [label_widget]
        if description:
            hint_label = ttk.Label(
                label_frame,
                text=f"({description})",
                foreground="#555",
                font=("TkDefaultFont", 8),
                wraplength=CONTROL_WRAP_LENGTH,
                justify="left",
            )
            hint_label.grid(row=1, column=0, sticky="w")
            self._attach_tooltip(hint_label, description)
            control_widgets.append(hint_label)

        slider = ttk.Scale(frame, from_=min_value, to=max_value, orient="horizontal", variable=variable)
        slider.grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=(2, 0))
        self._attach_tooltip(slider, description)
        control_widgets.append(slider)

        def format_value(value):
            try:
                if isinstance(variable, tk.IntVar) or resolution >= 1:
                    return f"{int(round(float(value)))}"
                return f"{float(value):.3g}"
            except Exception:
                return str(value)

        value_text = tk.StringVar(value=format_value(variable.get()))
        value_entry = ttk.Entry(frame, textvariable=value_text, width=9, justify="right")
        value_entry.grid(row=1, column=1, sticky="e", pady=(2, 0))
        self._attach_tooltip(value_entry, description)
        setattr(value_entry, "_slider_label", label)
        setattr(value_entry, "_slider_variable", variable)
        self._slider_value_entries.append(value_entry)
        control_widgets.append(value_entry)
        last_valid_text = value_text.get()

        def sync_entry_from_variable(*_):
            nonlocal last_valid_text
            if value_entry.focus_get() is value_entry:
                return
            last_valid_text = format_value(variable.get())
            value_text.set(last_valid_text)

        def commit_entry_value(_event=None):
            nonlocal last_valid_text
            raw_value = value_text.get().strip().replace(",", ".")
            try:
                parsed_value = float(raw_value)
            except ValueError:
                value_text.set(last_valid_text)
                return "break"

            if not np.isfinite(parsed_value):
                value_text.set(last_valid_text)
                return "break"

            parsed_value = min(max(parsed_value, float(min_value)), float(max_value))
            if isinstance(variable, tk.IntVar) or resolution >= 1:
                variable.set(int(round(parsed_value)))
            else:
                variable.set(parsed_value)
            last_valid_text = format_value(variable.get())
            value_text.set(last_valid_text)
            return "break"

        def restore_entry_value(_event=None):
            value_text.set(last_valid_text)
            return "break"

        setattr(value_entry, "_commit_value", commit_entry_value)
        variable.trace_add("write", sync_entry_from_variable)
        value_entry.bind("<Return>", commit_entry_value)
        value_entry.bind("<KP_Enter>", commit_entry_value)
        value_entry.bind("<FocusOut>", commit_entry_value)
        value_entry.bind("<Escape>", restore_entry_value)
        error_label = None
        if validation_key is not None:
            error_label = ttk.Label(
                frame,
                text="",
                foreground=VALIDATION_ERROR_COLOR,
                font=("TkDefaultFont", 8),
                wraplength=CONTROL_WRAP_LENGTH,
                justify="left",
            )
            error_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(1, 0))
            error_label.grid_remove()
            self._trajectory_error_labels[validation_key] = error_label
            self._slider_controls[validation_key] = {
                "frame": frame,
                "widgets": control_widgets,
                "entry": value_entry,
                "slider": slider,
                "error_label": error_label,
            }

    def _current_source_depth(self) -> int:
        return max(1, int(self.d_layer.get()))

    def _current_geometry(self) -> GeometryContext:
        return GeometryContext(
            lattice_constant_ang=float(self.a.get()),
            bohr_radius_ang=float(self.R_bohr.get()),
            alpha_deg=float(self.alpha_deg.get()),
            beta_deg=float(self.beta_deg.get()),
            lattice_radius=int(self.lattice_radius.get()),
            source_layer=self._current_source_depth() - 1,
            orbital_l=int(self.orbital_l.get()),
        )

    def _selected_formula_variant(self) -> str:
        return FORMULA_BY_LABEL[self.formula_variant_label.get()]

    def _phase_grid_request(self, impact_parameters_ang: list[float]) -> PhaseGridRequest:
        return PhaseGridRequest(
            Emin_eV=float(self.Emin.get()),
            Emax_eV=float(self.Emax.get()),
            N=int(self.Npts.get()),
            a_list_ang=impact_parameters_ang,
            Z=float(self.Z.get()),
            b_ang=float(self.b.get()),
            c1=float(self.c1.get()),
            c2=float(self.c2.get()),
            dr_ang=float(self.dr.get()),
            r_max_ang=float(self.rmax.get()),
            chi=interpolate_thomas_fermi_chi if self.use_table_chi.get() else exponential_chi,
            i3_mode="sum_avg" if self.i3_mode_sum.get() else "trapz",
        )

    def _boundary_energy_grid(self) -> np.ndarray:
        emin = float(self.boundary_Emin.get())
        emax = float(self.boundary_Emax.get())
        npts = int(self.boundary_Npts.get())
        if emin <= 0.0 or emax <= 0.0 or emax <= emin:
            raise ValueError("Для утилиты требуется 0 < Emin < Emax.")
        if npts < 2:
            raise ValueError("Для утилиты нужно минимум 2 точки по энергии.")
        return np.linspace(emin, emax, npts, dtype=float)

    def _boundary_selected_energy(self) -> float:
        energy_value = float(self.boundary_energy_point.get())
        if energy_value <= 0.0:
            raise ValueError("Выбранная энергия должна быть положительной.")
        return energy_value

    def _trajectory_sweep_mode(self) -> str:
        return TRAJECTORY_SWEEP_BY_LABEL[self.trajectory_sweep_label.get()]

    def _current_trajectory_request(self) -> TrajectorySweepRequest:
        return TrajectorySweepRequest(
            sweep_mode=self._trajectory_sweep_mode(),
            point_count=int(self.trajectory_Npts.get()),
            atomic_number=float(self.trajectory_Z.get()),
            mass_amu=float(self.trajectory_mass_amu.get()),
            energy_eV=float(self.trajectory_energy.get()),
            energy_min_eV=float(self.trajectory_Emin.get()),
            energy_max_eV=float(self.trajectory_Emax.get()),
            impact_parameter_ang=float(self.trajectory_impact.get()),
            impact_min_ang=float(self.trajectory_impact_min.get()),
            impact_max_ang=float(self.trajectory_impact_max.get()),
            r0_ang=float(self.trajectory_r0.get()),
            angle_step_deg=float(self.trajectory_angle_step_deg.get()),
            angle_step_min_deg=float(self.trajectory_angle_step_min_deg.get()),
            angle_step_max_deg=float(self.trajectory_angle_step_max_deg.get()),
            b_bohr=float(self.trajectory_b_bohr.get()),
            orbital_l=int(self.trajectory_orbital_l.get()),
            magnetic_m=int(self.trajectory_magnetic_m.get()),
            random_m=bool(self.trajectory_random_m.get()),
            min_steps=30,
            max_refinements=6,
            parallel_workers=int(self.trajectory_parallel_workers.get()),
        )

    def _current_rashba_request(self) -> RashbaSurfaceRequest:
        energy_min = float(self.rashba_Emin.get())
        energy_max = float(self.rashba_Emax.get())
        point_count = int(self.rashba_Npts.get())
        energies_eV = np.linspace(energy_min, energy_max, point_count, dtype=float)
        ver_up_to_down, ver_down_to_up = self._rashba_volume_flip_probabilities(energies_eV)
        return RashbaSurfaceRequest(
            energy_min_eV=energy_min,
            energy_max_eV=energy_max,
            point_count=point_count,
            layer_thickness_ang=float(self.rashba_layer_thickness.get()),
            rashba_alpha_au=float(self.rashba_alpha.get()),
            emission_angle_deg=float(self.rashba_theta_deg.get()),
            surface_potential_eV=float(self.rashba_surface_potential.get()),
            ver_up_to_down=ver_up_to_down,
            ver_down_to_up=ver_down_to_up,
        )

    def _rashba_volume_flip_probabilities(self, target_energies_eV: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        source = self.rashba_source_label.get()
        zeros = np.zeros_like(target_energies_eV, dtype=float)
        if source == RASHBA_SOURCE_ZERO:
            return zeros, zeros

        if source == RASHBA_SOURCE_SPECTRUM:
            payload = self._latest_plot_payload
            if payload is None:
                raise RuntimeError("Сначала постройте графики на вкладке «Спектры и формулы» или выберите Ver=0.")
            source_energies = payload.result.grid["E_eV"].to_numpy(dtype=float)
            spin_curves = payload.result.spin_curves
            ver_up_to_down = (
                np.asarray(spin_curves["sum_check_up"], dtype=float)
                - np.asarray(spin_curves["spin_mean_up"], dtype=float)
            ) / 2.0
            ver_down_to_up = (
                np.asarray(spin_curves["sum_check_dn"], dtype=float)
                + np.asarray(spin_curves["spin_mean_dn"], dtype=float)
            ) / 2.0
            return (
                self._interpolate_probability_curve(source_energies, ver_up_to_down, target_energies_eV),
                self._interpolate_probability_curve(source_energies, ver_down_to_up, target_energies_eV),
            )

        if source == RASHBA_SOURCE_TRAJECTORY:
            payload = self._latest_trajectory_payload
            if payload is None:
                raise RuntimeError("Сначала выполните траекторный расчёт или выберите Ver=0.")
            frame = payload.frame
            return (
                self._interpolate_probability_curve(
                    frame["energy_eV"].to_numpy(dtype=float),
                    frame["p_flip_initial_up"].to_numpy(dtype=float),
                    target_energies_eV,
                ),
                self._interpolate_probability_curve(
                    frame["energy_eV"].to_numpy(dtype=float),
                    frame["p_flip_initial_down"].to_numpy(dtype=float),
                    target_energies_eV,
                ),
            )

        raise RuntimeError(f"Неизвестный источник Ver: {source}")

    @staticmethod
    def _interpolate_probability_curve(
        source_energies_eV: np.ndarray,
        source_probabilities: np.ndarray,
        target_energies_eV: np.ndarray,
    ) -> np.ndarray:
        source_energies_eV = np.asarray(source_energies_eV, dtype=float)
        source_probabilities = np.asarray(source_probabilities, dtype=float)
        target_energies_eV = np.asarray(target_energies_eV, dtype=float)
        mask = np.isfinite(source_energies_eV) & np.isfinite(source_probabilities)
        source_energies_eV = source_energies_eV[mask]
        source_probabilities = np.clip(source_probabilities[mask], 0.0, 1.0)
        if source_energies_eV.size == 0:
            raise RuntimeError("В выбранном источнике Ver нет корректных точек.")

        order = np.argsort(source_energies_eV)
        sorted_energies = source_energies_eV[order]
        sorted_probabilities = source_probabilities[order]
        unique_energies, unique_indices = np.unique(sorted_energies, return_index=True)
        unique_probabilities = sorted_probabilities[unique_indices]
        if unique_energies.size == 1:
            return np.full_like(target_energies_eV, float(unique_probabilities[0]), dtype=float)
        return np.clip(
            np.interp(target_energies_eV, unique_energies, unique_probabilities),
            0.0,
            1.0,
        )

    def _trajectory_validation_errors(self) -> dict[str, list[str]]:
        errors: dict[str, list[str]] = {}

        def add(key: str, message: str) -> None:
            errors.setdefault(key, []).append(message)

        sweep_mode = self._trajectory_sweep_mode()
        z = float(self.trajectory_Z.get())
        mass = float(self.trajectory_mass_amu.get())
        energy = float(self.trajectory_energy.get())
        emin = float(self.trajectory_Emin.get())
        emax = float(self.trajectory_Emax.get())
        impact = float(self.trajectory_impact.get())
        impact_min = float(self.trajectory_impact_min.get())
        impact_max = float(self.trajectory_impact_max.get())
        r0 = float(self.trajectory_r0.get())
        angle_step = float(self.trajectory_angle_step_deg.get())
        angle_step_min = float(self.trajectory_angle_step_min_deg.get())
        angle_step_max = float(self.trajectory_angle_step_max_deg.get())
        b_bohr = float(self.trajectory_b_bohr.get())
        point_count = int(self.trajectory_Npts.get())
        orbital_l = int(self.trajectory_orbital_l.get())
        magnetic_m = int(self.trajectory_magnetic_m.get())
        parallel_workers = int(self.trajectory_parallel_workers.get())

        if z <= 0.0:
            add("atomic_number", "Z должен быть положительным.")
        if mass <= 0.0:
            add("mass_amu", "Масса должна быть положительной.")
        if energy <= 0.0:
            add("energy", "Энергия должна быть положительной.")
        if emin <= 0.0:
            add("energy_min", "Emin должен быть положительным.")
        if emax <= 0.0:
            add("energy_max", "Emax должен быть положительным.")
        if emax <= emin:
            add("energy_min", "Emin должен быть меньше Emax.")
            add("energy_max", "Emax должен быть больше Emin.")

        if impact <= 0.0:
            add("impact_fixed", "r_п должен быть положительным.")
        if impact_min <= 0.0:
            add("impact_min", "r_п min должен быть положительным.")
        if impact_max <= 0.0:
            add("impact_max", "r_п max должен быть положительным.")
        if impact_max <= impact_min:
            add("impact_min", "r_п min должен быть меньше r_п max.")
            add("impact_max", "r_п max должен быть больше r_п min.")

        largest_used_impact = impact_max if sweep_mode == TRAJECTORY_SWEEP_IMPACT else impact
        if r0 <= 0.0:
            add("r0", "r0 должен быть положительным.")
        elif r0 <= largest_used_impact:
            add("r0", f"r0 должен быть больше используемого r_п ({largest_used_impact:.4g} Å).")
            if sweep_mode == TRAJECTORY_SWEEP_IMPACT:
                add("impact_max", "r_п max должен быть меньше r0.")
            else:
                add("impact_fixed", "r_п фикс. должен быть меньше r0.")

        if angle_step <= 0.0:
            add("angle_step", "dθ фикс. должен быть положительным.")
        if angle_step_min <= 0.0:
            add("angle_step_min", "dθ min должен быть положительным.")
        if angle_step_max <= 0.0:
            add("angle_step_max", "dθ max должен быть положительным.")
        if angle_step_max <= angle_step_min:
            add("angle_step_min", "dθ min должен быть меньше dθ max.")
            add("angle_step_max", "dθ max должен быть больше dθ min.")

        if b_bohr <= 0.0:
            add("b_bohr", "b Thomas-Fermi должен быть положительным.")
        if point_count < 1:
            add("point_count", "Нужна хотя бы одна точка.")
        if orbital_l < 0:
            add("orbital_l", "L должен быть неотрицательным.")
        if not self.trajectory_random_m.get() and abs(magnetic_m) > orbital_l:
            add("magnetic_m", f"Для ручного M требуется -L <= M <= L; сейчас L={orbital_l}, M={magnetic_m}.")
        if parallel_workers < 1:
            add("parallel_workers", "Потоков должно быть минимум 1.")

        return errors

    def _apply_trajectory_validation_errors(self, errors: dict[str, list[str]]) -> None:
        for key, label in self._trajectory_error_labels.items():
            messages = errors.get(key, [])
            if messages:
                label.config(text=" ".join(messages))
                label.grid()
            else:
                label.config(text="")
                label.grid_remove()

    def _update_trajectory_validation_hints(self) -> dict[str, list[str]]:
        errors = self._trajectory_validation_errors()
        self._apply_trajectory_validation_errors(errors)
        return errors

    def _add_trajectory_runtime_hints(self, result: TrajectorySweepResult) -> None:
        errors = self._trajectory_validation_errors()
        failed_frame = result.frame[~result.frame["converged"].astype(bool)]
        if failed_frame.empty:
            self._apply_trajectory_validation_errors(errors)
            return

        status_text = " ".join(str(value) for value in failed_frame["status"].to_list())

        def add(key: str, message: str) -> None:
            errors.setdefault(key, [])
            if message not in errors[key]:
                errors[key].append(message)

        if "max_steps" in status_text:
            if result.request.sweep_mode == TRAJECTORY_SWEEP_IMPACT:
                add("impact_min", "Часть точек не сошлась: увеличьте r_п min, например до 0.3 Å.")
                add("angle_step", "Можно также увеличить dθ фикс. до 2-5°.")
            elif result.request.sweep_mode == TRAJECTORY_SWEEP_ANGLE_STEP:
                add("angle_step_min", "Часть точек не сошлась: увеличьте dθ min.")
            else:
                add("angle_step", "Часть точек не сошлась: увеличьте dθ фикс. до 2-5°.")
        if "r0" in status_text or "r_п" in status_text:
            add("r0", "Проверьте, что r0 больше всех используемых r_п.")

        self._apply_trajectory_validation_errors(errors)

    def _current_search_region(self, geometry: GeometryContext) -> tuple[LatticeSearchRegion, str]:
        if self.auto_n.get():
            estimate = estimate_lattice_search_region(
                lattice_constant_ang=geometry.lattice_constant_ang,
                bohr_radius_ang=geometry.bohr_radius_ang,
                alpha_rad=geometry.alpha_rad,
                beta_rad=geometry.beta_rad,
                source_layer=geometry.source_layer,
            )
            region = estimate.region
            summary = (
                f"Автообласть: nx={region.x_radius}, ny={region.y_radius}, "
                f"слои d=1..{region.z_max_layer + 1}"
            )
            if estimate.capped_by_max_atoms:
                req = estimate.required_region
                summary += (
                    f" | требуется nx={req.x_radius}, ny={req.y_radius}, слои d=1..{req.z_max_layer + 1},"
                    " но область урезана лимитом узлов."
                )
            return region, summary

        manual_n = max(1, int(self.lattice_radius.get()))
        region = LatticeSearchRegion(
            x_radius=manual_n,
            y_radius=manual_n,
            z_min_layer=0,
            z_max_layer=max(2 * manual_n, geometry.source_layer + manual_n),
        )
        return region, (
            f"Ручная область: x/y=[-{manual_n}..{manual_n}], "
            f"слои d=1..{region.z_max_layer + 1}."
        )

    def _prepare_plot_request(self) -> PlotComputationRequest:
        geometry = self._current_geometry()
        search_region, _ = self._current_search_region(geometry)
        atom_selection = collect_atom_selection(geometry, search_region=search_region)
        impact_parameters = atom_selection.impact_parameters_ang
        min_impact_parameter = 1e-4

        logger.info(
            "RIGHT | Emin=%.3g eV, Emax=%.3g eV, N=%d, Z=%.3g, lattice_a=%.3g Å, b=%.3g Å, "
            "c1=%.3g, c2=%.3g, dr=%.3g Å, rmax=%.3g Å, atoms(all)=%d, atoms(used)=%d, "
            "R_int=%.3g Å, n=%d, alpha=%.3f deg, beta=%.3f deg, L=%d, eps_a=%.3g",
            self.Emin.get(),
            self.Emax.get(),
            int(self.Npts.get()),
            self.Z.get(),
            geometry.lattice_constant_ang,
            self.b.get(),
            self.c1.get(),
            self.c2.get(),
            self.dr.get(),
            self.rmax.get(),
            len(atom_selection.all_atoms),
            len(impact_parameters),
            geometry.interaction_radius_ang,
            geometry.lattice_radius,
            geometry.alpha_deg,
            geometry.beta_deg,
            geometry.orbital_l,
            min_impact_parameter,
        )

        if not impact_parameters:
            raise RuntimeError(
                "Не найдено атомов для суммирования Φ(E). Увеличьте n или радиус взаимодействия (R_bohr), либо измените α/β."
            )

        max_impact_parameter = max(impact_parameters)
        if float(self.rmax.get()) <= max_impact_parameter:
            raise RuntimeError(
                f"Некорректно: r_max={float(self.rmax.get()):.6g} Å должен быть больше "
                f"max(a=d_прямой)={max_impact_parameter:.6g} Å."
            )

        return PlotComputationRequest(
            geometry=geometry,
            all_atom_count=len(atom_selection.all_atoms),
            formula_variant=self._selected_formula_variant(),
            formula_label=self.formula_variant_label.get(),
            phase_request=self._phase_grid_request(impact_parameters),
            use_table_chi=bool(self.use_table_chi.get()),
            i3_mode_sum=bool(self.i3_mode_sum.get()),
        )

    def _on_orbital_l_changed(self) -> None:
        self.update_output_left()
        self._schedule_right_update_if_auto()

    def _on_geometry_inputs_changed(self) -> None:
        if self._geometry_change_in_progress:
            return

        self._geometry_change_in_progress = True
        try:
            self._recompute_lattice_radius()
        finally:
            self._geometry_change_in_progress = False

        if self.auto.get():
            self._schedule_left_update(delay_ms=75)
        self._schedule_right_update_if_auto()

    def _on_calculation_inputs_changed(self) -> None:
        self._schedule_right_update_if_auto()

    def _on_formula_variant_changed(self) -> None:
        self._update_formula_hint()
        self._schedule_right_update_if_auto()

    def _on_auto_toggle(self) -> None:
        if self.auto.get():
            self._schedule_left_update(delay_ms=75)
            self._schedule_right_update(delay_ms=150)
        else:
            if self._scheduled_left_after is not None:
                self.after_cancel(self._scheduled_left_after)
                self._scheduled_left_after = None
            if self._scheduled_right_after is not None:
                self.after_cancel(self._scheduled_right_after)
                self._scheduled_right_after = None

    def _on_trajectory_inputs_changed(self) -> None:
        self._update_trajectory_control_states()
        errors = self._update_trajectory_validation_hints()
        if errors:
            if self._scheduled_trajectory_after is not None:
                self.after_cancel(self._scheduled_trajectory_after)
                self._scheduled_trajectory_after = None
            return
        if self.trajectory_auto.get():
            self._schedule_trajectory_update(delay_ms=450)

    def _on_trajectory_auto_toggle(self) -> None:
        errors = self._update_trajectory_validation_hints()
        if self.trajectory_auto.get():
            if errors:
                if self._scheduled_trajectory_after is not None:
                    self.after_cancel(self._scheduled_trajectory_after)
                    self._scheduled_trajectory_after = None
                return
            self._schedule_trajectory_update(delay_ms=150)
        elif self._scheduled_trajectory_after is not None:
            self.after_cancel(self._scheduled_trajectory_after)
            self._scheduled_trajectory_after = None

    def _update_trajectory_control_states(self) -> None:
        self._set_slider_control_enabled(
            "angle_step",
            self._trajectory_sweep_mode() != TRAJECTORY_SWEEP_ANGLE_STEP,
        )

    def _set_slider_control_enabled(self, validation_key: str, enabled: bool) -> None:
        control = self._slider_controls.get(validation_key)
        if not control:
            return
        state = ["!disabled"] if enabled else ["disabled"]
        for widget in control["widgets"]:
            try:
                widget.state(state)
            except (AttributeError, tk.TclError):
                continue

    def _update_formula_hint(self) -> None:
        self.formula_hint_text.set(FORMULA_HINTS[self._selected_formula_variant()])

    def _recompute_lattice_radius(self) -> None:
        if not self.auto_n.get():
            return

        geometry = self._current_geometry()
        estimate = estimate_lattice_search_region(
            lattice_constant_ang=geometry.lattice_constant_ang,
            bohr_radius_ang=geometry.bohr_radius_ang,
            alpha_rad=geometry.alpha_rad,
            beta_rad=geometry.beta_rad,
            source_layer=geometry.source_layer,
        )
        region = estimate.region
        auto_radius = max(region.x_radius, region.y_radius, abs(region.z_min_layer), abs(region.z_max_layer))
        if auto_radius != int(self.lattice_radius.get()):
            self.lattice_radius.set(auto_radius)

        if self.n_auto_label is not None:
            text = (
                f"Автообласть: nx={region.x_radius}, ny={region.y_radius}, "
                f"слои d=1..{region.z_max_layer + 1} (~{region.node_count:,} узлов)"
            )
            if estimate.capped_by_max_atoms:
                required = estimate.required_region
                text += (
                    f" | требуется nx={required.x_radius}, ny={required.y_radius}, "
                    f"слои d=1..{required.z_max_layer + 1}, но область урезана."
                )
            self.n_auto_label.config(text=text)

    def _schedule_right_update_if_auto(self) -> None:
        if self.auto.get():
            self._schedule_right_update(delay_ms=250)

    def _schedule_left_update(self, delay_ms: int = 0) -> None:
        if self._scheduled_left_after is not None:
            self.after_cancel(self._scheduled_left_after)
        self._scheduled_left_after = self.after(delay_ms, self._run_scheduled_left_update)

    def _run_scheduled_left_update(self) -> None:
        self._scheduled_left_after = None
        self.update_output_left()

    def _schedule_right_update(self, delay_ms: int = 0) -> None:
        if self._scheduled_right_after is not None:
            self.after_cancel(self._scheduled_right_after)
        self._scheduled_right_after = self.after(delay_ms, self._submit_right_update)

    def _schedule_trajectory_update(self, delay_ms: int = 0) -> None:
        if self._scheduled_trajectory_after is not None:
            self.after_cancel(self._scheduled_trajectory_after)
        self._scheduled_trajectory_after = self.after(delay_ms, self._submit_trajectory_update)

    def _submit_right_update(self) -> None:
        self._scheduled_right_after = None
        try:
            request = self._prepare_plot_request()
        except Exception as ex:
            self._display_right_error(ex)
            return

        if self._running_future is not None and not self._running_future.done():
            self._queued_request = request
            self.status_text.set("Расчёт уже идёт. Последнее изменение поставлено в очередь.")
            return

        self._start_right_update(request)

    def _start_right_update(self, request: PlotComputationRequest) -> None:
        self.status_text.set(f"Расчёт графиков: {request.formula_label}...")
        self._running_future = self._executor.submit(run_plot_calculation, request)
        self._running_future.add_done_callback(self._dispatch_right_update_finish)

    def _dispatch_right_update_finish(self, future: Future) -> None:
        if self._closing:
            return
        try:
            self.after(0, self._finish_right_update, future)
        except tk.TclError:
            return

    def _finish_right_update(self, future: Future) -> None:
        if not self.winfo_exists():
            return
        if future is not self._running_future:
            return

        self._running_future = None
        next_request = self._queued_request
        self._queued_request = None

        try:
            payload = future.result()
        except Exception as ex:
            logger.exception("RIGHT | ошибка при расчёте")
            self._display_right_error(ex)
        else:
            self._apply_plot_result(payload)

        if next_request is not None:
            self._start_right_update(next_request)

    def _apply_plot_result(self, payload: PlotComputationResult) -> None:
        result = payload.result
        request = payload.request
        energies_eV = result.grid["E_eV"].to_numpy(dtype=float)
        self._latest_plot_payload = payload
        draw_spin_plots(self.ax_sum, self.ax_spin, energies_eV, result.spin_curves)
        self.fig.tight_layout()
        self._default_view_limits = capture_view_limits(self.ax_sum, self.ax_spin)
        self.canvas.draw_idle()

        if result.fixed_lz is not None:
            lz_summary = f"Lz={result.fixed_lz} (фиксировано)"
        else:
            preview = ", ".join(str(value) for value in result.lz_chain[:8])
            suffix = "..." if len(result.lz_chain) > 8 else ""
            lz_summary = f"Lz=random [{preview}{suffix}]"

        self._append_output(
            f"\n[Часть 2 | {request.formula_label}] E∈[{request.phase_request.Emin_eV:.3g},{request.phase_request.Emax_eV:.3g}] эВ, "
            f"N={request.phase_request.N}, Z={request.phase_request.Z:.3g}, "
            f"lattice_a={request.geometry.lattice_constant_ang:.3g} Å, b={request.phase_request.b_ang:.3g} Å, "
            f"atoms={len(request.phase_request.a_list_ang)} из {request.all_atom_count}, "
            f"R_int={request.geometry.interaction_radius_ang:.3g} Å, n={request.geometry.lattice_radius}, "
            f"d={request.geometry.source_depth}, "
            f"α={request.geometry.alpha_deg:.3g}°, β={request.geometry.beta_deg:.3g}°, "
            f"χ={'table' if request.use_table_chi else 'exp'}, "
            f"I3={'sum' if request.i3_mode_sum else 'trapz'}, "
            f"L={request.geometry.orbital_l}, {lz_summary}\n"
        )
        self.status_text.set(
            f"Готово: {request.formula_label}, {len(energies_eV)} энергий, {len(request.phase_request.a_list_ang)} атомов."
        )
        logger.info("RIGHT | сетка рассчитана: %d точек", len(result.grid))
        if self.rashba_source_label.get() == RASHBA_SOURCE_SPECTRUM:
            self._update_rashba_surface()

    def _display_right_error(self, error: Exception) -> None:
        self._latest_plot_payload = None
        for axis in (self.ax_sum, self.ax_spin):
            axis.clear()
            axis.text(0.05, 0.95, f"Ошибка: {error}", transform=axis.transAxes, va="top", ha="left")
            axis.grid(True, which="both")
        if self.canvas is not None:
            self.canvas.draw_idle()
        self.status_text.set(f"Ошибка расчёта: {error}")
        self._append_output(f"\n[Ошибка расчёта] {error}\n")

    def _zoom_plots(self, factor: float) -> None:
        try:
            for axis in (self.ax_sum, self.ax_spin):
                zoom_axis(axis, factor)
            self.canvas.draw_idle()
        except Exception as ex:
            logger.exception("ZOOM | ошибка")
            self._append_output(f"\n[Zoom] Ошибка: {ex}\n")

    def _on_plot_scroll(self, event) -> None:
        if event.inaxes not in (self.ax_sum, self.ax_spin):
            return
        factor = 0.85 if event.button == "up" else 1.18
        try:
            if event.xdata is None or event.ydata is None:
                zoom_axis(event.inaxes, factor)
            else:
                zoom_axis_around_point(event.inaxes, factor, event.xdata, event.ydata)
            self.canvas.draw_idle()
        except Exception as ex:
            logger.exception("PLOT_SCROLL | ошибка")
            self._append_output(f"\n[Plot Scroll] Ошибка: {ex}\n")

    def _on_geometry_scroll(self, event) -> None:
        if event.inaxes not in (self.ax_geometry_3d, self.ax_geometry_xz, self.ax_geometry_xy):
            return

        factor = 0.85 if event.button == "up" else 1.18
        try:
            if event.inaxes is self.ax_geometry_3d:
                zoom_3d_axis(self.ax_geometry_3d, factor)
            elif event.xdata is None or event.ydata is None:
                zoom_axis(event.inaxes, factor)
            else:
                zoom_axis_around_point(event.inaxes, factor, event.xdata, event.ydata)
            if self.geometry_canvas is not None:
                self.geometry_canvas.draw_idle()
        except Exception as ex:
            logger.exception("GEOMETRY_SCROLL | ошибка")
            if self.geometry_output is not None:
                self.geometry_output.insert(tk.END, f"\n[Geometry Scroll] Ошибка: {ex}\n")
                self.geometry_output.see(tk.END)

    def _on_boundary_plot_scroll(self, event) -> None:
        if event.inaxes not in (self.ax_boundary_reflection, self.ax_boundary_angle):
            return
        factor = 0.85 if event.button == "up" else 1.18
        try:
            if event.xdata is None or event.ydata is None:
                zoom_axis(event.inaxes, factor)
            else:
                zoom_axis_around_point(event.inaxes, factor, event.xdata, event.ydata)
            if self.boundary_canvas is not None:
                self.boundary_canvas.draw_idle()
        except Exception as ex:
            logger.exception("BOUNDARY_SCROLL | ошибка")
            if self.boundary_output is not None:
                self.boundary_output.insert(tk.END, f"\n[Boundary Scroll] Ошибка: {ex}\n")
                self.boundary_output.see(tk.END)

    def _on_trajectory_plot_scroll(self, event) -> None:
        steps_axis = getattr(self.ax_trajectory_diagnostics, "_trajectory_steps_axis", None)
        if event.inaxes not in (self.ax_trajectory_phase, self.ax_trajectory_angle, self.ax_trajectory_diagnostics, steps_axis):
            return
        factor = 0.85 if event.button == "up" else 1.18
        try:
            target_axis = self.ax_trajectory_diagnostics if event.inaxes is steps_axis else event.inaxes
            if event.xdata is None or event.ydata is None:
                zoom_axis(target_axis, factor)
            else:
                zoom_axis_around_point(target_axis, factor, event.xdata, event.ydata)
            if self.trajectory_canvas is not None:
                self.trajectory_canvas.draw_idle()
        except Exception as ex:
            logger.exception("TRAJECTORY_SCROLL | ошибка")
            if self.trajectory_output is not None:
                self.trajectory_output.insert(tk.END, f"\n[Trajectory Scroll] Ошибка: {ex}\n")
                self.trajectory_output.see(tk.END)

    def _on_rashba_plot_scroll(self, event) -> None:
        if event.inaxes not in (self.ax_rashba_transmission, self.ax_rashba_polarization):
            return
        factor = 0.85 if event.button == "up" else 1.18
        try:
            if event.xdata is None or event.ydata is None:
                zoom_axis(event.inaxes, factor)
            else:
                zoom_axis_around_point(event.inaxes, factor, event.xdata, event.ydata)
            if self.rashba_canvas is not None:
                self.rashba_canvas.draw_idle()
        except Exception as ex:
            logger.exception("RASHBA_SCROLL | ошибка")
            if self.rashba_output is not None:
                self.rashba_output.insert(tk.END, f"\n[Rashba Scroll] Ошибка: {ex}\n")
                self.rashba_output.see(tk.END)

    def _reset_zoom(self) -> None:
        if not self._default_view_limits:
            return
        try:
            restore_view_limits(self._default_view_limits)
            self.canvas.draw_idle()
        except Exception as ex:
            logger.exception("RESET_ZOOM | ошибка")
            self._append_output(f"\n[Reset Zoom] Ошибка: {ex}\n")

    def _reset_boundary_zoom(self) -> None:
        if not self._boundary_view_limits:
            return
        try:
            restore_view_limits(self._boundary_view_limits)
            if self.boundary_canvas is not None:
                self.boundary_canvas.draw_idle()
        except Exception as ex:
            logger.exception("RESET_BOUNDARY_ZOOM | ошибка")
            if self.boundary_output is not None:
                self.boundary_output.insert(tk.END, f"\n[Reset Boundary Zoom] Ошибка: {ex}\n")
                self.boundary_output.see(tk.END)

    def _reset_trajectory_zoom(self) -> None:
        if not self._trajectory_view_limits:
            return
        try:
            restore_view_limits(self._trajectory_view_limits)
            if self.trajectory_canvas is not None:
                self.trajectory_canvas.draw_idle()
        except Exception as ex:
            logger.exception("RESET_TRAJECTORY_ZOOM | ошибка")
            if self.trajectory_output is not None:
                self.trajectory_output.insert(tk.END, f"\n[Reset Trajectory Zoom] Ошибка: {ex}\n")
                self.trajectory_output.see(tk.END)

    def _reset_rashba_zoom(self) -> None:
        if not self._rashba_view_limits:
            return
        try:
            restore_view_limits(self._rashba_view_limits)
            if self.rashba_canvas is not None:
                self.rashba_canvas.draw_idle()
        except Exception as ex:
            logger.exception("RESET_RASHBA_ZOOM | ошибка")
            if self.rashba_output is not None:
                self.rashba_output.insert(tk.END, f"\n[Reset Rashba Zoom] Ошибка: {ex}\n")
                self.rashba_output.see(tk.END)

    def _set_text_output(self, target: tk.Text | None, text: str) -> None:
        if target is None:
            return
        target.delete("1.0", tk.END)
        target.insert("1.0", text)
        target.see("1.0")

    def _update_boundary_utility(self) -> None:
        if self._closing:
            return
        if (
            self.boundary_fig is None
            or self.boundary_canvas is None
            or self.ax_boundary_reflection is None
            or self.ax_boundary_angle is None
        ):
            return
        try:
            energies_eV = self._boundary_energy_grid()
            point_energy = self._boundary_selected_energy()
            work_function_eV = float(self.boundary_work_function.get())
            incidence_angle_deg = float(self.boundary_alpha_deg.get())
            point_energy = min(max(point_energy, float(energies_eV[0])), float(energies_eV[-1]))

            curves = compute_boundary_reflection_curves(
                energies_eV,
                work_function_eV=work_function_eV,
                incidence_angle_deg=incidence_angle_deg,
            )
            point_result = compute_boundary_point(
                point_energy,
                work_function_eV=work_function_eV,
                incidence_angle_deg=incidence_angle_deg,
            )
        except Exception as ex:
            logger.exception("BOUNDARY | ошибка расчёта утилиты")
            if self.ax_boundary_reflection is not None and self.ax_boundary_angle is not None:
                for axis in (self.ax_boundary_reflection, self.ax_boundary_angle):
                    axis.clear()
                    axis.text(0.05, 0.95, f"Ошибка: {ex}", transform=axis.transAxes, va="top", ha="left")
                    axis.grid(True, which="both")
                if self.boundary_canvas is not None:
                    self.boundary_canvas.draw_idle()
            self._set_text_output(self.boundary_output, f"[Ошибка утилиты]\n{ex}\n")
            self.status_text.set(f"Ошибка утилиты границы: {ex}")
            return

        if self.boundary_energy_point.get() != point_energy:
            self.boundary_energy_point.set(point_energy)
            return

        draw_boundary_utility_plots(self.ax_boundary_reflection, self.ax_boundary_angle, curves, point_result)
        self.boundary_fig.tight_layout()
        self._boundary_view_limits = capture_view_limits(self.ax_boundary_reflection, self.ax_boundary_angle)
        self.boundary_canvas.draw_idle()

        summary = self._format_boundary_summary(curves, point_result)
        self._set_text_output(self.boundary_output, summary)

    def _update_rashba_surface(self) -> None:
        if self._closing:
            return
        if (
            self.rashba_fig is None
            or self.rashba_canvas is None
            or self.ax_rashba_transmission is None
            or self.ax_rashba_polarization is None
        ):
            return
        try:
            request = self._current_rashba_request()
            result = compute_rashba_surface(request)
        except Exception as ex:
            logger.exception("RASHBA | ошибка расчёта")
            self._latest_rashba_payload = None
            for axis in (self.ax_rashba_transmission, self.ax_rashba_polarization):
                axis.clear()
                axis.text(0.05, 0.95, f"Ошибка: {ex}", transform=axis.transAxes, va="top", ha="left")
                axis.grid(True, which="both")
            if self.rashba_canvas is not None:
                self.rashba_canvas.draw_idle()
            self._set_text_output(self.rashba_output, f"[Ошибка расчёта Рашбы]\n{ex}\n")
            self.status_text.set(f"Ошибка расчёта Рашбы: {ex}")
            return

        self._latest_rashba_payload = result
        draw_rashba_surface_plots(self.ax_rashba_transmission, self.ax_rashba_polarization, result.frame)
        self.rashba_fig.tight_layout()
        self._rashba_view_limits = capture_view_limits(self.ax_rashba_transmission, self.ax_rashba_polarization)
        self.rashba_canvas.draw_idle()
        self._set_text_output(self.rashba_output, self._format_rashba_summary(result))
        self.status_text.set(f"Расчёт Рашбы готов: {len(result.frame)} точек.")

    def _update_trajectory_utility(self) -> None:
        self._submit_trajectory_update()

    def _submit_trajectory_update(self) -> None:
        if self._closing:
            return
        if (
            self.trajectory_fig is None
            or self.trajectory_canvas is None
            or self.ax_trajectory_phase is None
            or self.ax_trajectory_angle is None
            or self.ax_trajectory_diagnostics is None
        ):
            return
        errors = self._update_trajectory_validation_hints()
        if errors:
            self._set_text_output(
                self.trajectory_output,
                "[Траекторный расчёт]\nИсправьте параметры, отмеченные красным в левой панели.\n",
            )
            self.status_text.set("Траекторный расчёт не запущен: есть ошибки в параметрах.")
            return
        try:
            request = self._current_trajectory_request()
        except Exception as ex:
            self._display_trajectory_error(ex)
            return

        if self._running_trajectory_future is not None and not self._running_trajectory_future.done():
            self._queued_trajectory_request = request
            self.status_text.set("Траекторный расчёт уже идёт. Последнее изменение поставлено в очередь.")
            return

        self._start_trajectory_update(request)

    def _start_trajectory_update(self, request: TrajectorySweepRequest) -> None:
        self.status_text.set(
            f"Траекторный расчёт: {TRAJECTORY_SWEEP_LABELS[request.sweep_mode]}, {request.point_count} точек..."
        )
        self._running_trajectory_future = self._trajectory_executor.submit(execute_trajectory_sweep, request)
        self.after(50, self._poll_trajectory_update)

    def _poll_trajectory_update(self) -> None:
        if self._closing:
            return
        future = self._running_trajectory_future
        if future is None:
            return
        if future.done():
            self._finish_trajectory_update(future)
            return
        self.after(50, self._poll_trajectory_update)

    def _finish_trajectory_update(self, future: Future) -> None:
        if not self.winfo_exists():
            return
        if future is not self._running_trajectory_future:
            return

        self._running_trajectory_future = None
        next_request = self._queued_trajectory_request
        self._queued_trajectory_request = None

        try:
            result = future.result()
        except Exception as ex:
            logger.exception("TRAJECTORY | ошибка расчёта")
            self._display_trajectory_error(ex)
        else:
            self._apply_trajectory_result(result)

        if next_request is not None:
            self._start_trajectory_update(next_request)

    def _apply_trajectory_result(self, result: TrajectorySweepResult) -> None:
        self._latest_trajectory_payload = result
        x_label = TRAJECTORY_AXIS_LABELS[result.request.sweep_mode]
        draw_trajectory_sweep_plots(
            self.ax_trajectory_phase,
            self.ax_trajectory_angle,
            self.ax_trajectory_diagnostics,
            result.frame,
            "sweep_value",
            x_label,
        )
        self.trajectory_fig.tight_layout()
        self._trajectory_view_limits = capture_view_limits(
            self.ax_trajectory_phase,
            self.ax_trajectory_angle,
            self.ax_trajectory_diagnostics,
        )
        self.trajectory_canvas.draw_idle()
        self._set_text_output(self.trajectory_output, self._format_trajectory_summary(result))
        self._add_trajectory_runtime_hints(result)
        converged_count = int(result.frame["converged"].sum())
        if converged_count == len(result.frame):
            self.status_text.set(
                f"Траекторный расчёт готов: {len(result.frame)} точек, {result.elapsed_ms:.0f} мс."
            )
        else:
            self.status_text.set(
                f"Траекторный расчёт готов с ошибками: {converged_count}/{len(result.frame)} точек, "
                f"{result.elapsed_ms:.0f} мс."
            )
        if self.rashba_source_label.get() == RASHBA_SOURCE_TRAJECTORY:
            self._update_rashba_surface()

    def _display_trajectory_error(self, error: Exception) -> None:
        self._latest_trajectory_payload = None
        for axis in (self.ax_trajectory_phase, self.ax_trajectory_angle, self.ax_trajectory_diagnostics):
            axis.clear()
            axis.text(0.05, 0.95, f"Ошибка: {error}", transform=axis.transAxes, va="top", ha="left")
            axis.grid(True, which="both")
        steps_axis = getattr(self.ax_trajectory_diagnostics, "_trajectory_steps_axis", None)
        if steps_axis is not None:
            steps_axis.clear()
        if self.trajectory_canvas is not None:
            self.trajectory_canvas.draw_idle()
        self._set_text_output(self.trajectory_output, f"[Ошибка траекторного расчёта]\n{error}\n")
        self.status_text.set(f"Ошибка траекторного расчёта: {error}")

    def _format_boundary_summary(self, curves, point_result) -> str:
        beta_text = "не реализуется" if point_result.transmission_angle_deg is None else f"{point_result.transmission_angle_deg:.4g}°"
        k_ratio_text = "не определён" if point_result.wavevector_ratio is None else f"{point_result.wavevector_ratio:.6g}"
        min_reflection = float(np.nanmin(curves.reflection_coefficient))
        max_reflection = float(np.nanmax(curves.reflection_coefficient))
        finite_beta_mask = np.isfinite(curves.transmission_angle_deg)
        if np.any(finite_beta_mask):
            beta_range = (
                f"{float(np.nanmin(curves.transmission_angle_deg[finite_beta_mask])):.4g}° .. "
                f"{float(np.nanmax(curves.transmission_angle_deg[finite_beta_mask])):.4g}°"
            )
        else:
            beta_range = "для выбранного диапазона энергий β не реализуется"

        return (
            "[Мини-утилита отражения от границы]\n"
            f"Диапазон энергий: {float(curves.energies_eV[0]):.6g} .. {float(curves.energies_eV[-1]):.6g} эВ "
            f"({len(curves.energies_eV)} точек)\n"
            f"A = {point_result.work_function_eV:.6g} эВ\n"
            f"α падения = {point_result.incidence_angle_deg:.6g}°\n"
            f"Выбранная точка: E = {point_result.energy_eV:.6g} эВ\n"
            f"β после прохождения = {beta_text}\n"
            f"k'/k = {k_ratio_text}\n"
            f"R = {point_result.reflection_coefficient:.6g}\n"
            f"Режим: {point_result.regime}\n"
            "\n"
            f"По диапазону: R в пределах {min_reflection:.6g} .. {max_reflection:.6g}\n"
            f"Диапазон реализуемых β: {beta_range}\n"
        )

    def _format_rashba_summary(self, result: RashbaSurfaceResult) -> str:
        request = result.request
        frame = result.frame

        def value_range(column: str) -> str:
            values = frame[column].to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                return "нет корректных точек"
            return f"{float(np.nanmin(finite_values)):.6g} .. {float(np.nanmax(finite_values)):.6g}"

        return (
            "[Рашба-поверхность]\n"
            f"Источник Ver: {self.rashba_source_label.get()}\n"
            f"E={request.energy_min_eV:.6g} .. {request.energy_max_eV:.6g} эВ, N={request.point_count}\n"
            f"d={request.layer_thickness_ang:.6g} Å, α_R={request.rashba_alpha_au:.6g} а.е., "
            f"θ={request.emission_angle_deg:.6g}°, U={request.surface_potential_eV:.6g} эВ\n"
            "\n"
            f"Ver(+→-): {value_range('ver_up_to_down')}\n"
            f"Ver(-→+): {value_range('ver_down_to_up')}\n"
            f"T_+^2: {value_range('transmission_up')}\n"
            f"T_-^2: {value_range('transmission_down')}\n"
            f"t_+^2: {value_range('t_plus_sq')}\n"
            f"t_-^2: {value_range('t_minus_sq')}\n"
            f"P: {value_range('polarization')}\n"
        )

    def _format_trajectory_summary(self, result: TrajectorySweepResult) -> str:
        frame = result.frame
        request = result.request
        converged_mask = frame["converged"].astype(bool).to_numpy()
        converged_count = int(frame["converged"].sum())
        failed_count = len(frame) - converged_count
        successful_frame = frame[converged_mask]
        has_success = not successful_frame.empty
        last = successful_frame.iloc[-1] if has_success else frame.iloc[-1]
        last_label = "Последняя успешная точка" if has_success else "Последняя точка"

        def value_range(column: str, suffix: str = "") -> str:
            values = frame[column].to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                return "нет успешных точек"
            return f"{float(np.nanmin(finite_values)):.6g}{suffix} .. {float(np.nanmax(finite_values)):.6g}{suffix}"

        steps_values = frame["steps"].to_numpy(dtype=float)
        finite_steps = steps_values[np.isfinite(steps_values)]
        if finite_steps.size:
            steps_range = f"{int(np.nanmin(finite_steps))} .. {int(np.nanmax(finite_steps))}"
        else:
            steps_range = "нет успешных точек"
        refinements_values = frame["refinements"].to_numpy(dtype=float)
        finite_refinements = refinements_values[np.isfinite(refinements_values)]
        refinements_text = str(int(np.nanmax(finite_refinements))) if finite_refinements.size else "нет"
        error_text = ""
        if failed_count:
            failed_status = str(frame.loc[~frame["converged"].astype(bool), "status"].iloc[0])
            error_text = f"Ошибок: {failed_count}. Первая ошибка: {failed_status}\n"

        return (
            "[Траекторный расчёт]\n"
            f"Режим: {TRAJECTORY_SWEEP_LABELS[request.sweep_mode]}, точек: {len(frame)}\n"
            f"Z={request.atomic_number:.6g}, масса={request.mass_amu:.8g} а.е.м, b={request.b_bohr:.6g} a0\n"
            f"r0={request.r0_ang:.6g} Å, min steps={request.min_steps}, max refinements={request.max_refinements}\n"
            f"L={request.orbital_l}, M={'random' if request.random_m else request.magnetic_m}\n"
            "\n"
            f"ϕ: {value_range('phase_rad', ' рад')}\n"
            f"θ: {value_range('theta_deg', '°')}\n"
            f"φ: {value_range('trajectory_phi_deg', '°')}\n"
            f"r_min: {value_range('r_min_ang', ' Å')}\n"
            f"Внутренние шаги интегрирования: {steps_range}, max уточнений dt: {refinements_text}\n"
            f"Сошлось по правилу steps >= {request.min_steps}: {converged_count}/{len(frame)}\n"
            f"{error_text}"
            f"Время расчёта: {result.elapsed_ms:.3g} мс\n"
            "\n"
            f"{last_label}:\n"
            f"E={float(last['energy_eV']):.6g} эВ, r_п={float(last['impact_parameter_ang']):.6g} Å, "
            f"dθ={float(last['angle_step_deg']):.6g}°\n"
            f"ϕ={float(last['phase_rad']):.6g} рад, θ={float(last['theta_deg']):.6g}°, "
            f"φ={float(last['trajectory_phi_deg']):.6g}°\n"
            f"P(no flip | ↑)={float(last['p_no_flip_initial_up']):.6g}, "
            f"P(no flip | ↓)={float(last['p_no_flip_initial_down']):.6g}\n"
            f"status={last['status']}\n"
        )

    def _append_output(self, text: str) -> None:
        target = self.spectrum_output or self.output
        if target is None:
            return
        target.insert(tk.END, text)
        target.see(tk.END)

    def _export_spectrum_data(self) -> None:
        payload = self._latest_plot_payload
        if payload is None:
            self.status_text.set("Нет рассчитанного спектра для экспорта. Сначала постройте графики.")
            self._append_output("\n[Экспорт] Нет рассчитанного спектра для экспорта.\n")
            return

        default_name = self._default_export_name(payload)
        selected_path = filedialog.asksaveasfilename(
            parent=self,
            title="Экспорт спектра",
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[
                ("JSON", "*.json"),
                ("Excel", "*.xlsx"),
                ("XML", "*.xml"),
                ("Все файлы", "*.*"),
            ],
        )
        if not selected_path:
            return

        selected = Path(selected_path)
        base_path = selected.parent / selected.stem if selected.suffix else selected
        result = payload.result
        energies_eV = result.grid["E_eV"].to_numpy(dtype=float)

        try:
            exported = export_spectrum_bundle(
                base_path=base_path,
                energies_eV=energies_eV,
                spin_curves=result.spin_curves,
                metadata=self._build_spectrum_export_metadata(payload),
            )
        except Exception as ex:
            logger.exception("EXPORT | ошибка экспорта спектра")
            self.status_text.set(f"Ошибка экспорта: {ex}")
            self._append_output(f"\n[Экспорт] Ошибка: {ex}\n")
            return

        exported_summary = ", ".join(f"{kind.upper()}={path.name}" for kind, path in exported.items())
        self.status_text.set(f"Экспорт выполнен: {base_path.stem}")
        self._append_output(f"\n[Экспорт] {exported_summary}\n")

    def _default_export_name(self, payload: PlotComputationResult) -> str:
        variant = payload.request.formula_variant.replace("_", "-")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"spectrum-{variant}-{timestamp}"

    def _build_spectrum_export_metadata(self, payload: PlotComputationResult) -> dict[str, object]:
        request = payload.request
        result = payload.result
        metadata: dict[str, object] = {
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "formula_label": request.formula_label,
            "formula_variant": request.formula_variant,
            "orbital_l": request.geometry.orbital_l,
            "energy_min_eV": request.phase_request.Emin_eV,
            "energy_max_eV": request.phase_request.Emax_eV,
            "energy_point_count": request.phase_request.N,
            "used_atom_count": len(request.phase_request.a_list_ang),
            "all_atom_count": request.all_atom_count,
            "lattice_constant_ang": request.geometry.lattice_constant_ang,
            "interaction_radius_ang": request.geometry.interaction_radius_ang,
            "impact_parameter_max_ang": max(request.phase_request.a_list_ang),
            "source_depth_layer": request.geometry.source_depth,
            "alpha_deg": request.geometry.alpha_deg,
            "beta_deg": request.geometry.beta_deg,
            "chi_model": "table" if request.use_table_chi else "exp",
            "i3_mode": "sum" if request.i3_mode_sum else "trapz",
        }
        if result.fixed_lz is not None:
            metadata["fixed_lz"] = result.fixed_lz
        else:
            metadata["lz_chain"] = list(result.lz_chain)
        return metadata

    def _export_trajectory_data(self) -> None:
        payload = self._latest_trajectory_payload
        if payload is None:
            self.status_text.set("Нет траекторного расчёта для экспорта. Сначала нажмите Рассчитать.")
            self._set_text_output(self.trajectory_output, "[Экспорт]\nНет траекторного расчёта для экспорта.\n")
            return

        selected_path = filedialog.asksaveasfilename(
            parent=self,
            title="Экспорт траекторного расчёта",
            initialfile=self._default_trajectory_export_name(payload),
            defaultextension=".json",
            filetypes=[
                ("JSON", "*.json"),
                ("Excel", "*.xlsx"),
                ("XML", "*.xml"),
                ("Все файлы", "*.*"),
            ],
        )
        if not selected_path:
            return

        selected = Path(selected_path)
        base_path = selected.parent / selected.stem if selected.suffix else selected
        try:
            exported = export_trajectory_bundle(
                base_path=base_path,
                frame=payload.frame,
                metadata=trajectory_export_metadata(payload),
            )
        except Exception as ex:
            logger.exception("EXPORT | ошибка экспорта траекторного расчёта")
            self.status_text.set(f"Ошибка экспорта траекторного расчёта: {ex}")
            if self.trajectory_output is not None:
                self.trajectory_output.insert(tk.END, f"\n[Экспорт] Ошибка: {ex}\n")
                self.trajectory_output.see(tk.END)
            return

        exported_summary = ", ".join(f"{kind.upper()}={path.name}" for kind, path in exported.items())
        self.status_text.set(f"Экспорт траекторного расчёта выполнен: {base_path.stem}")
        if self.trajectory_output is not None:
            self.trajectory_output.insert(tk.END, f"\n[Экспорт] {exported_summary}\n")
            self.trajectory_output.see(tk.END)

    def _default_trajectory_export_name(self, payload: TrajectorySweepResult) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"trajectory-{payload.request.sweep_mode}-{timestamp}"

    def _refresh_geometry_preview(self, geometry: GeometryContext, atom_selection, search_region: LatticeSearchRegion) -> None:
        if (
            self.geometry_fig is None
            or self.geometry_canvas is None
            or self.ax_geometry_3d is None
            or self.ax_geometry_xz is None
            or self.ax_geometry_xy is None
        ):
            return

        try:
            preview = build_geometry_preview_data(geometry, atom_selection, search_region=search_region)
            draw_geometry_preview(self.ax_geometry_3d, self.ax_geometry_xz, self.ax_geometry_xy, preview)
            self.geometry_fig.tight_layout(pad=1.2)
            self.geometry_canvas.draw_idle()
        except Exception:
            logger.exception("GEOMETRY_PREVIEW | ошибка обновления схемы")

    def _active_scroll_canvas_set(self, canvas: tk.Canvas | None) -> None:
        self._active_scroll_canvas = canvas

    def _on_controls_mousewheel(self, event) -> None:
        if self._active_scroll_canvas is None:
            return
        if getattr(event, "num", None) == 4:
            scroll_units = -3
        elif getattr(event, "num", None) == 5:
            scroll_units = 3
        else:
            scroll_units = -1 * int(event.delta / 120) if event.delta else 0
        if scroll_units:
            self._active_scroll_canvas.yview_scroll(scroll_units, "units")

    def update_output_left(self) -> None:
        geometry = self._current_geometry()
        search_region, region_summary = self._current_search_region(geometry)
        atom_selection = collect_atom_selection(geometry, search_region=search_region)
        matrices, inverses = build_transition_matrices(source_orbital_l=geometry.orbital_l)

        logger.info(
            "LEFT | a=%.4f Å, R_bohr=%.4f Å, interaction_radius=%.4f Å, "
            "alpha=%.4f deg (%.4f rad), beta=%.4f deg (%.4f rad), n=%d, d=%d (index=%d), L=%d",
            geometry.lattice_constant_ang,
            geometry.bohr_radius_ang,
            geometry.interaction_radius_ang,
            geometry.alpha_deg,
            geometry.alpha_rad,
            geometry.beta_deg,
            geometry.beta_rad,
            geometry.lattice_radius,
            geometry.source_depth,
            geometry.source_layer,
            geometry.orbital_l,
        )
        logger.info("LEFT | найдено атомов: %d", len(atom_selection.all_atoms))

        for lz, matrix in matrices.items():
            logger.info(
                "LEFT | D(Lz=%d), det=%.6e, invertible=%s",
                lz,
                np.linalg.det(matrix),
                inverses[lz] is not None,
            )

        preview_lines = [
            (
                f"{np.array2string(atom['coords'], precision=2, suppress_small=True)} -> "
                f"d_прямой={atom['distance_to_line']:.2f} Å, "
                f"d_исток={atom['distance_to_origin']:.2f} Å, "
                f"s={atom['longitudinal_distance']:.2f} Å"
            )
            for atom in atom_selection.all_atoms[:10]
        ]
        if len(atom_selection.all_atoms) > 10:
            preview_lines.append("...")

        matrix_blocks = []
        for lz, matrix in matrices.items():
            block = [f"\nLz = {lz}:\n{np.array2string(matrix, precision=4, suppress_small=True)}"]
            inverse = inverses[lz]
            if inverse is not None:
                block.append("\nD^-1:\n" + np.array2string(inverse, precision=4, suppress_small=True))
            else:
                block.append("\nОбратная матрица не существует.")
            matrix_blocks.append("".join(block))

        text = (
            f"{region_summary}\n\n"
            f"Глубина источника: d={geometry.source_depth} (поверхность z=0, старт z={geometry.source_z_ang:.2f} Å)\n\n"
            f"Ближайшие атомы (расстояние до прямой ≤ {geometry.interaction_radius_ang:.2f} Å):\n"
            f"Всего найдено: {len(atom_selection.all_atoms)}\n"
            + "\n".join(preview_lines)
            + "\n\n"
            + f"Параметры направления: α={geometry.alpha_deg:.2f}° ({geometry.alpha_rad:.4f} рад), "
              f"β={geometry.beta_deg:.2f}° ({geometry.beta_rad:.4f} рад)\n"
            + f"\nМатрицы переходов D (L = {geometry.orbital_l}):\n"
            + "".join(matrix_blocks)
            + "\n"
        )

        self._refresh_geometry_preview(geometry, atom_selection, search_region)

        if self.geometry_output is not None:
            self.geometry_output.delete(1.0, tk.END)
            self.geometry_output.insert(tk.END, text)
            self.geometry_output.see(tk.END)
        self.status_text.set(f"Геометрия обновлена: найдено {len(atom_selection.all_atoms)} атомов.")

    def update_output_right(self) -> None:
        self._schedule_right_update(delay_ms=0)

    def _on_close(self) -> None:
        self._closing = True
        if self._scheduled_left_after is not None:
            self.after_cancel(self._scheduled_left_after)
            self._scheduled_left_after = None
        if self._scheduled_right_after is not None:
            self.after_cancel(self._scheduled_right_after)
            self._scheduled_right_after = None
        if self._scheduled_trajectory_after is not None:
            self.after_cancel(self._scheduled_trajectory_after)
            self._scheduled_trajectory_after = None
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._trajectory_executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()


__all__ = ["App", "configure_logging"]
