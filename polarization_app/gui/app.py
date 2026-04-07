# -*- coding: utf-8 -*-
import logging
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from tkinter import ttk

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
from polarization_app.domain.lattice import LatticeSearchRegion, estimate_lattice_search_region
from polarization_app.domain.transitions import build_transition_matrices
from polarization_app.gui.plotting import (
    build_geometry_preview_data,
    capture_view_limits,
    draw_geometry_preview,
    draw_spin_plots,
    restore_view_limits,
    zoom_axis,
    zoom_3d_axis,
    zoom_axis_around_point,
)
from polarization_app.physics.phase_integrals import exponential_chi, interpolate_thomas_fermi_chi


logger = logging.getLogger(__name__)
CONTROL_PANEL_WIDTH = 360
CONTROL_WRAP_LENGTH = 300


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

        self._recompute_lattice_radius()
        self._update_formula_hint()
        self.update_output_left()
        self.after(0, self.update_output_right)

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
        self.status_text = tk.StringVar(value="Готово.")
        self.formula_hint_text = tk.StringVar(value="")

        self.geometry_output: tk.Text | None = None
        self.spectrum_output: tk.Text | None = None
        self.output: tk.Text | None = None
        self.n_auto_label: ttk.Label | None = None
        self.ax_sum = None
        self.ax_spin = None
        self.ax_geometry_3d = None
        self.ax_geometry_xz = None
        self.ax_geometry_xy = None
        self.canvas: FigureCanvasTkAgg | None = None
        self.fig: Figure | None = None
        self.geometry_canvas: FigureCanvasTkAgg | None = None
        self.geometry_fig: Figure | None = None
        self._default_view_limits = None
        self._scheduled_left_after: str | None = None
        self._scheduled_right_after: str | None = None
        self._running_future: Future | None = None
        self._queued_request: PlotComputationRequest | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="polarization")
        self._closing = False
        self._geometry_change_in_progress = False

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew")

        geometry_tab = ttk.Frame(notebook)
        spectrum_tab = ttk.Frame(notebook)
        notebook.add(geometry_tab, text="Геометрия и переходы")
        notebook.add(spectrum_tab, text="Спектры и формулы")

        self._build_geometry_tab(geometry_tab)
        self._build_spectrum_tab(spectrum_tab)

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
        ttk.Label(
            actions,
            text="Колесо мыши масштабирует график.",
            foreground="#555",
            wraplength=150,
            justify="left",
        ).pack(side="left", padx=(10, 0))

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

    def _make_slider(self, parent, label, variable, min_value, max_value, row, description="", resolution=0.01):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        frame.columnconfigure(0, weight=1)

        label_frame = ttk.Frame(frame)
        label_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(label_frame, text=label).grid(row=0, column=0, sticky="w")
        if description:
            ttk.Label(
                label_frame,
                text=f"({description})",
                foreground="#555",
                wraplength=CONTROL_WRAP_LENGTH,
                justify="left",
            ).grid(row=1, column=0, sticky="w")

        slider = ttk.Scale(frame, from_=min_value, to=max_value, orient="horizontal", variable=variable)
        slider.grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=(2, 0))

        def format_value(value):
            try:
                if isinstance(variable, tk.IntVar) or resolution >= 1:
                    return f"{int(round(float(value)))}"
                return f"{float(value):.3g}"
            except Exception:
                return str(value)

        value_label = ttk.Label(frame, text=format_value(variable.get()))
        value_label.grid(row=1, column=1, sticky="e", pady=(2, 0))
        variable.trace_add("write", lambda *_: value_label.config(text=format_value(variable.get())))

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

    def _display_right_error(self, error: Exception) -> None:
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

    def _reset_zoom(self) -> None:
        if not self._default_view_limits:
            return
        try:
            restore_view_limits(self._default_view_limits)
            self.canvas.draw_idle()
        except Exception as ex:
            logger.exception("RESET_ZOOM | ошибка")
            self._append_output(f"\n[Reset Zoom] Ошибка: {ex}\n")

    def _append_output(self, text: str) -> None:
        target = self.spectrum_output or self.output
        if target is None:
            return
        target.insert(tk.END, text)
        target.see(tk.END)

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

    def _on_controls_frame_configure(self, _event) -> None:
        if self._controls_canvas is not None:
            self._controls_canvas.configure(scrollregion=self._controls_canvas.bbox("all"))

    def _on_controls_canvas_configure(self, event) -> None:
        if self._controls_canvas is not None and self._controls_window_id is not None:
            self._controls_canvas.itemconfigure(self._controls_window_id, width=event.width)

    def _bind_controls_mousewheel(self, _event) -> None:
        self.bind_all("<MouseWheel>", self._on_controls_mousewheel)

    def _unbind_controls_mousewheel(self, _event) -> None:
        self.unbind_all("<MouseWheel>")

    def _on_controls_mousewheel(self, event) -> None:
        if self._controls_canvas is None:
            return
        scroll_units = -1 if event.delta > 0 else 1
        self._controls_canvas.yview_scroll(scroll_units, "units")

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
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()


__all__ = ["App", "configure_logging"]
