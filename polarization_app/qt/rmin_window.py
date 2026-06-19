# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from polarization_app.application.rmin_analysis import (
    RminAnalysisMetrics,
    compute_rmin_analysis_metrics,
    format_rmin_analysis_report,
)
from polarization_app.application.trajectory import (
    DEFAULT_PRECISE_TRAJECTORY_MAX_PHASE_STEP_RAD,
    DEFAULT_PRECISE_TRAJECTORY_MIN_STEPS,
    DEFAULT_TRAJECTORY_MAX_PHASE_STEP_RAD,
    DEFAULT_TRAJECTORY_MIN_STEPS,
    TRAJECTORY_SWEEP_IMPACT,
    TrajectorySweepRequest,
    TrajectorySweepResult,
    execute_trajectory_sweep,
    trajectory_export_metadata,
)
from polarization_app.application.trajectory_export import export_trajectory_file
from polarization_app.gui.plotting import draw_rmin_analysis_plots
from polarization_app.gui.theme import THEMES, AppTheme
from polarization_app.physics.compute_backend import cpu_worker_count
from polarization_app.physics.trajectory_phase import ELECTRON_MASS_AMU


try:
    from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
    from PySide6.QtGui import QAction, QIcon
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QSplitter,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

    PYSIDE6_AVAILABLE = True
    PYSIDE6_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised through availability checks.
    PYSIDE6_AVAILABLE = False
    PYSIDE6_IMPORT_ERROR = exc


def is_pyside6_available() -> bool:
    return PYSIDE6_AVAILABLE


def build_default_request() -> TrajectorySweepRequest:
    return TrajectorySweepRequest(
        sweep_mode=TRAJECTORY_SWEEP_IMPACT,
        point_count=600,
        atomic_number=82.0,
        mass_amu=0.0001,
        energy_eV=407.0,
        energy_min_eV=100.0,
        energy_max_eV=1000.0,
        impact_parameter_ang=0.2,
        impact_min_ang=0.05,
        impact_max_ang=0.5,
        r0_ang=3.0,
        angle_step_deg=3.0,
        orbital_l=6,
        magnetic_m=2,
        random_m=False,
        min_steps=DEFAULT_PRECISE_TRAJECTORY_MIN_STEPS,
        max_refinements=6,
        precise_mode=True,
        convergence_check=True,
        max_phase_step_rad=DEFAULT_PRECISE_TRAJECTORY_MAX_PHASE_STEP_RAD,
        parallel_workers=cpu_worker_count(),
    )


def build_qss(theme: AppTheme | None = None) -> str:
    theme = theme or THEMES["light"]
    return f"""
    QMainWindow {{
        background: {theme.background};
        color: {theme.text};
        font-family: Segoe UI, Arial, sans-serif;
        font-size: 10pt;
    }}
    QToolBar {{
        background: {theme.surface};
        border: 0;
        border-bottom: 1px solid {theme.border};
        spacing: 8px;
        padding: 8px 14px;
    }}
    QToolButton, QPushButton {{
        background: {theme.surface};
        color: {theme.text};
        border: 1px solid {theme.border};
        border-radius: 7px;
        padding: 7px 12px;
    }}
    QPushButton#primaryButton {{
        background: {theme.accent};
        color: {theme.on_accent};
        border-color: {theme.accent};
        font-weight: 600;
    }}
    QPushButton#primaryButton:hover {{
        background: {theme.accent_hover};
    }}
    QGroupBox, QFrame#card {{
        background: {theme.surface};
        color: {theme.text};
        border: 1px solid {theme.border};
        border-radius: 8px;
        margin-top: 10px;
        padding: 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 4px;
        font-weight: 700;
    }}
    QLabel#muted {{
        color: {theme.muted};
    }}
    QPlainTextEdit, QDoubleSpinBox, QSpinBox {{
        background: {theme.input_background};
        color: {theme.input_text};
        border: 1px solid {theme.border};
        border-radius: 5px;
        padding: 4px;
    }}
    QTableWidget {{
        background: {theme.surface};
        color: {theme.text};
        gridline-color: {theme.border};
        border: 1px solid {theme.border};
        border-radius: 6px;
    }}
    QHeaderView::section {{
        background: {theme.panel};
        color: {theme.text};
        border: 0;
        border-right: 1px solid {theme.border};
        padding: 6px;
        font-weight: 700;
    }}
    QStatusBar {{
        background: {theme.surface};
        color: {theme.muted};
        border-top: 1px solid {theme.border};
    }}
    """


if PYSIDE6_AVAILABLE:

    class TrajectoryWorker(QObject):
        finished = Signal(object)
        failed = Signal(str)

        def __init__(self, request: TrajectorySweepRequest):
            super().__init__()
            self._request = request

        @Slot()
        def run(self) -> None:
            try:
                self.finished.emit(execute_trajectory_sweep(self._request))
            except Exception as exc:
                self.failed.emit(str(exc))


    class RminAnalysisWindow(QMainWindow):
        def __init__(self, parent: QWidget | None = None):
            super().__init__(parent)
            self.setWindowTitle("Графики поляризации электрона - PySide6 прототип")
            self.resize(1620, 920)
            self._theme = THEMES["light"]
            self._latest_result: TrajectorySweepResult | None = None
            self._thread: QThread | None = None
            self._worker: TrajectoryWorker | None = None

            icon_path = Path(__file__).resolve().parents[1] / "assets" / "app_icon_256.png"
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))

            self.setStyleSheet(build_qss(self._theme))
            self._build_toolbar()
            self._build_body()
            self._update_empty_state()

        def _build_toolbar(self) -> None:
            toolbar = QToolBar("Основные действия")
            toolbar.setMovable(False)
            self.addToolBar(Qt.TopToolBarArea, toolbar)

            title = QLabel("Графики поляризации электрона")
            title.setStyleSheet("font-size: 13pt; font-weight: 700; padding-right: 20px;")
            toolbar.addWidget(title)
            toolbar.addSeparator()

            for label in ("Геометрия", "Спектры", "Траектория", "Рашба"):
                action = QAction(label, self)
                action.setEnabled(False)
                toolbar.addAction(action)
            active_action = QAction("Анализ r_min", self)
            active_action.setEnabled(False)
            toolbar.addAction(active_action)
            toolbar.addSeparator()

            self.run_button = QPushButton("Построить график")
            self.run_button.setObjectName("primaryButton")
            self.run_button.clicked.connect(self.run_calculation)
            toolbar.addWidget(self.run_button)

            copy_button = QPushButton("Скопировать вывод")
            copy_button.clicked.connect(self.copy_report)
            toolbar.addWidget(copy_button)

            png_button = QPushButton("Экспорт PNG")
            png_button.clicked.connect(self.export_png)
            toolbar.addWidget(png_button)

            xlsx_button = QPushButton("Экспорт XLSX")
            xlsx_button.clicked.connect(self.export_xlsx)
            toolbar.addWidget(xlsx_button)

            self.status_ready = QLabel("Расчёт готов")
            self.status_ready.setStyleSheet(f"color: {self._theme.success}; font-weight: 700; padding-left: 12px;")
            toolbar.addWidget(self.status_ready)

        def _build_body(self) -> None:
            splitter = QSplitter(Qt.Horizontal)
            splitter.setChildrenCollapsible(False)
            self.setCentralWidget(splitter)

            left_scroll = QScrollArea()
            left_scroll.setWidgetResizable(True)
            left_scroll.setFrameShape(QFrame.NoFrame)
            left_content = QWidget()
            self.left_layout = QVBoxLayout(left_content)
            self.left_layout.setContentsMargins(14, 14, 10, 14)
            self.left_layout.setSpacing(10)
            left_scroll.setWidget(left_content)
            splitter.addWidget(left_scroll)

            self._build_source_card()
            self._build_parameter_card()
            self._build_range_card()
            self._build_guides_card()
            self.left_layout.addStretch(1)

            center = QWidget()
            center_layout = QVBoxLayout(center)
            center_layout.setContentsMargins(8, 14, 8, 14)
            center_layout.setSpacing(10)
            splitter.addWidget(center)

            plot_title = QLabel("P(изменение спина) от минимального расстояния сближения")
            plot_title.setStyleSheet("font-size: 12pt; font-weight: 700;")
            center_layout.addWidget(plot_title)

            self.figure = Figure(figsize=(9.2, 6.8), dpi=100)
            grid = self.figure.add_gridspec(2, 2, height_ratios=[2.35, 1.05], hspace=0.48, wspace=0.22)
            self.ax_probability = self.figure.add_subplot(grid[0, :])
            self.ax_distribution = self.figure.add_subplot(grid[1, 0])
            self.ax_diagnostics = self.figure.add_subplot(grid[1, 1])
            self.canvas = FigureCanvas(self.figure)
            center_layout.addWidget(self.canvas, stretch=1)

            self.summary_table = QTableWidget(1, 7)
            self.summary_table.setHorizontalHeaderLabels(
                ["N точек", "r_TF (Å)", "Pmax", "r при P>0.5 (Å)", "точек внутри r_TF", "точек вне r_TF", "сходимость"]
            )
            self.summary_table.verticalHeader().setVisible(False)
            self.summary_table.setMinimumHeight(82)
            center_layout.addWidget(self.summary_table)

            formula = QLabel("ⓘ Методическая формула: r_TF = b*a0*Z^(-1/3), a0 = 0.529177 Å, b = 0.885341 a0.")
            formula.setObjectName("muted")
            center_layout.addWidget(formula)

            right = QWidget()
            right_layout = QVBoxLayout(right)
            right_layout.setContentsMargins(8, 14, 14, 14)
            right_layout.setSpacing(10)
            splitter.addWidget(right)

            self.report = QPlainTextEdit()
            self.report.setReadOnly(True)
            self._add_card(right_layout, "Вывод для записки", self.report, stretch=2)

            self.metrics_label = QLabel()
            self.metrics_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.metrics_label.setWordWrap(True)
            self._add_card(right_layout, "Ключевые метрики", self.metrics_label)

            self.validation_label = QLabel()
            self.validation_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.validation_label.setWordWrap(True)
            self._add_card(right_layout, "Проверки и валидация", self.validation_label)

            splitter.setSizes([330, 920, 360])

        def _build_source_card(self) -> None:
            card = self._group("1. Источник данных")
            layout = QVBoxLayout(card)
            label = QLabel("Экспериментальный PySide6-экран считает новый sweep по r_п через существующую модель траекторного расчёта.")
            label.setObjectName("muted")
            label.setWordWrap(True)
            layout.addWidget(label)
            self.left_layout.addWidget(card)

        def _build_parameter_card(self) -> None:
            card = self._group("2. Фиксированные параметры")
            layout = QGridLayout(card)
            request = build_default_request()
            self.z_spin = self._double_spin(1.0, 120.0, request.atomic_number, 1)
            self.energy_spin = self._double_spin(1.0, 10000.0, request.energy_eV, 1)
            self.mass_spin = self._double_spin(0.00001, 10.0, request.mass_amu, 0.0001, decimals=6)
            self.l_spin = self._int_spin(1, 20, request.orbital_l)
            self.m_spin = self._int_spin(-20, 20, request.magnetic_m)
            for row, (label, widget) in enumerate(
                (
                    ("Z (заряд ядра)", self.z_spin),
                    ("E (энергия), эВ", self.energy_spin),
                    ("Масса (а.е.м.)", self.mass_spin),
                    ("L (квант. число)", self.l_spin),
                    ("M (магн. кв. число)", self.m_spin),
                )
            ):
                layout.addWidget(QLabel(label), row, 0)
                layout.addWidget(widget, row, 1)
            self.left_layout.addWidget(card)

        def _build_range_card(self) -> None:
            card = self._group("3. Диапазон сближения")
            layout = QGridLayout(card)
            request = build_default_request()
            self.rmin_spin = self._double_spin(0.001, 20.0, request.impact_min_ang, 0.01)
            self.rmax_spin = self._double_spin(0.001, 20.0, request.impact_max_ang, 0.01)
            self.points_spin = self._int_spin(10, 5000, request.point_count)
            self.dtheta_spin = self._double_spin(0.01, 15.0, request.angle_step_deg, 0.01)
            for row, (label, widget) in enumerate(
                (
                    ("r_п min (Å)", self.rmin_spin),
                    ("r_п max (Å)", self.rmax_spin),
                    ("N точек", self.points_spin),
                    ("dθ (°)", self.dtheta_spin),
                )
            ):
                layout.addWidget(QLabel(label), row, 0)
                layout.addWidget(widget, row, 1)
            self.left_layout.addWidget(card)

        def _build_guides_card(self) -> None:
            card = self._group("4. Ориентиры на графике")
            layout = QVBoxLayout(card)
            self.show_tf = QCheckBox("показать r_TF")
            self.show_tf.setChecked(True)
            self.show_ba0 = QCheckBox("показать b*a0")
            self.show_ba0.setChecked(True)
            self.highlight_tf = QCheckBox("подсветить r_min <= r_TF")
            self.highlight_tf.setChecked(True)
            self.show_unstable = QCheckBox("показать неустойчивые точки")
            self.show_unstable.setChecked(True)
            for checkbox in (self.show_tf, self.show_ba0, self.highlight_tf, self.show_unstable):
                checkbox.stateChanged.connect(self.redraw)
                layout.addWidget(checkbox)
            self.left_layout.addWidget(card)

        def _add_card(self, parent_layout: QVBoxLayout, title: str, widget: QWidget, *, stretch: int = 0) -> None:
            card = QFrame()
            card.setObjectName("card")
            layout = QVBoxLayout(card)
            heading = QLabel(title)
            heading.setStyleSheet("font-weight: 700;")
            layout.addWidget(heading)
            layout.addWidget(widget)
            parent_layout.addWidget(card, stretch=stretch)

        def _group(self, title: str) -> QGroupBox:
            return QGroupBox(title)

        @staticmethod
        def _double_spin(minimum: float, maximum: float, value: float, step: float, *, decimals: int = 4) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
            spin.setValue(value)
            return spin

        @staticmethod
        def _int_spin(minimum: int, maximum: int, value: int) -> QSpinBox:
            spin = QSpinBox()
            spin.setRange(minimum, maximum)
            spin.setValue(value)
            return spin

        def _request_from_controls(self) -> TrajectorySweepRequest:
            precise = self.points_spin.value() >= 300
            return TrajectorySweepRequest(
                sweep_mode=TRAJECTORY_SWEEP_IMPACT,
                point_count=self.points_spin.value(),
                atomic_number=self.z_spin.value(),
                mass_amu=self.mass_spin.value(),
                energy_eV=self.energy_spin.value(),
                impact_parameter_ang=self.rmin_spin.value(),
                impact_min_ang=self.rmin_spin.value(),
                impact_max_ang=self.rmax_spin.value(),
                r0_ang=max(3.0, self.rmax_spin.value() * 6.0),
                angle_step_deg=self.dtheta_spin.value(),
                orbital_l=self.l_spin.value(),
                magnetic_m=self.m_spin.value(),
                random_m=False,
                min_steps=DEFAULT_PRECISE_TRAJECTORY_MIN_STEPS if precise else DEFAULT_TRAJECTORY_MIN_STEPS,
                max_refinements=6,
                precise_mode=precise,
                convergence_check=True,
                max_phase_step_rad=DEFAULT_PRECISE_TRAJECTORY_MAX_PHASE_STEP_RAD if precise else DEFAULT_TRAJECTORY_MAX_PHASE_STEP_RAD,
                parallel_workers=cpu_worker_count(),
            )

        @Slot()
        def run_calculation(self) -> None:
            request = self._request_from_controls()
            if request.impact_max_ang <= request.impact_min_ang:
                QMessageBox.warning(self, "Параметры", "r_п max должен быть больше r_п min.")
                return
            self.run_button.setEnabled(False)
            self.status_ready.setText("Расчёт идёт")
            self.statusBar().showMessage(f"Траекторный расчёт: {request.point_count} точек...")
            self._thread = QThread(self)
            self._worker = TrajectoryWorker(request)
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.finished.connect(self._on_result)
            self._worker.failed.connect(self._on_error)
            self._worker.finished.connect(self._thread.quit)
            self._worker.failed.connect(self._thread.quit)
            self._thread.finished.connect(self._thread.deleteLater)
            self._thread.start()

        @Slot(object)
        def _on_result(self, result: TrajectorySweepResult) -> None:
            self._latest_result = result
            self.run_button.setEnabled(True)
            self.status_ready.setText("Расчёт готов")
            self.statusBar().showMessage(f"Готово: {len(result.frame)} точек, {result.elapsed_ms:.0f} мс.")
            self.redraw()

        @Slot(str)
        def _on_error(self, message: str) -> None:
            self.run_button.setEnabled(True)
            self.status_ready.setText("Ошибка")
            self.statusBar().showMessage(message)
            QMessageBox.critical(self, "Ошибка расчёта", message)

        @Slot()
        def redraw(self) -> None:
            if self._latest_result is None:
                self._update_empty_state()
                return
            result = self._latest_result
            metrics = compute_rmin_analysis_metrics(result.frame)
            draw_rmin_analysis_plots(
                self.ax_probability,
                self.ax_distribution,
                self.ax_diagnostics,
                result.frame,
                show_tf=self.show_tf.isChecked(),
                show_ba0=self.show_ba0.isChecked(),
                highlight_tf=self.highlight_tf.isChecked(),
                show_unstable=self.show_unstable.isChecked(),
            )
            self.figure.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08, hspace=0.56, wspace=0.26)
            self.canvas.draw_idle()
            self.report.setPlainText(format_rmin_analysis_report(metrics))
            self.metrics_label.setText(self._format_metrics(metrics))
            self.validation_label.setText(self._format_validation(metrics, result))
            self._update_summary_table(metrics)

        def _update_empty_state(self) -> None:
            for axis in (self.ax_probability, self.ax_distribution, self.ax_diagnostics):
                axis.clear()
                axis.grid(True)
            self.ax_probability.set_title("P(изменение спина) от минимального расстояния сближения")
            self.ax_probability.text(0.5, 0.5, "Нажмите «Построить график»", transform=self.ax_probability.transAxes, ha="center", va="center")
            self.ax_distribution.set_title("Распределение точек относительно r_TF")
            self.ax_diagnostics.set_title("Диагностика шагов интегрирования")
            self.figure.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08, hspace=0.56, wspace=0.26)
            self.canvas.draw_idle()
            self.report.setPlainText("Нет данных. Настройте параметры слева и нажмите «Построить график».")
            self.metrics_label.setText("Нет данных.")
            self.validation_label.setText("Нет данных.")
            self._update_summary_table(None)

        @staticmethod
        def _format_metrics(metrics: RminAnalysisMetrics) -> str:
            r_tf = "—" if metrics.r_tf_ang is None else f"{metrics.r_tf_ang:.6g} Å"
            pmax = "—" if metrics.p_max is None else f"{metrics.p_max:.6g}"
            p_range = _format_optional_range(metrics.p_over_half_min_ang, metrics.p_over_half_max_ang)
            return (
                f"r_TF: {r_tf}\n"
                f"Pmax: {pmax}\n"
                f"r при P>0.5: {p_range}\n"
                f"точек внутри r_TF: {metrics.inside_tf_count}/{metrics.successful_count} "
                f"({100.0 * metrics.inside_tf_fraction:.3g}%)\n"
                f"сходимость: {'OK' if metrics.convergence_ok else 'внимание'}"
            )

        @staticmethod
        def _format_validation(metrics: RminAnalysisMetrics, result: TrajectorySweepResult) -> str:
            frame = result.frame
            probability_ok = True
            for column in ("p_flip_initial_up", "p_flip_initial_down"):
                if column in frame:
                    values = frame[column].dropna()
                    probability_ok &= bool(((values >= -1e-12) & (values <= 1.0 + 1e-12)).all())
            return (
                f"вероятности в [0,1]: {'OK' if probability_ok else 'внимание'}\n"
                f"ошибок: {metrics.failed_count}\n"
                f"неустойчивых точек: {metrics.unstable_count}\n"
                f"проверить d-θ: {'OK' if metrics.unstable_count == 0 else 'внимание'}"
            )

        def _update_summary_table(self, metrics: RminAnalysisMetrics | None) -> None:
            if metrics is None or metrics.total_count == 0:
                values = ["—"] * 7
            else:
                outside = max(metrics.successful_count - metrics.inside_tf_count, 0)
                values = [
                    str(metrics.successful_count),
                    _format_optional_number(metrics.r_tf_ang),
                    _format_optional_number(metrics.p_max),
                    _format_optional_range(metrics.p_over_half_min_ang, metrics.p_over_half_max_ang),
                    f"{metrics.inside_tf_count} ({100.0 * metrics.inside_tf_fraction:.3g}%)",
                    f"{outside} ({100.0 * outside / max(metrics.successful_count, 1):.3g}%)",
                    "OK" if metrics.convergence_ok else "внимание",
                ]
            for column, value in enumerate(values):
                self.summary_table.setItem(0, column, QTableWidgetItem(value))
            self.summary_table.resizeColumnsToContents()

        @Slot()
        def copy_report(self) -> None:
            QApplication.clipboard().setText(self.report.toPlainText())
            self.statusBar().showMessage("Вывод скопирован в буфер обмена.")

        @Slot()
        def export_png(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "Экспорт PNG", "rmin-analysis.png", "PNG (*.png)")
            if not path:
                return
            if not path.lower().endswith(".png"):
                path += ".png"
            self.figure.savefig(path, dpi=180, bbox_inches="tight")
            self.statusBar().showMessage(f"PNG сохранён: {Path(path).name}")

        @Slot()
        def export_xlsx(self) -> None:
            if self._latest_result is None:
                QMessageBox.information(self, "Экспорт", "Сначала выполните расчёт.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Экспорт XLSX", "rmin-analysis.xlsx", "Excel (*.xlsx)")
            if not path:
                return
            if not path.lower().endswith(".xlsx"):
                path += ".xlsx"
            exported = export_trajectory_file(path, self._latest_result.frame, trajectory_export_metadata(self._latest_result))
            self.statusBar().showMessage(f"XLSX сохранён: {exported.name}")


else:

    class RminAnalysisWindow:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(pyside6_missing_message())


def _format_optional_number(value: float | None) -> str:
    return "—" if value is None else f"{value:.6g}"


def _format_optional_range(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "—"
    if abs(low - high) <= max(abs(low), abs(high), 1.0) * 1e-12:
        return f"{low:.6g}"
    return f"{low:.6g} .. {high:.6g}"


def pyside6_missing_message() -> str:
    detail = "" if PYSIDE6_IMPORT_ERROR is None else f"\nИсходная ошибка: {PYSIDE6_IMPORT_ERROR}"
    return (
        "PySide6 не установлен. Установите экспериментальные зависимости командой:\n"
        "python -m pip install -r requirements-pyside6.txt"
        f"{detail}"
    )


def run(argv: list[str] | None = None) -> int:
    if not PYSIDE6_AVAILABLE:
        print(pyside6_missing_message(), file=sys.stderr)
        return 2
    app = QApplication(argv or sys.argv)
    window = RminAnalysisWindow()
    window.show()
    return int(app.exec())


__all__ = [
    "PYSIDE6_AVAILABLE",
    "RminAnalysisWindow",
    "build_default_request",
    "build_qss",
    "is_pyside6_available",
    "pyside6_missing_message",
    "run",
]
