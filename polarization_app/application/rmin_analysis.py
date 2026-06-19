# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polarization_app.physics.phase_integrals import BOHR_TO_ANGSTROM
from polarization_app.physics.trajectory_phase import DEFAULT_THOMAS_FERMI_B_BOHR


@dataclass(frozen=True)
class RminAnalysisMetrics:
    total_count: int
    successful_count: int
    failed_count: int
    unstable_count: int
    r_tf_ang: float | None
    inside_tf_count: int
    inside_tf_fraction: float
    p_max: float | None
    p_max_rmin_ang: float | None
    p_over_half_min_ang: float | None
    p_over_half_max_ang: float | None
    convergence_ok: bool


def compute_rmin_analysis_metrics(frame) -> RminAnalysisMetrics:
    total_count = int(len(frame))
    if total_count == 0:
        return RminAnalysisMetrics(
            total_count=0,
            successful_count=0,
            failed_count=0,
            unstable_count=0,
            r_tf_ang=None,
            inside_tf_count=0,
            inside_tf_fraction=0.0,
            p_max=None,
            p_max_rmin_ang=None,
            p_over_half_min_ang=None,
            p_over_half_max_ang=None,
            convergence_ok=False,
        )

    if "converged" in frame:
        converged = frame["converged"].fillna(False).astype(bool).to_numpy()
    else:
        converged = np.ones(total_count, dtype=bool)

    r_values = _column_to_float_array(frame, "r_min_ang")
    finite_r = np.isfinite(r_values)
    valid = converged & finite_r
    successful_count = int(valid.sum())
    failed_count = int(total_count - int(converged.sum()))

    if "convergence_unstable" in frame:
        unstable = frame["convergence_unstable"].fillna(False).astype(bool).to_numpy()
    else:
        unstable = np.zeros(total_count, dtype=bool)
    unstable_count = int((unstable & valid).sum())

    r_tf_ang = thomas_fermi_radius_ang_from_frame(frame)
    inside_tf_count = 0
    inside_tf_fraction = 0.0
    if r_tf_ang is not None and successful_count:
        inside_tf_count = int((r_values[valid] <= r_tf_ang).sum())
        inside_tf_fraction = inside_tf_count / successful_count

    p_up = _column_to_float_array(frame, "p_flip_initial_up")
    p_down = _column_to_float_array(frame, "p_flip_initial_down")
    p_max_by_point = np.fmax(p_up, p_down)
    valid_probability = valid & np.isfinite(p_max_by_point)

    p_max: float | None = None
    p_max_rmin_ang: float | None = None
    p_over_half_min_ang: float | None = None
    p_over_half_max_ang: float | None = None
    if np.any(valid_probability):
        valid_indices = np.flatnonzero(valid_probability)
        best_index = int(valid_indices[np.nanargmax(p_max_by_point[valid_probability])])
        p_max = float(p_max_by_point[best_index])
        p_max_rmin_ang = float(r_values[best_index])
        over_half = valid_probability & (p_max_by_point >= 0.5)
        if np.any(over_half):
            p_over_half_min_ang = float(np.nanmin(r_values[over_half]))
            p_over_half_max_ang = float(np.nanmax(r_values[over_half]))

    return RminAnalysisMetrics(
        total_count=total_count,
        successful_count=successful_count,
        failed_count=failed_count,
        unstable_count=unstable_count,
        r_tf_ang=r_tf_ang,
        inside_tf_count=inside_tf_count,
        inside_tf_fraction=inside_tf_fraction,
        p_max=p_max,
        p_max_rmin_ang=p_max_rmin_ang,
        p_over_half_min_ang=p_over_half_min_ang,
        p_over_half_max_ang=p_over_half_max_ang,
        convergence_ok=failed_count == 0 and unstable_count == 0 and successful_count > 0,
    )


def thomas_fermi_radius_ang_from_frame(frame) -> float | None:
    if "atomic_number" not in frame:
        return None
    atomic_numbers = frame["atomic_number"].to_numpy(dtype=float)
    atomic_numbers = atomic_numbers[np.isfinite(atomic_numbers) & (atomic_numbers > 0.0)]
    if atomic_numbers.size == 0:
        return None
    z_value = float(atomic_numbers[0])
    return DEFAULT_THOMAS_FERMI_B_BOHR * BOHR_TO_ANGSTROM / (z_value ** (1.0 / 3.0))


def format_rmin_analysis_report(metrics: RminAnalysisMetrics) -> str:
    if metrics.total_count == 0:
        return (
            "[Анализ r_min / Thomas-Fermi]\n"
            "Нет данных. Сначала выполните траекторный расчёт с sweep по r_п или энергии.\n"
        )

    r_tf_text = "не определён" if metrics.r_tf_ang is None else f"{metrics.r_tf_ang:.6g} Å"
    p_max_text = "нет данных" if metrics.p_max is None else f"{metrics.p_max:.6g}"
    p_max_r_text = "нет данных" if metrics.p_max_rmin_ang is None else f"{metrics.p_max_rmin_ang:.6g} Å"
    p_half_range = _format_optional_range(metrics.p_over_half_min_ang, metrics.p_over_half_max_ang)
    inside_percent = 100.0 * metrics.inside_tf_fraction
    convergence_text = "OK" if metrics.convergence_ok else "требует проверки"

    return (
        "[Анализ r_min / Thomas-Fermi]\n"
        f"Точек: {metrics.total_count}, успешно: {metrics.successful_count}, ошибок: {metrics.failed_count}\n"
        f"r_TF = {r_tf_text}; точек внутри r_TF: "
        f"{metrics.inside_tf_count}/{metrics.successful_count} ({inside_percent:.3g}%)\n"
        f"Pmax = {p_max_text} при r_min = {p_max_r_text}\n"
        f"Диапазон P >= 0.5: {p_half_range}\n"
        f"Неустойчивых точек по сходимости: {metrics.unstable_count}\n"
        f"Сходимость: {convergence_text}\n"
        "\n"
        "Вывод для записки:\n"
        "График показывает зависимость вероятности изменения спина от минимального расстояния "
        "сближения r_min. Вертикальная отметка r_TF = b*a0*Z^(-1/3) задаёт радиус "
        "экранирования Томаса-Ферми. Если максимум P расположен около или внутри области r_min <= r_TF, "
        "это согласуется с идеей, что наиболее сильное спин-орбитальное взаимодействие возникает "
        "в области сильного экранированного поля атома.\n"
    )


def format_rmin_metrics_panel(metrics: RminAnalysisMetrics) -> str:
    if metrics.total_count == 0:
        return "Нет данных.\nСначала выполните траекторный расчёт."

    r_tf_text = "—" if metrics.r_tf_ang is None else f"{metrics.r_tf_ang:.6g} Å"
    p_max_text = "—" if metrics.p_max is None else f"{metrics.p_max:.6g}"
    p_half_range = _format_optional_range(metrics.p_over_half_min_ang, metrics.p_over_half_max_ang)
    inside_percent = 100.0 * metrics.inside_tf_fraction
    convergence_text = "OK" if metrics.convergence_ok else "внимание"
    return (
        f"r_TF: {r_tf_text}\n"
        f"Pmax: {p_max_text}\n"
        f"r при P>=0.5: {p_half_range}\n"
        f"внутри r_TF: {metrics.inside_tf_count}/{metrics.successful_count} ({inside_percent:.3g}%)\n"
        f"ошибок: {metrics.failed_count}\n"
        f"неустойчивых: {metrics.unstable_count}\n"
        f"сходимость: {convergence_text}"
    )


def _column_to_float_array(frame, column: str) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), np.nan, dtype=float)
    return frame[column].to_numpy(dtype=float)


def _format_optional_range(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "не найден"
    if abs(low - high) <= max(abs(low), abs(high), 1.0) * 1e-12:
        return f"{low:.6g} Å"
    return f"{low:.6g} .. {high:.6g} Å"
