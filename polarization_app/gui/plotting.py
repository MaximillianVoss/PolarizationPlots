# -*- coding: utf-8 -*-
from dataclasses import dataclass

import numpy as np

from polarization_app.application.geometry import AtomSelection, GeometryContext
from polarization_app.domain.lattice import LatticeSearchRegion, build_lattice_points, direction_from_spherical_angles
from polarization_app.physics.boundary_reflection import BoundaryPointResult, BoundaryReflectionCurves


@dataclass(frozen=True)
class GeometryPreviewData:
    lattice_points: np.ndarray
    source_layer_points: np.ndarray
    interacting_points: np.ndarray
    selected_points: np.ndarray
    origin: np.ndarray
    direction: np.ndarray
    trajectory_end: np.ndarray
    region_min: np.ndarray
    region_max: np.ndarray
    alpha_deg: float
    beta_deg: float
    source_layer: int
    source_depth: int
    surface_z_ang: float
    interaction_radius_ang: float


def zoom_axis(axis, factor: float) -> None:
    zoom_axis_around_point(axis, factor)


def zoom_3d_axis(axis, factor: float) -> None:
    axis.set_xlim(_zoom_limits(axis.get_xlim(), factor, axis.get_xscale(), None))
    axis.set_ylim(_zoom_limits(axis.get_ylim(), factor, axis.get_yscale(), None))
    axis.set_zlim(_zoom_limits(axis.get_zlim(), factor, axis.get_zscale(), None))


def zoom_axis_around_point(axis, factor: float, x_anchor: float | None = None, y_anchor: float | None = None) -> None:
    axis.set_xlim(_zoom_limits(axis.get_xlim(), factor, axis.get_xscale(), x_anchor))
    axis.set_ylim(_zoom_limits(axis.get_ylim(), factor, axis.get_yscale(), y_anchor))


def _zoom_limits(limits, factor: float, scale: str, anchor: float | None):
    low, high = limits
    if scale == "log":
        low = max(low, 1e-300)
        high = max(high, 1e-300)
        log_low = np.log10(low)
        log_high = np.log10(high)
        log_anchor = np.log10(max(anchor, 1e-300)) if anchor is not None and anchor > 0 else (log_low + log_high) / 2.0
        return 10 ** (log_anchor - (log_anchor - log_low) * factor), 10 ** (log_anchor + (log_high - log_anchor) * factor)

    center = anchor if anchor is not None and np.isfinite(anchor) else (low + high) / 2.0
    return center - (center - low) * factor, center + (high - center) * factor


def draw_spin_plots(sum_axis, spin_axis, energies_eV: np.ndarray, spin_curves: dict[str, np.ndarray]) -> None:
    sum_axis.clear()
    sum_axis.set_title("Проверочный спектр: P↑+P↓  (начальный ↑ и ↓)")
    sum_axis.plot(energies_eV, spin_curves["sum_check_up"], label="начальный ↑")
    sum_axis.plot(energies_eV, spin_curves["sum_check_dn"], label="начальный ↓")
    sum_axis.set_xlabel("Энергия, эВ")
    sum_axis.set_ylabel("Σ вероятностей")
    sum_axis.grid(True, which="both")
    sum_axis.legend()

    spin_axis.clear()
    spin_axis.set_title("Средний удвоенный спин: P↑−P↓  (начальный ↑ и ↓)")
    spin_axis.plot(energies_eV, spin_curves["spin_mean_up"], label="начальный ↑")
    spin_axis.plot(energies_eV, spin_curves["spin_mean_dn"], label="начальный ↓")
    spin_axis.set_xlabel("Энергия, эВ")
    spin_axis.set_ylabel("P↑ − P↓")
    spin_axis.grid(True, which="both")
    spin_axis.legend()


def draw_boundary_utility_plots(
    reflection_axis,
    angle_axis,
    curves: BoundaryReflectionCurves,
    selected_point: BoundaryPointResult,
) -> None:
    reflection_axis.clear()
    reflection_axis.set_title("Отражение от границы раздела")
    reflection_axis.plot(curves.energies_eV, curves.reflection_coefficient, label="R, коэффициент отражения")
    reflection_axis.scatter(
        [selected_point.energy_eV],
        [selected_point.reflection_coefficient],
        s=36,
        c="#cf2f2f",
        zorder=3,
    )
    reflection_axis.set_xlabel("Энергия, эВ")
    reflection_axis.set_ylabel("Коэффициент")
    reflection_axis.grid(True, which="both")
    reflection_axis.legend()

    angle_axis.clear()
    angle_axis.set_title("Угол после прохождения через границу")
    finite_mask = np.isfinite(curves.transmission_angle_deg)
    if np.any(finite_mask):
        angle_axis.plot(
            curves.energies_eV[finite_mask],
            curves.transmission_angle_deg[finite_mask],
            label="β после границы, угол прохождения",
            color="#2f7f3f",
        )
    if selected_point.transmission_angle_deg is not None:
        angle_axis.scatter(
            [selected_point.energy_eV],
            [selected_point.transmission_angle_deg],
            s=36,
            c="#cf2f2f",
            zorder=3,
        )
    angle_axis.set_xlabel("Энергия, эВ")
    angle_axis.set_ylabel("β, °")
    angle_axis.grid(True, which="both")
    handles, labels = angle_axis.get_legend_handles_labels()
    if handles:
        angle_axis.legend()


def draw_trajectory_sweep_plots(phase_axis, angle_axis, diagnostic_axis, frame, x_column: str, x_label: str) -> None:
    x_values = frame[x_column].to_numpy(dtype=float)

    phase_axis.clear()
    phase_axis.set_title("Вероятность изменения спина после СОВ")
    phase_axis.plot(
        x_values,
        frame["p_flip_initial_up"].to_numpy(dtype=float),
        label="начальный ↑: ↑→↓",
        color="#2f5597",
    )
    phase_axis.plot(
        x_values,
        frame["p_flip_initial_down"].to_numpy(dtype=float),
        label="начальный ↓: ↓→↑",
        color="#cf2f2f",
    )
    phase_axis.set_xlabel(x_label)
    phase_axis.set_ylabel("P(изменение спина)")
    phase_axis.set_ylim(-0.03, 1.03)
    phase_axis.grid(True, which="both")
    phase_axis.legend()

    angle_axis.clear()
    angle_axis.set_title("Углы после взаимодействия")
    angle_axis.plot(x_values, frame["theta_deg"].to_numpy(dtype=float), label="θ, угол интегрирования", color="#2f7f3f")
    angle_axis.plot(
        x_values,
        frame["trajectory_phi_deg"].to_numpy(dtype=float),
        label="φ, угол траектории после взаимодействия",
        color="#cf2f2f",
    )
    angle_axis.set_xlabel(x_label)
    angle_axis.set_ylabel("угол, °")
    angle_axis.grid(True, which="both")
    angle_axis.legend()

    diagnostic_axis.clear()
    diagnostic_axis.set_title("r_min и число шагов")
    diagnostic_axis.plot(
        x_values,
        frame["r_min_ang"].to_numpy(dtype=float),
        label="r_min, минимальное сближение",
        color="#8a5a00",
    )
    diagnostic_axis.set_xlabel(x_label)
    diagnostic_axis.set_ylabel("r_min, Å")
    diagnostic_axis.grid(True, which="both")

    steps_axis = getattr(diagnostic_axis, "_trajectory_steps_axis", None)
    if steps_axis is None or steps_axis.figure is not diagnostic_axis.figure:
        steps_axis = diagnostic_axis.twinx()
        diagnostic_axis._trajectory_steps_axis = steps_axis
    else:
        steps_axis.clear()
    steps_axis.plot(
        x_values,
        frame["steps"].to_numpy(dtype=float),
        label="steps, шаги интегрирования",
        color="#6b4fa3",
        linestyle="--",
    )
    steps_axis.set_ylabel("шаги")

    handles, labels = diagnostic_axis.get_legend_handles_labels()
    step_handles, step_labels = steps_axis.get_legend_handles_labels()
    diagnostic_axis.legend(handles + step_handles, labels + step_labels, loc="best")
    if "converged" in frame:
        failed_count = int((~frame["converged"].astype(bool)).sum())
        if failed_count:
            diagnostic_axis.text(
                0.02,
                0.95,
                f"Ошибок в точках: {failed_count}. Подробности в сводке.",
                transform=diagnostic_axis.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                color="#9a3412",
                bbox={"facecolor": "#fff7ed", "edgecolor": "#fdba74", "pad": 4},
            )


def build_geometry_preview_data(
    geometry: GeometryContext,
    atom_selection: AtomSelection,
    *,
    search_region: LatticeSearchRegion | None = None,
    max_lattice_points: int = 4000,
    max_selected_points: int = 120,
) -> GeometryPreviewData:
    if search_region is None:
        search_region = LatticeSearchRegion(
            x_radius=geometry.lattice_radius,
            y_radius=geometry.lattice_radius,
            z_min_layer=-geometry.lattice_radius,
            z_max_layer=geometry.lattice_radius,
        )

    lattice_points = build_lattice_points(
        lattice_constant_ang=geometry.lattice_constant_ang,
        x_radius=search_region.x_radius,
        y_radius=search_region.y_radius,
        z_min_layer=search_region.z_min_layer,
        z_max_layer=search_region.z_max_layer,
    )
    preview_points = _sample_points(lattice_points, max_points=max_lattice_points)
    origin = np.array([0.0, 0.0, geometry.source_layer * geometry.lattice_constant_ang], dtype=float)
    direction = direction_from_spherical_angles(geometry.alpha_rad, geometry.beta_rad)
    direction /= max(np.linalg.norm(direction), 1e-12)

    all_points = _hits_to_points(atom_selection.all_atoms)
    selected_points = _hits_to_points(atom_selection.selected_atoms[:max_selected_points])
    source_mask = np.isclose(preview_points[:, 2], origin[2]) if len(preview_points) else np.array([], dtype=bool)
    source_layer_points = preview_points[source_mask] if len(preview_points) else np.empty((0, 3), dtype=float)

    longitudinal_max = 0.0
    if atom_selection.all_atoms:
        longitudinal_max = max(float(atom["longitudinal_distance"]) for atom in atom_selection.all_atoms)
    x_limit = search_region.x_radius * geometry.lattice_constant_ang
    y_limit = search_region.y_radius * geometry.lattice_constant_ang
    z_min = search_region.z_min_layer * geometry.lattice_constant_ang
    z_max = search_region.z_max_layer * geometry.lattice_constant_ang
    line_length = max(
        longitudinal_max + geometry.lattice_constant_ang,
        np.linalg.norm([x_limit, y_limit, z_max - z_min]) * 0.65,
        geometry.interaction_radius_ang * 1.5,
    )
    trajectory_end = origin + direction * line_length

    return GeometryPreviewData(
        lattice_points=preview_points,
        source_layer_points=source_layer_points,
        interacting_points=all_points,
        selected_points=selected_points,
        origin=origin,
        direction=direction,
        trajectory_end=trajectory_end,
        region_min=np.array([-x_limit, -y_limit, z_min], dtype=float),
        region_max=np.array([x_limit, y_limit, z_max], dtype=float),
        alpha_deg=geometry.alpha_deg,
        beta_deg=geometry.beta_deg,
        source_layer=geometry.source_layer,
        source_depth=geometry.source_depth,
        surface_z_ang=geometry.surface_z_ang,
        interaction_radius_ang=geometry.interaction_radius_ang,
    )


def draw_geometry_preview(space_axis, xz_axis, xy_axis, preview: GeometryPreviewData) -> None:
    for axis in (space_axis, xz_axis, xy_axis):
        axis.clear()

    _draw_prism_wireframe(space_axis, preview.region_min, preview.region_max)
    _draw_surface_plane(space_axis, preview.region_min, preview.region_max, preview.surface_z_ang)
    if len(preview.lattice_points):
        space_axis.scatter(
            preview.lattice_points[:, 0],
            preview.lattice_points[:, 1],
            preview.lattice_points[:, 2],
            s=6,
            c="#b8b8b8",
            alpha=0.18,
            depthshade=False,
            label="узлы решётки",
        )
    if len(preview.source_layer_points):
        space_axis.scatter(
            preview.source_layer_points[:, 0],
            preview.source_layer_points[:, 1],
            preview.source_layer_points[:, 2],
            s=16,
            c="#3a6ee8",
            alpha=0.35,
            depthshade=False,
            label=f"глубина d={preview.source_depth}",
        )
    if len(preview.interacting_points):
        space_axis.scatter(
            preview.interacting_points[:, 0],
            preview.interacting_points[:, 1],
            preview.interacting_points[:, 2],
            s=18,
            c="#f28e2b",
            alpha=0.65,
            depthshade=False,
            label="атомы в зоне взаимодействия",
        )
    if len(preview.selected_points):
        space_axis.scatter(
            preview.selected_points[:, 0],
            preview.selected_points[:, 1],
            preview.selected_points[:, 2],
            s=28,
            c="#cf2f2f",
            alpha=0.9,
            depthshade=False,
            label="атомы в расчёте",
        )

    trajectory = np.vstack([preview.origin, preview.trajectory_end])
    space_axis.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color="#2f7f3f", linewidth=2.2, label="траектория электрона")
    vector = preview.trajectory_end - preview.origin
    space_axis.quiver(
        preview.origin[0],
        preview.origin[1],
        preview.origin[2],
        vector[0],
        vector[1],
        vector[2],
        color="#2f7f3f",
        arrow_length_ratio=0.08,
        linewidth=1.6,
    )
    space_axis.scatter(
        [preview.origin[0]],
        [preview.origin[1]],
        [preview.origin[2]],
        s=70,
        c="#0f7c2b",
        edgecolors="black",
        linewidths=0.8,
        depthshade=False,
        label="старт электрона",
    )
    _set_3d_equal_limits(space_axis, preview)
    space_axis.set_title("3D схема решётки и траектории")
    space_axis.set_xlabel("x, Å")
    space_axis.set_ylabel("y, Å")
    space_axis.set_zlabel("глубина, Å")
    space_axis.view_init(elev=21, azim=-58)

    _draw_projection_xz(xz_axis, preview)
    _draw_projection_xy(xy_axis, preview)


def capture_view_limits(*axes) -> dict:
    return {axis: {"xlim": axis.get_xlim(), "ylim": axis.get_ylim()} for axis in axes}


def restore_view_limits(view_limits: dict) -> None:
    for axis, limits in view_limits.items():
        axis.set_xlim(limits["xlim"])
        axis.set_ylim(limits["ylim"])


def _sample_points(points: np.ndarray, *, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return points[indices]


def _hits_to_points(atom_hits: list[dict[str, float | np.ndarray]]) -> np.ndarray:
    if not atom_hits:
        return np.empty((0, 3), dtype=float)
    return np.asarray([np.asarray(atom["coords"], dtype=float) for atom in atom_hits], dtype=float)


def _draw_prism_wireframe(axis, region_min: np.ndarray, region_max: np.ndarray) -> None:
    x_min, y_min, z_min = region_min
    x_max, y_max, z_max = region_max
    vertices = np.array(
        [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ],
        dtype=float,
    )
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    for start, end in edges:
        axis.plot(
            vertices[[start, end], 0],
            vertices[[start, end], 1],
            vertices[[start, end], 2],
            color="black",
            linewidth=0.8,
            alpha=0.65,
        )


def _draw_surface_plane(axis, region_min: np.ndarray, region_max: np.ndarray, surface_z_ang: float) -> None:
    x_min, y_min = region_min[:2]
    x_max, y_max = region_max[:2]
    xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
    zz = np.full_like(xx, surface_z_ang, dtype=float)
    axis.plot_surface(
        xx,
        yy,
        zz,
        color="#7a7a7a",
        alpha=0.08,
        linewidth=0,
        shade=False,
    )
    axis.plot(
        [x_min, x_max],
        [y_min, y_min],
        [surface_z_ang, surface_z_ang],
        color="#6b6b6b",
        linewidth=1.0,
        alpha=0.75,
    )


def _set_3d_equal_limits(axis, preview: GeometryPreviewData) -> None:
    points = [
        preview.origin[np.newaxis, :],
        preview.trajectory_end[np.newaxis, :],
        preview.region_min[np.newaxis, :],
        preview.region_max[np.newaxis, :],
        np.array([[0.0, 0.0, preview.surface_z_ang]], dtype=float),
    ]
    for cloud in (preview.lattice_points, preview.interacting_points, preview.selected_points):
        if len(cloud):
            points.append(cloud)

    stacked = np.vstack(points)
    minimums = stacked.min(axis=0)
    maximums = stacked.max(axis=0)
    centers = (minimums + maximums) / 2.0
    half_span = max(np.max(maximums - minimums) / 2.0, preview.interaction_radius_ang)

    axis.set_xlim(centers[0] - half_span, centers[0] + half_span)
    axis.set_ylim(centers[1] - half_span, centers[1] + half_span)
    axis.set_zlim(centers[2] + half_span, centers[2] - half_span)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def _draw_projection_xz(axis, preview: GeometryPreviewData) -> None:
    if len(preview.lattice_points):
        axis.scatter(preview.lattice_points[:, 0], preview.lattice_points[:, 2], s=6, c="#b8b8b8", alpha=0.18)
    if len(preview.interacting_points):
        axis.scatter(preview.interacting_points[:, 0], preview.interacting_points[:, 2], s=16, c="#f28e2b", alpha=0.65)
    if len(preview.selected_points):
        axis.scatter(preview.selected_points[:, 0], preview.selected_points[:, 2], s=24, c="#cf2f2f", alpha=0.9)

    axis.plot(
        [preview.origin[0], preview.trajectory_end[0]],
        [preview.origin[2], preview.trajectory_end[2]],
        color="#2f7f3f",
        linewidth=2.0,
    )
    axis.plot(
        [preview.origin[0], preview.origin[0]],
        [preview.origin[2], preview.origin[2] + np.sign(preview.direction[2] or 1.0) * max(abs(preview.trajectory_end[2] - preview.origin[2]), 1.0)],
        color="#3a6ee8",
        linestyle="--",
        linewidth=1.1,
    )
    axis.axhline(preview.surface_z_ang, color="#6b6b6b", linestyle="-.", linewidth=1.0, alpha=0.9)
    axis.axhline(preview.origin[2], color="#3a6ee8", linestyle=":", linewidth=1.0, alpha=0.8)
    axis.scatter([preview.origin[0]], [preview.origin[2]], s=60, c="#0f7c2b", edgecolors="black", linewidths=0.8)
    axis.set_title("Проекция XZ: полярный угол и глубина")
    axis.set_xlabel("x, Å")
    axis.set_ylabel("глубина, Å")
    axis.text(
        0.97,
        0.08,
        "поверхность",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color="#666",
        fontsize=9,
    )
    axis.text(
        0.03,
        0.95,
        f"alpha = {preview.alpha_deg:.1f}°\nглубина d = {preview.source_depth}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "#bbbbbb"},
    )
    _set_projection_limits(axis, preview, x_index=0, y_index=2, invert_y=True)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.35)


def _draw_projection_xy(axis, preview: GeometryPreviewData) -> None:
    if len(preview.source_layer_points):
        axis.scatter(preview.source_layer_points[:, 0], preview.source_layer_points[:, 1], s=10, c="#3a6ee8", alpha=0.35)
    if len(preview.interacting_points):
        axis.scatter(preview.interacting_points[:, 0], preview.interacting_points[:, 1], s=16, c="#f28e2b", alpha=0.45)
    if len(preview.selected_points):
        axis.scatter(preview.selected_points[:, 0], preview.selected_points[:, 1], s=24, c="#cf2f2f", alpha=0.9)

    axis.plot(
        [preview.origin[0], preview.trajectory_end[0]],
        [preview.origin[1], preview.trajectory_end[1]],
        color="#2f7f3f",
        linewidth=2.0,
    )
    horizontal_length = max(np.linalg.norm(preview.trajectory_end[:2] - preview.origin[:2]), preview.interaction_radius_ang)
    axis.plot(
        [preview.origin[0], preview.origin[0] + horizontal_length],
        [preview.origin[1], preview.origin[1]],
        color="#3a6ee8",
        linestyle="--",
        linewidth=1.1,
    )
    axis.scatter([preview.origin[0]], [preview.origin[1]], s=60, c="#0f7c2b", edgecolors="black", linewidths=0.8)
    axis.set_title("Проекция XY: азимутальный угол")
    axis.set_xlabel("x, Å")
    axis.set_ylabel("y, Å")
    axis.text(
        0.03,
        0.95,
        f"beta = {preview.beta_deg:.1f}°\nR_int = {preview.interaction_radius_ang:.2f} Å",
        transform=axis.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "#bbbbbb"},
    )
    _set_projection_limits(axis, preview, x_index=0, y_index=1)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.35)


def _set_projection_limits(axis, preview: GeometryPreviewData, *, x_index: int, y_index: int, invert_y: bool = False) -> None:
    points = [
        preview.origin[[x_index, y_index]][np.newaxis, :],
        preview.trajectory_end[[x_index, y_index]][np.newaxis, :],
        preview.region_min[[x_index, y_index]][np.newaxis, :],
        preview.region_max[[x_index, y_index]][np.newaxis, :],
    ]
    for cloud in (preview.lattice_points, preview.interacting_points, preview.selected_points, preview.source_layer_points):
        if len(cloud):
            points.append(cloud[:, [x_index, y_index]])
    stacked = np.vstack(points)
    minimums = stacked.min(axis=0)
    maximums = stacked.max(axis=0)
    centers = (minimums + maximums) / 2.0
    half_span = max(np.max(maximums - minimums) / 2.0, preview.interaction_radius_ang)
    axis.set_xlim(centers[0] - half_span, centers[0] + half_span)
    if invert_y:
        axis.set_ylim(centers[1] + half_span, centers[1] - half_span)
    else:
        axis.set_ylim(centers[1] - half_span, centers[1] + half_span)


__all__ = [
    "GeometryPreviewData",
    "zoom_axis",
    "zoom_3d_axis",
    "zoom_axis_around_point",
    "draw_spin_plots",
    "draw_boundary_utility_plots",
    "build_geometry_preview_data",
    "draw_geometry_preview",
    "capture_view_limits",
    "restore_view_limits",
]
