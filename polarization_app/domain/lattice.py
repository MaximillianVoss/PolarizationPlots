# -*- coding: utf-8 -*-
from dataclasses import dataclass
import logging
import math

import numpy as np


logger = logging.getLogger(__name__)

AtomHit = dict[str, float | np.ndarray]


@dataclass(frozen=True)
class LatticeRadiusEstimate:
    radius: int
    required_radius: int
    capped_by_max_atoms: bool


@dataclass(frozen=True)
class LatticeSearchRegion:
    x_radius: int
    y_radius: int
    z_min_layer: int
    z_max_layer: int

    @property
    def node_count(self) -> int:
        return (2 * self.x_radius + 1) * (2 * self.y_radius + 1) * (self.z_max_layer - self.z_min_layer + 1)


@dataclass(frozen=True)
class LatticeRegionEstimate:
    region: LatticeSearchRegion
    required_region: LatticeSearchRegion
    capped_by_max_atoms: bool


def build_lattice_points(
    lattice_constant_ang: float,
    radius: int = 3,
    *,
    x_radius: int | None = None,
    y_radius: int | None = None,
    z_min_layer: int | None = None,
    z_max_layer: int | None = None,
) -> np.ndarray:
    """Генерация атомов в прямоугольной области решетки."""
    x_radius = int(radius if x_radius is None else x_radius)
    y_radius = int(radius if y_radius is None else y_radius)
    z_min_layer = int(-radius if z_min_layer is None else z_min_layer)
    z_max_layer = int(radius if z_max_layer is None else z_max_layer)

    atoms = []
    for i in range(-x_radius, x_radius + 1):
        for j in range(-y_radius, y_radius + 1):
            for k in range(z_min_layer, z_max_layer + 1):
                atoms.append(np.array([i * lattice_constant_ang, j * lattice_constant_ang, k * lattice_constant_ang]))
    return np.array(atoms)


def direction_from_spherical_angles(alpha_rad: float, beta_rad: float) -> np.ndarray:
    """Преобразование углов вылета в вектор направления.

    Ось z трактуется как глубина в веществе, поэтому при alpha=0 электрон
    вылетает к поверхности и имеет отрицательную z-компоненту.
    """
    x = np.sin(alpha_rad) * np.cos(beta_rad)
    y = np.sin(alpha_rad) * np.sin(beta_rad)
    z = -np.cos(alpha_rad)
    return np.array([x, y, z], dtype=float)


def point_to_line_distance(point: np.ndarray, line_point: np.ndarray, line_dir: np.ndarray) -> float:
    """Расстояние от точки до прямой."""
    return float(np.linalg.norm(np.cross(point - line_point, line_dir)) / np.linalg.norm(line_dir))


def find_interacting_atoms(
    lattice_constant_ang: float,
    interaction_radius_ang: float,
    alpha_rad: float,
    beta_rad: float,
    radius: int = 3,
    source_layer: int = 0,
    max_longitudinal: float | None = None,
    search_region: LatticeSearchRegion | None = None,
) -> list[AtomHit]:
    """
    Возвращает атомы в пределах interaction_radius_ang от луча движения.
    """
    logger.info(
        "lattice.find_interacting_atoms | a=%.4f, R_int=%.4f, alpha=%.4f, beta=%.4f, n=%d, d=%d, s_max=%s",
        lattice_constant_ang,
        interaction_radius_ang,
        alpha_rad,
        beta_rad,
        radius,
        source_layer,
        "None" if max_longitudinal is None else f"{max_longitudinal:.4f}",
    )

    origin = np.array([0.0, 0.0, float(source_layer) * lattice_constant_ang], dtype=float)
    direction = direction_from_spherical_angles(alpha_rad, beta_rad)
    norm = np.linalg.norm(direction)
    if norm <= 1e-15:
        raise ValueError("Направляющий вектор траектории имеет почти нулевую длину.")

    direction_unit = direction / norm
    if search_region is None:
        atoms = build_lattice_points(lattice_constant_ang, radius=radius)
    else:
        atoms = build_lattice_points(
            lattice_constant_ang,
            x_radius=search_region.x_radius,
            y_radius=search_region.y_radius,
            z_min_layer=search_region.z_min_layer,
            z_max_layer=search_region.z_max_layer,
        )

    results: list[AtomHit] = []
    for atom in atoms:
        if np.allclose(atom, origin):
            continue

        displacement = atom - origin
        longitudinal = float(np.dot(displacement, direction_unit))
        if longitudinal < 0.0:
            continue
        if max_longitudinal is not None and longitudinal > max_longitudinal:
            continue

        distance_to_line = float(np.linalg.norm(np.cross(displacement, direction_unit)))
        if distance_to_line <= interaction_radius_ang:
            results.append(
                {
                    "coords": atom,
                    "distance_to_line": distance_to_line,
                    "distance_to_origin": float(np.linalg.norm(displacement)),
                    "longitudinal_distance": longitudinal,
                }
            )

    results.sort(key=lambda item: float(item["longitudinal_distance"]))
    logger.info(
        "lattice.find_interacting_atoms | origin=%s, dir=%s, всего узлов=%d, отобрано=%d",
        np.array2string(origin, precision=4, suppress_small=True),
        np.array2string(direction_unit, precision=4, suppress_small=True),
        len(atoms),
        len(results),
    )
    return results


def estimate_lattice_radius(
    lattice_constant_ang: float,
    bohr_radius_ang: float,
    alpha_rad: float,
    beta_rad: float,
    source_layer: int,
    margin_bohr: float = 5.0,
    max_atoms: int = 100_000,
) -> int:
    return estimate_lattice_radius_details(
        lattice_constant_ang=lattice_constant_ang,
        bohr_radius_ang=bohr_radius_ang,
        alpha_rad=alpha_rad,
        beta_rad=beta_rad,
        source_layer=source_layer,
        margin_bohr=margin_bohr,
        max_atoms=max_atoms,
    ).radius


def estimate_lattice_radius_details(
    lattice_constant_ang: float,
    bohr_radius_ang: float,
    alpha_rad: float,
    beta_rad: float,
    source_layer: int,
    margin_bohr: float = 5.0,
    max_atoms: int = 100_000,
) -> LatticeRadiusEstimate:
    region_estimate = estimate_lattice_search_region(
        lattice_constant_ang=lattice_constant_ang,
        bohr_radius_ang=bohr_radius_ang,
        alpha_rad=alpha_rad,
        beta_rad=beta_rad,
        source_layer=source_layer,
        margin_bohr=margin_bohr,
        max_atoms=max_atoms,
    )
    region = region_estimate.region
    required_region = region_estimate.required_region
    radius = max(region.x_radius, region.y_radius, abs(region.z_min_layer), abs(region.z_max_layer))
    required_radius = max(
        required_region.x_radius,
        required_region.y_radius,
        abs(required_region.z_min_layer),
        abs(required_region.z_max_layer),
    )
    return LatticeRadiusEstimate(
        radius=radius,
        required_radius=required_radius,
        capped_by_max_atoms=region_estimate.capped_by_max_atoms,
    )


def estimate_lattice_search_region(
    lattice_constant_ang: float,
    bohr_radius_ang: float,
    alpha_rad: float,
    beta_rad: float,
    source_layer: int,
    margin_bohr: float = 5.0,
    max_atoms: int = 100_000,
) -> LatticeRegionEstimate:
    """
    Возвращает прямоугольную область узлов, покрывающую рабочий объём вдоль траектории.
    """
    interaction_radius = margin_bohr * bohr_radius_ang
    direction = direction_from_spherical_angles(alpha_rad, beta_rad)
    vz = direction[2]
    if abs(vz) <= 1e-8:
        vz = 1e-8

    origin = np.array([0.0, 0.0, source_layer * lattice_constant_ang], dtype=float)
    travel_distance = interaction_radius / abs(vz)
    endpoint = origin + direction * travel_distance

    x_min = min(origin[0], endpoint[0]) - interaction_radius
    x_max = max(origin[0], endpoint[0]) + interaction_radius
    y_min = min(origin[1], endpoint[1]) - interaction_radius
    y_max = max(origin[1], endpoint[1]) + interaction_radius
    z_min = 0.0
    z_max = max(origin[2], endpoint[2]) + interaction_radius

    required_region = LatticeSearchRegion(
        x_radius=max(1, int(math.ceil(max(abs(x_min), abs(x_max)) / lattice_constant_ang))),
        y_radius=max(1, int(math.ceil(max(abs(y_min), abs(y_max)) / lattice_constant_ang))),
        z_min_layer=0,
        z_max_layer=int(math.ceil(z_max / lattice_constant_ang)),
    )

    region = required_region
    if region.node_count > max_atoms:
        scale = (max_atoms / region.node_count) ** (1.0 / 3.0)
        region = LatticeSearchRegion(
            x_radius=max(1, int(math.floor(required_region.x_radius * scale))),
            y_radius=max(1, int(math.floor(required_region.y_radius * scale))),
            z_min_layer=0,
            z_max_layer=max(source_layer, int(math.floor(required_region.z_max_layer * scale))),
        )

    return LatticeRegionEstimate(
        region=region,
        required_region=required_region,
        capped_by_max_atoms=region != required_region,
    )


def generate_lattice(a, n=3):
    return build_lattice_points(a, radius=n)


def spherical_to_cartesian(alpha, beta):
    return direction_from_spherical_angles(alpha, beta)


def distance_point_to_line(point, line_point, line_dir):
    return point_to_line_distance(point, line_point, line_dir)


def nearest_atoms(a, interaction_radius, alpha, beta, n=3, d_layer=0, max_longitudinal=None):
    return find_interacting_atoms(
        lattice_constant_ang=a,
        interaction_radius_ang=interaction_radius,
        alpha_rad=alpha,
        beta_rad=beta,
        radius=n,
        source_layer=d_layer,
        max_longitudinal=max_longitudinal,
    )


def compute_lattice_n_auto(
    a_ang: float,
    R_bohr: float,
    alpha: float,
    beta: float,
    d_layer: int,
    margin_bohr: float = 5.0,
    max_atoms: int = 100_000,
) -> int:
    return estimate_lattice_radius(
        lattice_constant_ang=a_ang,
        bohr_radius_ang=R_bohr,
        alpha_rad=alpha,
        beta_rad=beta,
        source_layer=d_layer,
        margin_bohr=margin_bohr,
        max_atoms=max_atoms,
    )


__all__ = [
    "AtomHit",
    "LatticeRadiusEstimate",
    "LatticeSearchRegion",
    "LatticeRegionEstimate",
    "build_lattice_points",
    "direction_from_spherical_angles",
    "point_to_line_distance",
    "find_interacting_atoms",
    "estimate_lattice_radius",
    "estimate_lattice_radius_details",
    "estimate_lattice_search_region",
    "generate_lattice",
    "spherical_to_cartesian",
    "distance_point_to_line",
    "nearest_atoms",
    "compute_lattice_n_auto",
]
