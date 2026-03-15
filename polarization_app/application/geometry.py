# -*- coding: utf-8 -*-
from dataclasses import dataclass

import numpy as np

from polarization_app.domain.lattice import AtomHit, LatticeSearchRegion, find_interacting_atoms


@dataclass(frozen=True)
class GeometryContext:
    lattice_constant_ang: float
    bohr_radius_ang: float
    alpha_deg: float
    beta_deg: float
    lattice_radius: int
    source_layer: int
    orbital_l: int

    @property
    def alpha_rad(self) -> float:
        return float(np.deg2rad(self.alpha_deg))

    @property
    def beta_rad(self) -> float:
        return float(np.deg2rad(self.beta_deg))

    @property
    def interaction_radius_ang(self) -> float:
        return 5.0 * self.bohr_radius_ang

    @property
    def source_depth(self) -> int:
        return self.source_layer + 1

    @property
    def source_z_ang(self) -> float:
        return self.source_layer * self.lattice_constant_ang

    @property
    def surface_z_ang(self) -> float:
        return 0.0


@dataclass(frozen=True)
class AtomSelection:
    all_atoms: list[AtomHit]
    selected_atoms: list[AtomHit]
    impact_parameters_ang: list[float]


def collect_atom_selection(
    geometry: GeometryContext,
    *,
    max_atoms: int = 200,
    min_impact_parameter_ang: float = 1e-4,
    search_region: LatticeSearchRegion | None = None,
) -> AtomSelection:
    all_atoms = find_interacting_atoms(
        lattice_constant_ang=geometry.lattice_constant_ang,
        interaction_radius_ang=geometry.interaction_radius_ang,
        alpha_rad=geometry.alpha_rad,
        beta_rad=geometry.beta_rad,
        radius=geometry.lattice_radius,
        source_layer=geometry.source_layer,
        search_region=search_region,
    )
    selected_atoms = sorted(all_atoms, key=lambda item: float(item["distance_to_line"]))[:int(max_atoms)]
    impact_parameters = [
        float(item["distance_to_line"])
        for item in selected_atoms
        if float(item["distance_to_line"]) > float(min_impact_parameter_ang)
    ]
    return AtomSelection(
        all_atoms=all_atoms,
        selected_atoms=selected_atoms,
        impact_parameters_ang=impact_parameters,
    )


__all__ = [
    "GeometryContext",
    "AtomSelection",
    "collect_atom_selection",
]
