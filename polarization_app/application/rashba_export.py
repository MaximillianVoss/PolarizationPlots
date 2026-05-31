# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from polarization_app.application.table_export import export_table_file
from polarization_app.physics.rashba_surface import RashbaSurfaceResult


def rashba_surface_export_metadata(
    result: RashbaSurfaceResult,
    *,
    source_label: str,
) -> dict[str, object]:
    request = result.request
    return {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "source_label": source_label,
        "energy_min_eV": request.energy_min_eV,
        "energy_max_eV": request.energy_max_eV,
        "point_count": request.point_count,
        "layer_thickness_ang": request.layer_thickness_ang,
        "rashba_alpha_au": request.rashba_alpha_au,
        "emission_angle_deg": request.emission_angle_deg,
        "surface_potential_eV": request.surface_potential_eV,
    }


def export_rashba_surface_file(
    path: str | Path,
    result: RashbaSurfaceResult,
    *,
    source_label: str,
) -> Path:
    return export_table_file(
        path,
        result.frame,
        rashba_surface_export_metadata(result, source_label=source_label),
        root_tag="rashba_surface_export",
        data_sheet_name="rashba_surface_data",
    )


__all__ = ["export_rashba_surface_file", "rashba_surface_export_metadata"]
