# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import pandas as pd

from polarization_app.application.table_export import export_table_bundle, export_table_file


def export_trajectory_file(
    path: str | Path,
    frame: pd.DataFrame,
    metadata: dict[str, object] | None = None,
) -> Path:
    return export_table_file(
        path,
        frame,
        metadata,
        root_tag="trajectory_export",
        data_sheet_name="trajectory_data",
    )


def export_trajectory_bundle(
    base_path: str | Path,
    frame: pd.DataFrame,
    metadata: dict[str, object] | None = None,
) -> dict[str, Path]:
    return export_table_bundle(
        base_path,
        frame,
        metadata,
        root_tag="trajectory_export",
        data_sheet_name="trajectory_data",
    )


__all__ = ["export_trajectory_bundle", "export_trajectory_file"]
