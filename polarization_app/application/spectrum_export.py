# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from polarization_app.application.table_export import export_table_bundle, export_table_file


EXPORT_COLUMN_MAP = OrderedDict(
    [
        ("energy_eV", "energy_eV"),
        ("sum_check_up", "sum_probability_initial_up"),
        ("sum_check_dn", "sum_probability_initial_down"),
        ("spin_mean_up", "spin_mean_initial_up"),
        ("spin_mean_dn", "spin_mean_initial_down"),
    ]
)


def build_spectrum_export_frame(
    energies_eV: np.ndarray,
    spin_curves: dict[str, np.ndarray],
) -> pd.DataFrame:
    energies = np.asarray(energies_eV, dtype=float)
    data: dict[str, np.ndarray] = {"energy_eV": energies}

    for source_name, export_name in list(EXPORT_COLUMN_MAP.items())[1:]:
        if source_name not in spin_curves:
            raise KeyError(f"В наборе кривых отсутствует ключ {source_name!r}.")
        values = np.asarray(spin_curves[source_name], dtype=float)
        if len(values) != len(energies):
            raise ValueError(f"Длина кривой {source_name!r} не совпадает с длиной сетки энергий.")
        data[export_name] = values

    return pd.DataFrame(data)


def export_spectrum_bundle(
    base_path: str | Path,
    energies_eV: np.ndarray,
    spin_curves: dict[str, np.ndarray],
    metadata: dict[str, object] | None = None,
) -> dict[str, Path]:
    frame = build_spectrum_export_frame(energies_eV, spin_curves)
    return export_table_bundle(
        base_path,
        frame,
        metadata,
        root_tag="spectrum_export",
        data_sheet_name="spectrum_data",
    )


def export_spectrum_file(
    path: str | Path,
    energies_eV: np.ndarray,
    spin_curves: dict[str, np.ndarray],
    metadata: dict[str, object] | None = None,
) -> Path:
    frame = build_spectrum_export_frame(energies_eV, spin_curves)
    return export_table_file(
        path,
        frame,
        metadata,
        root_tag="spectrum_export",
        data_sheet_name="spectrum_data",
    )


__all__ = [
    "EXPORT_COLUMN_MAP",
    "build_spectrum_export_frame",
    "export_spectrum_bundle",
    "export_spectrum_file",
]
