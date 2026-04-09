# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import json
import zipfile
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


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
    base_path = Path(base_path)
    frame = build_spectrum_export_frame(energies_eV, spin_curves)
    metadata = _normalize_metadata(metadata or {})

    json_path = base_path.with_suffix(".json")
    xml_path = base_path.with_suffix(".xml")
    xlsx_path = base_path.with_suffix(".xlsx")

    json_path.parent.mkdir(parents=True, exist_ok=True)

    _write_json_export(json_path, frame, metadata)
    _write_xml_export(xml_path, frame, metadata)
    _write_xlsx_export(xlsx_path, frame, metadata)

    return {
        "json": json_path,
        "xml": xml_path,
        "xlsx": xlsx_path,
    }


def _normalize_metadata(metadata: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in metadata.items():
        if isinstance(value, np.generic):
            normalized[str(key)] = value.item()
        elif isinstance(value, tuple):
            normalized[str(key)] = list(value)
        else:
            normalized[str(key)] = value
    return normalized


def _write_json_export(path: Path, frame: pd.DataFrame, metadata: dict[str, object]) -> None:
    payload = {
        "metadata": metadata,
        "columns": list(frame.columns),
        "rows": [
            {column: _json_value(row[column]) for column in frame.columns}
            for _, row in frame.iterrows()
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_xml_export(path: Path, frame: pd.DataFrame, metadata: dict[str, object]) -> None:
    root = ET.Element("spectrum_export")
    metadata_node = ET.SubElement(root, "metadata")
    for key, value in metadata.items():
        item = ET.SubElement(metadata_node, "item", key=str(key))
        item.text = _stringify(value)

    columns_node = ET.SubElement(root, "columns")
    for column in frame.columns:
        node = ET.SubElement(columns_node, "column")
        node.text = column

    rows_node = ET.SubElement(root, "rows")
    for index, (_, row) in enumerate(frame.iterrows()):
        row_node = ET.SubElement(rows_node, "row", index=str(index))
        for column in frame.columns:
            value_node = ET.SubElement(row_node, column)
            value_node.text = _stringify(row[column])

    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _write_xlsx_export(path: Path, frame: pd.DataFrame, metadata: dict[str, object]) -> None:
    metadata_frame = pd.DataFrame(
        [{"key": key, "value": _stringify(value)} for key, value in metadata.items()]
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _xlsx_content_types())
        archive.writestr("_rels/.rels", _xlsx_root_rels())
        archive.writestr("xl/workbook.xml", _xlsx_workbook())
        archive.writestr("xl/_rels/workbook.xml.rels", _xlsx_workbook_rels())
        archive.writestr("xl/worksheets/sheet1.xml", _xlsx_sheet_xml(frame, "spectrum_data"))
        archive.writestr("xl/worksheets/sheet2.xml", _xlsx_sheet_xml(metadata_frame, "metadata"))


def _xlsx_content_types() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"""


def _xlsx_root_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""


def _xlsx_workbook() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="spectrum_data" sheetId="1" r:id="rId1"/>
    <sheet name="metadata" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>"""


def _xlsx_workbook_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
</Relationships>"""


def _xlsx_sheet_xml(frame: pd.DataFrame, sheet_name: str) -> str:
    del sheet_name
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        "  <sheetData>",
        _xlsx_row_xml(1, list(frame.columns)),
    ]
    for row_index, row in enumerate(frame.itertuples(index=False, name=None), start=2):
        lines.append(_xlsx_row_xml(row_index, list(row)))
    lines.extend(["  </sheetData>", "</worksheet>"])
    return "\n".join(lines)


def _xlsx_row_xml(row_index: int, values: list[object]) -> str:
    cells = []
    for column_index, value in enumerate(values, start=1):
        reference = f"{_excel_column_name(column_index)}{row_index}"
        if isinstance(value, str):
            escaped = escape(value)
            cells.append(f'<c r="{reference}" t="inlineStr"><is><t>{escaped}</t></is></c>')
        elif isinstance(value, (bool, np.bool_)):
            cells.append(f'<c r="{reference}" t="b"><v>{int(bool(value))}</v></c>')
        else:
            numeric = _stringify(value)
            cells.append(f'<c r="{reference}"><v>{numeric}</v></c>')
    return f'    <row r="{row_index}">' + "".join(cells) + "</row>"


def _excel_column_name(index: int) -> str:
    letters = []
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters.append(chr(65 + remainder))
    return "".join(reversed(letters))


def _json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _stringify(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return f"{value:.16g}"
    if isinstance(value, (list, tuple)):
        return json.dumps([_json_value(item) for item in value], ensure_ascii=False)
    return str(value)


__all__ = [
    "EXPORT_COLUMN_MAP",
    "build_spectrum_export_frame",
    "export_spectrum_bundle",
]
