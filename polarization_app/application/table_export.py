# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import json
import zipfile
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


SUPPORTED_EXPORT_SUFFIXES = (".json", ".xml", ".xlsx")


def export_table_file(
    path: str | Path,
    frame: pd.DataFrame,
    metadata: dict[str, object] | None = None,
    *,
    root_tag: str,
    data_sheet_name: str,
) -> Path:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXPORT_SUFFIXES:
        raise ValueError("Поддерживаются только расширения .json, .xml и .xlsx.")

    normalized_metadata = normalize_metadata(metadata or {})
    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".json":
        write_json(path, frame, normalized_metadata)
    elif suffix == ".xml":
        write_xml(path, frame, normalized_metadata, root_tag=root_tag)
    else:
        write_xlsx(path, frame, normalized_metadata, data_sheet_name=data_sheet_name)
    return path


def export_table_bundle(
    base_path: str | Path,
    frame: pd.DataFrame,
    metadata: dict[str, object] | None = None,
    *,
    root_tag: str,
    data_sheet_name: str,
) -> dict[str, Path]:
    base_path = Path(base_path)
    return {
        "json": export_table_file(
            base_path.with_suffix(".json"),
            frame,
            metadata,
            root_tag=root_tag,
            data_sheet_name=data_sheet_name,
        ),
        "xml": export_table_file(
            base_path.with_suffix(".xml"),
            frame,
            metadata,
            root_tag=root_tag,
            data_sheet_name=data_sheet_name,
        ),
        "xlsx": export_table_file(
            base_path.with_suffix(".xlsx"),
            frame,
            metadata,
            root_tag=root_tag,
            data_sheet_name=data_sheet_name,
        ),
    }


def normalize_metadata(metadata: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in metadata.items():
        normalized[str(key)] = json_value(value)
    return normalized


def write_json(path: Path, frame: pd.DataFrame, metadata: dict[str, object]) -> None:
    payload = {
        "metadata": metadata,
        "columns": list(frame.columns),
        "rows": [
            {column: json_value(row[column]) for column in frame.columns}
            for _, row in frame.iterrows()
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_xml(path: Path, frame: pd.DataFrame, metadata: dict[str, object], *, root_tag: str) -> None:
    root = ET.Element(root_tag)
    metadata_node = ET.SubElement(root, "metadata")
    for key, value in metadata.items():
        item = ET.SubElement(metadata_node, "item", key=str(key))
        item.text = stringify(value)

    columns_node = ET.SubElement(root, "columns")
    for column in frame.columns:
        node = ET.SubElement(columns_node, "column")
        node.text = str(column)

    rows_node = ET.SubElement(root, "rows")
    for index, (_, row) in enumerate(frame.iterrows()):
        row_node = ET.SubElement(rows_node, "row", index=str(index))
        for column in frame.columns:
            value_node = ET.SubElement(row_node, str(column))
            value_node.text = stringify(row[column])

    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def write_xlsx(path: Path, frame: pd.DataFrame, metadata: dict[str, object], *, data_sheet_name: str) -> None:
    metadata_frame = pd.DataFrame(
        [{"key": key, "value": stringify(value)} for key, value in metadata.items()]
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _xlsx_content_types())
        archive.writestr("_rels/.rels", _xlsx_root_rels())
        archive.writestr("xl/workbook.xml", _xlsx_workbook(data_sheet_name))
        archive.writestr("xl/_rels/workbook.xml.rels", _xlsx_workbook_rels())
        archive.writestr("xl/worksheets/sheet1.xml", _xlsx_sheet_xml(frame))
        archive.writestr("xl/worksheets/sheet2.xml", _xlsx_sheet_xml(metadata_frame))


def json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return [json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return value


def stringify(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.16g}"
    if isinstance(value, np.ndarray):
        return json.dumps([json_value(item) for item in value.tolist()], ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return json.dumps([json_value(item) for item in value], ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(json_value(value), ensure_ascii=False)
    return str(value)


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


def _xlsx_workbook(data_sheet_name: str) -> str:
    sheet_name = escape(data_sheet_name)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="{sheet_name}" sheetId="1" r:id="rId1"/>
    <sheet name="metadata" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>"""


def _xlsx_workbook_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
</Relationships>"""


def _xlsx_sheet_xml(frame: pd.DataFrame) -> str:
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
        if _is_blank_cell(value):
            cells.append(f'<c r="{reference}"/>')
        elif isinstance(value, str):
            cells.append(f'<c r="{reference}" t="inlineStr"><is><t>{escape(value)}</t></is></c>')
        elif isinstance(value, (bool, np.bool_)):
            cells.append(f'<c r="{reference}" t="b"><v>{int(bool(value))}</v></c>')
        else:
            cells.append(f'<c r="{reference}"><v>{stringify(value)}</v></c>')
    return f'    <row r="{row_index}">' + "".join(cells) + "</row>"


def _excel_column_name(index: int) -> str:
    letters = []
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters.append(chr(65 + remainder))
    return "".join(reversed(letters))


def _is_blank_cell(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, np.generic):
        value = value.item()
    return isinstance(value, float) and not np.isfinite(value)


__all__ = [
    "SUPPORTED_EXPORT_SUFFIXES",
    "export_table_bundle",
    "export_table_file",
    "json_value",
    "normalize_metadata",
    "stringify",
]
