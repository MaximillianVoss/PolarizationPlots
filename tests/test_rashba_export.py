import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from polarization_app.application.rashba_export import export_rashba_surface_file, rashba_surface_export_metadata
from polarization_app.physics.rashba_surface import RashbaSurfaceRequest, compute_rashba_surface


class RashbaExportTestCase(unittest.TestCase):
    def _sample_result(self):
        return compute_rashba_surface(
            RashbaSurfaceRequest(
                energy_min_eV=100.0,
                energy_max_eV=200.0,
                point_count=3,
                layer_thickness_ang=1.0,
                rashba_alpha_au=0.02,
                emission_angle_deg=45.0,
                surface_potential_eV=1.0,
            )
        )

    def test_metadata_describes_request_and_source(self):
        result = self._sample_result()

        metadata = rashba_surface_export_metadata(result, source_label="Ver=0")

        self.assertEqual(metadata["source_label"], "Ver=0")
        self.assertEqual(metadata["point_count"], 3)
        self.assertEqual(metadata["rashba_alpha_au"], 0.02)

    def test_export_file_writes_json_xml_and_xlsx_by_extension(self):
        result = self._sample_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = export_rashba_surface_file(
                Path(tmpdir) / "rashba.json",
                result,
                source_label="Ver=0",
            )
            xml_path = export_rashba_surface_file(
                Path(tmpdir) / "rashba.xml",
                result,
                source_label="Ver=0",
            )
            xlsx_path = export_rashba_surface_file(
                Path(tmpdir) / "rashba.xlsx",
                result,
                source_label="Ver=0",
            )

            json_payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertIn("polarization", json_payload["columns"])
            self.assertEqual(json_payload["metadata"]["source_label"], "Ver=0")

            xml_root = ET.parse(xml_path).getroot()
            self.assertEqual(xml_root.tag, "rashba_surface_export")

            with zipfile.ZipFile(xlsx_path) as archive:
                workbook_xml = archive.read("xl/workbook.xml").decode("utf-8")
                self.assertIn("rashba_surface_data", workbook_xml)
                self.assertIn("xl/worksheets/sheet1.xml", archive.namelist())
                self.assertIn("xl/worksheets/sheet2.xml", archive.namelist())


if __name__ == "__main__":
    unittest.main()
