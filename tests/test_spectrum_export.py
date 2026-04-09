import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from polarization_app.application.spectrum_export import (
    build_spectrum_export_frame,
    export_spectrum_bundle,
)


class SpectrumExportTestCase(unittest.TestCase):
    def test_build_export_frame_uses_expected_columns(self):
        frame = build_spectrum_export_frame(
            energies_eV=np.array([10.0, 20.0]),
            spin_curves={
                "sum_check_up": np.array([1.0, 1.0]),
                "sum_check_dn": np.array([1.0, 1.0]),
                "spin_mean_up": np.array([0.25, 0.5]),
                "spin_mean_dn": np.array([-0.25, -0.5]),
            },
        )

        self.assertEqual(
            list(frame.columns),
            [
                "energy_eV",
                "sum_probability_initial_up",
                "sum_probability_initial_down",
                "spin_mean_initial_up",
                "spin_mean_initial_down",
            ],
        )
        np.testing.assert_allclose(frame["energy_eV"].to_numpy(dtype=float), np.array([10.0, 20.0]))
        np.testing.assert_allclose(frame["spin_mean_initial_up"].to_numpy(dtype=float), np.array([0.25, 0.5]))

    def test_export_bundle_writes_json_xml_and_xlsx(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "spectrum_export"
            exported = export_spectrum_bundle(
                base_path=base_path,
                energies_eV=np.array([10.0, 20.0]),
                spin_curves={
                    "sum_check_up": np.array([1.0, 1.0]),
                    "sum_check_dn": np.array([1.0, 1.0]),
                    "spin_mean_up": np.array([0.25, 0.5]),
                    "spin_mean_dn": np.array([-0.25, -0.5]),
                },
                metadata={
                    "formula_label": "Тестовая модель",
                    "orbital_l": 1,
                    "lz_chain": [1, 0, -1],
                },
            )

            self.assertEqual(set(exported.keys()), {"json", "xml", "xlsx"})
            for path in exported.values():
                self.assertTrue(path.exists(), msg=f"Не найден файл {path}")

            json_payload = json.loads(exported["json"].read_text(encoding="utf-8"))
            self.assertEqual(json_payload["metadata"]["formula_label"], "Тестовая модель")
            self.assertEqual(json_payload["columns"][0], "energy_eV")
            self.assertEqual(json_payload["rows"][1]["spin_mean_initial_down"], -0.5)

            xml_root = ET.parse(exported["xml"]).getroot()
            self.assertEqual(xml_root.tag, "spectrum_export")
            rows = xml_root.find("rows")
            self.assertIsNotNone(rows)
            self.assertEqual(len(list(rows)), 2)
            first_row = rows[0]
            self.assertEqual(first_row.findtext("energy_eV"), "10")
            self.assertEqual(first_row.findtext("spin_mean_initial_up"), "0.25")

            with zipfile.ZipFile(exported["xlsx"]) as archive:
                names = set(archive.namelist())
                self.assertIn("[Content_Types].xml", names)
                self.assertIn("xl/workbook.xml", names)
                self.assertIn("xl/worksheets/sheet1.xml", names)
                self.assertIn("xl/worksheets/sheet2.xml", names)

                sheet1 = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
                sheet2 = archive.read("xl/worksheets/sheet2.xml").decode("utf-8")

            self.assertIn("energy_eV", sheet1)
            self.assertIn("spin_mean_initial_down", sheet1)
            self.assertIn(">20<", sheet1)
            self.assertIn("formula_label", sheet2)
            self.assertIn("Тестовая модель", sheet2)


if __name__ == "__main__":
    unittest.main()
