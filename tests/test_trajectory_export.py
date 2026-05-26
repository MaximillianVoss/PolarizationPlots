import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from polarization_app.application.trajectory import (
    TRAJECTORY_SWEEP_ENERGY,
    TRAJECTORY_SWEEP_IMPACT,
    TrajectorySweepRequest,
    execute_trajectory_sweep,
    trajectory_export_metadata,
)
from polarization_app.application.trajectory_export import export_trajectory_bundle


class TrajectoryExportTestCase(unittest.TestCase):
    def test_export_bundle_writes_json_xml_and_xlsx(self):
        result = execute_trajectory_sweep(
            TrajectorySweepRequest(
                sweep_mode=TRAJECTORY_SWEEP_ENERGY,
                point_count=1,
                atomic_number=29.0,
                energy_min_eV=100.0,
                energy_max_eV=101.0,
                impact_parameter_ang=0.8,
                r0_ang=10.0,
                angle_step_deg=3.0,
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            exported = export_trajectory_bundle(
                Path(tmpdir) / "trajectory",
                frame=result.frame,
                metadata=trajectory_export_metadata(result),
            )

            self.assertEqual(set(exported), {"json", "xml", "xlsx"})
            for path in exported.values():
                self.assertTrue(path.exists())

            json_payload = json.loads(exported["json"].read_text(encoding="utf-8"))
            self.assertEqual(json_payload["metadata"]["sweep_mode"], TRAJECTORY_SWEEP_ENERGY)
            self.assertIn("phase_rad", json_payload["columns"])

            xml_root = ET.parse(exported["xml"]).getroot()
            self.assertEqual(xml_root.tag, "trajectory_export")

            with zipfile.ZipFile(exported["xlsx"]) as archive:
                self.assertIn("xl/worksheets/sheet1.xml", archive.namelist())
                self.assertIn("xl/worksheets/sheet2.xml", archive.namelist())

    def test_export_failed_rows_without_nan_literals(self):
        result = execute_trajectory_sweep(
            TrajectorySweepRequest(
                sweep_mode=TRAJECTORY_SWEEP_IMPACT,
                point_count=2,
                atomic_number=29.0,
                energy_eV=100.0,
                impact_min_ang=0.8,
                impact_max_ang=11.0,
                r0_ang=10.0,
                angle_step_deg=3.0,
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            exported = export_trajectory_bundle(
                Path(tmpdir) / "trajectory",
                frame=result.frame,
                metadata=trajectory_export_metadata(result),
            )

            json_payload = json.loads(exported["json"].read_text(encoding="utf-8"))
            failed_row = json_payload["rows"][1]
            self.assertIsNone(failed_row["phase_rad"])
            with zipfile.ZipFile(exported["xlsx"]) as archive:
                sheet_xml = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
            self.assertNotIn(">nan<", sheet_xml.lower())


if __name__ == "__main__":
    unittest.main()
