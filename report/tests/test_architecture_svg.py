"""Validate the supplied-SVG replacement without needing CairoSVG at test time."""
import hashlib
import json
from pathlib import Path
import unittest
from xml.etree import ElementTree as ET

import fitz

ROOT = Path(__file__).resolve().parents[2]


class ArchitectureSvgTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "rag/results/report-architecture/svg-summary.json").read_text())

    def test_source_converter_and_asset_hashes_match(self):
        for prefix in ("source", "converter", "pdf", "preview"):
            data = (ROOT / self.manifest[prefix + "_path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), self.manifest[prefix + "_sha256"])
        self.assertFalse(self.manifest["source_modified"])

    def test_vector_pdf_preserves_all_twelve_arrowheads(self):
        svg = ET.parse(ROOT / self.manifest["source_path"]).getroot()
        markers = sum("marker-end" in node.attrib for node in svg.iter())
        with fitz.open(ROOT / self.manifest["pdf_path"]) as doc:
            self.assertEqual(len(doc), 1)
            page = doc[0]
            self.assertEqual(page.get_images(), [])
            triangles = [d for d in page.get_drawings() if d["fill"] is not None
                         and len(d["items"]) == 3
                         and all(item[0] == "l" for item in d["items"])]
            self.assertEqual(markers, 12)
            self.assertEqual(len(triangles), markers)
            self.assertAlmostEqual(page.rect.width / page.rect.height, 1800 / 875)
            self.assertIn("GLM-ASR", page.get_text())
            self.assertNotIn("System / Data-Flow Overview", page.get_text())

    def test_report_uses_replacement_not_superseded_diagram(self):
        tex = (ROOT / "report/latex_template/report.tex").read_text()
        self.assertIn(r"\includegraphics[width=\linewidth]{figures/ASR_RAG_System_Architecture_No_Title.pdf}", tex)
        self.assertNotIn(r"\includegraphics[width=\linewidth]{figures/system-architecture.pdf}", tex)


if __name__ == "__main__":
    unittest.main()
