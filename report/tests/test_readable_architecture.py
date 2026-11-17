import hashlib
import json
from pathlib import Path
import unittest

import fitz

ROOT = Path(__file__).resolve().parents[2]


class ReadableArchitectureTests(unittest.TestCase):
    def test_hashes_and_original_retention(self):
        s = json.loads((ROOT / "rag/results/report-architecture/readable-summary.json").read_text())
        for prefix in ("source", "pdf", "png"):
            self.assertEqual(hashlib.sha256((ROOT / s[prefix + "_path"]).read_bytes()).hexdigest(),
                             s[prefix + "_sha256"])
        self.assertFalse(s["original_svg_modified"])
        self.assertGreaterEqual(s["minimum_font_size_pt_at_396pt_report_width"], 6.5)

    def test_reflow_is_vector_and_labels_fit_at_report_scale(self):
        with fitz.open(ROOT / "report/latex_template/figures/architecture-report.pdf") as doc:
            page = doc[0]
            self.assertFalse(page.get_images())
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        self.assertGreaterEqual(span["size"] * 396 / page.rect.width, 6.5)
                        self.assertTrue(page.rect.contains(fitz.Rect(span["bbox"])))
