"""Regression checks for the report's offline diagnostic evidence."""
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "rag/results/practice-error-analysis-v1/summary.json"


class ErrorAnalysisEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads(SUMMARY.read_text())
        cls.notes = json.loads((ROOT / "rag/experiments/practice-error-analysis-v1.json").read_text())
        cls.appendix = (ROOT / "report/latex_template/practice_error_analysis.tex").read_text()
        cls.report = (ROOT / "report/latex_template/report.tex").read_text()

    def test_report_population_counts_match_saved_judgements(self):
        for arm in ("dense5", "qwen4_matched5"):
            path = ROOT / f"rag/results/qwen4-answers-3724290/judge_json_{arm}/answer_judge_per_query.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            zeros = sum(row["normalised_answer_score"] == 0 for row in rows)
            pop = self.summary["population"][arm]
            self.assertEqual(len(rows), pop["queries"])
            self.assertEqual(zeros, pop["all_zero"])
            self.assertEqual(sum(pop["zero_by_qrel_coverage"].values()), zeros)
        self.assertEqual(self.summary["population"]["qwen4_matched5"]["zero_by_qrel_coverage"]["full"], 30)

    def test_six_flags_are_not_six_rescored_answers(self):
        flags = sum("judge_text_conflict" in case["tags"] for case in self.notes["cases"])
        self.assertEqual(flags, 6)
        self.assertEqual(self.summary["qualitative_tags_nonexclusive"]["judge_text_conflict"], flags)
        self.assertIn("not six corrected scores", self.appendix)
        self.assertIn("All original scores and paired intervals are retained", self.appendix)

    def test_appendix_query_examples_belong_to_fixed_sample(self):
        ids = {q.replace(r"\_", "_") for q in re.findall(r"q(?:\\_[a-f0-9]+)", self.appendix)}
        self.assertGreaterEqual(len(ids), 6)
        self.assertTrue(ids <= set(self.summary["sampled_query_ids"]))
        self.assertEqual(len(self.summary["sampled_query_ids"]), 20)

    def test_provenance_and_uncertainty_are_preserved(self):
        for path, digest in self.summary["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest, path)
        script = ROOT / "rag/extensions/analyze_practice_errors.py"
        self.assertEqual(hashlib.sha256(script.read_bytes()).hexdigest(), self.summary["script_sha256"])
        self.assertIn("[-0.0007,0.1127]", self.report)
        self.assertIn("unblinded AI-assisted", self.report)
        self.assertIn("practice_error_analysis.tex", self.report)


if __name__ == "__main__":
    unittest.main()
