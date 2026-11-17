"""Stdlib-only tests; do not load models, make requests, or import API clients."""
import importlib.util
import copy
import json
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "practice_errors", Path(__file__).resolve().parents[1] / "extensions/analyze_practice_errors.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class PracticeErrorAnalysisTests(unittest.TestCase):
    def test_coverage_uses_ids_not_number_of_selected_slots(self):
        self.assertEqual(analysis.coverage(["a", "a"], {"a": 1, "b": 1}), "partial")
        self.assertEqual(analysis.coverage(["c"], {"a": 1}), "none")
        self.assertEqual(analysis.coverage(["a", "c"], {"a": 1}), "full")

    def test_sample_is_unique_deterministic_and_order_invariant(self):
        ids = [str(i) for i in range(121)]
        a = analysis.sample_ids(ids)
        self.assertEqual(len(a), 20)
        self.assertEqual(len(set(a)), 20)
        self.assertEqual(a, analysis.sample_ids(reversed(ids)))

    def test_insufficient_sample_population_fails(self):
        with self.assertRaises(ValueError):
            analysis.sample_ids(["a"], size=2)

    def test_paired_score_states_keep_both_directions(self):
        dense = {q: {"normalised_answer_score": s} for q, s in zip("abcd", [0, 0, .5, 1])}
        qwen = {q: {"normalised_answer_score": s} for q, s in zip("abcd", [0, .2, 0, 1])}
        self.assertEqual(analysis.score_states(dense, qwen),
                         {"zero_zero": 1, "zero_positive": 1, "positive_zero": 1, "positive_positive": 1})

    def test_mismatched_pair_ids_fail(self):
        with self.assertRaises(AssertionError):
            analysis.score_states({"a": {}}, {"b": {}})

    def test_descriptive_counts_keep_overlap_and_sample_scope(self):
        notes = json.loads((analysis.ROOT / analysis.NOTES).read_text())
        recheck = json.loads((analysis.ROOT / analysis.RECHECK).read_text())
        result = analysis.summarize_diagnostics(notes, recheck)
        self.assertEqual(result["category_counts"],
                         {"evidence_use": 9, "topic_entity_mismatch": 3, "reference_alignment": 12,
                          "judge_text_conflict": 6, "source_quality_risk": 1})
        self.assertEqual(result["confidence_counts"], {"confirmed": 3, "possible": 9, "ambiguous": 8})
        self.assertEqual(sum(result["confidence_counts"].values()), 20)
        self.assertGreater(sum(result["category_counts"].values()), 20)
        self.assertFalse(result["complete_20_case_human_review"])

    def test_unconfirmed_case_cannot_be_promoted_to_confirmed(self):
        notes = json.loads((analysis.ROOT / analysis.NOTES).read_text())
        recheck = json.loads((analysis.ROOT / analysis.RECHECK).read_text())
        notes["cases"][0]["confidence"] = "confirmed"
        with self.assertRaises(AssertionError):
            analysis.summarize_diagnostics(notes, recheck)

    def test_duplicate_category_cannot_inflate_case_counts(self):
        notes = json.loads((analysis.ROOT / analysis.NOTES).read_text())
        recheck = json.loads((analysis.ROOT / analysis.RECHECK).read_text())
        bad = copy.deepcopy(notes)
        bad["cases"][0]["diagnostic_categories"] *= 2
        with self.assertRaises(AssertionError):
            analysis.summarize_diagnostics(bad, recheck)

    def test_quote_is_checked_against_named_source(self):
        case = {"query_id": "q", "gold_documents": [{"record_id": "g", "text": "gold text"}],
                "arms": {"qwen4_matched5": {"answer": "candidate answer",
                         "judge_explanations": ["judge explanation"],
                         "selected_documents": [{"record_id": "s", "text": "selected text"}]}}}
        analysis.verify_quotes(case, {"quotes": [{"source": "selected", "record_id": "s", "quote": "selected"}]})
        with self.assertRaises(AssertionError):
            analysis.verify_quotes(case, {"quotes": [{"source": "gold", "record_id": "g", "quote": "selected"}]})


if __name__ == "__main__":
    unittest.main()
