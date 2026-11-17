"""Stdlib-only tests; do not load models, make requests, or import API clients."""
import importlib.util
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
