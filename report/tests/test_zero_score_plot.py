import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("zeros", ROOT / "report/plot_zero_scores.py")
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


class ZeroScorePlotTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "rag/results/practice-error-analysis-v1/summary.json").read_text())

    def test_bar_denominator_is_all_queries_not_only_zeros(self):
        data = plot.extract(self.data)
        self.assertEqual(data["arm_segments"]["dense5"], [212, 90, 29, 19])
        self.assertEqual(data["arm_segments"]["qwen4_matched5"], [229, 62, 29, 30])
        self.assertTrue(all(sum(v) == 350 for v in data["arm_segments"].values()))

    def test_paired_transition_counts(self):
        self.assertEqual(plot.extract(self.data)["paired_states"],
                         {"zero_zero": 85, "zero_positive": 53, "positive_zero": 36, "positive_positive": 176})

    def test_invalid_coverage_total_is_rejected(self):
        bad = copy.deepcopy(self.data)
        bad["population"]["dense5"]["zero_by_qrel_coverage"]["full"] += 1
        with self.assertRaises(AssertionError):
            plot.extract(bad)

    def test_inconsistent_matrix_marginal_is_rejected(self):
        bad = copy.deepcopy(self.data)
        bad["paired_score_states"]["zero_positive"] -= 1
        bad["paired_score_states"]["positive_positive"] += 1
        with self.assertRaises(AssertionError):
            plot.extract(bad)
