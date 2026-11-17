import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "quality_cost", Path(__file__).resolve().parents[1] / "plot_quality_cost.py")
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


def fixture():
    return {
        "queries": 350,
        "arms": {
            "dense5": {"mean_normalised_answer_score": .30,
                       "mean_attributed_selection_plus_generation_ms": 2000,
                       "mean_server_input_tokens": 1290},
            "qwen4_matched5": {"mean_normalised_answer_score": .36,
                              "mean_attributed_selection_plus_generation_ms": 5000,
                              "mean_server_input_tokens": 1267},
        },
        "paired_qwen4_minus_dense": {
            "normalised_answer_score": {"mean_delta": .06,
                                       "query_bootstrap_95pct_interval": [.01, .10]}}
    }


class QualityCostTests(unittest.TestCase):
    def test_mean_milliseconds_are_converted_to_seconds(self):
        result = plot.extract(fixture())
        self.assertEqual([p["mean_attributed_seconds"] for p in result["points"]], [2, 5])
        self.assertEqual(result["additional_attributed_seconds"], 3)

    def test_relative_gain_and_cost_use_the_fresh_dense_denominator(self):
        result = plot.extract(fixture())
        self.assertAlmostEqual(result["relative_mean_score_gain_percent"], 20)
        self.assertAlmostEqual(result["relative_attributed_cost_increase_percent"], 150)

    def test_paired_interval_is_not_converted_to_arm_intervals(self):
        source = fixture()
        result = plot.extract(source)
        self.assertEqual(result["paired_score_delta"],
                         source["paired_qwen4_minus_dense"]["normalised_answer_score"])
        self.assertTrue(all("interval" not in p for p in result["points"]))

    def test_invalid_score_or_cost_is_rejected(self):
        for key, value in (("mean_normalised_answer_score", 1.2),
                           ("mean_attributed_selection_plus_generation_ms", 0)):
            source = fixture()
            source["arms"]["dense5"][key] = value
            with self.assertRaises(AssertionError):
                plot.extract(source)


if __name__ == "__main__":
    unittest.main()
