from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extensions.run_qwen_dev_validation import validate_input_rows


class DevValidationTests(unittest.TestCase):
    def fixtures(self):
        ranked = [{"query_id": f"q{i}", "transcript": f"Question {i}",
                   "record_ids": [f"r{j}" for j in range(100)]} for i in range(2)]
        sample = [{"query_id": f"q{i}", "transcript": f"Question {i}", "split": "dev"} for i in range(2)]
        return ranked, sample

    def test_complete_matching_inputs(self):
        ranked, sample = self.fixtures()
        validate_input_rows(ranked, sample, expected=2)

    def test_practice_split_is_rejected(self):
        ranked, sample = self.fixtures()
        sample[0]["split"] = "practice"
        with self.assertRaises(ValueError):
            validate_input_rows(ranked, sample, expected=2)

    def test_transcript_change_is_rejected(self):
        ranked, sample = self.fixtures()
        ranked[0]["transcript"] = "Changed"
        with self.assertRaises(ValueError):
            validate_input_rows(ranked, sample, expected=2)

    def test_duplicate_candidates_are_rejected(self):
        ranked, sample = self.fixtures()
        ranked[0]["record_ids"][1] = ranked[0]["record_ids"][0]
        with self.assertRaises(ValueError):
            validate_input_rows(ranked, sample, expected=2)

    def test_missing_or_duplicate_queries_are_rejected(self):
        ranked, sample = self.fixtures()
        with self.assertRaises(ValueError):
            validate_input_rows(ranked[:1], sample, expected=2)
        with self.assertRaises(ValueError):
            validate_input_rows(ranked + ranked[:1], sample, expected=2)


if __name__ == "__main__":
    unittest.main()
