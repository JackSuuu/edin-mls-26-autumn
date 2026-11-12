"""Output-format option preserves default rubric, parser and request semantics."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.answer_judge import Judge


class JudgeProtocolTests(unittest.TestCase):
    def call(self, json_mode):
        captured = {}
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content='{"correctness":3,"grounding":2,"completeness":1,"explanation":"reason"}'))])
        judge = Judge.__new__(Judge)
        judge.model, judge.retries = "test-model", 1
        judge.json_mode, judge.raw_log_path = json_mode, None
        judge.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = judge.evaluate("Question", "Answer", [{"record_id": "r", "text": "Evidence"}])
        return captured, result

    def test_default_request_is_unchanged(self):
        args, result = self.call(False)
        self.assertNotIn("response_format", args)
        self.assertEqual(args["temperature"], 0.0)
        self.assertEqual(result["correctness"], 3)

    def test_json_mode_changes_only_response_format(self):
        default, _ = self.call(False)
        structured, _ = self.call(True)
        self.assertEqual(structured.pop("response_format"), {"type": "json_object"})
        self.assertEqual(structured, default)

    def test_invalid_scores_and_json_remain_rejected(self):
        for raw in ('{"correctness":true,"grounding":2,"completeness":1}',
                    '{"correctness":5,"grounding":2,"completeness":1}', '{broken'):
            with self.assertRaises((ValueError, KeyError)):
                Judge.parse(raw)


if __name__ == "__main__":
    unittest.main()
