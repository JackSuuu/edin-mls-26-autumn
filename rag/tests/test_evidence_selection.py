"""Check evidence fidelity and hard limits, without models or API requests."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extensions.evidence_selection import render_prompt, select_context, chat_input_tokens
from src.context_builder import SimpleContextBuilder


class SelectionTests(unittest.TestCase):
    def test_chat_counter_handles_transformers_mapping_and_list_outputs(self):
        from collections import UserDict
        class Tokenizer:
            def __init__(self, result):
                self.result = result
            def apply_chat_template(self, *args, **kwargs):
                return self.result
        for encoded in ([1, 2, 3, 4], {"input_ids": [1, 2, 3, 4], "attention_mask": [1] * 4},
                        UserDict({"input_ids": [[1, 2, 3, 4]], "attention_mask": [[1] * 4]})):
            self.assertEqual(chat_input_tokens(Tokenizer(encoded), "S", "Q"), 4)

    def test_original_top5_prompt_is_unchanged(self):
        records = {str(i): f"Original text {i}." for i in range(7)}
        class Data:
            def record_text(self, rid):
                return records[rid]
        expected = SimpleContextBuilder().build("Question?", list(records)[:5], Data())
        actual = select_context("Question?", list(records), records, len, top_k=5)
        self.assertEqual(expected["prompt"], actual["prompt"])

    def test_budget_truncates_last_document_without_changing_prefix(self):
        records = {"a": "A" * 20, "b": "B" * 100, "c": "C" * 20}
        budget = len(render_prompt("Q", [records["a"], "B" * 12]))
        result = select_context("Q", list(records), records, len, budget=budget)
        self.assertEqual(result["retrieved_record_ids"], ["a", "b"])
        self.assertEqual(result["local_input_tokens"], budget)
        self.assertTrue(result["selected_documents"][-1]["truncated"])
        self.assertEqual(result["selected_documents"][-1]["included_characters"], 12)

    def test_budget_smaller_than_scaffold_fails(self):
        with self.assertRaises(ValueError):
            select_context("Q", ["a"], {"a": "A"}, len, budget=1)

    def test_no_room_for_document_header(self):
        result = select_context("Q", ["a"], {"a": "A" * 100}, len,
                                budget=len(render_prompt("Q", [])))
        self.assertEqual(result["retrieved_record_ids"], [])
        self.assertEqual(result["prompt"], render_prompt("Q", []))

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            select_context("Q", ["a", "a"], {"a": "A"}, len)


if __name__ == "__main__":
    unittest.main()
