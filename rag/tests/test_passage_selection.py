"""Check source fidelity, overlap handling and complete-chat budget constraints."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extensions.passage_selection import sentence_spans, make_windows, materialise, pack_windows
from extensions.run_passage_pilot import read_generator_system_prompt


class PassageSelectionTests(unittest.TestCase):
    def test_gpu_prompt_reader_needs_no_dotenv_or_api_import(self):
        with patch.dict(sys.modules, {'dotenv': None}):
            value = read_generator_system_prompt()
        self.assertEqual(value, "Use the supplied evidence to answer the user's question. Do not invent facts that are unsupported by the evidence.")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'generator.py'
            source.write_text('SYSTEM_PROMPT = make_prompt()\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                read_generator_system_prompt(source)

    def test_sentence_offsets_preserve_abbreviations_decimal_and_negation(self):
        text = 'Dr. Smith is in the U.S. and pays 3.5 percent. This applies only to firms. It does not apply to pensions. Final sentence.'
        spans = sentence_spans(text)
        self.assertEqual(len(spans), 4)
        windows = make_windows(['r'], {'r': text})
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]['sentence_stop'], 3)
        self.assertEqual(windows[1]['sentence_start'], 2)
        for w in windows:
            self.assertEqual(w['text'], text[w['start']:w['end']])
        self.assertIn('does not apply', windows[0]['text'])
        self.assertIn('does not apply', windows[1]['text'])

    def test_overlapping_windows_do_not_duplicate_text_and_keep_source_order(self):
        records = {'a': 'First sentence. Second sentence. Third sentence. Fourth sentence.', 'b': 'Other source.'}
        windows = make_windows(['a', 'b'], records)
        built = pack_windows('Q?', ['a', 'b'], records, windows, [1, 2, 9], len, 1000)
        self.assertEqual(built['retrieved_record_ids'], ['a', 'b'])
        self.assertEqual(built['prompt'].count('Third sentence.'), 1)
        self.assertEqual(built['selected_documents'][0]['character_ranges'], [[0, len(records['a'])]])

    def test_budget_checks_full_prompt_and_skips_nonfitting_window(self):
        records = {'a': 'Long words ' * 100 + '.', 'b': 'Short evidence.'}
        windows = make_windows(['a', 'b'], records)
        short = materialise('Q?', ['a', 'b'], records, [windows[-1]], len)
        built = pack_windows('Q?', ['a', 'b'], records, windows, [9, 1], len, short['local_input_tokens'])
        self.assertEqual(built['retrieved_record_ids'], ['b'])
        self.assertLessEqual(built['local_input_tokens'], built['input_budget'])
        with self.assertRaises(ValueError):
            pack_windows('Q?', ['a', 'b'], records, windows, [9, 1], len, 1)

    def test_noncontiguous_ranges_have_explicit_omission_marker(self):
        records = {'r': 'Alpha. Beta. Gamma. Delta. Epsilon.'}
        windows = make_windows(['r'], records, sentences=1, overlap=0)
        built = materialise('Q?', ['r'], records, [windows[0], windows[-1]], len)
        self.assertIn('[...]', built['prompt'])
        self.assertNotIn('Gamma', built['prompt'])
        with self.assertRaises(ValueError):
            pack_windows('Q?', ['r'], records, windows, [float('nan')] * len(windows), len, 1000)


if __name__ == '__main__':
    unittest.main()
