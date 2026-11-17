import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "architecture", Path(__file__).resolve().parents[1] / "draw_system_architecture.py")
architecture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(architecture)


class ArchitectureTests(unittest.TestCase):
    def test_spec_has_unique_nodes_and_known_edges(self):
        architecture.validate_spec()

    def test_qrels_feed_only_offline_evaluation(self):
        edges = [edge for edge in architecture.EDGES if edge["from"] == "qrels"]
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["to"], "judge")
        self.assertEqual(edges[0]["kind"], "evaluation")

    def test_no_judge_feedback_into_selection(self):
        self.assertFalse(any(edge["from"] == "judge" for edge in architecture.EDGES))
        self.assertTrue(any(edge["from"] == "generator" and edge["to"] == "judge"
                            and edge["kind"] == "evaluation" for edge in architecture.EDGES))

    def test_offline_evaluation_receives_rankings_answers_and_qrels(self):
        sources = {edge["from"] for edge in architecture.EDGES
                   if edge["to"] == "judge" and edge["kind"] == "evaluation"}
        self.assertEqual(sources, {"selector", "generator", "qrels"})

    def test_full_corpus_supplies_context_and_selector(self):
        targets = {edge["to"] for edge in architecture.EDGES if edge["from"] == "corpus"}
        self.assertEqual(targets, {"dense", "selector", "context"})

    def test_arrowhead_points_back_from_tip(self):
        heads = architecture.arrowhead((0, 0), (10, 0))
        self.assertTrue(all(x < 10 for x, _ in heads))
        self.assertAlmostEqual(heads[0][1], -heads[1][1])


if __name__ == "__main__":
    unittest.main()
