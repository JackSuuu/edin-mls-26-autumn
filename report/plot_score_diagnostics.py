"""Plot paired score changes for the practice runs."""

import json
from collections import Counter
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "rag/results"
ANSWERS = RESULTS / "qwen4-answers-3724290"
FIGURE = ROOT / "report/latex_template/figures/score-diagnostics.pdf"


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def count_outcomes():
    scores = {}
    for arm in ("dense5", "qwen4_matched5"):
        rows = read_jsonl(ANSWERS / f"judge_json_{arm}/answer_judge_per_query.jsonl")
        scores[arm] = {row["query_id"]: row["normalised_answer_score"] for row in rows}
    plans = {
        (row["query_id"], row["arm"]): row["retrieved_record_ids"]
        for row in read_jsonl(ANSWERS / "plans.jsonl")
    }
    rankings = {
        row["query_id"]: row
        for row in read_jsonl(RESULTS / "qwen4-retrieval-3724174/full/rankings.jsonl")
    }
    if not rankings.keys() == scores["dense5"].keys() == scores["qwen4_matched5"].keys():
        raise ValueError("Ranking and answer records must cover the same practice queries")

    totals, unchanged_recall = Counter(), Counter()
    for query, row in rankings.items():
        dense, qwen = row["dense"], row["qwen_matched"]
        dense_ids, qwen_ids = dense["record_ids"], qwen["record_ids"]
        if (dense_ids[:5] != plans[query, "dense5"]
                or qwen_ids[:5] != plans[query, "qwen4_matched5"]):
            raise ValueError(f"Rankings and generation plans disagree on {query}")

        dense_score, qwen_score = scores["dense5"][query], scores["qwen4_matched5"][query]
        change = (qwen_score > dense_score) - (qwen_score < dense_score)
        totals[change] += 1
        if dense["metrics"]["Recall@5"] == qwen["metrics"]["Recall@5"]:
            unchanged_recall[change] += 1

    return totals, unchanged_recall


def draw_figure(totals, unchanged_recall):
    black = (0, 0, 0)
    grid = (0.89, 0.89, 0.89)
    same_fill = (0.24, 0.41, 0.59)
    changed_fill = (0.77, 0.85, 0.92)
    font = fitz.Font("tiro")
    with fitz.open() as document:
        page = document.new_page(width=288, height=106)
        left, right, top, bottom = 47, 275, 18, 78
        scale = (right - left) / 170

        def text(x, y, label, size=10, align=0, color=black):
            x -= align * font.text_length(label, fontsize=size)
            page.insert_text((x, y), label, fontsize=size, color=color, fontname="tiro")

        for x, fill, label in ((47, same_fill, "Recall@5 unchanged"),
                               (150, changed_fill, "Recall@5 changed")):
            page.draw_rect((x, 5, x + 8, 12), color=None, fill=fill)
            text(x + 11, 12, label, size=9)

        for tick in (0, 50, 100, 150):
            x = left + tick * scale
            page.draw_line((x, top), (x, bottom), color=grid, width=.45)
            page.draw_line((x, bottom), (x, bottom + 3), color=black, width=.5)
            text(x, 90, str(tick), size=9, align=.5)
        page.draw_line((left, bottom), (right, bottom), color=black, width=.6)
        page.draw_line((left, top), (left, bottom), color=black, width=.6)
        text((left + right) / 2, 103, "Queries", align=.5)

        for y, change, label in ((27, 1, "Higher"), (48, 0, "Tied"), (69, -1, "Lower")):
            same_end = left + unchanged_recall[change] * scale
            total_end = left + totals[change] * scale
            page.draw_rect((left, y - 5.5, same_end, y + 5.5), color=None, fill=same_fill)
            page.draw_rect((same_end, y - 5.5, total_end, y + 5.5), color=None, fill=changed_fill)
            text(left - 5, y + 3.5, label, align=1)
            text((left + same_end) / 2, y + 3.1, str(unchanged_recall[change]),
                 size=9, align=.5, color=(1, 1, 1))
            text(total_end + 4, y + 3.3, str(totals[change]), size=9)

        document.set_metadata({"title": "Paired answer-score changes"})
        document.save(FIGURE, garbage=4, deflate=True, no_new_id=True)


if __name__ == "__main__":
    draw_figure(*count_outcomes())
