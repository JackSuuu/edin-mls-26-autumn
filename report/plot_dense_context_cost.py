"""Plot practice first-hit ranks and the Dense5/Dense20 generation comparison."""
import json
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
RANKINGS = ROOT / "rag/results/diagnostic-retrieval-3713758/rankings.jsonl"
COMPARISON = ROOT / "rag/results/selection-run-full/comparison.json"
FIGURE = ROOT / "report/latex_template/figures/dense-context-cost.pdf"


def first_hit_counts(rankings):
    through_5, through_10, through_20, through_100 = (
        sum(row[f"Recall@{k}"] > 0 for row in rankings) for k in (5, 10, 20, 100)
    )
    return (
        through_5,
        through_10 - through_5,
        through_20 - through_10,
        through_100 - through_20,
        len(rankings) - through_100,
    )


def main():
    with RANKINGS.open(encoding="utf-8") as source:
        rankings = [json.loads(line) for line in source if line.strip()]
    data = json.loads(COMPARISON.read_text(encoding="utf-8"))
    paired = data["paired"]["dense20 minus dense5"]["query_differences"]
    if {row["query_id"] for row in rankings} != {row["query_id"] for row in paired}:
        raise ValueError("The ranking and generation archives cover different queries")

    counts = first_hit_counts(rankings)
    arms = [data["per_arm"][name] for name in ("dense5", "dense20")]
    tokens = [arm["mean_local_input_tokens"] for arm in arms]
    scores = [arm["mean_judge_normalised_answer_score"] for arm in arms]

    with fitz.open() as document:
        page = document.new_page(width=450, height=141)
        font = fitz.Font("tiro")
        black = (0, 0, 0)
        grid = (0.89, 0.89, 0.89)
        rank_fill = (0.56, 0.65, 0.74)
        dense5_fill = (0.26, 0.40, 0.55)
        dense20_fill = (0.79, 0.49, 0.30)
        top, bottom = 13, 108

        def text(x, y, label, size=10, align=.5):
            x -= align * font.text_length(label, fontsize=size)
            page.insert_text((x, y), label, fontname="tiro", fontsize=size, color=black)

        def bars(left, right, label_x, ylabel, ticks, labels, values, colors, value_format):
            scale = (bottom - top) / ticks[-1]
            step = (right - left) / len(labels)
            centre_y = (top + bottom) / 2
            centre_y += font.text_length(ylabel, fontsize=10) / 2
            page.insert_text((label_x, centre_y), ylabel, rotate=90,
                             fontname="tiro", fontsize=10, color=black)
            for value in ticks:
                y = bottom - value * scale
                page.draw_line((left, y), (right, y), color=grid, width=.45)
                page.draw_line((left - 3, y), (left, y), color=black, width=.6)
                tick = f"{value:,g}"
                text(left - 5, y + 3, tick, size=9, align=1)
            page.draw_line((left, top), (left, bottom), color=black, width=.65)
            page.draw_line((left, bottom), (right, bottom), color=black, width=.65)

            for index, (name, value, color) in enumerate(zip(labels, values, colors)):
                x = left + (index + .5) * step
                y = bottom - value * scale
                half_width = step * .26
                if value:
                    page.draw_rect((x - half_width, y, x + half_width, bottom),
                                   color=None, fill=color)
                page.draw_line((x, bottom), (x, bottom + 3), color=black, width=.6)
                text(x, bottom + 13, name, size=9)
                text(x, y - 4, format(value, value_format), size=9)

        bins = ("1-5", "6-10", "11-20", "21-100", ">100")
        bars(35, 187, 10, "Queries", (0, 50, 100, 150, 200, 250),
             bins, counts, (rank_fill,) * len(bins), "d")
        text(111, 136, "First labelled rank")
        choices = ("Dense5", "Dense20")
        colors = (dense5_fill, dense20_fill)
        bars(241, 320, 199, "Mean input (tokens)", (0, 3000, 6000),
             choices, tokens, colors, ",.0f")
        bars(363, 443, 331, "Mean judge score", (0, .5, 1),
             choices, scores, colors, ".4f")
        text(342, 136, "Context selection")
        document.set_metadata({"title": "First labelled hit, input length, and answer score"})
        document.save(FIGURE, garbage=4, deflate=True, no_new_id=True)


if __name__ == "__main__":
    main()
