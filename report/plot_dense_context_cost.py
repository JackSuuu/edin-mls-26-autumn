"""Plot the measured Dense5 and Dense20 configurations without model calls."""
import json
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
SPLITS = (
    ("Practice", "selection-run-full"),
    ("Dev", "dev-validation"),
)
FIGURE = ROOT / "report/latex_template/figures/dense-context-cost.pdf"


def main():
    with fitz.open() as document:
        page = document.new_page(width=432, height=132)
        ink = (0.12, 0.16, 0.20)
        grid = (0.84, 0.87, 0.90)
        blue = (0.13, 0.32, 0.54)
        font = fitz.Font("tiro")

        def text(x, y, value, size=9, align=0):
            x -= align * font.text_length(value, fontsize=size)
            page.insert_text((x, y), value, fontname="tiro", fontsize=size, color=ink)

        text(216, 11, "Mean normalised judge score", 10, align=.5)
        for panel, (split, run) in enumerate(SPLITS):
            source = ROOT / "rag/results" / run / "comparison.json"
            data = json.loads(source.read_text(encoding="utf-8"))
            left = 28 + panel * 216
            right, top, bottom = left + 172, 30, 96
            text((left + right) / 2, 24, f"{split} ({data['queries']} queries)", 10, align=.5)
            for value in (0, .5, 1):
                y = bottom - value * (bottom - top)
                page.draw_line((left, y), (right, y), color=grid, width=.45)
                text(left - 5, y + 3, f"{value:g}", align=1)
            for value in (0, 2500, 5000):
                x = left + value / 5500 * (right - left)
                page.draw_line((x, bottom), (x, bottom + 3), color=ink, width=.6)
                text(x, bottom + 13, f"{value:,}", align=.5)
            page.draw_line((left, top), (left, bottom), color=ink, width=.7)
            page.draw_line((left, bottom), (right, bottom), color=ink, width=.7)

            points = []
            for arm, label in (("dense5", "Dense5"), ("dense20", "Dense20")):
                metrics = data["per_arm"][arm]
                score = metrics["mean_judge_normalised_answer_score"]
                tokens = metrics["mean_local_input_tokens"]
                x = left + tokens / 5500 * (right - left)
                y = bottom - score * (bottom - top)
                points.append((x, y))
                text(x, y - 28, label, align=.5)
                text(x, y - 17, f"{score:.4f}", align=.5)
                text(x, y + 13, f"{tokens:,.0f}", size=8, align=.5)
            page.draw_line(*points, color=blue, width=1.2)
            for point in points:
                page.draw_circle(point, 2.1, color=blue, fill=blue)

        text(216, 126, "Mean generator input (tokens per query)", 10, align=.5)
        document.set_metadata({"title": "Dense context cost and answer score"})
        document.save(FIGURE, garbage=4, deflate=True, no_new_id=True)


if __name__ == "__main__":
    main()
