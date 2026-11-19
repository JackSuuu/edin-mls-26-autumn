"""Plot practice retrieval quality and reranking time from the full rerun."""

import json
from pathlib import Path

import pymupdf


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "rag/results/manual-full-20261118-135559"
FIGURE = ROOT / "report/latex_template/figures/reranker-tradeoff.pdf"


def load_arms(model):
    source = RUN / f"{model}-practice/summary.json"
    return json.loads(source.read_text(encoding="utf-8"))["arms"]


def main():
    small = load_arms("qwen06")
    large = load_arms("qwen4")
    # Use the MiniLM control measured in the same pass as the two 4B arms.
    points = (
        (large["minilm_512"], "MiniLM-L6", 8, 3, 0, True),
        (small["qwen_matched"], "Qwen-0.6B matched", 8, 3, 0, True),
        (large["qwen_matched"], "Qwen-4B matched", -8, 3, 1, True),
        (large["qwen_native2048"], "Qwen-4B native", 3, -8, 1, False),
    )
    black = (0, 0, 0)
    blue = (0.13, 0.32, 0.54)
    grid = (0.86, 0.88, 0.90)
    baseline = (0.39, 0.42, 0.45)
    font = pymupdf.Font("tiro")
    left, right, top, bottom = 46, 404, 7, 92

    def x_at(seconds):
        return left + seconds / 8 * (right - left)

    def y_at(percent):
        return bottom - (percent - 36) / 20 * (bottom - top)

    with pymupdf.open() as document:
        page = document.new_page(width=416, height=125)

        def label(x, y, value, *, size=10, align=0, rotate=0):
            offset = align * font.text_length(value, fontsize=size)
            position = (x, y + offset) if rotate else (x - offset, y)
            page.insert_text(position, value, fontname="tiro", fontsize=size,
                             color=black, rotate=rotate)

        for percent in range(36, 57, 4):
            y = y_at(percent)
            page.draw_line((left, y), (right, y), color=grid, width=0.45)
            page.draw_line((left - 3, y), (left, y), color=black, width=0.65)
            label(left - 6, y + 3, str(percent), align=1)

        for seconds in range(0, 9, 2):
            x = x_at(seconds)
            page.draw_line((x, bottom), (x, bottom + 3), color=black, width=0.65)
            label(x, bottom + 14, str(seconds), align=0.5)

        dense_recall = 100 * large["dense"]["Recall@5"]
        dense_y = y_at(dense_recall)
        page.draw_line((left, dense_y), (right, dense_y), color=baseline,
                       width=0.8, dashes="[4 3] 0")
        label(right - 3, dense_y - 4, f"Dense5 ({dense_recall:.1f}%)", align=1)

        for arm, name, dx, dy, align, filled in points:
            x = x_at(arm["median_rerank_ms"] / 1000)
            y = y_at(100 * arm["Recall@5"])
            page.draw_circle((x, y), 3, color=blue, width=1.1,
                             fill=blue if filled else (1, 1, 1))
            label(x + dx, y + dy, name, align=align)

        page.draw_line((left, top), (left, bottom), color=black, width=0.75)
        page.draw_line((left, bottom), (right, bottom), color=black, width=0.75)
        label((left + right) / 2, 122, "Median reranking time (s)", size=11, align=0.5)
        label(13, (top + bottom) / 2, "Recall@5 (%)", size=11, align=0.5, rotate=90)

        document.set_metadata({"title": "Practice recall and reranking time"})
        document.save(FIGURE, garbage=4, deflate=True, no_new_id=True)


if __name__ == "__main__":
    main()
