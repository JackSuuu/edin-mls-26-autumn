"""Draw the two measured practice operating points; no model/API calls.

Run from the repository root with the Python used by the PDF format checker.
PyMuPDF is already a dependency of that checker; no plotting install is needed.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "rag/results/qwen4-answers-3724290/comparison.json"
FIGURE = ROOT / "report/latex_template/figures/qwen-quality-cost.pdf"
SUMMARY = ROOT / "rag/results/report-quality-cost/summary.json"


def extract(data):
    points = []
    for arm, label in (("dense5", "Fresh Dense5"),
                       ("qwen4_matched5", "Qwen 4B matched5")):
        values = data["arms"][arm]
        points.append({
            "arm": arm, "label": label,
            "mean_judge_score": values["mean_normalised_answer_score"],
            "mean_attributed_seconds": values["mean_attributed_selection_plus_generation_ms"] / 1000,
            "mean_input_tokens": values["mean_server_input_tokens"],
        })
    first, second = points
    assert all(0 <= p["mean_judge_score"] <= 1 and p["mean_attributed_seconds"] > 0 for p in points)
    assert first["mean_judge_score"] > 0
    return {
        "queries": data["queries"], "points": points,
        "relative_mean_score_gain_percent": 100 * (
            second["mean_judge_score"] / first["mean_judge_score"] - 1),
        "relative_attributed_cost_increase_percent": 100 * (
            second["mean_attributed_seconds"] / first["mean_attributed_seconds"] - 1),
        "additional_attributed_seconds": second["mean_attributed_seconds"] - first["mean_attributed_seconds"],
        "paired_score_delta": data["paired_qwen4_minus_dense"]["normalised_answer_score"],
        "caveats": [
            "Practice exploratory results; one generation/judge per answer",
            "Mean judge score is not answer accuracy",
            "Costs combine stages from different runs; not measured audio-to-answer latency",
            "No individual-arm or timing confidence intervals inferred from the paired score interval",
            "No Qwen 0.6B answer-quality point is available",
            "Only two observed configurations, not a latency/quality utility model",
        ],
    }


def main():
    import fitz
    data = extract(json.loads(SOURCE.read_text()))
    data["source_path"] = str(SOURCE.relative_to(ROOT))
    data["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    data["plot_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    doc = fitz.open()
    page = doc.new_page(width=450, height=180)
    left, top, right, bottom = 49, 24, 429, 133
    ink, grid = (0.12, 0.16, 0.20), (0.87, 0.89, 0.91)

    def text(x, y, value, size=9, color=ink):
        page.insert_text((x, y), value, fontsize=size, fontname="Times-Roman", color=color)

    text(left, 12, "Mean normalised judge score (full 0-1 scale)", 11)
    for value in (0, .25, .5, .75, 1):
        y = bottom - value * (bottom - top)
        page.draw_line((left, y), (right, y), color=grid, width=.5)
        text(23, y + 3, f"{value:.2f}", 9)
    for value in (0, 3, 6, 9, 12):
        x = left + value / 12 * (right - left)
        page.draw_line((x, bottom), (x, bottom + 3), color=ink, width=.7)
        text(x - 2, bottom + 13, str(value), 9)
    page.draw_line((left, top), (left, bottom), color=ink, width=.8)
    page.draw_line((left, bottom), (right, bottom), color=ink, width=.8)
    text(87, 162, "Mean attributed selection + generation time (seconds)", 11)
    colors = ((0.15, 0.35, 0.60), (0.72, 0.29, 0.12))
    for point, color in zip(data["points"], colors):
        x = left + point["mean_attributed_seconds"] / 12 * (right - left)
        y = bottom - point["mean_judge_score"] * (bottom - top)
        page.draw_circle((x, y), 3.2, color=color, fill=color)
        box = fitz.Rect(x - 78, y - 39, x + 68, y - 3)
        value = (point["label"] + "\n"
                 + f'{point["mean_judge_score"]:.4f}; {point["mean_attributed_seconds"]:.3f} s')
        assert page.insert_textbox(box, value, fontsize=10, fontname="Times-Roman",
                                   color=color, align=1) >= 0
    text(70, 176, "Stage-attributed costs, not measured audio-to-answer latency.", 9)
    doc.set_metadata({"title": "Practice answer quality versus attributed stage cost",
                      "subject": "Two observed configurations; source: " + data["source_path"]})
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    doc.save(FIGURE, garbage=4, deflate=True, no_new_id=True)
    data["figure_path"] = str(FIGURE.relative_to(ROOT))
    data["figure_sha256"] = hashlib.sha256(FIGURE.read_bytes()).hexdigest()
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
