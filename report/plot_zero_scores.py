"""Plot saved practice judge-zero states; no models, API calls or new scores."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "rag/results/practice-error-analysis-v1/summary.json"
PDF = ROOT / "report/latex_template/figures/practice-zero-scores.pdf"
PNG = PDF.with_suffix(".png")
SUMMARY = ROOT / "rag/results/report-zero-scores/summary.json"


def extract(data):
    arms = {}
    for arm in ("dense5", "qwen4_matched5"):
        p = data["population"][arm]
        c = p["zero_by_qrel_coverage"]
        values = [p["queries"] - p["all_zero"], c["none"], c["partial"], c["full"]]
        assert all(type(v) is int and v >= 0 for v in values)
        assert sum(values) == p["queries"] == 350
        assert sum(values[1:]) == p["all_zero"]
        arms[arm] = values
    states = data["paired_score_states"]
    assert sum(states.values()) == 350
    assert states["zero_zero"] + states["zero_positive"] == sum(arms["dense5"][1:])
    assert states["zero_zero"] + states["positive_zero"] == sum(arms["qwen4_matched5"][1:])
    return {"arm_segments": arms, "paired_states": states, "denominator": 350,
            "segment_order": ["nonzero", "zero_no_qrel_ids", "zero_partial_qrel_ids", "zero_all_qrel_ids"]}


def main():
    import fitz
    data = extract(json.loads(SOURCE.read_text()))
    doc = fitz.open()
    page = doc.new_page(width=450, height=165)
    ink = (.10, .14, .20)
    colors = [(0.85, .90, .95), (.90, .57, .35), (.95, .77, .43), (.65, .55, .78)]

    def text(x, y, value, size=9):
        page.insert_text((x, y), value, fontsize=size, fontname="Helvetica", color=ink)

    text(4, 13, "A. Score / qrel-ID coverage (350 per arm)", 10)
    text(275, 13, "B. Paired judge-score states", 10)
    for y, arm, label in [(47, "dense5", "Dense5"), (88, "qwen4_matched5", "Qwen 4B")]:
        text(3, y + 14, label)
        x = 58
        for value, color in zip(data["arm_segments"][arm], colors):
            width = value / 350 * 202
            page.draw_rect((x, y, x + width, y + 23), color=(1, 1, 1), fill=color, width=.4)
            digits = str(value)
            tw = fitz.get_text_length(digits, fontsize=8, fontname="Helvetica")
            text(x + (width - tw) / 2, y + 15, digits, 8)
            x += width
    for value in (0, 175, 350):
        x = 58 + value / 350 * 202
        page.draw_line((x, 114), (x, 117), color=ink, width=.5)
        text(x - 6, 127, str(value), 8)
    for x, y, color, label in [(5, 139, colors[0], "Nonzero"),
                               (118, 139, colors[1], "Zero: no qrel IDs"),
                               (5, 154, colors[2], "Zero: partial IDs"),
                               (118, 154, colors[3], "Zero: all qrel IDs")]:
        page.draw_rect((x, y - 7, x + 7, y), color=color, fill=color)
        text(x + 11, y, label, 8)
    text(331, 37, "Qwen = 0", 8)
    text(390, 37, "Qwen > 0", 8)
    text(274, 66, "Dense = 0", 8)
    text(274, 103, "Dense > 0", 8)
    states = data["paired_states"]
    for row, col, key, color in [
        (0, 0, "zero_zero", (.93, .93, .93)),
        (0, 1, "zero_positive", (.78, .90, .82)),
        (1, 0, "positive_zero", (.97, .84, .76)),
        (1, 1, "positive_positive", (.87, .91, .95)),
    ]:
        x, y = 330 + col * 56, 45 + row * 37
        page.draw_rect((x, y, x + 56, y + 37), color=(1, 1, 1), fill=color, width=1)
        value = str(states[key])
        tw = fitz.get_text_length(value, fontsize=13, fontname="Helvetica")
        text(x + (56 - tw) / 2, y + 23, value, 13)
    text(276, 142, f'{states["zero_positive"]} leave zero; {states["positive_zero"]} enter zero', 9)
    text(276, 156, "States are not correctness labels.", 8)
    doc.set_metadata({"title": "Practice judge-zero composition and paired transitions"})
    PDF.parent.mkdir(parents=True, exist_ok=True)
    doc.save(PDF, garbage=4, deflate=True, no_new_id=True)
    page.get_pixmap(matrix=fitz.Matrix(2, 2)).save(PNG)
    data.update({
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pdf_path": str(PDF.relative_to(ROOT)), "pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
        "png_path": str(PNG.relative_to(ROOT)), "png_sha256": hashlib.sha256(PNG.read_bytes()).hexdigest(),
        "not_claimed": ["Human-verified error rates", "Causal reranking effects", "Snippet benefits"],
    })
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(data, indent=2) + "\n")
    print(PDF)


if __name__ == "__main__":
    main()
