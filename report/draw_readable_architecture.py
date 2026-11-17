"""Reflow the group's architecture for report-scale labels; original SVG unchanged."""
from pathlib import Path
import hashlib
import json
import math
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "report/latex_template/figures/ASR_RAG_System_Architecture_No_Title.svg"
PDF = SOURCE.parent / "architecture-report.pdf"
PNG = PDF.with_suffix(".png")
MANIFEST = ROOT / "rag/results/report-architecture/readable-summary.json"
# Abbreviated labels; full CUDA task details remain in the adjacent report text.
NODES = [
    ("audio", (8, 10, 66, 64), "Audio", ["Spoken", "question"], "blue"),
    ("asr", (79, 10, 181, 64), "GLM-ASR", ["PyTorch / CUDA", "GPU kernels"], "blue"),
    ("cache", (194, 10, 296, 64), "Transcript cache", ["Saved transcript", "Fixed RAG input"], "blue"),
    ("encoding", (309, 10, 411, 64), "Query encoding", ["enc_a", "CPU"], "blue"),
    ("dense", (424, 10, 588, 64), "Dense search / cache", ["CPU cosine / top-100", "Candidate IDs reused"], "blue"),
    ("qrels", (8, 80, 145, 116), "Qrels / references", ["Offline evaluation only"], "purple"),
    ("corpus", (405, 80, 588, 116), "Corpus + embeddings", ["24,000 records + full text"], "purple"),
    ("eval", (8, 132, 145, 184), "Offline evaluation", ["Metrics / LLM judge", "EuroLLM via ELM"], "red"),
    ("gen", (160, 132, 290, 184), "Answer generation", ["EuroLLM via ELM", "Generated answer"], "blue"),
    ("context", (305, 132, 435, 184), "Context builder", ["Full / budgeted text", "Fixed prompt"], "blue"),
    ("selection", (450, 132, 588, 184), "Evidence selection", ["MiniLM / Qwen (GPU)", "Dense / top-k / budget"], "orange"),
]
EDGES = [
    ([(66, 37), (79, 37)], False), ([(181, 37), (194, 37)], False),
    ([(296, 37), (309, 37)], False), ([(411, 37), (424, 37)], False),
    ([(588, 37), (597, 37), (597, 158), (588, 158)], False),
    ([(488, 80), (488, 64)], False), ([(552, 116), (552, 132)], False),
    ([(405, 99), (370, 99), (370, 132)], False),
    ([(450, 158), (435, 158)], False), ([(305, 158), (290, 158)], False),
    ([(160, 158), (145, 158)], True), ([(76, 116), (76, 132)], True),
    ([(520, 184), (520, 197), (132, 197), (132, 184)], True),
]


def main():
    import fitz
    # Ensure the supplied conceptual components are still present in the input.
    svg = ET.parse(SOURCE)
    source_text = " ".join("".join(e.itertext()) for e in svg.iter() if e.tag.endswith("text"))
    for value in ("GLM-ASR", "enc_a", "Evidence", "Selection", "Qrels / References", "24,000 records"):
        assert value in source_text
    palette = {
        "blue": ((.945, .965, 1), (.19, .37, .74)),
        "purple": ((.973, .953, .996), (.46, .27, .63)),
        "orange": ((1, .97, .91), (.69, .37, .04)),
        "red": ((1, .95, .95), (.71, .23, .29)),
    }
    doc = fitz.open()
    page = doc.new_page(width=600, height=205)
    for _, coords, title, lines, role in NODES:
        rect = fitz.Rect(coords)
        fill, color = palette[role]
        page.draw_rect(rect, color=color, fill=fill, width=.8)
        for i, value in enumerate([title] + lines):
            size, font = (11, "Helvetica-Bold") if i == 0 else (10.5, "Helvetica")
            width = fitz.get_text_length(value, fontsize=size, fontname=font)
            assert width <= rect.width - 10, value
            page.insert_text((rect.x0 + (rect.width - width) / 2, rect.y0 + 14 + i * 14),
                             value, fontsize=size, fontname=font, color=color if i == 0 else (.10, .14, .20))
    for points, evaluation in EDGES:
        color = palette["red"][1] if evaluation else (.20, .26, .34)
        page.draw_polyline(points, color=color, width=.9, **({"dashes": "[3 2] 0"} if evaluation else {}))
        a, b = points[-2:]
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        for offset in (-math.pi / 6, math.pi / 6):
            page.draw_line(b, (b[0] - 4 * math.cos(angle + offset), b[1] - 4 * math.sin(angle + offset)),
                           color=color, width=.9)
    doc.set_metadata({"title": "Report-scale reflow of group-supplied ASR–RAG architecture"})
    doc.save(PDF, garbage=4, deflate=True, no_new_id=True)
    page.get_pixmap(matrix=fitz.Matrix(2, 2)).save(PNG)
    data = {
        "source_path": str(SOURCE.relative_to(ROOT)), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pdf_path": str(PDF.relative_to(ROOT)), "pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
        "png_path": str(PNG.relative_to(ROOT)), "png_sha256": hashlib.sha256(PNG.read_bytes()).hexdigest(),
        "original_svg_modified": False, "minimum_font_size_pt_at_396pt_report_width": 10.5 * 396 / 600,
        "adaptation": "Two-row reflow and abbreviated labels using the supplied palette/components; "
                      "explicit corpus and ranking-evaluation links from implementation; timing caveat in caption.",
        "nodes": NODES, "edges": EDGES,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n")
    print(PDF)


if __name__ == "__main__":
    main()
