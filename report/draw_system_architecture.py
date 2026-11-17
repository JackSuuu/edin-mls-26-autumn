"""Reproducible vector architecture diagram, not an end-to-end timing claim.

Uses the PDF checker's existing PyMuPDF dependency. No model/API calls.
Edit NODES/EDGES to revise the layout; the PDF contains editable vector objects.
"""
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
FIGURE = ROOT / "report/latex_template/figures/system-architecture.pdf"
PREVIEW = ROOT / "report/latex_template/figures/system-architecture.png"
SUMMARY = ROOT / "rag/results/report-architecture/summary.json"

NODES = (
    {"id": "asr", "box": (12, 32, 145, 92), "role": "gpu",
     "title": "GLM-ASR", "lines": ["PyTorch / CUDA", "GPU kernel choices"]},
    {"id": "transcripts", "box": (165, 32, 290, 92), "role": "cache",
     "title": "Transcript cache", "lines": ["Fixed for RAG arms", "ASR reused, not timed"]},
    {"id": "embedding", "box": (310, 32, 435, 92), "role": "local",
     "title": "Query encoding", "lines": ["enc_a on CPU", "Query embedding"]},
    {"id": "dense", "box": (455, 32, 588, 92), "role": "cache",
     "title": "Dense search/cache", "lines": ["CPU cosine / top-100", "Query + candidate IDs"]},
    {"id": "qrels", "box": (12, 114, 145, 153), "role": "evaluation",
     "title": "Qrels / references", "lines": ["Public relevance labels", "Offline evaluation only"]},
    {"id": "corpus", "box": (455, 114, 588, 153), "role": "asset",
     "title": "Corpus + embeddings", "lines": ["24,000 records", "enc_a vectors + full text"]},
    {"id": "selector", "box": (455, 176, 588, 238), "role": "selection",
     "title": "Evidence selection", "lines": ["MiniLM / Qwen (GPU)", "Dense / top-k / budget"]},
    {"id": "context", "box": (310, 176, 435, 238), "role": "local",
     "title": "Context builder", "lines": ["Full / budgeted text", "Fixed evidence prompt"]},
    {"id": "generator", "box": (165, 176, 290, 238), "role": "remote",
     "title": "Answer generation", "lines": ["EuroLLM via ELM", "Generated answer"]},
    {"id": "judge", "box": (12, 176, 145, 238), "role": "evaluation",
     "title": "Offline evaluation", "lines": ["Qrel metrics / judge", "EuroLLM + paired CIs"]},
)
EDGES = (
    {"from": "asr", "to": "transcripts", "points": ((145, 62), (165, 62)), "kind": "online"},
    {"from": "transcripts", "to": "embedding", "points": ((290, 62), (310, 62)), "kind": "online"},
    {"from": "embedding", "to": "dense", "points": ((435, 62), (455, 62)), "kind": "online"},
    {"from": "dense", "to": "selector", "points": ((588, 62), (597, 62), (597, 207), (588, 207)), "kind": "online"},
    {"from": "corpus", "to": "dense", "points": ((477, 114), (477, 92)), "kind": "online"},
    {"from": "corpus", "to": "selector", "points": ((565, 153), (565, 176)), "kind": "online"},
    {"from": "corpus", "to": "context", "points": ((455, 134), (372, 134), (372, 176)), "kind": "online"},
    {"from": "selector", "to": "context", "points": ((455, 207), (435, 207)), "kind": "online"},
    {"from": "context", "to": "generator", "points": ((310, 207), (290, 207)), "kind": "online"},
    {"from": "generator", "to": "judge", "points": ((165, 207), (145, 207)), "kind": "evaluation"},
    {"from": "qrels", "to": "judge", "points": ((78, 153), (78, 176)), "kind": "evaluation"},
    {"from": "selector", "to": "judge", "points": ((520, 238), (520, 262), (138, 262), (138, 238)), "kind": "evaluation"},
)
SOURCE_PATHS = (
    "hw1-asr/glm_asr_cuda/ops.py",
    "rag/src/asr_adapter.py", "rag/src/encoder.py", "rag/src/retriever.py",
    "rag/src/context_builder.py", "rag/src/generator.py",
    "rag/src/evaluation/answer_judge.py",
    "rag/src/metrics.py",
    "rag/extensions/evaluate_qwen_reranker.py",
    "rag/extensions/evidence_selection.py",
    "rag/extensions/run_qwen_answer_experiment.py",
    "rag/results/qwen4-dev-validation-20261117-v1/protocol.json",
)


def arrowhead(start, end, length=5):
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    return tuple((end[0] - length * math.cos(angle + offset),
                  end[1] - length * math.sin(angle + offset))
                 for offset in (-math.pi / 6, math.pi / 6))


def validate_spec():
    ids = {node["id"] for node in NODES}
    assert len(ids) == len(NODES)
    for edge in EDGES:
        assert edge["from"] in ids and edge["to"] in ids
        assert len(edge["points"]) >= 2
        assert all(0 <= x <= 600 and 0 <= y <= 302 for x, y in edge["points"])
        if edge["from"] == "qrels":
            assert edge["to"] == "judge" and edge["kind"] == "evaluation"
        assert edge["from"] != "judge"  # no judge-feedback/tuning loop is claimed


def main():
    import fitz
    validate_spec()
    doc = fitz.open()
    page = doc.new_page(width=600, height=302)
    ink = (.16, .20, .24)
    palette = {
        "gpu": ((.91, .95, .99), (.24, .43, .65)),
        "cache": ((.99, .97, .88), (.61, .48, .16)),
        "local": ((.96, .97, .98), (.42, .48, .53)),
        "asset": ((.95, .96, .97), (.42, .48, .53)),
        "selection": ((1, .94, .88), (.72, .39, .16)),
        "remote": ((.91, .97, .93), (.24, .50, .33)),
        "evaluation": ((.96, .93, .99), (.46, .34, .62)),
    }

    def text(x, y, value, size=10, color=ink, bold=False):
        page.insert_text((x, y), value, fontsize=size,
                         fontname="Helvetica-Bold" if bold else "Helvetica", color=color)

    def arrow(points, evaluation=False):
        color = palette["evaluation"][1] if evaluation else ink
        kwargs = {"dashes": "[3 2] 0"} if evaluation else {}
        page.draw_polyline(points, color=color, width=1, **kwargs)
        for point in arrowhead(points[-2], points[-1]):
            page.draw_line(points[-1], point, color=color, width=1)

    for node in NODES:
        box = fitz.Rect(node["box"])
        fill, border = palette[node["role"]]
        kwargs = {"dashes": "[3 2] 0"} if node["role"] == "evaluation" else {}
        page.draw_rect(box, color=border, fill=fill, width=.9, **kwargs)
        compact = box.height < 45
        title_y = box.y0 + (13 if compact else 17)
        title_size = 10.5 if compact or node["id"] == "dense" else 11.5
        text(box.x0 + 6, title_y, node["title"], title_size, border, True)
        step = 11 if compact else 15
        for index, line in enumerate(node["lines"], 1):
            size = 10 if compact else 11
            assert fitz.get_text_length(line, fontname="Helvetica", fontsize=size) <= box.width - 12
            text(box.x0 + 6, title_y + index * step, line, size)
    for edge in EDGES:
        arrow(edge["points"], edge["kind"] == "evaluation")
    text(49, 15, "Audio input", 11, bold=True)
    arrow(((79, 19), (79, 32)))
    text(174, 126, "RAG runs reuse saved", 9)
    text(174, 140, "transcripts + candidate IDs", 9)
    text(383, 128, "full text", 8.5)
    arrow(((12, 253), (42, 253)))
    text(49, 256, "System data flow", 9)
    arrow(((180, 253), (210, 253)), evaluation=True)
    text(217, 256, "Evaluation only", 9)
    page.draw_rect((373, 246, 383, 256), color=palette["cache"][1], fill=palette["cache"][0])
    text(390, 256, "Cached RAG inputs", 9)
    page.draw_rect((12, 267, 588, 298), color=(.8, .82, .84), fill=(.97, .98, .99), width=.5)
    text(18, 280, "Timing: ASR benchmark separate; RAG costs are attributed selection + generation stages.", 11)
    text(18, 292, "Frozen transcripts/candidates; no measured audio-to-answer latency or online judge feedback.", 11)
    doc.set_metadata({"title": "CUDA ASR and frozen-evidence RAG architecture",
                      "subject": "System data flow, caches, offline evaluation and timing boundary"})
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    doc.save(FIGURE, garbage=4, deflate=True, no_new_id=True)
    page.get_pixmap(matrix=fitz.Matrix(2, 2)).save(PREVIEW)
    evidence = {
        "nodes": NODES, "edges": EDGES,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in SOURCE_PATHS},
        "diagram_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "figure_path": str(FIGURE.relative_to(ROOT)),
        "figure_sha256": hashlib.sha256(FIGURE.read_bytes()).hexdigest(),
        "preview_path": str(PREVIEW.relative_to(ROOT)),
        "preview_sha256": hashlib.sha256(PREVIEW.read_bytes()).hexdigest(),
        "scope": "Conceptual pipeline; evidence-selection runs reuse transcript/candidate caches",
        "not_claimed": ["Measured audio-to-answer timing", "Qrels used for evidence selection",
                        "Judge feedback used to tune the frozen dev configuration"],
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(evidence, indent=2) + "\n")
    print(FIGURE)
    print(SUMMARY)


if __name__ == "__main__":
    main()
