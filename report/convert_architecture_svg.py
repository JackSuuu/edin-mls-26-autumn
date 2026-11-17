"""Convert the supplied architecture SVG to a vector PDF and PNG preview.

CairoSVG preserves SVG marker arrowheads (MuPDF's direct SVG import does not).
Requires CairoSVG and PyMuPDF; no model or API calls. Source SVG is never edited.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "report/latex_template/figures/ASR_RAG_System_Architecture_No_Title.svg"
PDF = SOURCE.with_suffix(".pdf")
PNG = PDF.with_suffix(".png")
MANIFEST = ROOT / "rag/results/report-architecture/svg-summary.json"


def main():
    import cairosvg
    import fitz

    source = SOURCE.read_bytes()
    pdf_bytes = cairosvg.svg2pdf(bytestring=source)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    assert len(doc) == 1
    assert not doc[0].get_images(), "Architecture must remain vector, not a screenshot"
    doc.set_metadata({
        "title": "ASR–RAG System / Data-Flow Overview",
        "creator": "CairoSVG + PyMuPDF; report/convert_architecture_svg.py",
        "creationDate": "",
        "modDate": "",
    })
    PDF.parent.mkdir(parents=True, exist_ok=True)
    doc.save(PDF, garbage=4, deflate=True, no_new_id=True)
    cairosvg.svg2png(bytestring=source, write_to=str(PNG))
    manifest = {
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "converter_path": str(Path(__file__).relative_to(ROOT)),
        "converter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pdf_path": str(PDF.relative_to(ROOT)),
        "pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
        "preview_path": str(PNG.relative_to(ROOT)),
        "preview_sha256": hashlib.sha256(PNG.read_bytes()).hexdigest(),
        "cairosvg_version": cairosvg.__version__,
        "pymupdf_version": fitz.VersionBind,
        "source_modified": False,
        "note": "User-supplied SVG; marker arrowheads preserved by CairoSVG; vector PDF included by LaTeX.",
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(PDF)
    print(PNG)
    print(MANIFEST)


if __name__ == "__main__":
    main()
