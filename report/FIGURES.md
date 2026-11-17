# Report figures

## System architecture

![Report-scale CUDA ASR and RAG data flow](latex_template/figures/architecture-report.png)

- **Main-body version:** `report/latex_template/figures/architecture-report.pdf`.
  `python3 report/draw_readable_architecture.py` makes a two-row reflow using the
  supplied components/palette, abbreviates redundant labels and moves timing
  explanation into the caption. Its smallest labels are about 6.9 pt at the
  396-pt report width, versus roughly 3.1 pt in the original wide SVG.
  Explicit corpus-to-context/selection and ranking-to-evaluation links reflect
  the implementation. Adaptation and hashes:
  `rag/results/report-architecture/readable-summary.json`.
- **Unmodified original design**, retained as a vector PDF:
  `report/latex_template/figures/ASR_RAG_System_Architecture_No_Title.pdf`.
- Editable source: the group-supplied
  `report/latex_template/figures/ASR_RAG_System_Architecture_No_Title.svg`,
  pulled in commit `07fdab9`. The supplied SVG and its full conversion are
  retained unchanged; Figure 1 uses the explicitly labelled readable reflow.
- Conversion: install `CairoSVG==2.9.1` and PyMuPDF in a Python environment, then
  run `python report/convert_architecture_svg.py` from the repository root.
  The PDF contains vector shapes and text, not a raster screenshot.
  CairoSVG preserves SVG marker arrowheads; direct MuPDF SVG import loses them.
  CairoSVG also requires the platform Cairo library.
- Source/conversion/asset hashes and converter versions:
  `rag/results/report-architecture/svg-summary.json`.
- Solid arrows show data flow; dashed arrows denote cached/reused artifacts.
  In the reflow, dashed red arrows instead explicitly mark offline-evaluation
  inputs; cache reuse is labelled in the nodes.
- Qrels do **not** feed online evidence selection; no judge-to-selector tuning
  loop is shown.
- Transcript/candidate caches mark the controlled RAG experimental boundary.
  Timing notes distinguish standalone ASR benchmarks from attributed RAG stage
  costs. The diagram does not claim measured audio-to-answer latency.
- Top-k generation uses full selected records; token-budget arms may use
  document prefixes. The context-builder label deliberately covers both.
- The previous `draw_system_architecture.py` and `figures/system-architecture.*`
  are retained as superseded assets; they are no longer included by LaTeX.
  Their original provenance remains in `report-architecture/summary.json`.
  The earlier titled SVG in `latex_template/` is likewise superseded.

## Quality versus attributed stage cost

- Vector PDF: `report/latex_template/figures/qwen-quality-cost.pdf`.
- Generator: `report/plot_quality_cost.py`.
- Data/provenance: `rag/results/report-quality-cost/summary.json`.
- Practice/dev are separate panels with full 0–1 judge-score axes.
  Annotated intervals are for paired score differences, not individual arms or
  timing. The dev interval includes zero.

## Judge-zero composition and paired transitions

![Practice score states](latex_template/figures/practice-zero-scores.png)

- Main-body vector PDF: `report/latex_template/figures/practice-zero-scores.pdf`.
- Generator: `python3 report/plot_zero_scores.py`, after running
  `python3 rag/extensions/analyze_practice_errors.py`.
- Input: `rag/results/practice-error-analysis-v1/summary.json`;
  arithmetic/source/asset hashes: `rag/results/report-zero-scores/summary.json`.
- Each left-panel bar includes all 350 practice queries, not just zero-score
  queries. Qrel-ID coverage subdivides zero-score answers only.
- The right panel preserves query pairing: 85 zero/zero, 53 zero/nonzero,
  36 nonzero/zero, 176 nonzero/nonzero.
- These are existing judge-score states, not verified correctness, error rates,
  causal effects or new judgments. No dev diagnostic is added.

## Layout and course requirements

Adding the architecture without moving content exceeded the six-page body
limit. The four selected ASR answers were therefore moved **without changing
their content** into `report/latex_template/asr_selected_answers.tex`, included
as Appendix A. Their numbered labels remain linked from the introduction.
Core ASR implementation, correctness, profiling, GEMM/cuBLAS comparison and
end-to-end findings remain in the main body.

This follows the course announcement permitting selected answers anywhere in
the report. Optional Try This experiments were not added.
