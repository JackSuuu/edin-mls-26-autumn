"""Collect full practice rankings using frozen ASR transcripts, without ELM/ASR.

Run in the existing ICF RAG environment, from the repository root. CPU only;
submit through the cluster scheduler rather than computing on the login node.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))
from src.data import CourseData, load_jsonl
from src.context_builder import SimpleContextBuilder
from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transcript-cache", type=Path, required=True)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--saved-answers", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Choose a new output directory; existing results are never overwritten")
    data = CourseData(ROOT / "rag/data/rag_corpus")
    items = data.audio_items("practice")
    qrels = data.load_qrels("practice")
    cached = load_jsonl(args.transcript_cache)
    transcripts = {r["query_id"]: r["transcript"] for r in cached}
    if len(transcripts) != len(cached) or set(transcripts) != {r["query_id"] for r in items}:
        raise ValueError("Frozen cache must cover practice exactly, without duplicates")
    sample_rows = load_jsonl(args.sample)
    sample_ids = {r["query_id"] for r in sample_rows}
    saved_rows = load_jsonl(args.saved_answers)
    saved = {r["query_id"]: r for r in saved_rows}
    if len(sample_ids) != len(sample_rows) or not sample_ids <= set(transcripts):
        raise ValueError("Invalid diagnostic sample")
    if len(saved) != len(saved_rows) or not set(saved) <= set(transcripts):
        raise ValueError("Invalid saved answers")
    for row in sample_rows + saved_rows:
        if row.get("split", "practice") != "practice" or row["transcript"] != transcripts[row["query_id"]]:
            raise ValueError("Sample/saved transcript differs from frozen practice cache")

    from src.encoder import LocalEncoder
    from src.retriever import ExactDenseRetriever
    import numpy as np
    import torch
    from importlib.metadata import version
    encoder = LocalEncoder(ROOT / "rag/models/enc_a", device="cpu")
    retriever = ExactDenseRetriever(
        ROOT / "rag/data/rag_corpus/embeddings/corpus_enc_a.npy",
        ROOT / "rag/data/rag_corpus/embeddings/corpus_enc_a_ids.json")
    args.output.mkdir(parents=True)
    config = {"split": "practice", "candidate_depth": 100, "device": "cpu",
              "python": platform.python_version(), "torch": torch.__version__,
              "numpy": np.__version__, "sentence_transformers": version("sentence-transformers"),
              "encoder_parameter_dtype": str(next(encoder.model.parameters()).dtype),
              "timing_note": "Single sequential pass, no warmup/repeats; diagnostic timings, not a benchmark",
              "no_asr_or_elm_calls": True,
              "input_sha256": {str(p): digest(p) for p in
                               (args.transcript_cache, args.sample, args.saved_answers, Path(__file__),
                                ROOT / "rag/src/encoder.py", ROOT / "rag/src/retriever.py",
                                ROOT / "rag/src/context_builder.py",
                                ROOT / "rag/data/rag_corpus/corpus.jsonl",
                                ROOT / "rag/data/rag_corpus/qrels_practice.tsv",
                                ROOT / "rag/data/rag_corpus/embeddings/corpus_enc_a.npy",
                                ROOT / "rag/data/rag_corpus/embeddings/corpus_enc_a_ids.json")},
              "encoder_files_sha256": {str(p.relative_to(ROOT / "rag/models/enc_a")): digest(p)
                                       for p in sorted((ROOT / "rag/models/enc_a").rglob("*")) if p.is_file()}}
    (args.output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    metrics = []
    drift = []
    with (args.output / "rankings.jsonl").open("x", encoding="utf-8") as rankings, \
         (args.output / "diagnostic_packets.jsonl").open("x", encoding="utf-8") as packets:
        for i, item in enumerate(items, 1):
            qid = item["query_id"]
            transcript = transcripts[qid]
            start = time.perf_counter()
            vector = encoder.encode(transcript)
            embedding_ms = (time.perf_counter() - start) * 1000
            ret = retriever.retrieve(vector, 100)
            ids = ret["record_ids"]
            rel = qrels[qid]
            ranks = {rid: ids.index(rid) + 1 if rid in ids else None for rid in rel}
            row = {"query_id": qid, "transcript": transcript, **ret,
                   "embedding_ms": embedding_ms, "positive_qrel_ranks": ranks,
                   **{f"Recall@{k}": recall_at_k(ids, rel, k) for k in (5, 10, 20, 100)},
                   "MRR@10": mrr_at_k(ids, rel, 10), "NDCG@10": ndcg_at_k(ids, rel, 10)}
            metrics.append(row)
            rankings.write(json.dumps(row) + "\n")
            rankings.flush()
            if qid in saved:
                drift.append({"query_id": qid, "same_ordered_top5": ids[:5] == saved[qid]["retrieved_record_ids"],
                              "saved_top5": saved[qid]["retrieved_record_ids"], "current_top5": ids[:5]})
            if qid in sample_ids or qid in saved:
                context = SimpleContextBuilder().build(transcript, ids[:5], data)
                packet = {"query_id": qid, "transcript": transcript,
                          "subset": "existing-five" if qid in saved else "fixed-next-30",
                          "retrieved_documents": [{"rank": j, "record_id": rid,
                                                    "qrel_positive": rid in rel,
                                                    "text": data.record_text(rid)}
                                                   for j, rid in enumerate(ids[:5], 1)],
                          "reference_documents": [{"record_id": rid, "rank_in_100": ranks[rid],
                                                   "text": data.record_text(rid)} for rid in sorted(rel)],
                          "reconstructed_prompt": context["prompt"],
                          "prompt_characters": len(context["prompt"]),
                          "answer_status": "saved baseline available" if qid in saved else "not yet generated"}
                packets.write(json.dumps(packet) + "\n")
                packets.flush()
            print(f"[{i}/{len(items)}] {qid} R@5={row['Recall@5']:.3f} R@100={row['Recall@100']:.3f}", flush=True)
    summary = {"queries": len(metrics), "completed": True,
               "metrics": {key: float(np.mean([r[key] for r in metrics])) for key in
                           ("Recall@5", "Recall@10", "Recall@20", "Recall@100", "MRR@10", "NDCG@10")},
               "counts": {
                   "additional_positive_evidence_at_6_to_100": sum(r["Recall@100"] > r["Recall@5"] for r in metrics),
                   "no_positive_at_5_but_some_at_100": sum(r["Recall@5"] == 0 < r["Recall@100"] for r in metrics),
                   "no_positive_at_100": sum(r["Recall@100"] == 0 for r in metrics),
                   "all_positive_at_5": sum(r["Recall@5"] == 1 for r in metrics)},
               "saved_top5_comparison": drift,
               "interpretation": "These counts describe positive qrels, not human-assessed evidence sufficiency"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
