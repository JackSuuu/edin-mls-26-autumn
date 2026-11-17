"""Offline evidence audit; uses saved outputs, never calls ASR or an API.

Run from the repository root:
    python rag/extensions/audit_baseline.py --output rag/results/diagnostic-01
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))
from src.context_builder import SimpleContextBuilder


def jsonl(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        if value in result:
            raise ValueError(f"Duplicate {key}: {value}")
        result[value] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "rag-assets.zip")
    parser.add_argument("--baseline", type=Path,
                        default=ROOT / "rag/results/icf-3713067")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", default="2026-11-12-evidence-audit-v1")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    if args.sample_size < 1:
        parser.error("Sample size must be positive")

    # Read only practice assets: leave dev untouched during method selection.
    members = ["data/rag_corpus/" + name for name in
               ("corpus.jsonl", "student_audio_manifest.jsonl", "qrels_practice.tsv")]
    with zipfile.ZipFile(args.assets) as archive:
        raw = {name: archive.read(name) for name in members}
    corpus = unique(jsonl(raw[members[0]].decode("utf-8-sig")), "record_id")
    manifest = jsonl(raw[members[1]].decode("utf-8-sig"))
    items = unique([r for r in manifest if r["split"] == "practice"], "query_id")
    qrels = {}
    for row in csv.DictReader(io.StringIO(raw[members[2]].decode("utf-8-sig")),
                              delimiter="\t"):
        if float(row.get("score", 1)) > 0:
            qrels.setdefault(row["query_id"], {})[row["record_id"]] = float(row.get("score", 1))

    candidate_path = args.baseline / "sample_answers.jsonl"
    judge_path = args.baseline / "sample_judge/answer_judge_per_query.jsonl"
    cache_path = args.baseline / "practice_transcripts.jsonl"
    answers = unique(jsonl(candidate_path.read_text(encoding="utf-8")), "query_id")
    judges = unique(jsonl(judge_path.read_text(encoding="utf-8")), "query_id")
    transcripts = unique(jsonl(cache_path.read_text(encoding="utf-8")), "query_id")
    if set(transcripts) != set(items):
        raise ValueError("Transcript cache must cover exactly the practice split")

    class Records:
        def record_text(self, rid):
            return corpus[rid]["text"]

    packets = []
    for qid, answer in answers.items():
        if answer.get("split") != "practice" or qid not in items:
            raise ValueError(f"Not a practice candidate: {qid}")
        ranked = answer["retrieved_record_ids"]
        if len(ranked) != 5 or len(set(ranked)) != 5:
            raise ValueError(f"Expected five unique baseline records: {qid}")
        rels = qrels[qid]
        context = SimpleContextBuilder().build(answer["transcript"], ranked, Records())
        hit = [rid for rid in ranked if rid in rels]
        packet = {
            "query_id": qid, "transcript": answer["transcript"],
            "matches_retrieval_transcript_cache": answer["transcript"] == transcripts[qid]["transcript"],
            "answer": answer["answer"], "judge": judges[qid],
            "positive_qrel_count": len(rels), "positive_hits_at_5": hit,
            "recall_at_5": len(hit) / len(rels),
            "context_characters": len(context["prompt"]),
            "context_whitespace_words": len(context["prompt"].split()),
            "length_note": "Characters/whitespace words are not model token counts.",
            "retrieved_documents": [{"rank": i, "record_id": rid,
                                     "qrel_positive": rid in rels,
                                     "text": corpus[rid]["text"]}
                                    for i, rid in enumerate(ranked, 1)],
            "reference_documents": [{"record_id": rid, "text": corpus[rid]["text"]}
                                    for rid in sorted(rels)],
            "reconstructed_prompt": context["prompt"],
        }
        packets.append(packet)

    # Fixed hash ordering gives a reproducible subset, chosen without viewing
    # its retrieval/answer outcomes. This is exploratory practice selection.
    eligible = sorted(set(items) - set(answers))
    if args.sample_size > len(eligible):
        raise ValueError("Sample exceeds eligible practice queries")
    selected = sorted(eligible, key=lambda qid: hashlib.sha256(
        (args.seed + ":" + qid).encode()).hexdigest())[:args.sample_size]
    sample = [{"query_id": qid, "transcript": transcripts[qid]["transcript"],
               "audio_file": items[qid]["audio_file"]} for qid in selected]
    summary = {
        "stage": "offline audit of existing five answers; no new model inference",
        "practice_queries": len(items), "existing_answer_count": len(packets),
        "mean_recall_at_5_existing_subset": sum(p["recall_at_5"] for p in packets) / len(packets),
        "transcripts_match_cache": all(p["matches_retrieval_transcript_cache"] for p in packets),
        "next_sample_size": len(sample), "selection_seed": args.seed,
        "selection_rule": "Ascending SHA256(seed + ':' + query_id), exclude existing answers; first N",
        "limitations": ["Existing five are first-in-manifest, not a representative sample",
                        "Positive qrels are not a completeness or sufficiency guarantee",
                        "Full top-100 rankings were not saved in the baseline outputs",
                        "No claim about ASR accuracy: original audio was not audited",
                        "Prompt reconstructed with current context builder; source hash recorded"],
        "files_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha256(p)
                         for p in (args.assets, candidate_path, judge_path, cache_path,
                                   ROOT / "rag/src/context_builder.py", Path(__file__))},
        "archive_members_sha256": {name: hashlib.sha256(blob).hexdigest()
                                   for name, blob in raw.items()},
    }
    args.output.mkdir(parents=True)
    for name, rows in (("evidence_packets.jsonl", packets), ("next_practice_sample.jsonl", sample)):
        (args.output / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                        encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    lines = ["# Baseline evidence audit", "", "Existing first-five practice answers; no new inference.",
             "References below are public qrels, not exhaustive truth. Lengths are not token counts.", ""]
    for p in packets:
        lines += ["## " + p["query_id"], "", "Question: " + p["transcript"], "",
                  f"Positive hits: {len(p['positive_hits_at_5'])}/{p['positive_qrel_count']}; "
                  f"Recall@5: {p['recall_at_5']:.6f}", "", "### Saved answer", "", p["answer"],
                  "", "### Saved judge explanations", "", *p["judge"]["judge_explanations"], "",
                  "### Retrieved evidence (rank order)", ""]
        for doc in p["retrieved_documents"]:
            lines += [f"#### {doc['rank']}. {doc['record_id']} (positive qrel: {doc['qrel_positive']})",
                      "", doc["text"], ""]
        lines += ["### Qrels reference evidence", ""]
        for doc in p["reference_documents"]:
            lines += ["#### " + doc["record_id"], "", doc["text"], ""]
    (args.output / "evidence_review.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": [
        {"query_id": p["query_id"], "recall_at_5": p["recall_at_5"],
         "context_characters": p["context_characters"]} for p in packets],
        "next_sample_size": len(sample)}, indent=2))


if __name__ == "__main__":
    main()
