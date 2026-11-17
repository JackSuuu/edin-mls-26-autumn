"""Inventory saved zero-score answers offline; coverage is not a causal label.

Read frozen practice outputs and the course asset ZIP. Never call a model,
change a judgement, or use gold evidence in the inference path.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("dense5", "qwen4_matched5")
DIMS = ("correctness", "grounding", "completeness")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def indexed(path):
    rows = read_rows(path)
    result = {row["query_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate query IDs: {path}")
    return result


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def analyse(run, rankings_path, asset_zip):
    paths = {"rankings": rankings_path, "asset_zip": asset_zip}
    for arm in ARMS:
        paths[arm + "_answers"] = run / "answers" / (arm + ".jsonl")
        paths[arm + "_judgements"] = run / ("judge_json_" + arm) / "answer_judge_per_query.jsonl"
    answers = {arm: indexed(paths[arm + "_answers"]) for arm in ARMS}
    scores = {arm: indexed(paths[arm + "_judgements"]) for arm in ARMS}
    rankings = indexed(rankings_path)
    ids = list(answers[ARMS[1]])
    if len(ids) != 350 or any(set(data) != set(ids) for data in (*answers.values(), *scores.values(), rankings)):
        raise ValueError("Expected identical 350-query coverage in every source")
    with zipfile.ZipFile(asset_zip) as archive:
        corpus = {r["record_id"]: r["text"] for r in map(json.loads, archive.read("data/rag_corpus/corpus.jsonl").decode("utf-8-sig").splitlines())}
        qrels = defaultdict(dict)
        text = archive.read("data/rag_corpus/qrels_practice.tsv").decode("utf-8-sig")
        for row in csv.DictReader(io.StringIO(text), delimiter="\t"):
            if float(row["score"]) > 0:
                qrels[row["query_id"]][row["record_id"]] = float(row["score"])
    coverage = defaultdict(Counter)
    pool_status = Counter()
    positions = Counter()
    regressions = Counter()
    cases = []
    for qid in ids:
        answer = answers[ARMS[1]][qid]
        for arm in ARMS:
            score = scores[arm][qid]
            if abs(sum(score[d] for d in DIMS) / 12 - score["normalised_answer_score"]) > 1e-12:
                raise ValueError(f"Score mismatch: {qid}/{arm}")
            source = "dense" if arm == ARMS[0] else "qwen_matched"
            if answers[arm][qid]["retrieved_record_ids"] != rankings[qid][source]["record_ids"][:5]:
                raise ValueError(f"Selection mismatch: {qid}/{arm}")
        gold = qrels[qid]
        selected = answer["retrieved_record_ids"]
        order = rankings[qid]["qwen_matched"]["record_ids"]
        present = set(gold) & set(selected)
        pool = set(gold) & set(rankings[qid]["dense"]["record_ids"])
        state = "none" if not present else "full" if len(present) == len(gold) else "partial"
        zero = scores[ARMS[1]][qid]["normalised_answer_score"] == 0
        coverage[state]["questions"] += 1
        coverage[state]["zeros"] += int(zero)
        if not zero:
            continue
        status = "no_gold_in_pool100" if not pool else "gold_in_pool_but_none_selected" if not present else "some_gold_selected"
        pool_status[status] += 1
        gold_positions = {rid: order.index(rid) + 1 if rid in order else None for rid in gold}
        first = min((p for p in gold_positions.values() if p is not None), default=None)
        position_group = "absent" if first is None else "1-5" if first <= 5 else "6-10" if first <= 10 else "11-20" if first <= 20 else "21-100"
        positions[position_group] += 1
        dense_score = scores[ARMS[0]][qid]["normalised_answer_score"]
        if dense_score > 0:
            regressions[state] += 1
        cases.append({
            "query_id": qid, "question": answer["transcript"], "labelled_top5_coverage": state,
            "pool_status": status, "positive_qrels": len(gold), "selected_positive_qrels": len(present),
            "pool100_positive_qrels": len(pool), "gold_reranked_positions": gold_positions,
            "qwen_score": scores[ARMS[1]][qid], "dense_score": scores[ARMS[0]][qid],
            "answer": answer["answer"], "dense_answer": answers[ARMS[0]][qid]["answer"],
            "input_tokens": answer["local_input_tokens"], "finish_reason": answer["finish_reason"],
            "selected_text_truncated": any(d["truncated"] for d in answer["selected_documents"]),
            "gold_documents": [{"record_id": rid, "relevance": rel, "selected_rank": selected.index(rid) + 1 if rid in selected else None, "text": corpus[rid]} for rid, rel in gold.items()],
            "selected_documents": [{"record_id": rid, "is_positive_qrel": rid in gold, "text": corpus[rid]} for rid in selected],
            "review_status": "mechanical_inventory_only; see separate source-review notes",
        })
    for row in coverage.values():
        row["zero_rate"] = row["zeros"] / row["questions"]
    summary = {
        "schema": "qwen-zero-diagnosis-v1", "queries": len(ids), "zero_questions": len(cases),
        "coverage_groups_all_350": dict(coverage), "zero_pool_status": dict(pool_status),
        "zero_first_gold_reranked_position": dict(positions), "positive_to_zero_by_coverage": dict(regressions),
        "zero_dimension_counts_all_350": {dim: sum(r[dim] == 0 for r in scores[ARMS[1]].values()) for dim in DIMS},
        "zero_finish_reasons": dict(Counter(c["finish_reason"] for c in cases)),
        "zero_selected_text_truncated": sum(c["selected_text_truncated"] for c in cases),
        "zero_input_tokens": {"min": min(c["input_tokens"] for c in cases), "max": max(c["input_tokens"] for c in cases), "mean": statistics.mean(c["input_tokens"] for c in cases)},
        "limitations": [
            "Qrel coverage is a document-ID observation, not sufficient-answer coverage or a causal failure label.",
            "Source review is AI-assisted and score-conditioned; it is not blinded independent human validation.",
            "Original answers, scores and evaluation code remain unchanged. No model calls were made.",
            "No clean reference transcripts are supplied in the asset manifest; suspected ASR mechanisms need audio verification.",
        ],
        "input_sha256": {label: digest(path) for label, path in paths.items()},
    }
    if len(cases) != 121 or sum(pool_status.values()) != 121 or sum(regressions.values()) != 36:
        raise ValueError("The sources do not reproduce the frozen 121-zero/36-regression run")
    return summary, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "rag/results/qwen4-answers-3724290")
    parser.add_argument("--rankings", type=Path, default=ROOT / "rag/results/qwen4-retrieval-3724174/full/rankings.jsonl")
    parser.add_argument("--asset-zip", type=Path, default=ROOT / "rag-assets.zip")
    parser.add_argument("--out-dir", type=Path, required=True, help="Fresh directory; frozen outputs are never overwritten")
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("Choose a fresh output directory")
    summary, cases = analyse(args.run, args.rankings, args.asset_zip)
    args.out_dir.mkdir(parents=True)
    save(args.out_dir / "summary.json", summary)
    with (args.out_dir / "cases.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("input_sha256", "limitations")}, indent=2))


if __name__ == "__main__":
    main()
