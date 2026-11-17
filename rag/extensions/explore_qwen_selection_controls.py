"""Exploratory retrieval-only controls on frozen dense/Qwen rankings.

No model/API calls. These rankings do not produce counterfactual answer scores.
Policies use ranks only; qrels and saved answer scores are evaluation inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))
from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k


def policies(dense, qwen):
    if len(dense) != 100 or len(set(dense)) != 100 or set(dense) != set(qwen):
        raise ValueError("Expected the same 100 unique candidates")
    dense_rank = {rid: rank for rank, rid in enumerate(dense, 1)}
    qwen_rank = {rid: rank for rank, rid in enumerate(qwen, 1)}
    rrf = sorted(dense, key=lambda rid: (-(1 / (60 + dense_rank[rid]) + 1 / (60 + qwen_rank[rid])), dense_rank[rid]))
    # Retain Qwen's best four, reserve the fifth slot for dense top-1 if absent.
    first = qwen[:4]
    if dense[0] not in first:
        first = first + [dense[0]]
    else:
        first = qwen[:5]
    anchor = first + [rid for rid in qwen if rid not in first]
    return {"dense": dense, "qwen_matched": qwen, "rrf60_equal": rrf, "qwen4_plus_dense1": anchor}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Choose a fresh output file")
    source = ROOT / "rag/results/qwen4-retrieval-3724174/full/rankings.jsonl"
    qrel_path = source.parents[1] / "qrels_practice.tsv"
    answer_path = ROOT / "rag/results/qwen4-answers-3724290/judge_json_qwen4_matched5/answer_judge_per_query.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    answers = {r["query_id"]: r for r in map(json.loads, answer_path.read_text(encoding="utf-8").splitlines())}
    qrels = {}
    with qrel_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if float(row["score"]) > 0:
                qrels.setdefault(row["query_id"], {})[row["record_id"]] = float(row["score"])
    if len(rows) != 350 or {r["query_id"] for r in rows} != set(answers):
        raise ValueError("Expected complete paired practice coverage")
    per_query = []
    for row in rows:
        qid = row["query_id"]
        ranked = policies(row["dense"]["record_ids"], row["qwen_matched"]["record_ids"])
        metric = {}
        for arm, order in ranked.items():
            assert len(order) == len(set(order)) == 100 and set(order) == set(ranked["dense"])
            metric[arm] = {**{f"Recall@{k}": recall_at_k(order, qrels[qid], k) for k in (5, 10, 20, 100)}, "MRR@10": mrr_at_k(order, qrels[qid], 10), "NDCG@10": ndcg_at_k(order, qrels[qid], 10)}
            if arm in ("dense", "qwen_matched"):
                for name, value in metric[arm].items():
                    if not math.isclose(value, row[arm]["metrics"][name], abs_tol=1e-12):
                        raise ValueError(f"Frozen baseline metric mismatch: {qid}/{arm}/{name}")
        per_query.append({"query_id": qid, "qwen_answer_was_zero": answers[qid]["normalised_answer_score"] == 0,
                          "positive_qrel_count": len(qrels[qid]),
                          "qwen_selected_positive_count": len(set(ranked["qwen_matched"][:5]) & set(qrels[qid])),
                          "qwen_first_positive_rank": next((i for i, rid in enumerate(ranked["qwen_matched"], 1) if rid in qrels[qid]), None),
                          "selected_record_ids": {a: r[:5] for a, r in ranked.items()}, "metrics": metric})
    summary = {}
    for arm in per_query[0]["metrics"]:
        delta = [r["metrics"][arm]["Recall@5"] - r["metrics"]["qwen_matched"]["Recall@5"] for r in per_query]
        zeros = [r for r in per_query if r["qwen_answer_was_zero"]]
        summary[arm] = {
            "means": {m: statistics.mean(r["metrics"][arm][m] for r in per_query) for m in per_query[0]["metrics"][arm]},
            "recall5_vs_qwen": dict(Counter("improved" if d > 1e-12 else "worse" if d < -1e-12 else "tied" for d in delta)),
            "zero_subset_no_qrel_hit_to_hit": sum(r["metrics"]["qwen_matched"]["Recall@5"] == 0 < r["metrics"][arm]["Recall@5"] for r in zeros),
            "zero_subset_hit_to_no_qrel_hit": sum(r["metrics"][arm]["Recall@5"] == 0 < r["metrics"]["qwen_matched"]["Recall@5"] for r in zeros),
        }
    zero_rows = [r for r in per_query if r["qwen_answer_was_zero"]]
    diagnostics = {
        "zero_first_positive_rank_histogram": dict(Counter(str(r["qwen_first_positive_rank"]) for r in zero_rows)),
        "zero_top1_is_positive_qrel": sum(r["qwen_first_positive_rank"] == 1 for r in zero_rows),
        "full_coverage_zero_top1_is_positive_qrel": sum(r["qwen_first_positive_rank"] == 1 and r["qwen_selected_positive_count"] == r["positive_qrel_count"] for r in zero_rows),
        "limitations": "Ranks describe evidence already present in saved top-5 answers, not newly generated top-1 answers."
    }
    output = {"status": "exploratory_retrieval_only", "queries": 350,
              "policies": {"rrf60_equal": "sum of reciprocal ranks with constant 60, equal dense/Qwen weights, dense-rank tie break",
                           "qwen4_plus_dense1": "Qwen top 4 plus dense top 1 if absent; otherwise Qwen top 5; remaining Qwen order unchanged"},
              "limitations": ["Practice exploration after observing failures; no parameter sweep was run.", "Same candidate pool; no absent document can be recovered.", "No new answers or judgements. Label-hit changes do not imply zero answers are repaired."],
              "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, qrel_path, answer_path)},
              "summary": summary, "diagnostics": diagnostics, "per_query": per_query}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"summary": summary, "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
