"""Compare raw, normalized, and fused dense top-100 retrieval on practice."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from statistics import mean
import sys
import tempfile
from zipfile import ZipFile

RAG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAG))
from extensions import query_normalization
from src.evaluation.answer_judge import load_jsonl, load_qrels, write_json_atomic, write_jsonl_atomic
from src.metrics import recall_at_k

DEPTH = 100
RRF_CONSTANT = 60
ARMS = ("raw", "normalized", "raw_normalized_rrf")
RESULTS = RAG / "results"
FROZEN_RANKINGS = RESULTS / "qwen4-retrieval-3724174/full/rankings.jsonl"
FROZEN_ZEROS = RESULTS / "qwen4-answers-3724290/zero-diagnosis-v1/cases.jsonl"


def fuse(raw, normalized):
    raw_ranks = {rid: rank for rank, rid in enumerate(raw, 1)}
    normalized_ranks = {rid: rank for rank, rid in enumerate(normalized, 1)}

    def order(rid):
        raw_rank = raw_ranks.get(rid)
        normalized_rank = normalized_ranks.get(rid)
        score = sum(1 / (RRF_CONSTANT + rank) for rank in (raw_rank, normalized_rank) if rank)
        return (-score, raw_rank or DEPTH + 1, normalized_rank or DEPTH + 1)

    return sorted(raw_ranks.keys() | normalized_ranks.keys(), key=order)[:DEPTH]


def retrieve(inputs, assets, work):
    from src.encoder import LocalEncoder
    from src.retriever import ExactDenseRetriever
    import torch

    torch.set_num_threads(2)
    with ZipFile(assets) as archive:
        files = ["data/rag_corpus/embeddings/corpus_enc_a.npy",
                 "data/rag_corpus/embeddings/corpus_enc_a_ids.json"]
        files.extend(name for name in archive.namelist() if name.startswith("models/enc_a/")
                     and not name.endswith("/"))
        for name in files:
            archive.extract(name, work)
    encoder = LocalEncoder(work / "models/enc_a", device="cpu")
    embeddings = work / "data/rag_corpus/embeddings"
    retriever = ExactDenseRetriever(embeddings / "corpus_enc_a.npy", embeddings / "corpus_enc_a_ids.json")
    rankings = []
    for position, row in enumerate(inputs, 1):
        original = row["transcript"]
        normalized = query_normalization.normalize_financial_query(original)
        raw = retriever.retrieve(encoder.encode(original), k=DEPTH)["record_ids"]
        if normalized == original:
            orders = dict.fromkeys(ARMS, raw)
        else:
            canonical = retriever.retrieve(encoder.encode(normalized), k=DEPTH)["record_ids"]
            orders = dict(zip(ARMS, (raw, canonical, fuse(raw, canonical))))
        rankings.append({
            "query_id": row["query_id"], "transcript": original, "normalized_query": normalized,
            "orders": orders,
            "raw_matches_frozen_order": raw == row["dense"]["record_ids"],
            "raw_matches_frozen_pool": set(raw) == set(row["dense"]["record_ids"]),
        })
        if position % 50 == 0:
            print(f"Retrieved {position}/{len(inputs)} practice questions", flush=True)
    return rankings


def score(rankings, assets, work):
    # Relevance labels enter only after candidate generation is finished.
    with ZipFile(assets) as archive:
        qrel_path = Path(archive.extract("data/rag_corpus/qrels_practice.tsv", work))
    qrels = load_qrels(qrel_path)
    historical_misses = {
        row["query_id"] for row in load_jsonl(FROZEN_ZEROS)
        if row["pool_status"] == "no_gold_in_pool100"
    }
    if len(historical_misses) != 32 or set(qrels) != {row["query_id"] for row in rankings}:
        raise ValueError("Unexpected practice or zero-case coverage")
    for row in rankings:
        gold = qrels[row["query_id"]]
        row["historical_zero_and_top100_miss"] = row["query_id"] in historical_misses
        row["metrics"] = {}
        for arm, order in row["orders"].items():
            if len(set(order)) != DEPTH or len(order) != DEPTH:
                raise ValueError("Expected 100 unique candidates per query and arm")
            row["metrics"][arm] = {
                "recall_at_100": recall_at_k(order, gold, DEPTH),
                "positive_ranks": {rid: rank for rank, rid in enumerate(order, 1) if rid in gold},
            }
    return rankings


def summarize(rows):
    cohorts = {
        "all_practice": rows,
        "query_changed": [row for row in rows if row["transcript"] != row["normalized_query"]],
        "historical_zero_and_top100_miss": [row for row in rows if row["historical_zero_and_top100_miss"]],
    }
    metrics = {}
    for name, cohort in cohorts.items():
        table = {}
        for arm in ARMS:
            baseline = [row["metrics"]["raw"]["recall_at_100"] for row in cohort]
            values = [row["metrics"][arm]["recall_at_100"] for row in cohort]
            table[arm] = {
                "recall_at_100": mean(values) if values else None,
                "questions_with_a_hit": sum(value > 0 for value in values),
                "miss_to_hit": sum(a == 0 < b for a, b in zip(baseline, values)),
                "hit_to_miss": sum(b == 0 < a for a, b in zip(baseline, values)),
            }
        metrics[name] = {"queries": len(cohort), "arms": table}
    return {
        "scope": "Exploratory practice analysis after inspecting examples; normalization changes retrieval only.",
        "same_order_as_frozen_baseline": sum(row["raw_matches_frozen_order"] for row in rows),
        "same_pool_as_frozen_baseline": sum(row["raw_matches_frozen_pool"] for row in rows),
        "candidate_depth": DEPTH,
        "rrf_constant": RRF_CONSTANT,
        "cohorts": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    inputs = load_jsonl(FROZEN_RANKINGS)
    if len(inputs) != 350 or len({row["query_id"] for row in inputs}) != 350:
        raise ValueError("Expected the 350 frozen practice queries")
    assets = RAG.parent / "rag-assets.zip"
    with tempfile.TemporaryDirectory(prefix="query-normalization-pilot-") as directory:
        work = Path(directory)
        rankings = score(retrieve(inputs, assets, work), assets, work)
    result = summarize(rankings)
    inputs_to_hash = (assets, FROZEN_RANKINGS, FROZEN_ZEROS, Path(__file__),
                      Path(query_normalization.__file__), RAG / "src/encoder.py", RAG / "src/retriever.py")
    result["sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs_to_hash}
    args.output.mkdir(parents=True)
    write_jsonl_atomic(args.output / "per_query.jsonl", rankings)
    write_json_atomic(args.output / "summary.json", result)
    print(result["cohorts"], flush=True)


if __name__ == "__main__":
    main()
