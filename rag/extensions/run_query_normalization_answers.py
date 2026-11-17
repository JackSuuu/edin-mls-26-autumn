"""Generate and score answers after the query-normalization reranking pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys
import tempfile
from zipfile import ZipFile

RAG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAG))
from extensions.evidence_selection import render_prompt
from src.evaluation.answer_judge import append_cache, load_jsonl, write_json_atomic, write_jsonl_atomic

ARMS = ("raw", "raw_normalized_rrf")
SOURCE = RAG / "results/qwen4-answers-3724290"
RANKINGS = RAG / "results/qwen4-retrieval-3724174/full/rankings.jsonl"


def prepare(run, reranked, corpus):
    baseline = {r["query_id"]: r for r in load_jsonl(SOURCE / "answers/qwen4_matched5.jsonl")}
    old_rankings = {r["query_id"]: r for r in load_jsonl(RANKINGS)}
    new = {r["query_id"]: r for r in reranked}
    inputs = load_jsonl(run / "rrf_candidates.jsonl")
    if len(new) != len(reranked) or set(new) != {r["query_id"] for r in inputs}:
        raise ValueError("Incomplete or duplicated reranking outputs")
    records = {r["record_id"]: r["text"] for r in load_jsonl(corpus)}
    plans = []
    for row in inputs:
        query_id = row["query_id"]
        base, result = baseline[query_id], new[query_id]
        transcript = row["transcript"]
        if (base["transcript"] != transcript or result["transcript"] != transcript
                or result["dense"]["record_ids"] != row["record_ids"]
                or set(result["qwen_matched"]["record_ids"]) != set(row["record_ids"])):
            raise ValueError(f"Changed question or candidate set: {query_id}")
        top5 = result["qwen_matched"]["record_ids"][:5]
        raw_qwen = old_rankings[query_id]["qwen_matched"]
        if base["retrieved_record_ids"] != raw_qwen["record_ids"][:5]:
            raise ValueError(f"Frozen raw top-five mismatch: {query_id}")
        plans.append({
            "query_id": query_id, "split": "practice", "transcript": transcript,
            "retrieved_record_ids": dict(zip(ARMS, (base["retrieved_record_ids"], top5))),
            "prompts": dict(zip(ARMS, (base["prompt"], render_prompt(transcript, [records[r] for r in top5])))),
            "qwen_recall_at_5": dict(zip(ARMS, (raw_qwen["metrics"]["Recall@5"],
                                               result["qwen_matched"]["metrics"]["Recall@5"]))),
        })
    return plans


def generate(run, plans, generator):
    paths = {arm: run / f"answers_{arm}.jsonl" for arm in ARMS}
    done = {arm: {r["query_id"]: r for r in load_jsonl(path)} if path.exists() else {}
            for arm, path in paths.items()}
    for index, plan in enumerate(plans):
        query_id = plan["query_id"]
        for arm in ARMS if index % 2 == 0 else ARMS[::-1]:
            prompt = plan["prompts"][arm]
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            prior = done[arm].get(query_id)
            if prior:
                if prior["prompt_sha256"] != prompt_hash or prior["finish_reason"] != "stop":
                    raise ValueError(f"Incompatible cached answer for {query_id}/{arm}")
                continue
            answer = generator.generate(prompt)
            if not answer["answer"].strip() or answer["finish_reason"] != "stop":
                raise ValueError(f"Malformed answer for {query_id}/{arm}")
            row = {**answer, "query_id": query_id, "split": "practice", "transcript": plan["transcript"],
                   "prompt_sha256": prompt_hash, "retrieved_record_ids": plan["retrieved_record_ids"][arm]}
            append_cache(paths[arm], row)
            done[arm][query_id] = row
            print(f"Answer ready {query_id}/{arm}", flush=True)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run
    with tempfile.TemporaryDirectory(prefix="query-normalization-answers-") as directory:
        build = Path(directory)
        with ZipFile(RAG.parent / "rag-assets.zip") as archive:
            for name in ("corpus.jsonl", "qrels_practice.tsv"):
                (build / name).write_bytes(archive.read("data/rag_corpus/" + name))
        plans = prepare(run, load_jsonl(run / "reranked/rankings.jsonl"), build / "corpus.jsonl")
        plan_path = run / "answer_plans.jsonl"
        if plan_path.exists():
            if load_jsonl(plan_path) != plans:
                raise ValueError("Answer plans changed on rerun")
        else:
            write_jsonl_atomic(plan_path, plans)
        from dotenv import load_dotenv
        from src.generator import ELMGenerator, SYSTEM_PROMPT

        reference = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
        load_dotenv(RAG / ".env")
        generator = ELMGenerator()
        if (SYSTEM_PROMPT, generator.model) != (reference["system_prompt"], reference["generator_model"]):
            raise ValueError("Generator prompt or model differs from the historical baseline")
        paths = generate(run, plans, generator)
        scores = {}
        for arm in ARMS:
            out = run / f"judge_{arm}"
            subprocess.run([
                sys.executable, "-u", str(RAG / "scripts/evaluate_answers.py"),
                "--split", "practice", "--allow-subset", "--json-mode", "--api-retries", "1",
                "--data-dir", str(build), "--model", reference["judge_model"],
                "--candidates", str(paths[arm]), "--out-dir", str(out),
            ], check=True)
            scores[arm] = {r["query_id"]: r["normalised_answer_score"]
                           for r in load_jsonl(out / "answer_judge_per_query.jsonl")}

    rows = [{"query_id": plan["query_id"], "qwen_recall_at_5": plan["qwen_recall_at_5"],
             "same_top5": plan["retrieved_record_ids"][ARMS[0]] == plan["retrieved_record_ids"][ARMS[1]],
             "scores": {arm: scores[arm][plan["query_id"]] for arm in ARMS}} for plan in plans]
    summary = {"scope": "Eight practice queries altered by the normalization rules, original system prompt.",
               "queries": len(rows), "mean": {arm: mean(scores[arm].values()) for arm in ARMS},
               "zeros": {arm: sum(v == 0 for v in scores[arm].values()) for arm in ARMS},
               "per_query": rows}
    write_json_atomic(run / "answer_comparison.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
