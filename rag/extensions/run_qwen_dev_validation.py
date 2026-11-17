"""Frozen Qwen-4B matched vs fresh dense5 on previously inspected dev.

Requires explicit authorization to make new dev generation/judge calls.
No model selection or score-dependent decisions are performed by this runner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))
from extensions.run_qwen_answer_experiment import ARMS, MODEL, indexed, digest
from src.data import load_jsonl

SOURCE_FILES = (
    "rag/extensions/run_qwen_dev_validation.py",
    "rag/extensions/evaluate_qwen_reranker.py",
    "rag/extensions/run_qwen_answer_experiment.py",
    "rag/extensions/evidence_selection.py",
    "rag/src/generator.py", "rag/src/metrics.py", "rag/src/data.py",
    "rag/scripts/evaluate_answers.py", "rag/src/evaluation/answer_judge.py",
)


def validate_input_rows(rankings, samples, expected=125):
    ranked, sample = indexed(rankings), indexed(samples)
    if len(ranked) != expected or ranked.keys() != sample.keys():
        raise ValueError("Require exact complete dev input coverage")
    for qid, row in ranked.items():
        if sample[qid].get("split") != "dev" or row["transcript"] != sample[qid]["transcript"]:
            raise ValueError("Dev split/transcript mismatch")
        ids = row["record_ids"]
        if len(ids) != 100 or len(set(ids)) != 100:
            raise ValueError("Require 100 unique frozen candidates")


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def offline_audit(run, retrieval):
    """Check raw/cache/plan identities and paired score summary; no API calls."""
    import math
    import random
    import statistics
    from src.evaluation.answer_judge import build_public_gold, cache_key, Judge

    config = json.loads((run / "config.json").read_text())
    assert config["split"] == "dev" and config["queries"] == 125
    qids = config["query_ids"]
    plans = {(p["query_id"], p["arm"]): p for p in load_jsonl(run / "plans.jsonl")}
    ranks = indexed(load_jsonl(retrieval / "rankings.jsonl"))
    gold = build_public_gold(ROOT / "rag/data/rag_corpus", "dev")
    fingerprint = json.loads((run / "answers/generation_config.json").read_text())
    comparison = json.loads((run / "comparison.json").read_text())
    judged = {}
    for arm in ARMS:
        answers = indexed(load_jsonl(run / "answers" / f"{arm}.jsonl"))
        judged[arm] = indexed(load_jsonl(run / f"judge_json_{arm}/answer_judge_per_query.jsonl"))
        caches = indexed(load_jsonl(run / f"judge_json_{arm}/judge_cache.jsonl"))
        raw = load_jsonl(run / f"judge_json_{arm}/judge_raw.jsonl")
        assert len(answers) == len(judged[arm]) == len(caches) == 125
        assert set(answers) == set(judged[arm]) == set(caches) == set(qids) == set(gold)
        for qid in qids:
            answer, judge, cache = answers[qid], judged[arm][qid], caches[qid]
            plan = plans[qid, arm]
            assert answer["split"] == judge["split"] == "dev" and answer["finish_reason"] == "stop"
            assert answer["prompt_sha256"] == digest_text(answer["prompt"]) == plan["prompt_sha256"]
            key = "dense" if arm == "dense5" else "qwen_matched"
            assert answer["retrieved_record_ids"] == ranks[qid][key]["record_ids"][:5]
            assert answer["transcript"] == ranks[qid]["transcript"] == plan["transcript"]
            tag = MODEL + "@" + fingerprint["endpoint"] + "#response_format=json_object"
            assert cache_key(tag, answer["transcript"], answer["answer"],
                             gold[qid]["gold_documents"], 0) == cache["cache_key"]
            candidate_hash = digest_text(answer["answer"])
            parsed = []
            for entry in raw:
                if entry.get("question") == answer["transcript"] and entry.get("candidate_sha256") == candidate_hash:
                    try:
                        parsed.append(Judge.parse(entry["raw"]))
                    except (ValueError, KeyError, TypeError):
                        pass
            assert cache["result"] in parsed
            for metric in ("correctness", "grounding", "completeness"):
                assert judge[metric] == cache["result"][metric] and 0 <= judge[metric] <= 4
            assert math.isclose(judge["normalised_answer_score"],
                                sum(judge[m] for m in ("correctness", "grounding", "completeness")) / 12,
                                abs_tol=1e-12)
        for metric in ("correctness", "grounding", "completeness", "normalised_answer_score"):
            assert math.isclose(statistics.mean(r[metric] for r in judged[arm].values()),
                                comparison["arms"][arm]["mean_" + metric], abs_tol=1e-12)
    rng = random.Random(20261116)
    samples = [rng.choices(qids, k=125) for _ in range(2000)]
    for metric in ("correctness", "grounding", "completeness", "normalised_answer_score"):
        delta = {q: judged[ARMS[1]][q][metric] - judged[ARMS[0]][q][metric] for q in qids}
        estimates = sorted(statistics.mean(delta[q] for q in sample) for sample in samples)
        recorded = comparison["paired_qwen4_minus_dense"][metric]
        assert math.isclose(statistics.mean(delta.values()), recorded["mean_delta"], abs_tol=1e-12)
        assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(
            (estimates[50], estimates[1949]), recorded["query_bootstrap_95pct_interval"]))
    result = {"passed": True, "split": "dev", "queries": 125,
              "fresh_answers_each": 125, "fresh_judgements_each": 125,
              "frozen_prompts_candidates_caches_and_raw_scores_verified": True,
              "paired_means_and_intervals_recomputed": True,
              "qualification": "Previously inspected dev, not untouched independent holdout"}
    write_json(run / "offline-audit.json", result)


def digest_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def audit_retrieval(path, original):
    import math
    from src.data import CourseData
    from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k
    source = indexed(load_jsonl(original))
    output = indexed(load_jsonl(path / "rankings.jsonl"))
    summary = json.loads((path / "summary.json").read_text())
    assert set(source) == set(output) and len(output) == 125
    assert summary["split"] == "dev" and summary["completed"] and summary["full_split"]
    qrels = CourseData(ROOT / "rag/data/rag_corpus").load_qrels("dev")
    arms = ("dense", "qwen_matched")
    aggregates = {arm: [] for arm in arms}
    for qid, row in output.items():
        assert row["transcript"] == source[qid]["transcript"]
        assert row["dense"]["record_ids"] == source[qid]["record_ids"]
        for arm in arms:
            ids = row[arm]["record_ids"]
            assert len(ids) == len(set(ids)) == 100 and set(ids) == set(source[qid]["record_ids"])
            if arm == "qwen_matched":
                scores = dict(zip(ids, row[arm]["scores"], strict=True))
                expected = sorted(source[qid]["record_ids"],
                                  key=lambda rid: (-scores[rid], source[qid]["record_ids"].index(rid)))
                assert ids == expected and row[arm]["truncated_documents"] == 0
            vals = {**{f"Recall@{k}": recall_at_k(ids, qrels[qid], k) for k in (5, 10, 20, 100)},
                    "MRR@10": mrr_at_k(ids, qrels[qid], 10), "NDCG@10": ndcg_at_k(ids, qrels[qid], 10)}
            assert all(math.isclose(value, row[arm]["metrics"][key], abs_tol=1e-12)
                       for key, value in vals.items())
            aggregates[arm].append(vals)
    for arm, values in aggregates.items():
        assert all(math.isclose(sum(v[key] for v in values) / 125,
                                summary["arms"][arm][key], abs_tol=1e-12)
                   for key in values[0])
    write_json(path / "offline-audit.json", {
        "passed": True, "queries": 125, "split": "dev",
        "frozen_candidates_and_transcripts_verified": True,
        "all_metrics_recomputed_and_stable_ranking_verified": True,
        "matched_prefix_not_truncated_again": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--authorize-dev-api-calls", action="store_true", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    if run.exists() and not args.resume and any(run.iterdir()):
        raise ValueError("Choose a fresh run directory; existing files are not overwritten")
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError("Another validation process holds this run lock")

    from dotenv import dotenv_values
    values = dotenv_values(ROOT / "rag/.env")
    if not values.get("ELM_API_KEY"):
        raise ValueError("ELM credentials missing (secret not logged)")
    if values.get("ELM_MODEL", MODEL) != MODEL:
        raise ValueError("ELM model differs from frozen practice model")
    if values.get("ELM_BASE_URL", "https://elm.edina.ac.uk/api/v1").rstrip("/") != "https://elm.edina.ac.uk/api/v1":
        raise ValueError("ELM endpoint differs from frozen practice endpoint")
    ranking_path = ROOT / "rag/results/bm25-compare-dev/dense_rankings.jsonl"
    sample_path = ROOT / "rag/experiments/dev125.jsonl"
    validate_input_rows(load_jsonl(ranking_path), load_jsonl(sample_path))
    source_hashes = {name: digest(ROOT / name) for name in SOURCE_FILES}
    input_hashes = {str(p.relative_to(ROOT)): digest(p) for p in (
        ranking_path, sample_path, ROOT / "rag/data/rag_corpus/corpus.jsonl",
        ROOT / "rag/data/rag_corpus/qrels_dev.tsv")}
    model_manifest = ROOT / "rag/models/qwen3-reranker-4b/experiment_manifest.json"
    manifest = json.loads(model_manifest.read_text())
    assert manifest["repo"] == "Qwen/Qwen3-Reranker-4B"
    assert manifest["revision"] == "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
    protocol = {
        "split": "dev", "queries": 125, "arms": list(ARMS),
        "authorization": "Explicit user authorization on 2026-11-17 for fresh dev generation/judging",
        "qualification": "Frozen-config validation on previously inspected dev; not untouched holdout",
        "reranker": manifest, "primary_ranking": "qwen_matched",
        "candidate_pool": 100, "selected_documents": 5,
        "batch_size": 8, "qwen_dtype": "bfloat16", "generator": MODEL,
        "temperature": .1, "judge": MODEL, "json_mode": True, "judge_repeats": 1,
        "bootstrap": {"resamples": 2000, "seed": 20261116, "paired": True},
        "no_score_dependent_tuning_or_stopping": True,
        "smoke_checks": "2 queries/arm, only structural validity; no required score or performance threshold",
        "fresh_logical_answers": 250, "fresh_logical_judgements": 250,
        "smoke_cache_reused_in_full_run": True, "api_retries_may_add_requests": True,
        "no_old_answers_or_judge_cache_reused": True, "no_new_asr": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.version.split()[0], "python_executable": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "numpy", "openai", "python-dotenv", "huggingface-hub")},
        "source_sha256": source_hashes, "input_sha256": input_hashes,
    }
    protocol_file = run / "protocol.json"
    if protocol_file.exists():
        if not args.resume or json.loads(protocol_file.read_text()) != protocol:
            raise ValueError("Existing run needs --resume and identical frozen protocol")
    else:
        if args.resume:
            raise ValueError("Cannot resume a run without a frozen protocol")
        write_json(protocol_file, protocol)
        (run / "inputs").mkdir()
        shutil.copyfile(ranking_path, run / "inputs/dev_rankings.jsonl")
        shutil.copyfile(sample_path, run / "inputs/dev125.jsonl")
        for name in SOURCE_FILES:
            target = run / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
    assert digest(run / "inputs/dev_rankings.jsonl") == digest(ranking_path)
    status_file = run / "status.json"
    status = json.loads(status_file.read_text()) if status_file.exists() else {
        "started_at": now(), "completed_stages": [], "run": str(run)}
    status.update(pid=os.getpid(), state="running", updated_at=now())
    write_json(status_file, status)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
               PYTHONDONTWRITEBYTECODE="1")
    env.update({key: values[key] for key in ("ELM_API_KEY", "ELM_BASE_URL", "ELM_MODEL") if values.get(key)})
    env.pop("OPENAI_LOG", None)

    def call(*command):
        for name, expected in source_hashes.items():
            if digest(ROOT / name) != expected:
                raise ValueError("Frozen source changed during validation: " + name)
        subprocess.run([sys.executable, "-u", *map(str, command)], cwd=ROOT, env=env, check=True)

    def stage(name, function):
        if name in status["completed_stages"]:
            print("SKIP completed stage:", name, flush=True)
            return
        status.update(stage=name, updated_at=now())
        write_json(status_file, status)
        print(now(), "START", name, flush=True)
        function()
        status["completed_stages"].append(name)
        status.update(updated_at=now())
        write_json(status_file, status)
        print(now(), "DONE", name, flush=True)

    retrieve_script = "rag/extensions/evaluate_qwen_reranker.py"
    answers_script = "rag/extensions/run_qwen_answer_experiment.py"
    judge_script = "rag/scripts/evaluate_answers.py"
    rankings = run / "inputs/dev_rankings.jsonl"
    answers = run / "answer-evaluation"
    retrieval = run / "retrieval"
    common = (retrieve_script, "run", "--model", "Qwen/Qwen3-Reranker-4B",
              "--split", "dev", "--primary-only", "--rankings", rankings,
              "--batch-size", "8", "--bootstrap", "2000")

    def generate(smoke=False):
        command = [answers_script, "generate", "--run", answers]
        if smoke:
            command.extend(("--max-new-queries", "2"))
        if (answers / "answers/generation_config.json").exists():
            command.append("--resume")
        call(*command)

    def judge(smoke=False):
        for arm in ARMS:
            candidates = answers / "answers" / (("smoke_" if smoke else "") + arm + ".jsonl")
            command = [judge_script, "--split", "dev", "--model", MODEL, "--json-mode",
                       "--candidates", candidates, "--out-dir", answers / f"judge_json_{arm}"]
            if smoke:
                command.extend(("--allow-subset", "--api-retries", "1"))
            call(*command, "--dry-run")
            call(*command)

    try:
        stage("retrieval-smoke", lambda: call(*common, "--output", run / "retrieval-smoke", "--limit", "2"))
        stage("retrieval-full", lambda: call(*common, "--output", retrieval))
        stage("retrieval-offline-audit", lambda: audit_retrieval(retrieval, rankings))
        stage("prepare-fresh-answer-plans", lambda: call(
            answers_script, "prepare", "--split", "dev", "--rankings", retrieval / "rankings.jsonl", "--run", answers))
        stage("generation-smoke", lambda: generate(smoke=True))
        stage("extract-smoke", lambda: call(answers_script, "smoke", "--run", answers))
        stage("judge-smoke", lambda: judge(smoke=True))
        stage("generation-full", generate)
        stage("judge-full", judge)
        stage("summarise", lambda: call(answers_script, "summarise", "--run", answers))
        stage("offline-audit", lambda: offline_audit(answers, retrieval))
        status.update(state="completed", completed_at=now(), updated_at=now())
        write_json(status_file, status)
        print("FROZEN_QWEN_DEV_VALIDATION_COMPLETE", flush=True)
    except BaseException as error:
        status.update(state="failed", error_type=type(error).__name__, updated_at=now())
        write_json(status_file, status)
        raise


if __name__ == "__main__":
    main()
