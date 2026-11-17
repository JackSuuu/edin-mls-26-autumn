"""Compare two system prompts on the same 24 frozen Qwen top-five contexts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
import subprocess
import sys
import tempfile
import time
from zipfile import ZipFile

RAG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAG))
from src.evaluation.answer_judge import append_cache, load_jsonl, write_json_atomic

SOURCE = RAG / "results/qwen4-answers-3724290"
ARMS = ("baseline", "evidence_conditions")
DIMENSIONS = ("correctness", "grounding", "completeness", "normalised_answer_score")
VARIANT = """Answer the question using the supplied documents.

Identify what the user asks for: a value, a time, a source, a procedure,
or a decision. Check all the documents for a passage that answers it
directly before concluding that evidence is insufficient.

Use a claim only when its conditions match the question. Pay attention
to the jurisdiction, account type, transaction, time period, and the
role of each party.

Give the direct answer first. For procedures, include the supported
steps and any prerequisites. For a value, time, or source, give the
specific detail only if the evidence provides it. Cite the document
numbers that support the main claims.

If only part of the question can be answered, give that part and name
the exact detail or condition that is missing. If passages describe
different situations, explain when each applies."""


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed(rows):
    result = {row["query_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate query ID")
    return result


def prepare(run):
    source = read_json(SOURCE / "config.json")
    marker = read_json(SOURCE / "prepared.json")
    for path in ("config.json", "plans.jsonl"):
        if digest(SOURCE / path) != marker[path.split(".")[0] + "_sha256"]:
            raise ValueError(f"Frozen input changed: {path}")
    saved = indexed([row for row in load_jsonl(SOURCE / "plans.jsonl")
                     if row["arm"] == "qwen4_matched5"])
    review_key = SOURCE / "zero-review-v1/review-key.json"
    sample = indexed([row for row in read_json(review_key) if row["arm"] == "qwen4_matched5"])
    strata = Counter(row["review_stratum"] for row in sample.values())
    if strata != {"zero_none": 6, "zero_partial": 6, "zero_full": 6, "positive_control": 6}:
        raise ValueError("Unexpected frozen review sample")

    plans = []
    for query_id, review in sample.items():
        row = saved[query_id]
        plan = {key: row[key] for key in ("query_id", "transcript", "split", "prompt",
                                         "prompt_sha256", "retrieved_record_ids")}
        if len(set(plan["retrieved_record_ids"])) != 5:
            raise ValueError(f"Expected five records for {query_id}")
        plan["stratum"] = review["review_stratum"]
        plans.append(plan)
    random.Random(20261117).shuffle(plans)
    first_arms = list(ARMS) * (len(plans) // len(ARMS))
    random.Random(20261118).shuffle(first_arms)
    for plan, first in zip(plans, first_arms, strict=True):
        plan["first_arm"] = first

    run.mkdir(parents=True, exist_ok=False)
    config = {
        "system_prompts": dict(zip(ARMS, (source["system_prompt"], VARIANT))),
        "generator_model": source["generator_model"],
        "judge_model": source["judge_model"],
        "temperature": source["generation_temperature"],
        "endpoint": read_json(SOURCE / "answers/generation_config.json")["endpoint"],
        "runner_sha256": digest(Path(__file__)),
        "evaluator_sha256": digest(RAG / "scripts/evaluate_answers.py"),
        "judge_sha256": digest(RAG / "src/evaluation/answer_judge.py"),
        "source_run": SOURCE.name,
        "source_plans_sha256": marker["plans_sha256"],
        "sample_sha256": digest(review_key),
        "stratum_counts": dict(strata),
    }
    write_json_atomic(run / "config.json", config)
    for plan in plans:
        append_cache(run / "plans.jsonl", plan)
    write_json_atomic(run / "prepared.json", {name: digest(run / name)
                      for name in ("config.json", "plans.jsonl")})
    print(f"Prepared {len(plans)} frozen question/context pairs in {run}", flush=True)


def load_prepared(run):
    for name, expected in read_json(run / "prepared.json").items():
        if digest(run / name) != expected:
            raise ValueError(f"Frozen pilot file changed: {name}")
    config = read_json(run / "config.json")
    for key, path in (
        ("runner_sha256", Path(__file__)),
        ("evaluator_sha256", RAG / "scripts/evaluate_answers.py"),
        ("judge_sha256", RAG / "src/evaluation/answer_judge.py"),
    ):
        if digest(path) != config[key]:
            raise ValueError(f"Frozen implementation changed: {path.name}")
    return config, indexed(load_jsonl(run / "plans.jsonl"))


def generate(run, config, plans):
    from dotenv import load_dotenv
    from src.generator import ELMGenerator

    load_dotenv(RAG / ".env")
    generator = ELMGenerator()
    if (generator.model, str(generator.client.base_url)) != (config["generator_model"], config["endpoint"]):
        raise ValueError("Generator model or endpoint differs from the original run")
    client = generator.client.with_options(max_retries=1, timeout=60)
    paths = {arm: run / f"{arm}.jsonl" for arm in ARMS}
    done = {arm: indexed(load_jsonl(path)) if path.exists() else {}
            for arm, path in paths.items()}
    for arm, rows in done.items():
        for query_id, row in rows.items():
            plan = plans[query_id]
            if (row["arm"] != arm or row["prompt_sha256"] != plan["prompt_sha256"]
                    or row["transcript"] != plan["transcript"]
                    or row["retrieved_record_ids"] != plan["retrieved_record_ids"]
                    or not row["answer"].strip() or row["finish_reason"] != "stop"):
                raise ValueError(f"Incompatible cached answer: {query_id}/{arm}")

    for query_id, plan in plans.items():
        for arm in sorted(ARMS, key=lambda arm: arm != plan["first_arm"]):
            if query_id in done[arm]:
                continue
            start = time.perf_counter()
            response = client.chat.completions.create(
                model=config["generator_model"], temperature=config["temperature"],
                messages=[
                    {"role": "system", "content": config["system_prompts"][arm]},
                    {"role": "user", "content": plan["prompt"]},
                ],
            )
            choice = response.choices[0]
            answer = (choice.message.content or "").strip()
            if not answer or choice.finish_reason != "stop":
                raise RuntimeError(f"Invalid completion for {query_id}/{arm}: {choice.finish_reason}")
            row = {key: plan[key] for key in ("query_id", "split", "transcript", "prompt_sha256",
                                                "retrieved_record_ids")}
            row.update(arm=arm, answer=answer, finish_reason=choice.finish_reason,
                       response_model=response.model, latency_ms=(time.perf_counter() - start) * 1000,
                       usage=response.usage.model_dump() if response.usage else None)
            append_cache(paths[arm], row)
            done[arm][query_id] = row
            print(f"Generated {sum(map(len, done.values()))}/{2 * len(plans)} {query_id} {arm}", flush=True)
    return done


def evaluate(run, config):
    with tempfile.TemporaryDirectory(prefix="system-prompt-pilot-") as directory:
        build = Path(directory)
        with ZipFile(RAG.parent / "rag-assets.zip") as assets:
            for name in ("corpus.jsonl", "qrels_practice.tsv"):
                (build / name).write_bytes(assets.read("data/rag_corpus/" + name))
        for arm in ARMS:
            out = run / f"judge_{arm}"
            subprocess.run([
                sys.executable, "-u", str(RAG / "scripts/evaluate_answers.py"),
                "--split", "practice", "--allow-subset", "--json-mode", "--api-retries", "1",
                "--data-dir", str(build), "--model", config["judge_model"],
                "--candidates", str(run / f"{arm}.jsonl"), "--out-dir", str(out),
            ], check=True)


def summarise(run, plans, answers):
    judged = {arm: indexed(load_jsonl(run / f"judge_{arm}/answer_judge_per_query.jsonl"))
              for arm in ARMS}
    if any(set(rows) != set(plans) for rows in judged.values()):
        raise ValueError("Incomplete matched judge results")
    base, variant = ARMS
    pairs = []
    for query_id, plan in plans.items():
        left, right = (judged[arm][query_id][DIMENSIONS[-1]] for arm in ARMS)
        pairs.append({"query_id": query_id, "stratum": plan["stratum"], "baseline": left,
                      "variant": right, "difference": right - left})

    def paired(rows):
        return {
            "questions": len(rows),
            "mean_change": mean(row["difference"] for row in rows),
            "wins": sum(row["difference"] > 0 for row in rows),
            "ties": sum(row["difference"] == 0 for row in rows),
            "losses": sum(row["difference"] < 0 for row in rows),
            "zero_to_positive": sum(row["baseline"] == 0 < row["variant"] for row in rows),
            "positive_to_zero": sum(row["variant"] == 0 < row["baseline"] for row in rows),
        }

    arms = {}
    for arm in ARMS:
        results = judged[arm].values()
        generated = answers[arm].values()
        arms[arm] = {key: mean(row[key] for row in results) for key in DIMENSIONS}
        arms[arm]["zeros"] = sum(row[DIMENSIONS[-1]] == 0 for row in results)
        arms[arm]["mean_latency_ms"] = mean(row["latency_ms"] for row in generated)
        for field in ("prompt_tokens", "completion_tokens"):
            values = [row["usage"][field] for row in generated if row["usage"] and field in row["usage"]]
            arms[arm]["mean_" + field] = mean(values) if values else None

    summary = {
        "status": "completed_exploratory_pilot", "arms": arms, "paired": paired(pairs),
        "by_original_score_stratum": {
            stratum: paired([row for row in pairs if row["stratum"] == stratum])
            for stratum in sorted({row["stratum"] for row in pairs})
        },
        "per_query": pairs,
        "scope": "Frozen, score-conditioned practice sample; one new generation and judge per arm.",
    }
    write_json_atomic(run / "comparison.json", summary)
    print(json.dumps({"arms": summary["arms"], "paired": summary["paired"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.run)
    else:
        config, plans = load_prepared(args.run)
        answers = generate(args.run, config, plans)
        evaluate(args.run, config)
        summarise(args.run, plans, answers)


if __name__ == "__main__":
    main()
