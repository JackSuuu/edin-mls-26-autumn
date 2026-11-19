"""Fresh, coverage-stratified top-20 document/chunk experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import run_passage_pilot as runner
from evaluate_qwen_reranker import ROOT, ranked, qwen_input
from extensions.evidence_selection import chat_input_tokens, select_context
from extensions.passage_selection import make_windows, pack_windows
from src.data import CourseData

ARMS = ("A_document", "B_fixed5_chunks", "C_rerank20_chunks")
BUDGET = 768
SEED = 2026111901
STRATA = ("present", "outside5")
DIMS = ("correctness", "grounding", "completeness")


def prepare(run, ranking_file):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data_dir = ROOT / "rag/data/rag_corpus"
    data = CourseData(data_dir)
    ranks = runner.index(runner.load(ranking_file))
    old = ROOT / "rag/experiments/passage-pilot-v1.json"
    excluded = set(json.loads(old.read_text())["query_ids"])
    expected = {r["query_id"] for r in data.audio_items("practice")}
    assert set(ranks) == expected and len(excluded) == 24 and excluded <= expected
    qrels = data.load_qrels("practice")
    assets = ROOT / "rag/models"
    directory = assets / "qwen3-reranker-4b"
    manifest = json.loads((directory / "experiment_manifest.json").read_text())
    assert manifest["repo"] == "Qwen/Qwen3-Reranker-4B"
    assert manifest["revision"] == "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
    for name, h in manifest["files_sha256"].items():
        assert runner.digest(directory / name) == h
    token_manifest = json.loads((assets / "selection/manifest.json").read_text())["tokenizer"]
    assert token_manifest["repo"] == runner.MODEL
    for name, h in token_manifest["files_sha256"].items():
        assert runner.digest(assets / "selection/tokenizer" / name) == h
    system = runner.read_generator_system_prompt()
    gen_tok = AutoTokenizer.from_pretrained(assets / "selection/tokenizer", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(directory, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").eval().cuda()
    yes, no = (tok.convert_tokens_to_ids(w) for w in ("yes", "no"))
    assert tok.encode("yes", add_special_tokens=False) == [yes] and tok.encode("no", add_special_tokens=False) == [no]
    torch.set_num_threads(2)

    @torch.inference_mode()
    def score(question, texts):
        values, flags = [], []
        for i in range(0, len(texts), 8):
            encoded, truncated = zip(*(qwen_input(tok, question, text) for text in texts[i:i+8]))
            batch = tok.pad(list(encoded), padding=True, return_tensors="pt").to("cuda")
            logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
            values.extend((logits[:, yes] - logits[:, no]).cpu().tolist())
            flags.extend(truncated)
        return values, flags

    count = lambda prompt: chat_input_tokens(gen_tok, system, prompt)
    assert score("Capital of France?", ["Paris is the capital of France.", "A banana is fruit."])[0][0] > score(
        "Capital of France?", ["A banana is fruit."])[0][0]
    order = sorted(expected - excluded)
    random.Random(SEED).shuffle(order)
    run.mkdir(parents=True, exist_ok=False)
    protocol = {
        "split": "practice", "sampling": "Another sample from the existing course split; all answers freshly generated",
        "seed": SEED, "sample_rule": "First 12 eligible per stratum in a pre-shuffled ID list, without answer scores",
        "excluded_old_pilot_ids": sorted(excluded), "budget": BUDGET, "arms": ARMS,
        "scorer": manifest["repo"], "revision": manifest["revision"], "scorer_max_tokens": 2048,
        "scorer_policy": "Full source offered to scorer; Qwen template total capped at 2048; only document/chunk text may truncate",
        "window": "3 sentences, overlap 1; rank by score; skip non-fitting windows, merge overlaps, render in parent/source order",
        "generator": runner.MODEL, "judge": runner.MODEL, "judge_mode": "json_object", "judge_repeats": 1,
        "files_sha256": {str(p.relative_to(ROOT)): runner.digest(p) for p in (
            ranking_file, old, data_dir / "corpus.jsonl", data_dir / "qrels_practice.tsv")},
    }
    runner.save(run / "protocol.json", protocol)
    chosen = {s: [] for s in STRATA}
    plans = []
    with (run / "screening.jsonl").open("w") as screen, (run / "chunk_scores.jsonl").open("w") as chunks:
        for i, q in enumerate(order, 1):
            row = ranks[q]
            question = row["transcript"]
            pool = row["dense"]["record_ids"][:20]
            assert len(set(pool)) == 20 and question.strip()
            t = time.perf_counter()
            ds, truncated = score(question, [data.records[r] for r in pool])
            document_ms = (time.perf_counter() - t) * 1000
            doc_rank = ranked(pool, ds)
            top = doc_rank[:5]
            t = time.perf_counter()
            a = select_context(question, top, data.records, count, top_k=5, budget=BUDGET)
            a.update(context_build_ms=(time.perf_counter() - t) * 1000, cached_reranking_ms=document_ms)
            gold = set(qrels.get(q, {}))
            stratum = ("present" if gold & set(a["retrieved_record_ids"]) else
                       "outside5" if not gold & set(top) and gold & set(doc_rank[5:]) else None)
            take = bool(stratum and len(chosen[stratum]) < 12)
            screen.write(json.dumps(dict(query_id=q, candidate_ids=pool, doc_rank=doc_rank, scores=ds,
                                         document_scoring_ms=document_ms, truncated=truncated,
                                         stratum=stratum, selected=take)) + "\n")
            screen.flush()
            if not take:
                continue
            chosen[stratum].append(q)
            started = time.perf_counter()
            bwindows = make_windows(top, data.records, 3, 1)
            t = time.perf_counter()
            bs, bt = score(question, [w["text"] for w in bwindows])
            bm = (time.perf_counter() - t) * 1000
            b = pack_windows(question, top, data.records, bwindows, bs, count, BUDGET)
            b.update(passage_scoring_ms=bm, context_build_ms=(time.perf_counter() - started) * 1000,
                     cached_reranking_ms=document_ms)
            started = time.perf_counter()
            cwindows = make_windows(pool, data.records, 3, 1)
            t = time.perf_counter()
            cs, ct = score(question, [w["text"] for w in cwindows])
            cm = (time.perf_counter() - t) * 1000
            best = {r: float("-inf") for r in pool}
            for w, v in zip(cwindows, cs):
                best[w["record_id"]] = max(best[w["record_id"]], v)
            ctop = sorted(pool, key=lambda r: -best[r])[:5]
            allowed = set(ctop)
            # Match B's tie-breaking: parent rank first, then source offset.
            parent = {rid: i for i, rid in enumerate(ctop)}
            eligible = sorted(((w, v) for w, v in zip(cwindows, cs) if w["record_id"] in allowed),
                              key=lambda pair: (parent[pair[0]["record_id"]], pair[0]["start"]))
            c = pack_windows(question, ctop, data.records, [w for w, _ in eligible], [v for _, v in eligible], count, BUDGET)
            c.update(passage_scoring_ms=cm, context_build_ms=(time.perf_counter() - started) * 1000,
                     cached_reranking_ms=0)
            for key, windows, scores, flags in ((ARMS[1], bwindows, bs, bt), (ARMS[2], cwindows, cs, ct)):
                chunks.write(json.dumps(dict(query_id=q, arm=key, windows=[
                    {**w, "score": v, "scorer_truncated": f} for w, v, f in zip(windows, scores, flags)])) + "\n")
            chunks.flush()
            refs = dict(zip(ARMS, [(a, top), (b, top), (c, ctop)]))
            arm_order = list(ARMS)
            random.Random(f"{SEED}:{q}").shuffle(arm_order)
            for arm in arm_order:
                built, parents = refs[arm]
                assert 0 < built["local_input_tokens"] <= BUDGET
                plans.append(dict(query_id=q, split="practice", arm=arm, transcript=question,
                                  stratum=stratum, parent_record_ids=parents, **built,
                                  prompt_sha256=hashlib.sha256(built["prompt"].encode()).hexdigest()))
            print(f"Sampled {q} {stratum} (screened {i}): "
                  f"{ {s:len(chosen[s]) for s in STRATA} }", flush=True)
            if all(len(v) == 12 for v in chosen.values()):
                break
    assert all(len(v) == 12 for v in chosen.values()), chosen
    with (run / "plans.jsonl").open("w") as f:
        for plan in plans:
            f.write(json.dumps(plan, ensure_ascii=False) + "\n")
    protocol["query_ids"] = [p["query_id"] for p in plans[::3]]
    protocol["strata"] = chosen
    runner.save(run / "protocol.json", protocol)
    sources = [Path(__file__).resolve(), *runner.sources()]
    runner.save(run / "config.json", dict(protocol=protocol, source_sha256={
        str(p.relative_to(ROOT)): runner.digest(p) for p in sources}))
    runner.save(run / "prepared.json", {n: runner.digest(run/n) for n in
                                        ("config.json", "protocol.json", "plans.jsonl", "screening.jsonl", "chunk_scores.jsonl")})
    print("Frozen plans ready: 24 queries × 3 arms", flush=True)


def evaluate(run, resume):
    runner.ARMS = ARMS
    runner.generate(argparse.Namespace(run=run, resume=resume, max_new_queries=None))
    for arm in ARMS:
        subprocess.run([sys.executable, "-u", str(ROOT / "rag/scripts/evaluate_answers.py"),
                        "--split", "practice", "--allow-subset", "--json-mode", "--model", runner.MODEL,
                        "--candidates", str(run / "answers" / f"{arm}.jsonl"),
                        "--out-dir", str(run / f"judge_json_{arm}")], check=True)
    summarise(run)


def summarise(run):
    protocol = json.loads((run/"protocol.json").read_text())
    ids = set(protocol["query_ids"])
    qrels = CourseData(ROOT/"rag/data/rag_corpus").load_qrels("practice")
    per_arm = {}
    for arm in ARMS:
        a = runner.index(runner.load(run/"answers"/f"{arm}.jsonl"))
        j = runner.index(runner.load(run/f"judge_json_{arm}"/"answer_judge_per_query.jsonl"))
        assert set(a) == set(j) == ids and len(ids) == 24
        out = {}
        for q, r in a.items():
            s = j[q]["normalised_answer_score"]
            assert s == sum(j[q][d] for d in DIMS)/12
            assert r["usage"]["prompt_tokens"] == r["local_input_tokens"] <= BUDGET
            gold = set(qrels[q])
            out[q] = dict(score=s, parent_hit=bool(gold & set(r["parent_record_ids"])),
                          prompt_hit=bool(gold & set(r["retrieved_record_ids"])),
                          gold_in_prompt=sorted(gold & set(r["retrieved_record_ids"])),
                          input_tokens=r["local_input_tokens"], generation_ms=r["generation_ms"],
                          passage_scoring_ms=r.get("passage_scoring_ms"),
                          document_scoring_ms=r["cached_reranking_ms"])
        per_arm[arm] = out
    result = {"strata": {}}
    for stratum in STRATA:
        group = protocol["strata"][stratum]
        summary = {}
        for arm, records in per_arm.items():
            rows = [records[q] for q in group]
            summary[arm] = {k: sum(row[k] for row in rows) for k in ("parent_hit", "prompt_hit")}
            for key in ("score", "input_tokens", "generation_ms", "document_scoring_ms", "passage_scoring_ms"):
                values = [row[key] for row in rows if row[key] is not None]
                summary[arm]["mean_"+key] = st.mean(values) if values else None
        old, new = (ARMS[:2] if stratum == "present" else ARMS[1:])
        delta = {q: per_arm[new][q]["score"] - per_arm[old][q]["score"] for q in group}
        result["strata"][stratum] = dict(n=len(group), arms=summary, primary=f"{new} minus {old}",
                                         deltas=delta, mean_delta=st.mean(delta.values()),
                                         wins=sum(x>1e-12 for x in delta.values()),
                                         losses=sum(x< -1e-12 for x in delta.values()),
                                         ties=sum(abs(x)<=1e-12 for x in delta.values()))
    result["limits"] = "Prompt hit is labelled-source text presence, not a verified answer-bearing span; single generation/judge per arm."
    runner.save(run/"per_query.json", per_arm)
    runner.save(run/"summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("prepare", "evaluate", "summarise"))
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--rankings", type=Path, default=ROOT/"rag/results/manual-full-20261118-135559/qwen4-practice/rankings.jsonl")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.stage == "prepare":
        prepare(args.run.resolve(), args.rankings.resolve())
    elif args.stage == "evaluate":
        evaluate(args.run, args.resume)
    else:
        summarise(args.run)


if __name__ == "__main__":
    main()
