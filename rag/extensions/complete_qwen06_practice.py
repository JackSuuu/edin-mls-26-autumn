"""Complete the matched-source 0.6B arm against existing manual-rerun controls."""
import argparse
import csv
import json
import random
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

from run_qwen_answer_experiment import ROOT, MODEL, digest, indexed, save
from extensions.evidence_selection import chat_input_tokens, select_context
from src.data import CourseData, load_jsonl
from src.generator import ELMGenerator, SYSTEM_PROMPT
from dotenv import load_dotenv
from transformers import AutoTokenizer


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    args = cli.parse_args()
    base = args.base.resolve()
    out = base / "qwen06-practice-answers"
    out.mkdir(exist_ok=True)
    previous = base / "qwen4-practice-answers"
    rank = base / "qwen06-practice/rankings.jsonl"
    cfg = json.loads((base / "qwen06-practice/config.json").read_text())
    assert cfg["model"]["repo"] == "Qwen/Qwen3-Reranker-0.6B" and cfg["split"] == "practice"
    rankings = indexed(load_jsonl(rank))
    dense = indexed(load_jsonl(previous / "answers/dense5.jsonl"))
    data = CourseData(ROOT / "rag/data/rag_corpus")
    assert len(rankings) == 350 and set(rankings) == set(dense) == {r["query_id"] for r in data.audio_items("practice")}
    assert digest(ROOT / "rag/data/rag_corpus/corpus.jsonl") == cfg["input_sha256"]["corpus"]
    tok = AutoTokenizer.from_pretrained(ROOT / "rag/models/selection/tokenizer", local_files_only=True)
    load_dotenv(ROOT / "rag/.env")
    gen = ELMGenerator()
    assert gen.model == MODEL
    protocol = dict(model=MODEL, rank_hash=digest(rank), endpoint=str(gen.client.base_url),
                    controls=str(previous), arm="qwen06_matched5", split="practice", n=350,
                    temperature=0.1, judge_json=True, repeats=1, script_sha256=digest(Path(__file__)))
    if (out / "protocol.json").exists():
        assert json.loads((out / "protocol.json").read_text()) == protocol
    else:
        save(out / "protocol.json", protocol)
    path = out / "answers.jsonl"
    done = indexed(load_jsonl(path)) if path.exists() else {}
    for i, (q, row) in enumerate(rankings.items(), 1):
        arm = row["qwen_matched"]
        assert row["transcript"] == dense[q]["transcript"] and row["dense"]["record_ids"][:5] == dense[q]["retrieved_record_ids"]
        start = time.perf_counter()
        built = select_context(row["transcript"], arm["record_ids"], data.records,
                               lambda prompt: chat_input_tokens(tok, SYSTEM_PROMPT, prompt), top_k=5)
        build_ms = (time.perf_counter() - start) * 1000
        if q in done:
            assert done[q]["prompt"] == built["prompt"] and done[q]["finish_reason"] == "stop"
            continue
        result = gen.generate(built["prompt"])
        assert result["finish_reason"] == "stop" and result["answer"].strip() and result["response_model"] == MODEL
        a = dict(query_id=q, arm="qwen06_matched5", split="practice", transcript=row["transcript"],
                 **built, **result, generation_ms=result["latency_ms"], context_build_ms=build_ms,
                 cached_reranking_ms=arm["rerank_ms"], attributed_selection_plus_generation_ms=
                 arm["rerank_ms"] + build_ms + result["latency_ms"])
        with path.open("a") as stream:
            stream.write(json.dumps(a) + "\n")
        print(f"Generated {i}/350 {q} {result['latency_ms']:.0f} ms", flush=True)
    judge = out / "judge_json_qwen06_matched5"
    subprocess.run([sys.executable, "-u", str(ROOT / "rag/scripts/evaluate_answers.py"),
                    "--split", "practice", "--model", MODEL, "--json-mode",
                    "--candidates", str(path), "--out-dir", str(judge)], check=True)
    rows = []
    scores = {}
    for method, ap, jp, rp, name in [
        ("Dense5", previous/"answers/dense5.jsonl", previous/"judge_json_dense5", rank, "dense"),
        ("Qwen-0.6B", path, judge, rank, "qwen_matched"),
        ("Qwen-4B", previous/"answers/qwen4_matched5.jsonl", previous/"judge_json_qwen4_matched5",
         base/"qwen4-practice/rankings.jsonl", "qwen_matched")]:
        a = indexed(load_jsonl(ap))
        j = indexed(load_jsonl(jp/"answer_judge_per_query.jsonl"))
        rs = indexed(load_jsonl(rp))
        assert set(a) == set(j) == set(rs) == set(rankings)
        assert all(a[q]["retrieved_record_ids"] == rs[q][name]["record_ids"][:5] for q in a)
        scores[method] = {q:v["normalised_answer_score"] for q,v in j.items()}
        rows.append(dict(method=method, queries=len(a), recall5=st.mean(rs[q][name]["metrics"]["Recall@5"] for q in a),
                         answer_score=st.mean(scores[method].values()),
                         zeros=sum(v==0 for v in scores[method].values()),
                         **{key:st.mean(v[key] for v in a.values()) for key in
                            ("cached_reranking_ms","generation_ms","attributed_selection_plus_generation_ms")}))
    with (out/"three_points.csv").open("w") as f:
        w = csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    diffs = {}
    for m1, m2 in [("Qwen-0.6B","Dense5"),("Qwen-4B","Qwen-0.6B")]:
        d = [scores[m1][q]-scores[m2][q] for q in rankings]
        rng = random.Random(20261116)
        vals = sorted(sum(rng.choices(d,k=len(d)))/len(d) for _ in range(2000))
        diffs[f"{m1} minus {m2}"] = [st.mean(d),vals[50],vals[1949]]
    save(out/"comparison.json", dict(rows=rows,paired_delta_and_query_ci95=diffs,
         note="0.6B measured in a later ELM window; controls reused, timings attributed from stages."))
    print(json.dumps(rows, indent=2), json.dumps(diffs), flush=True)


if __name__ == "__main__":
    main()
