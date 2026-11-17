"""Read cached judgements and original texts; no API calls or model inference."""
import csv
import hashlib
import json
from pathlib import Path
import sys

repo = Path.cwd()
sys.path.insert(0, str(repo / "rag"))
from src.evaluation.answer_judge import cache_key, build_public_gold
run = repo / "rag/results/qwen4-answers-3724290"
def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines(keepends=True)
            if line.strip() and line.endswith("\n")]

targets = ["q_f78418e691979c64", "q_371c6ca10bf1bf5e", "q_9d78dff597245697",
           "q_890a6df9ce937b08", "q_9a684db4ff009865", "q_0a12138b81eb7e2a",
           "q_6382e94af1377ac6", "q_6eb08803469fb277"]
answers = {arm: {r["query_id"]: r for r in rows(run / "answers" / (arm + ".jsonl"))}
           for arm in ("dense5", "qwen4_matched5")}
entries = {arm: {r["query_id"]: r for r in rows(run / ("judge_json_" + arm) / "judge_cache.jsonl")}
           for arm in answers}
caches = {arm: {qid: r["result"] for qid, r in cache.items()} for arm, cache in entries.items()}
gold = build_public_gold(repo / "rag/data/rag_corpus", "practice")
generation_config = json.loads((run / "answers/generation_config.json").read_text())
raw = rows(run / "judge_json_dense5/judge_raw.jsonl")
qrels = {}
with (repo / "rag/data/rag_corpus/qrels_practice.tsv").open(encoding="utf-8-sig", newline="") as stream:
    for rel in csv.DictReader(stream, delimiter="\t"):
        if float(rel["score"]) > 0:
            qrels.setdefault(rel["query_id"], []).append(rel["record_id"])
needed = {rid for qid in targets for rid in qrels[qid]}
texts = {}
with (repo / "rag/data/rag_corpus/corpus.jsonl").open(encoding="utf-8") as stream:
    for line in stream:
        record = json.loads(line)
        if record["record_id"] in needed:
            texts[record["record_id"]] = record["text"]
snapshot = {"stage": "in_progress_cached_judge_diagnostic", "cases": [],
            "scored_snapshot": {arm: {"queries": len(cache), "all_zero": sum(
                all(r[k] == 0 for k in ("correctness", "grounding", "completeness")) for r in cache.values())}
                for arm, cache in caches.items()}}
for qid in targets:
    case = {"query_id": qid, "question": answers["dense5"][qid]["transcript"],
            "gold_documents": [{"record_id": rid, "text": texts[rid]} for rid in qrels[qid]], "arms": {}}
    for arm in answers:
        answer = answers[arm][qid]
        expected_key = cache_key("utter-project/EuroLLM-22B-Instruct-2512@" +
            generation_config["endpoint"] + "#response_format=json_object",
            answer["transcript"], answer["answer"], gold[qid]["gold_documents"], 0)
        case["arms"][arm] = {"answer": answer["answer"], "judge": caches[arm].get(qid),
                             "cache_matches_current_answer_and_gold": entries[arm].get(qid, {}).get("cache_key") == expected_key,
                             "retrieved_record_ids": answer["retrieved_record_ids"],
                             "relevant_ids_retrieved": sorted(set(answer["retrieved_record_ids"]) & set(qrels[qid]))}
    case["raw_judge_response_matches_current_answer"] = any(r["question"] == case["question"] and
        r["candidate_sha256"] == hashlib.sha256(case["arms"]["dense5"]["answer"].encode()).hexdigest()
        for r in raw)
    snapshot["cases"].append(case)
output = run / "zero-cases-diagnostic.json"
output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(snapshot["scored_snapshot"]))
for case in snapshot["cases"]:
    print("\nQUERY", case["query_id"], case["question"])
    print("DENSE JUDGE", json.dumps(case["arms"]["dense5"]["judge"], ensure_ascii=False))
    print("CACHE_AND_RAW_MATCH", case["arms"]["dense5"]["cache_matches_current_answer_and_gold"],
          case["raw_judge_response_matches_current_answer"])
    print("DENSE RELEVANT", case["arms"]["dense5"]["relevant_ids_retrieved"])
    print("DENSE ANSWER", case["arms"]["dense5"]["answer"][:2000])
    print("GOLD", json.dumps(case["gold_documents"], ensure_ascii=False)[:5500])
