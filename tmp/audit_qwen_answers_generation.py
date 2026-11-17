import collections
import hashlib
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parents[1]
run = root / "rag/results/qwen4-answers-3724290"
load = lambda p: [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
config = json.loads((run / "config.json").read_text(encoding="utf-8"))
prepared = json.loads((run / "prepared.json").read_text(encoding="utf-8"))
for name, suffix in (("config", ".json"), ("plans", ".jsonl")):
    assert hashlib.sha256((run / (name + suffix)).read_bytes()).hexdigest() == prepared[name + "_sha256"]
plans = {(r["query_id"], r["arm"]): r for r in load(run / "plans.jsonl")}
assert len(plans) == 700
audit = {"generation_complete": True, "answer_evaluation_complete": False, "arms": {}}
for arm in ("dense5", "qwen4_matched5"):
    rows = load(run / "answers" / (arm + ".jsonl"))
    assert len(rows) == len({r["query_id"] for r in rows}) == 350
    assert {r["query_id"] for r in rows} == set(config["query_ids"])
    for row in rows:
        plan = plans[row["query_id"], arm]
        assert row["transcript"] == plan["transcript"]
        assert row["retrieved_record_ids"] == plan["retrieved_record_ids"]
        assert len(row["retrieved_record_ids"]) == 5
        assert row["answer"].strip() and row["finish_reason"] == "stop"
        assert hashlib.sha256(row["prompt"].encode()).hexdigest() == plan["prompt_sha256"] == row["prompt_sha256"]
    audit["arms"][arm] = {"queries": len(rows), "all_prompt_hashes_and_selected_records_match_plan": True,
        "all_finish_reasons_stop": True,
        "observed_response_models": dict(collections.Counter(r["response_model"] for r in rows)),
        "mean_generation_ms": statistics.mean(r["generation_ms"] for r in rows),
        "mean_local_input_tokens": statistics.mean(r["local_input_tokens"] for r in rows),
        "mean_server_input_tokens": statistics.mean(r["usage"]["prompt_tokens"] for r in rows),
        "mean_server_output_tokens": statistics.mean(r["usage"]["completion_tokens"] for r in rows),
        "server_local_token_mismatches": sum(r["usage"]["prompt_tokens"] != r["local_input_tokens"] for r in rows)}
(run / "generation-audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(audit, indent=2))
