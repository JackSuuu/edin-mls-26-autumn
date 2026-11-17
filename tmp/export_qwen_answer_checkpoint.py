"""Read-only snapshot of matched completed answers; never changes live run files."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

run = Path(sys.argv[1])
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
target = run / "checkpoints" / stamp
target.mkdir(parents=True)
config = json.loads((run / "config.json").read_text(encoding="utf-8"))
arms = ("dense5", "qwen4_matched5")
data = {}
for arm in arms:
    lines = (run / "answers" / (arm + ".jsonl")).read_text(encoding="utf-8").splitlines(keepends=True)
    rows = [json.loads(line) for line in lines if line.endswith("\n") and line.strip()]
    data[arm] = {row["query_id"]: row for row in rows}
    if len(data[arm]) != len(rows):
        raise ValueError("Duplicate query IDs")
common = set(data[arms[0]]) & set(data[arms[1]])
queries = [qid for qid in config["query_ids"] if qid in common]
for arm in arms:
    with (target / (arm + ".jsonl")).open("x", encoding="utf-8") as stream:
        for qid in queries:
            row = data[arm][qid]
            if row["finish_reason"] != "stop" or hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
                raise ValueError("Incomplete answer or prompt hash mismatch")
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
shutil.copyfile(run / "config.json", target / "experiment_config.json")
shutil.copyfile(run / "answers/generation_config.json", target / "generation_config.json")
for arm in arms:
    source = run / ("judge_json_" + arm)
    summary = json.loads((source / "answer_judge_summary.json").read_text())
    destination = target / ("smoke_judge_" + arm if summary["is_subset"] else "judge_" + arm)
    destination.mkdir()
    for name in ("answer_judge_summary.json", "answer_judge_per_query.jsonl"):
        shutil.copyfile(source / name, destination / name)
metadata = {"captured_utc": stamp, "source_job": "3724290", "status": "in_progress_checkpoint",
            "generation_complete": len(queries) == 350, "full_answer_evaluation_complete": (run / "comparison.json").exists(),
            "expected_queries_per_arm": 350, "matched_queries_per_arm": len(queries),
            "observed_complete_answers_per_arm": {arm: len(data[arm]) for arm in arms},
            "query_ids": queries, "note": "Fixed paired snapshot, not a resume directory or a final benchmark result",
            "files_sha256": {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(target.rglob("*")) if p.is_file()}}
(target / "checkpoint.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print(str(target), flush=True)
print(json.dumps({k: v for k, v in metadata.items() if k not in ("query_ids", "files_sha256")}), flush=True)
