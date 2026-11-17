import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
checkpoint = root / "rag/results/qwen4-answers-3724290/checkpoints/20261116T220346Z"
metadata = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
for relative, expected in metadata["files_sha256"].items():
    assert hashlib.sha256((checkpoint / relative).read_bytes()).hexdigest() == expected, relative
for arm in ("dense5", "qwen4_matched5"):
    rows = [json.loads(line) for line in (checkpoint / (arm + ".jsonl")).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == metadata["matched_queries_per_arm"] == 240
    assert [row["query_id"] for row in rows] == metadata["query_ids"]
    for row in rows:
        assert row["finish_reason"] == "stop"
        assert hashlib.sha256(row["prompt"].encode()).hexdigest() == row["prompt_sha256"]
assert metadata["full_answer_evaluation_complete"] is False
print("PUBLICATION_CHECKPOINT_VERIFIED: 240 paired queries, hashes and prompts intact, full evaluation pending")
