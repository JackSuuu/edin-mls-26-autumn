import hashlib
import json
import re
import subprocess

paths = subprocess.check_output(["git", "diff", "--cached", "--name-only", "-z"]).decode().strip("\0").split("\0")
assert len(paths) == 39
total = 0
for path in paths:
    assert not path.startswith(("tmp/", "gpu.md", "report/AGENTS.md"))
    assert not any(part in (".env", ".venv", "models", "__pycache__") for part in path.split("/"))
    assert not path.endswith((".safetensors", ".bin", ".pyc", ".log"))
    payload = subprocess.check_output(["git", "show", ":" + path])
    total += len(payload)
    assert len(payload) < 20_000_000, path
    assert not re.search(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{24,}", payload), path
    if path.endswith(".json"):
        json.loads(payload)
    if path.endswith(".jsonl"):
        for row in payload.splitlines():
            json.loads(row)
    if path.startswith(("rag/results/qwen", "rag/results/diagnostic-retrieval-3713758/rankings.jsonl")):
        with open(path, "rb") as stream:
            assert payload == stream.read(), "Frozen bytes changed in index: " + path
base = "rag/results/qwen4-answers-3724290/checkpoints/20261116T220346Z/"
checkpoint = json.loads(subprocess.check_output(["git", "show", ":" + base + "checkpoint.json"]))
for relative, expected in checkpoint["files_sha256"].items():
    assert hashlib.sha256(subprocess.check_output(["git", "show", ":" + base + relative])).hexdigest() == expected
print(f"STAGED_PUBLICATION_VERIFIED: {len(paths)} files, {total / 1024**2:.2f} MiB; frozen bytes and checkpoint hashes preserved")
