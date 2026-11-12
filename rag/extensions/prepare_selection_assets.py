"""Download pinned reranker and generator tokenizer only, on the login node."""
from pathlib import Path
import hashlib
import json
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rag/models/selection"


def main():
    manifest = {}
    for name, repo, revision, patterns in (
        ("reranker", "cross-encoder/ms-marco-MiniLM-L6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a",
         ["config.json", "model.safetensors", "tokenizer*", "special_tokens_map.json", "vocab.txt"]),
        ("tokenizer", "utter-project/EuroLLM-22B-Instruct-2512", "8c6ed5c8d18527d9daed994f645032b86c53f0bb",
         ["config.json", "tokenizer*", "special_tokens_map.json", "chat_template*", "added_tokens.json"]),
    ):
        local = OUT / name
        snapshot_download(repo, revision=revision, local_dir=local, allow_patterns=patterns)
        hashes = {str(p.relative_to(local)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(local.rglob("*")) if p.is_file() and ".cache" not in p.parts}
        if name == "reranker" and "model.safetensors" not in hashes:
            raise RuntimeError("Reranker safetensors missing")
        manifest[name] = {"repo": repo, "revision": revision, "files_sha256": hashes}
        print(name, revision, len(hashes), flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
