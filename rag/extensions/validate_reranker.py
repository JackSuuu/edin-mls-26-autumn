"""Check score orientation and reproduce cached scores independently."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))
from src.data import load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("Output exists")
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(2)
    path = str(ROOT / "rag/models/selection/reranker")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).eval()
    def scores(pairs):
        features = tokenizer([p[0] for p in pairs], [p[1] for p in pairs], padding=True,
                             truncation="only_second", max_length=512, return_tensors="pt")
        with torch.inference_mode():
            return model(**features).logits.flatten()
    toy = scores([
        ("How many people live in Berlin?", "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers."),
        ("How many people live in Berlin?", "New York City is famous for the Metropolitan Museum of Art."),
    ])
    if not toy[0] > toy[1]:
        raise AssertionError("Score orientation sanity check failed")
    records = {r["record_id"]: r["text"] for r in load_jsonl(ROOT / "rag/data/rag_corpus/corpus.jsonl")}
    plans = load_jsonl(args.plan / "plans.jsonl")
    ranked = {r["query_id"]: r for r in load_jsonl(args.plan / "rerankings.jsonl")}
    checks = []
    for qid in ("q_b7390fb1b4a9ad79", "q_8ef57dbb7e83f8d2", "q_d40345916def841c"):
        plan = next(p for p in plans if p["query_id"] == qid)
        cached = ranked[qid]
        score_map = dict(zip(cached["record_ids"], cached["scores"]))
        ids = cached["record_ids"][:5] + cached["record_ids"][-5:]
        actual = scores([(plan["transcript"], records[rid]) for rid in ids])
        expected = torch.tensor([score_map[rid] for rid in ids])
        matches = torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
        checks.append({"query_id": qid, "pairs": len(ids), "matches": matches,
                       "max_absolute_difference": float((actual - expected).abs().max())})
        if not matches:
            raise AssertionError("Cached scores do not reproduce")
    result = {"passed": True, "toy_logits": toy.tolist(), "cached_score_checks": checks,
              "source": "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2",
              "note": "Implementation validation, not a domain-quality benchmark"}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
