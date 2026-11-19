"""Time warm, complete practice audio-to-answer requests for two selectors."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate_qwen_reranker import ROOT, ranked, visible_prefix, qwen_input
from src.asr_adapter import ASRAdapter
from src.context_builder import SimpleContextBuilder
from src.data import CourseData
from src.encoder import LocalEncoder
from src.generator import ELMGenerator
from src.retriever import ExactDenseRetriever


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    base = ROOT / "rag"
    assets = base / "models"
    data_dir = base / "data/rag_corpus"
    data = CourseData(data_dir)
    items = random.Random(20261118).sample(sorted(data.audio_items("practice"), key=lambda r: r["query_id"]), 11)
    (out / "sample.json").write_text(json.dumps([i["query_id"] for i in items]))
    load_dotenv(base / ".env")
    gen = ELMGenerator()
    assert gen.model == "utter-project/EuroLLM-22B-Instruct-2512"
    enc = LocalEncoder(assets / "enc_a", device="cpu")
    retr = ExactDenseRetriever(data_dir / "embeddings/corpus_enc_a.npy", data_dir / "embeddings/corpus_enc_a_ids.json")
    mini = AutoTokenizer.from_pretrained(assets / "selection/reranker", local_files_only=True, use_fast=True)
    qdir = assets / "qwen3-reranker-4b"
    tok = AutoTokenizer.from_pretrained(qdir, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(qdir, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()
    yes, no = [tok.convert_tokens_to_ids(w) for w in ("yes", "no")]
    rows = []
    with ASRAdapter(cuda_groups="all") as asr:
        for i, item in enumerate(items):
            pair = []
            for arm in (("dense", "qwen") if i % 2 == 0 else ("qwen", "dense")):
                start = time.perf_counter()
                text = asr.transcribe(data_dir / item["audio_file"])
                ids = retr.retrieve(enc.encode(text), k=100)["record_ids"]
                t1 = time.perf_counter()
                if arm == "qwen":
                    scores = []
                    for k in range(0, 100, 8):
                        docs = [visible_prefix(mini, text, data.record_text(x)) for x in ids[k:k+8]]
                        batch = tok.pad([qwen_input(tok, text, d, forbid_truncation=True)[0] for d in docs],
                                        padding=True, return_tensors="pt").to("cuda")
                        logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
                        scores.extend((logits[:, yes]-logits[:, no]).cpu().tolist())
                    ids = ranked(ids, scores)
                t2 = time.perf_counter()
                prompt = SimpleContextBuilder().build(text, ids[:5], data)["prompt"]
                t3 = time.perf_counter()
                result = gen.generate(prompt)
                stop = time.perf_counter()
                assert result["finish_reason"] == "stop" and result["answer"]
                row = dict(qid=item["query_id"], arm=arm, warmup=(i == 0), transcript=text,
                           selected=ids[:5], upstream=t1-start, rerank=t2-t1,
                           generation=stop-t3, total=stop-start, answer=result["answer"], usage=result["usage"])
                with (out / "rows.jsonl").open("a") as f:
                    f.write(json.dumps(row)+"\n")
                print(i, arm, round(row["total"], 3), flush=True)
                pair.append(row)
            assert pair[0]["transcript"] == pair[1]["transcript"]
            if i:
                rows.extend(pair)
    summary = {arm:{k:statistics.mean(r[k] for r in rows if r["arm"]==arm)
                    for k in ("upstream","rerank","generation","total")} for arm in ("dense","qwen")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
