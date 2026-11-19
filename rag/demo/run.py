"""Send live audio RAG stages (or labelled recorded results) to the dashboard."""
import argparse
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

RAG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAG))


def live(args, emit, data):
    from dotenv import load_dotenv
    from src.asr_adapter import ASRAdapter
    from src.context_builder import SimpleContextBuilder
    from src.encoder import LocalEncoder
    from src.generator import ELMGenerator
    from src.retriever import ExactDenseRetriever

    items = data.audio_items(args.split)
    item = next((x for x in items if not args.query_id or x["query_id"] == args.query_id), None)
    if not args.audio and item is None:
        raise ValueError("No matching course query")
    audio = args.audio or data.data_dir / item["audio_file"]
    load_dotenv(RAG / ".env")
    generator = ELMGenerator()
    expected = "utter-project/EuroLLM-22B-Instruct-2512"
    if generator.model != expected:
        raise ValueError("Expected the coursework EuroLLM generator")
    models = RAG / "models"
    encoder = LocalEncoder(models / "enc_a", device="cpu")
    search = ExactDenseRetriever(data.data_dir / "embeddings/corpus_enc_a.npy",
                                 data.data_dir / "embeddings/corpus_enc_a_ids.json")
    score = None
    if args.method == "qwen":
        emit("init", "Loading the Qwen‑4B reranker")
        score = load_qwen(models)

    with ASRAdapter() as asr:
        emit("init", "Loading the GLM-ASR worker (not included in request time)")
        asr.prepare()
        start = time.perf_counter()
        emit("asr", "GLM-ASR is transcribing the audio…", source=audio.name if args.audio else item["query_id"])
        transcript = asr.transcribe(audio)
        emit("retrieve", "Searching the dense index…", transcript=transcript,
             asr_ms=(time.perf_counter() - start) * 1000)
        t = time.perf_counter()
        ids = search.retrieve(encoder.encode(transcript), k=100)["record_ids"]
        emit("retrieve", f"Dense search found {len(ids)} candidates", candidates=len(ids),
             retrieval_ms=(time.perf_counter() - t) * 1000, dense_top5=ids[:5])
        if score:
            t = time.perf_counter()
            ids = score(transcript, ids, data, emit)
            emit("rerank", f"Qwen‑4B ranked {len(ids)} candidates; keeping five",
                 rerank_ms=(time.perf_counter() - t) * 1000)
        else:
            emit("rerank", "Dense5 keeps the original order", rerank_ms=0)
        docs = ids[:5]
        prompt = SimpleContextBuilder().build(transcript, docs, data)["prompt"]
        emit("generate", "EuroLLM is generating an answer…", documents=previews(data, docs))
        result = generator.generate(prompt)
        if result["finish_reason"] != "stop" or not result["answer"]:
            raise RuntimeError(f"Generation did not complete normally ({result['finish_reason']})")
        total_ms = (time.perf_counter() - start) * 1000
        emit("generate", "EuroLLM returned the complete answer", answer=result["answer"],
             generation_ms=result["latency_ms"], total_ms=total_ms,
             tokens=(result["usage"] or {}).get("prompt_tokens"))
        if args.judge:
            judge_answer(data, args.split, item["query_id"], transcript, result["answer"], expected, emit)
        else:
            emit("done", "Answer ready · evaluation not requested")


def judge_answer(data, split, query_id, transcript, answer, model, emit):
    """Use the coursework gold-evidence rubric, after the response timer stops."""
    from src.evaluation.answer_judge import Judge

    rels = data.load_qrels(split).get(query_id, {})
    if not rels:
        emit("done", "Answer ready · no reference evidence for this query", judge_status="unavailable")
        return
    gold = [{"record_id": rid, "text": data.record_text(rid)}
            for rid, _ in sorted(rels.items(), key=lambda pair: (-pair[1], pair[0]))]
    emit("judge", "EuroLLM judge is comparing the answer with labelled reference evidence…",
         judge_status="running")
    start = time.perf_counter()
    try:
        score = Judge(model, json_mode=True).evaluate(transcript, answer, gold)
    except Exception as exc:
        print(f"Judge failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        emit("done", "Answer ready · judge request failed (no score assigned)", judge_status="failed")
        return
    dimensions = ("correctness", "grounding", "completeness")
    emit("done", "Answer and evaluation ready", judge_status="scored",
         scores={key: score[key] for key in dimensions},
         normalised_score=sum(score[key] for key in dimensions) / 12,
         explanation=score["explanation"], judge_ms=(time.perf_counter() - start) * 1000)


def load_qwen(models):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(RAG / "extensions"))
    from evaluate_qwen_reranker import qwen_input, ranked, visible_prefix

    mini = AutoTokenizer.from_pretrained(models / "selection/reranker", local_files_only=True, use_fast=True)
    directory = models / "qwen3-reranker-4b"
    tok = AutoTokenizer.from_pretrained(directory, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").eval().cuda()
    yes, no = (tok.convert_tokens_to_ids(word) for word in ("yes", "no"))

    @torch.inference_mode()
    def score(question, ids, data, emit):
        scores = []
        emit("rerank", f"Qwen‑4B scoring 0/{len(ids)} candidates", progress=0)
        for i in range(0, len(ids), 8):
            docs = [visible_prefix(mini, question, data.record_text(rid)) for rid in ids[i:i+8]]
            encoded = [qwen_input(tok, question, doc, forbid_truncation=True)[0] for doc in docs]
            batch = tok.pad(encoded, padding=True, return_tensors="pt").to("cuda")
            logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
            scores.extend((logits[:, yes] - logits[:, no]).cpu().tolist())
            emit("rerank", f"Qwen‑4B scoring {len(scores)}/{len(ids)} candidates",
                 progress=len(scores) / len(ids))
        return ranked(ids, scores)

    return score


def previews(data, ids):
    return [{"id": rid, "text": " ".join(data.record_text(rid).split())[:460]} for rid in ids]


def replay(args, emit, data):
    from src.data import load_jsonl
    rows = load_jsonl(args.replay)
    row = next((r for r in rows if r["arm"] == args.method and not r.get("warmup")
                and (not args.query_id or r["qid"] == args.query_id)), None)
    if row is None:
        raise ValueError("No matching recorded query/selector")
    emit("asr", "Showing a saved audio request", source=row["qid"])
    time.sleep(0.7)
    emit("retrieve", "Saved transcript; upstream timing was recorded as a single stage",
         transcript=row["transcript"], upstream_ms=row["upstream"] * 1000)
    time.sleep(0.7)
    emit("rerank", "Saved selector timing", rerank_ms=row["rerank"] * 1000)
    time.sleep(0.7)
    emit("generate", "Showing the five documents from the recorded request",
         documents=previews(data, row["selected"]))
    time.sleep(0.7)
    emit("done", "Replay complete (no inference performed)", answer=row["answer"], judge_status="unavailable",
         generation_ms=row["generation"] * 1000, total_ms=row["total"] * 1000,
         tokens=(row["usage"] or {}).get("prompt_tokens"))


def main():
    from src.data import CourseData
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--method", choices=("dense", "qwen"), default="dense")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--query-id")
    inputs.add_argument("--audio", type=Path)
    parser.add_argument("--split", choices=("practice", "dev"), default="practice")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--judge", action="store_true",
                        help="Also run the EuroLLM judge against the selected course query's qrels")
    args = parser.parse_args()
    if args.replay and args.audio:
        parser.error("--replay uses recorded queries, not an audio file")
    if args.judge and (args.replay or args.audio):
        parser.error("--judge requires a live course query with labelled evidence")
    args.events.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    request_started = None

    def emit(stage, message, **fields):
        nonlocal request_started
        if stage == "asr" and not args.replay:
            request_started = time.perf_counter()
        if request_started is not None:
            fields["elapsed_ms"] = (time.perf_counter() - request_started) * 1000
        row = dict(run=run_id, mode="replay" if args.replay else "live", method=args.method,
                   stage=stage, message=message, **fields)
        with args.events.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{stage}] {message}", flush=True)

    emit("init", "Loading the system", judge_status="queued" if args.judge else "skipped")
    try:
        data = CourseData(RAG / "data/rag_corpus")
        (replay if args.replay else live)(args, emit, data)
    except (Exception, KeyboardInterrupt) as exc:
        emit("error", f"Run stopped ({type(exc).__name__}); see the terminal for details.")
        raise


if __name__ == "__main__":
    main()
