"""Offline diagnostic sample from frozen practice answers; no inference or API calls."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RUN = Path("rag/results/qwen4-answers-3724290")
ARMS = ("dense5", "qwen4_matched5")
SEED = 20261118
SAMPLE_SIZE = 20
NOTES = Path("rag/experiments/practice-error-analysis-v1.json")
RECHECK = Path("rag/experiments/error-case-recheck.json")
DIMENSIONS = ("correctness", "grounding", "completeness")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def coverage(selected, gold):
    hits = len(set(selected) & set(gold))
    return "none" if hits == 0 else "full" if hits == len(gold) else "partial"


def sample_ids(eligible, size=SAMPLE_SIZE, seed=SEED):
    return sorted(random.Random(seed).sample(sorted(eligible), size))


def score_states(dense, qwen):
    assert set(dense) == set(qwen)
    result = {key: 0 for key in ("zero_zero", "zero_positive", "positive_zero", "positive_positive")}
    for qid in dense:
        label = "_".join("zero" if rows[qid]["normalised_answer_score"] == 0 else "positive"
                         for rows in (dense, qwen))
        result[label] += 1
    return result


def verify_quotes(case, note):
    """Every review quotation must be present verbatim in its named saved source."""
    arm = case["arms"]["qwen4_matched5"]
    for quote in note["quotes"]:
        source = quote["source"]
        if source == "answer":
            text = arm["answer"]
        elif source == "judge":
            text = "\n".join(arm["judge_explanations"])
        else:
            assert source in ("selected", "gold"), source
            docs = arm["selected_documents"] if source == "selected" else case["gold_documents"]
            text = next(d["text"] for d in docs if d["record_id"] == quote["record_id"])
        assert quote["quote"] in text, (case["query_id"], quote)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "rag/results/practice-error-analysis-v1")
    args = parser.parse_args()
    config = json.loads((ROOT / RUN / "config.json").read_text())
    judge_source = Path("rag/src/evaluation/answer_judge.py")
    judge_bytes = (ROOT / judge_source).read_bytes()
    judge_lf = judge_bytes.replace(b"\r\n", b"\n")
    assert config["source_sha256"][str(judge_source)] in {
        hashlib.sha256(judge_lf).hexdigest(),
        hashlib.sha256(judge_lf.replace(b"\n", b"\r\n")).hexdigest(),
    }
    sources = [RUN / "config.json", judge_source]
    answers, judges = {}, {}
    for arm in ARMS:
        af = RUN / "answers" / f"{arm}.jsonl"
        jf = RUN / f"judge_json_{arm}" / "answer_judge_per_query.jsonl"
        sources.extend((af, jf))
        answers[arm] = {r["query_id"]: r for r in rows(ROOT / af)}
        judges[arm] = {r["query_id"]: r for r in rows(ROOT / jf)}
        assert len(rows(ROOT / af)) == len(rows(ROOT / jf)) == len(config["query_ids"]) == 350
        assert set(answers[arm]) == set(judges[arm]) == set(config["query_ids"])
    with zipfile.ZipFile(ROOT / "rag-assets.zip") as archive:
        corpus_bytes = archive.read("data/rag_corpus/corpus.jsonl")
        qrel_bytes = archive.read("data/rag_corpus/qrels_practice.tsv")
    assert hashlib.sha256(corpus_bytes).hexdigest() == config["input_sha256"]["corpus"]
    corpus = {r["record_id"]: r["text"] for r in map(json.loads, corpus_bytes.decode().splitlines())}
    qrels = collections.defaultdict(dict)
    for row in csv.DictReader(io.StringIO(qrel_bytes.decode("utf-8-sig")), delimiter="\t"):
        if float(row["score"]) > 0:
            qrels[row["query_id"]][row["record_id"]] = float(row["score"])
    summaries = {}
    for arm in ARMS:
        counts = collections.Counter()
        cf, rf = RUN / f"judge_json_{arm}/judge_cache.jsonl", RUN / f"judge_json_{arm}/judge_raw.jsonl"
        sources.extend((cf, rf))
        cached = {r["query_id"]: r["result"] for r in rows(ROOT / cf)}
        raw_results = collections.defaultdict(list)
        for row in rows(ROOT / rf):
            try:
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", row["raw"].strip(), flags=re.I)
                parsed = json.loads(raw)
                assert all(type(parsed[d]) is int and 0 <= parsed[d] <= 4 for d in DIMENSIONS)
            except (ValueError, KeyError, TypeError, AssertionError):
                continue
            raw_results[row["question"], row["candidate_sha256"]].append(parsed)
        for qid, a in answers[arm].items():
            assert a["split"] == "practice"
            j = judges[arm][qid]
            assert all(j[d] == cached[qid][d] for d in DIMENSIONS)
            assert cached[qid] in raw_results[a["transcript"], hashlib.sha256(a["answer"].encode()).hexdigest()]
            assert all(0 <= j[d] <= 4 for d in ("correctness", "grounding", "completeness"))
            assert math.isclose(j["normalised_answer_score"],
                                sum(j[d] for d in ("correctness", "grounding", "completeness")) / 12)
            assert a["retrieved_record_ids"] == [d["record_id"] for d in a["selected_documents"]]
            assert len(a["retrieved_record_ids"]) == 5
            texts = [corpus[rid] for rid in a["retrieved_record_ids"]]
            evidence = "\n\n".join(f"[Document {i}]\n{text}" for i, text in enumerate(texts, 1))
            prompt = ("Answer the question using the retrieved evidence below. "
                      "If the evidence is insufficient, say so.\n\n"
                      f"Question:\n{a['transcript']}\n\nEvidence:\n{evidence}")
            assert a["prompt"] == prompt
            assert hashlib.sha256(prompt.encode()).hexdigest() == a["prompt_sha256"]
            assert all(not d["truncated"] and d["original_characters"] == d["included_characters"]
                       for d in a["selected_documents"])
            assert qrels[qid]
            if judges[arm][qid]["normalised_answer_score"] == 0:
                counts[coverage(a["retrieved_record_ids"], qrels[qid])] += 1
        summaries[arm] = {"queries": len(answers[arm]), "all_zero": sum(counts.values()),
                          "zero_by_qrel_coverage": dict(sorted(counts.items())),
                          "dimension_zero_counts": {d: sum(j[d] == 0 for j in judges[arm].values())
                                                    for d in DIMENSIONS},
                          "mixed_zero_dimension_answers": sum(
                              any(j[d] == 0 for d in DIMENSIONS) and not all(j[d] == 0 for d in DIMENSIONS)
                              for j in judges[arm].values()),
                          "empty_answers": sum(not a["answer"].strip() for a in answers[arm].values()),
                          "finish_reasons": dict(collections.Counter(a["finish_reason"] for a in answers[arm].values())),
                          "all_scores_match_cached_and_raw_judge_responses": True}
    eligible = sorted(q for q, j in judges[ARMS[1]].items() if j["normalised_answer_score"] == 0)
    sampled = sample_ids(eligible)
    cases = []
    for qid in sampled:
        cases.append({
            "query_id": qid, "question": answers[ARMS[1]][qid]["transcript"],
            "qwen_qrel_coverage": coverage(answers[ARMS[1]][qid]["retrieved_record_ids"], qrels[qid]),
            "gold_documents": [{"record_id": rid, "relevance": rel, "text": corpus[rid]}
                               for rid, rel in sorted(qrels[qid].items())],
            "arms": {arm: {
                "answer": answers[arm][qid]["answer"],
                "score": judges[arm][qid]["normalised_answer_score"],
                "judge_explanations": judges[arm][qid]["judge_explanations"],
                "prompt_sha256": answers[arm][qid]["prompt_sha256"],
                "selected_documents": [{"record_id": rid, "text": corpus[rid],
                                        "qrel_positive": rid in qrels[qid]}
                                       for rid in answers[arm][qid]["retrieved_record_ids"]],
            } for arm in ARMS}
        })
    notes = json.loads((ROOT / NOTES).read_text())
    indexed_notes = {n["query_id"]: n for n in notes["cases"]}
    assert len(indexed_notes) == len(notes["cases"]) == len(sampled)
    assert set(indexed_notes) == set(sampled)
    for case in cases:
        verify_quotes(case, indexed_notes[case["query_id"]])
    recheck = json.loads((ROOT / RECHECK).read_text())
    indexed_cases = {c["query_id"]: c for c in cases}
    for note in recheck["cases"]:
        verify_quotes(indexed_cases[note["query_id"]], note)
    assert type(recheck["author_confirmation_outstanding"]) is bool and not recheck["scores_changed"]
    sources.extend((NOTES, RECHECK))
    tag_counts = collections.Counter(tag for n in notes["cases"] for tag in n["tags"])
    summary = {
        "schema": "practice-error-analysis-v1", "seed": SEED, "sample_size": SAMPLE_SIZE,
        "sampling": "Uniform without replacement from sorted Qwen practice all-zero judge IDs",
        "eligible_queries": len(eligible), "sampled_query_ids": sampled,
        "population": summaries,
        "paired_score_states": score_states(judges[ARMS[0]], judges[ARMS[1]]),
        "judge_zero_template_risk": {
            "source_matches_frozen_run_modulo_lf_crlf": True,
            "schema_literal_present": '{"correctness":0,"grounding":0,"completeness":0,' in (ROOT / judge_source).read_text(),
            "interpretation": "Possible formatting/anchoring risk, not a measured causal explanation; no rejudging.",
        },
        "sample_qrel_coverage": dict(collections.Counter(c["qwen_qrel_coverage"] for c in cases)),
        "all_700_saved_prompts_match_full_corpus_text": True,
        "all_700_prompt_hashes_match": True,
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "practice_qrels_sha256": hashlib.sha256(qrel_bytes).hexdigest(),
        "source_sha256": {str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "qualitative_tags_nonexclusive": dict(sorted(tag_counts.items())),
        "reviewer": notes["reviewer"],
        "review_quotations_verified_verbatim": True,
        "representative_case_recheck": recheck,
        "limitations": [
            "Judge-zero is a sampling label, not human-verified incorrectness.",
            "AI-assisted unblinded qualitative review; not independent human adjudication.",
            "Qrel coverage is document-ID coverage, not semantic evidence sufficiency.",
            "Saved outgoing prompts do not prove what the remote server internally processed.",
            "No snippet intervention or repeated generation/judging; no causal attribution.",
            "Practice-only exploratory diagnostic; no new benchmark improvement claim.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("summary.json", summary), ("cases.json", cases)):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    report = [
        "# Practice error diagnostic — offline, exploratory",
        "",
        f"Reviewer: {notes['reviewer']}.",
        "",
        f"Uniform sample: {SAMPLE_SIZE}/{len(eligible)} Qwen judge-zero practice answers; seed {SEED}.",
        "No new inference, generation, judging or dev analysis. Original scores unchanged.",
        "Gold = positive-qrel documents; selected = actual outgoing prompt documents.",
        "A missing qrel ID does not prove missing answer evidence. No causal snippet effect is measured.",
        "",
        "## Population counts (not inferred from the sample)",
        "",
        "| Arm | Queries | Judge all-zero | No qrel ID | Partial | All qrel IDs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, pop in summaries.items():
        c = pop["zero_by_qrel_coverage"]
        report.append(f"| {arm} | {pop['queries']} | {pop['all_zero']} | "
                      f"{c.get('none', 0)} | {c.get('partial', 0)} | {c.get('full', 0)} |")
    report.extend([
        "",
        "All 700 saved prompts exactly match full corpus text and their saved hashes.",
        "This rules out local context-builder prefix truncation for these runs, not remote-server truncation.",
        "All 700 scores match saved raw judge responses and caches; no empty answers.",
        "The three dimension-zero sets coincide in each arm; this flags calibration, not proven judge bias.",
        "The frozen judge prompt uses a zero-filled JSON example; anchoring is possible but untested.",
        "Paired score states: " + json.dumps(summary["paired_score_states"], sort_keys=True) + ".",
        "",
        "## Sample review",
        "",
        "Sample qrel-ID coverage: " + json.dumps(summary["sample_qrel_coverage"], sort_keys=True) + ".",
        "Tags are non-exclusive observations/hypotheses, not verified causes or population error rates.",
        "Judge-text conflicts are review flags, not corrected scores.",
        "",
    ])
    for i, case in enumerate(cases, 1):
        note = indexed_notes[case["query_id"]]
        report.extend([
            f"### {i}. `{case['query_id']}`", "",
            f"**Question:** {case['question']}", "",
            f"**Qrel-ID coverage:** {case['qwen_qrel_coverage']}; "
            f"Qwen score: 0; paired Dense5 score: {case['arms']['dense5']['score']:.4f}.", "",
            f"**Tags:** {', '.join(note['tags'])}.", "",
            note["finding"], "", f"**Snippet assessment:** {note['snippet_assessment']}", "",
        ])
        for q in note["quotes"]:
            origin = q["source"] + ("/" + q["record_id"] if "record_id" in q else "")
            report.extend([f"> [{origin}] {q['quote']}", ""])
    report.extend(["## Second source check (not human adjudication)", "", recheck["reviewer"], ""])
    if not recheck["author_confirmation_outstanding"]:
        report.extend(["The user reports author confirmation of these three representative cases; "
                       "this is not a complete 20-case mechanism taxonomy or causal intervention.", ""])
    for note in recheck["cases"]:
        report.extend([f"### `{note['query_id']}`", "", note["finding"], ""])
        for quote in note["quotes"]:
            report.extend([f"> [{quote['source']}] {quote['quote']}", ""])
    (args.output / "REVIEW.md").write_text("\n".join(report).rstrip() + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
