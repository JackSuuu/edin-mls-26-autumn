"""Deterministic original-text sentence windows and budgeted passage packing.

No qrels, labels, model calls or generative rewriting. Sentence segmentation is
a conservative heuristic, not a guarantee that every qualification stays nearby.
"""
from __future__ import annotations

import math
import re

from extensions.evidence_selection import render_prompt

ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "vs.", "etc.", "inc.", "corp.", "no.", "fig.", "st.", "jr.", "sr."}
BOUNDARY = re.compile(r'''[.!?]["”’')\]]*\s+|\n[ \t]*\n''')


def sentence_spans(text):
    """Return nonempty character spans; keep abbreviations and decimals intact."""
    spans, start = [], 0
    for match in BOUNDARY.finditer(text):
        end = match.end()
        preceding = text[start:end].rstrip().rstrip('"”\'’)]')
        last = preceding.split()[-1].lower() if preceding.split() else ""
        paragraph = "\n\n" in match.group() or bool(re.fullmatch(r"\n[ \t]*\n", match.group()))
        if not paragraph and (last in ABBREVIATIONS or re.search(r"(?:[a-z]\.){2,}$", last)
                              or re.fullmatch(r"\d+\.", preceding.strip())):
            continue
        left, right = start, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if right > left:
            spans.append((left, right))
        start = end
    left, right = start, len(text)
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    if right > left:
        spans.append((left, right))
    return spans


def make_windows(record_ids, records, sentences=3, overlap=1):
    if sentences < 1 or overlap < 0 or overlap >= sentences:
        raise ValueError("Require 0 <= overlap < sentences")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Duplicate parent records")
    windows = []
    for parent_rank, rid in enumerate(record_ids, 1):
        text = records[rid]
        spans = sentence_spans(text)
        start = 0
        while start < len(spans):
            stop = min(start + sentences, len(spans))
            left, right = spans[start][0], spans[stop - 1][1]
            windows.append({"window_id": f"{rid}:{left}:{right}", "record_id": rid,
                            "parent_rank": parent_rank, "sentence_start": start, "sentence_stop": stop,
                            "start": left, "end": right, "text": text[left:right]})
            if stop == len(spans):
                break
            start += sentences - overlap
    return windows


def merge_ranges(ranges):
    merged = []
    for left, right in sorted(ranges):
        if left < 0 or right <= left:
            raise ValueError("Invalid character range")
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return merged


def materialise(question, record_ids, records, selected, count_tokens):
    grouped = {rid: [] for rid in record_ids}
    for w in selected:
        rid = w["record_id"]
        if rid not in grouped or records[rid][w["start"]:w["end"]] != w["text"]:
            raise ValueError("Window differs from original text")
        grouped[rid].append((w["start"], w["end"]))
    texts, documents = [], []
    for parent_rank, rid in enumerate(record_ids, 1):
        ranges = merge_ranges(grouped[rid])
        if not ranges:
            continue
        # Explicit gap markers prevent pretending noncontiguous spans are adjacent.
        text = "\n\n[...]\n\n".join(records[rid][a:b] for a, b in ranges)
        texts.append(text)
        included = sum(b - a for a, b in ranges)
        documents.append({"record_id": rid, "parent_rank": parent_rank, "original_characters": len(records[rid]),
                          "included_characters": included, "character_ranges": ranges,
                          "truncated": included != len(records[rid])})
    prompt = render_prompt(question, texts)
    return {"prompt": prompt, "retrieved_record_ids": [d["record_id"] for d in documents],
            "selected_documents": documents, "local_input_tokens": count_tokens(prompt)}


def pack_windows(question, record_ids, records, windows, scores, count_tokens, budget):
    if budget < 1 or len(windows) != len(scores) or not all(math.isfinite(s) for s in scores):
        raise ValueError("Invalid budget or passage scores")
    if len({w["window_id"] for w in windows}) != len(windows):
        raise ValueError("Duplicate windows")
    empty = materialise(question, record_ids, records, [], count_tokens)
    if empty["local_input_tokens"] > budget:
        raise ValueError("Fixed prompt does not fit budget")
    selected, accepted = [], []
    built = empty
    for i in sorted(range(len(windows)), key=lambda i: (-scores[i], i)):
        trial = materialise(question, record_ids, records, selected + [windows[i]], count_tokens)
        if trial["local_input_tokens"] <= budget:
            selected.append(windows[i])
            accepted.append(windows[i]["window_id"])
            built = trial
    if not built["retrieved_record_ids"]:
        raise ValueError("No complete sentence window fits the budget; inspect rather than truncate silently")
    built.update({"input_budget": budget, "accepted_window_ids": accepted,
                  "window_candidates": len(windows), "selected_windows": len(selected),
                  "packing": "descending score, skip windows that do not fit, merge overlaps, original parent/span order"})
    return built
