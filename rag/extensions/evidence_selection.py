"""Evidence selection independent of qrels and of model/API implementations."""
from __future__ import annotations


def chat_input_tokens(tokenizer, system, prompt):
    """Handle Transformers versions returning IDs or a BatchEncoding mapping."""
    from collections.abc import Mapping
    encoded = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("Expected a single chat input")
        ids = ids[0]
    if not isinstance(ids, list) or not all(type(value) is int for value in ids):
        raise ValueError("Tokenizer must return a single list of integer IDs")
    return len(ids)


def render_prompt(question, texts):
    evidence = "\n\n".join(f"[Document {i}]\n{text}" for i, text in enumerate(texts, 1))
    return ("Answer the question using the retrieved evidence below. "
            "If the evidence is insufficient, say so.\n\n"
            f"Question:\n{question}\n\nEvidence:\n{evidence}")


def select_context(question, ranked_ids, records, count_tokens, *, top_k=None, budget=None):
    """Rank-prefix packing; truncate only the last admitted document, then stop.

    Budget covers the full serialized chat input as measured by count_tokens.
    Truncation uses an original-text character prefix found by bisection, with
    a final hard budget check. BPE counts need not be monotonic: this is a
    feasible-prefix rule, not a claim of maximal packing or semantic compression.
    """
    if top_k is not None and (type(top_k) is not int or top_k < 1):
        raise ValueError("top_k must be a positive integer")
    if budget is not None and (type(budget) is not int or budget < 1):
        raise ValueError("budget must be a positive integer")
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("Duplicate ranked IDs")
    if budget is not None and count_tokens(render_prompt(question, [])) > budget:
        raise ValueError("Budget cannot fit question and fixed prompt")
    texts, selected = [], []
    for rid in ranked_ids[:top_k] if top_k else ranked_ids:
        text = records[rid]
        if budget is None or count_tokens(render_prompt(question, texts + [text])) <= budget:
            texts.append(text)
            selected.append({"record_id": rid, "original_characters": len(text),
                             "included_characters": len(text), "truncated": False})
            continue
        low, high, best = 1, len(text) - 1, 0
        while low <= high:
            middle = (low + high) // 2
            if count_tokens(render_prompt(question, texts + [text[:middle]])) <= budget:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best:
            texts.append(text[:best])
            selected.append({"record_id": rid, "original_characters": len(text),
                             "included_characters": best, "truncated": True})
        break
    prompt = render_prompt(question, texts)
    tokens = count_tokens(prompt)
    if budget is not None and tokens > budget:
        raise AssertionError("Context exceeds hard input budget")
    return {"prompt": prompt, "selected_documents": selected,
            "retrieved_record_ids": [row["record_id"] for row in selected],
            "local_input_tokens": tokens, "input_budget": budget}
