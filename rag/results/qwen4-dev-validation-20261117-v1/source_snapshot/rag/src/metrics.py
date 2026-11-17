from __future__ import annotations

import math


def recall_at_k(ranked_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked_ids[:k])) / len(relevant)


def mrr_at_k(ranked_ids, relevant_ids, k=10):
    relevant = set(relevant_ids)
    for rank, rid in enumerate(ranked_ids[:k], start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids, relevance, k=10):
    def gain(rel, rank):
        return (2.0 ** rel - 1.0) / math.log2(rank + 1.0)

    dcg = sum(
        gain(float(relevance.get(rid, 0.0)), rank)
        for rank, rid in enumerate(ranked_ids[:k], start=1)
        if float(relevance.get(rid, 0.0)) > 0
    )

    ideal_rels = sorted(
        [float(v) for v in relevance.values() if float(v) > 0],
        reverse=True,
    )[:k]
    idcg = sum(gain(rel, rank) for rank, rel in enumerate(ideal_rels, start=1))

    return dcg / idcg if idcg > 0 else 0.0
