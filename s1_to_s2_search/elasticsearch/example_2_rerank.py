#!/usr/bin/env python3
"""BOILERPLATE 2 — Retrieve-then-rerank. Copy this file anywhere.

Pipeline:  ES Top-K (fast) -> cross-encoder rescoring (accurate) -> final Top-N.

    pip install sentence-transformers   # once
    python example_2_rerank.py --s1_id S1-965667 --topk 20 --final-k 5

Swap in any cross-encoder: pass ``model=...`` (anything with ``predict(pairs)``)
to ``retrieve_rerank()``. For many queries, load the model ONCE and reuse it.
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))  # .../elasticsearch
_PKG_DIR = os.path.dirname(_HERE)  # .../s1_to_s2_search
_ROOT_DIR = os.path.dirname(_PKG_DIR)  # .../student_resource
for _p in (_HERE, _PKG_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from s1_to_s2_search.elasticsearch import search_s1_id

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker(model_name=DEFAULT_MODEL):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise RuntimeError("pip install sentence-transformers to run the reranker")
    return CrossEncoder(model_name)


def retrieve_rerank(s1_id, model, source="train", topk=50, final_k=5,
                    corpus="s2", mode="lexical", index=None):
    """One call: retrieve candidates, rescore, return final ranking.

    Returns (query_dict, final_df with ce_score, timings_dict).
    """
    query, cands, es_ms = search_s1_id(s1_id, source, topk, index, mode, corpus)
    if len(cands) == 0:
        return query, cands, {"es_ms": es_ms, "ce_ms": 0.0}
    q = f"{query['business_name']} [SEP] {query['business_address']}"
    pairs = [[q, f"{r.business_name} [SEP] {r.business_address}"]
             for _, r in cands.iterrows()]
    t0 = time.perf_counter()
    scores = list(model.predict(pairs))
    ce_ms = (time.perf_counter() - t0) * 1000.0
    out = cands.copy()
    out["ce_score"] = scores
    return query, out.sort_values("ce_score", ascending=False
                                  ).head(final_k).reset_index(drop=True), \
        {"es_ms": es_ms, "ce_ms": ce_ms}


def main():
    ap = argparse.ArgumentParser(description="ES retrieve -> cross-encoder rerank")
    ap.add_argument("--s1_id", required=True)
    ap.add_argument("--source", default="train")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--final-k", type=int, default=5)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--corpus", default="s2", choices=["s2", "s3"])
    args = ap.parse_args()

    model = load_reranker(args.model)
    query, final, t = retrieve_rerank(args.s1_id, model, args.source,
                                      args.topk, args.final_k, args.corpus)
    print(f"QUERY {query['entity_id']} (ES {t['es_ms']:.0f} ms, CE {t['ce_ms']:.0f} ms)")
    print(final[["entity_id", "ce_score", "_score"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
