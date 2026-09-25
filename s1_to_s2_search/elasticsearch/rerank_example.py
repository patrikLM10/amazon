#!/usr/bin/env python3
"""Retrieve-then-rerank pattern: ES Top-50 -> cross-encoder -> final Top-K.

Stage 1 (fast, indexed):  S1 -> search_s1_id() -> 50 S2 candidates
Stage 2 (accurate, slow): cross-encoder scores each (query, candidate) pair

Usage:
    pip install sentence-transformers   # once
    python rerank_example.py --s1_id S1-965667 --source train --topk 50 --final-k 5

    # S1 -> S3 variant (index the S3 corpus first):
    #   python elasticsearch_index.py --source train --corpus s3 --recreate
    python rerank_example.py --s1_id S1-965667 --corpus s3 --topk 50 --final-k 5
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_HERE)
_ROOT_DIR = os.path.dirname(_PKG_DIR)
for _p in (_HERE, _PKG_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _pair_text(name, address):
    return f"{name or ''} [SEP] {address or ''}"


def rerank(s1_id, source="train", topk=50, final_k=5, model_name=None,
           corpus="s2", mode="lexical", index=None, model=None):
    """Full retrieve->rerank pipeline. Returns (query, final_df).

    Pass your own cross-encoder ``model`` (must expose ``predict(pairs)``)
    to reuse a loaded instance across queries; otherwise one is loaded from
    ``model_name``.
    """
    try:
        from s1_to_s2_search.elasticsearch.elasticsearch_search import search_s1_id
    except ImportError:
        from elasticsearch_search import search_s1_id

    if model is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise RuntimeError("pip install sentence-transformers to run the reranker")
        model = CrossEncoder(
            model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2")

    query, cands, es_ms = search_s1_id(s1_id, source, topk, index, mode, corpus)
    if len(cands) == 0:
        return query, cands, {"es_ms": es_ms, "ce_ms": 0.0}

    import time
    q = _pair_text(query["business_name"], query["business_address"])
    pairs = [[q, _pair_text(r["business_name"], r["business_address"])]
             for _, r in cands.iterrows()]
    t0 = time.perf_counter()
    scores = list(model.predict(pairs))
    ce_ms = (time.perf_counter() - t0) * 1000.0

    out = cands.copy()
    out["ce_score"] = scores
    out = out.sort_values("ce_score", ascending=False
                          ).head(final_k).reset_index(drop=True)
    return query, out, {"es_ms": es_ms, "ce_ms": ce_ms}


def main():
    ap = argparse.ArgumentParser(description="ES retrieve -> cross-encoder rerank")
    ap.add_argument("--s1_id", required=True)
    ap.add_argument("--source", default="train", choices=["train", "test"])
    ap.add_argument("--topk", type=int, default=50, help="ES retrieval depth")
    ap.add_argument("--final-k", type=int, default=5, help="Candidates after rerank")
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--corpus", default="s2", choices=["s2", "s3"])
    ap.add_argument("--mode", default="lexical", choices=["lexical", "ngram"])
    args = ap.parse_args()

    query, final, t = rerank(args.s1_id, args.source, args.topk, args.final_k,
                             args.model, args.corpus, args.mode)
    print(f"QUERY {query['entity_id']}: {query['business_name']} | "
          f"{query['business_address']} [{query['country']}]")
    print(f"ES {t['es_ms']:.0f} ms -> rerank {t['ce_ms']:.0f} ms\n")
    for i, r in final.iterrows():
        print(f"{i+1:2d}. {r['entity_id']} | ce={r['ce_score']:.3f} "
              f"| es={r['_score']:.2f} | [{r['country']}]")
        print(f"     name: {r['business_name']}")
        print(f"     addr: {r['business_address']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
