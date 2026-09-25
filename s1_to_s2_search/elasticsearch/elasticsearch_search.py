#!/usr/bin/env python3
"""Elasticsearch S1 -> S2 retrieval API + single-query CLI.

Indexed retrieval only: one ES query per call, no S2 TSV scanning.

Usage as a library:
    from elasticsearch_search import search_elasticsearch, search_s1_id
    results, latency_ms = search_elasticsearch(name, address, country, topk=50)

Usage as CLI:
    python elasticsearch_search.py --s1_id S1-965667 --source train --topk 50
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

import pandas as pd

try:
    from s1_to_s2_search.elasticsearch.es_config import (
        INDEX_TEST, INDEX_TRAIN, build_query, default_index, get_client,
    )
except ImportError:
    from es_config import INDEX_TEST, INDEX_TRAIN, build_query, default_index, get_client

_es = None


def _client():
    global _es
    if _es is None:
        _es = get_client()
    return _es


def search_elasticsearch(business_name, business_address, country,
                         topk=50, index=None, mode="lexical"):
    """One ES query -> (results DataFrame, latency_ms). No TSV access.

    DataFrame columns: entity_id, business_name, business_address, country,
    _score (ES ranking order preserved).
    """
    index = index or INDEX_TRAIN
    body = build_query(business_name, business_address, country, topk, mode)
    t0 = time.perf_counter()
    resp = _client().search(index=index, body=body)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    rows = [{
        "entity_id": h["_source"].get("entity_id", h["_id"]),
        "business_name": h["_source"].get("business_name", ""),
        "business_address": h["_source"].get("business_address", ""),
        "country": h["_source"].get("country", ""),
        "_score": h.get("_score", 0.0),
    } for h in resp["hits"]["hits"]]
    return pd.DataFrame(rows, columns=["entity_id", "business_name",
                                       "business_address", "country", "_score"]), latency_ms


def search_s1_id(s1_id, source="train", topk=50, index=None, mode="lexical",
                 corpus="s2"):
    """Look up the S1 record, then run the ES query. Returns (query, df, ms).

    ``corpus`` selects the default target index (``s2``/``s3``); an explicit
    ``index`` always wins.
    """
    try:
        from s1_to_s2_search.search_core import find_s1, resolve_paths
    except ImportError:
        from search_core import find_s1, resolve_paths
    s1_p, _ = resolve_paths(source, None, None, None)
    query = find_s1(s1_p, s1_id)
    if query is None:
        raise ValueError(f"{s1_id} not found in {s1_p} (wrong --source?)")
    index = index or default_index(source, corpus)
    df, ms = search_elasticsearch(query["business_name"], query["business_address"],
                                  query["country"], topk, index, mode)
    return query, df, ms


def main():
    ap = argparse.ArgumentParser(description="Single-query ES retrieval")
    ap.add_argument("--s1_id", required=True)
    ap.add_argument("--source", default="train", choices=["train", "test"])
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--index", default=None)
    ap.add_argument("--mode", default="lexical", choices=["lexical", "ngram"])
    ap.add_argument("--corpus", default="s2", choices=["s2", "s3"])
    args = ap.parse_args()

    query, df, ms = search_s1_id(args.s1_id, args.source, args.topk,
                                 args.index, args.mode, args.corpus)
    print(f"QUERY {query['entity_id']}: {query['business_name']} | "
          f"{query['business_address']} [{query['country']}]")
    print(f"mode={args.mode} latency={ms:.1f} ms, {len(df)} hits\n")
    for i, r in df.iterrows():
        print(f"{i+1:2d}. {r['entity_id']} | score={r['_score']:.2f} | [{r['country']}]")
        print(f"     name: {r['business_name']}")
        print(f"     addr: {r['business_address']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
