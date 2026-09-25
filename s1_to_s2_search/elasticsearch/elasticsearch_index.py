#!/usr/bin/env python3
"""Bulk-index an S2/S3 TSV corpus into Elasticsearch (one-time setup per corpus).

Usage:
    python elasticsearch_index.py --source train                  # S2 (default)
    python elasticsearch_index.py --source train --corpus s3      # S3 corpus
    python elasticsearch_index.py --source train --index s2_train --batch 2000
    python elasticsearch_index.py --source train --limit 100000   # DEBUG partial

Streams the TSV (never full-loads it) and uses Elasticsearch bulk indexing.
S2 and S3 files share the same schema, so either can be indexed.
"""

import argparse
import csv
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))  # .../elasticsearch
_PKG_DIR = os.path.dirname(_HERE)  # .../s1_to_s2_search
_ROOT_DIR = os.path.dirname(_PKG_DIR)  # .../student_resource
for _p in (_HERE, _PKG_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from s1_to_s2_search.elasticsearch.es_config import (
        ES_HOST, INDEX_TEST, INDEX_TRAIN, INDEX_SETTINGS, get_client,
        default_index, streaming_bulk_iter as streaming_bulk,
    )
except ImportError:
    from es_config import (
        ES_HOST, INDEX_TEST, INDEX_TRAIN, INDEX_SETTINGS, get_client,
        default_index, streaming_bulk_iter as streaming_bulk,
    )


def corpus_path(source, corpus="s2"):
    prefix = "train" if source == "train" else "test"
    return os.path.join(_ROOT_DIR, "dataset", source, f"{prefix}_source{corpus[1]}.tsv")


def doc_stream(tsv_path, limit=None):
    """Yield ES bulk actions, streaming the TSV."""
    with open(tsv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            eid = (row.get("entity_id") or "").strip()
            if not eid:
                continue
            yield {
                "_id": eid,
                "entity_id": eid,
                "business_name": row.get("business_name") or "",
                "business_address": row.get("business_address") or "",
                "country": (row.get("country") or "").strip(),
            }


def main():
    ap = argparse.ArgumentParser(description="Bulk-index S2 corpus into Elasticsearch")
    ap.add_argument("--source", default="train", choices=["train", "test"])
    ap.add_argument("--corpus", default="s2", choices=["s2", "s3"],
                    help="Which source corpus to index (default: s2)")
    ap.add_argument("--index", default=None)
    ap.add_argument("--batch", type=int, default=2000, help="Bulk batch size")
    ap.add_argument("--limit", type=int, default=None, help="DEBUG: index first N rows only")
    ap.add_argument("--recreate", action="store_true", help="Delete index if it exists")
    args = ap.parse_args()

    index = args.index or default_index(args.source, args.corpus)
    path = corpus_path(args.source, args.corpus)
    if not os.path.isfile(path):
        print(f"{args.corpus.upper()} file not found: {path}", file=sys.stderr)
        return 2

    es = get_client()
    info = es.info()
    ver = info.body["version"]["number"] if hasattr(info, "body") else info["version"]["number"]
    print(f"Elasticsearch {ver} at {ES_HOST}")

    if es.indices.exists(index=index):
        if args.recreate:
            es.indices.delete(index=index)
            print(f"Deleted existing index {index}")
        else:
            print(f"Index {index} exists; use --recreate to rebuild.", file=sys.stderr)
            return 2
    es.indices.create(index=index, body=INDEX_SETTINGS)
    print(f"Created index {index}")

    t0 = time.perf_counter()
    n_ok, n_fail = 0, 0
    for ok, _ in streaming_bulk(es, doc_stream(path, args.limit),
                                index=index, chunk_size=args.batch,
                                max_retries=3, raise_on_error=False):
        if ok:
            n_ok += 1
        else:
            n_fail += 1
        if (n_ok + n_fail) % 200_000 == 0:
            dt = time.perf_counter() - t0
            print(f"  ... {n_ok + n_fail:,} docs ({(n_ok + n_fail) / dt:,.0f} docs/s)",
                  flush=True)

    es.indices.refresh(index=index)
    dt = time.perf_counter() - t0
    count = es.count(index=index)["count"]
    print("\n========================================")
    print("INDEXING BENCHMARK")
    print("========================================")
    print(f"Index name     : {index}")
    print(f"{args.corpus.upper()} documents  : {count:,}")
    print(f"Indexing time  : {dt:.1f} sec")
    print(f"Throughput     : {count / dt:,.0f} docs/sec")
    print(f"Bulk failures  : {n_fail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
