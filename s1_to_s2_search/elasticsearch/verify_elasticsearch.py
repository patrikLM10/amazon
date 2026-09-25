#!/usr/bin/env python3
"""Recall@K + latency evaluation for Elasticsearch S1 -> S2 retrieval.

Same ground-truth file and rules as verify_recall.py:
  - ground truth columns: source1_entity_id, matched_entity_ids
  - only S2-* IDs count (S3-only / singleton rows skipped, never failures)
  - HIT if any GT S2 ID appears in retrieved Top-K; rank = first match

Usage:
    python verify_elasticsearch.py --source train --topk 50 --limit 20
    python verify_elasticsearch.py --source train --topk 50 --limit 100 --mode ngram
"""

import argparse
import csv
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))  # .../elasticsearch
_PKG_DIR = os.path.dirname(_HERE)  # .../s1_to_s2_search
_ROOT_DIR = os.path.dirname(_PKG_DIR)  # .../student_resource
for _p in (_HERE, _PKG_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from s1_to_s2_search.elasticsearch.elasticsearch_search import search_s1_id
    from s1_to_s2_search.elasticsearch.es_config import INDEX_TEST, INDEX_TRAIN
except ImportError:
    from elasticsearch_search import search_s1_id
    from es_config import INDEX_TEST, INDEX_TRAIN

K_LIST = (1, 5, 10, 20, 50)


def parse_gt_s2_ids(raw):
    if raw is None:
        raise ValueError("missing matched_entity_ids cell")
    raw = str(raw).strip()
    if not raw:
        return [], 0
    parts = [p.strip() for p in raw.split(",")]
    if any(p == "" for p in parts):
        raise ValueError(f"malformed id list (empty token): {raw!r}")
    s2 = [p for p in parts if p.startswith("S2-")]
    return s2, len(parts) - len(s2)


def load_gt_rows(gt_path, limit=None):
    import pandas as pd
    n_read = 0
    for chunk in pd.read_csv(gt_path, sep="\t", encoding="utf-8",
                             chunksize=50_000, dtype=str, keep_default_na=False):
        if "source1_entity_id" not in chunk.columns or "matched_entity_ids" not in chunk.columns:
            raise ValueError(f"Unexpected GT header {list(chunk.columns)}")
        for _, row in chunk.iterrows():
            yield str(row["source1_entity_id"]).strip(), row["matched_entity_ids"]
            n_read += 1
            if limit is not None and n_read >= limit:
                return


def main():
    ap = argparse.ArgumentParser(description="Recall@K + latency eval for ES retrieval")
    ap.add_argument("--ground_truth", default=None)
    ap.add_argument("--source", default="train", choices=["train", "test"])
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate first N ground-truth rows (same S1 population "
                         "order as verify_recall.py for a fair comparison)")
    ap.add_argument("--output", default=None)
    ap.add_argument("--index", default=None)
    ap.add_argument("--mode", default="lexical", choices=["lexical", "ngram"])
    args = ap.parse_args()

    default_gt = os.path.join(_ROOT_DIR, "dataset", args.source,
                              f"{args.source}_ground_truth.tsv")
    gt_path = args.ground_truth or default_gt
    if not os.path.isfile(gt_path):
        print(f"Ground-truth file not found: {gt_path}", file=sys.stderr)
        return 2
    index = args.index or (INDEX_TRAIN if args.source == "train" else INDEX_TEST)
    out_path = args.output or os.path.join(_HERE, "verify_elasticsearch_results.tsv")

    topk = args.topk
    ks = [k for k in K_LIST if k <= topk]
    rows_out, lat = [], []
    n_hits = n_eval = n_errors = 0
    n_skipped_no_s2 = n_skipped_malformed = 0
    hits_at = {k: 0 for k in ks}
    t0 = time.perf_counter()

    for idx, (s1_id, raw_matched) in enumerate(load_gt_rows(gt_path, args.limit), 1):
        if not s1_id:
            n_skipped_malformed += 1
            continue
        try:
            gt_s2_ids, _ = parse_gt_s2_ids(raw_matched)
        except ValueError as exc:
            n_skipped_malformed += 1
            print(f"[{idx}] SKIP {s1_id}: malformed GT ({exc})")
            continue
        if not gt_s2_ids:
            n_skipped_no_s2 += 1
            continue
        try:
            query, df, ms = search_s1_id(s1_id, args.source, topk, index, args.mode)
            retrieved = ([str(x) for x in df["entity_id"].tolist()]
                         if df is not None and len(df) and "entity_id" in df.columns else [])
            status = "ok"
        except Exception as exc:
            retrieved, query, ms, status = [], {}, 0.0, f"retrieval_error: {exc}"
            n_errors += 1
            print(f"[{idx}] ERROR {s1_id}: {status}")
        if status == "ok":
            lat.append(ms)
            n_eval += 1
            gt_set = set(gt_s2_ids)
            rank = next((i for i, r in enumerate(retrieved, 1) if r in gt_set), "")
            hit = 1 if rank != "" else 0
            n_hits += hit
            for k in ks:
                if rank != "" and rank <= k:
                    hits_at[k] += 1
            print(f"[{idx}] {'HIT ' if hit else 'MISS'} {s1_id} rank={rank or '-'} "
                  f"({ms:.0f} ms)")
        else:
            rank, hit = "", 0
        rows_out.append({
            "s1_id": s1_id, "country": (query or {}).get("country", ""),
            "ground_truth_s2_ids": ",".join(gt_s2_ids), "hit": hit,
            "match_rank": rank, "num_retrieved": len(retrieved),
            "retrieved_s2_ids": ",".join(retrieved),
            "query_business_name": (query or {}).get("business_name", ""),
            "query_business_address": (query or {}).get("business_address", ""),
            "latency_ms": f"{ms:.1f}", "status": status,
        })

    cols = ["s1_id", "country", "ground_truth_s2_ids", "hit", "match_rank",
            "num_retrieved", "retrieved_s2_ids", "query_business_name",
            "query_business_address", "latency_ms", "status"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nPer-query results -> {out_path}")

    print("\n========================================")
    print("ELASTICSEARCH RETRIEVAL EVALUATION")
    print("========================================\n")
    print(f"Index                       : {index} (mode={args.mode})")
    print(f"Queries evaluated           : {n_eval}")
    print(f"Top-K                       : {topk}\n")
    for k in ks:
        print(f"Recall@{k:<4} : {100.0 * hits_at[k] / n_eval:6.2f}%" if n_eval else f"Recall@{k}: n/a")
    if n_eval:
        print(f"\nRecall@{topk} = {100.0 * n_hits / n_eval:.2f}%")
    if lat:
        lat_sorted = sorted(lat)
        p95 = lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))]
        print("\nLatency (per-query ES search):")
        print(f"mean              : {statistics.mean(lat):.1f} ms")
        print(f"median            : {statistics.median(lat):.1f} ms")
        print(f"p95               : {p95:.1f} ms")
        print(f"queries/sec       : {n_eval / (time.perf_counter() - t0):.1f}")
    print(f"\nSuccessful queries        : {n_eval}")
    print(f"Retrieval errors          : {n_errors}")
    print(f"Skipped (no S2 GT)        : {n_skipped_no_s2}")
    print(f"Skipped (malformed GT)    : {n_skipped_malformed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
