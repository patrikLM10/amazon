#!/usr/bin/env python3
"""Check which Top-K retrieved hits match ground truth — for one bucket only.

Given S1 id(s) and a bucket (s2/s3): retrieve Top-K from that bucket's ES
index, keep only ground-truth IDs with the same bucket prefix, and report
exactly which retrieved IDs match, at which ranks.

Usage:
    python check_match.py --s1_id S1-965667 --corpus s3 --topk 50
    python check_match.py --s1_ids S1-965667,S1-55344266 --corpus s2 --topk 50
    python check_match.py --s1_id S1-965667 --corpus s2 --source train
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))  # .../elasticsearch
_PKG_DIR = os.path.dirname(_HERE)  # .../s1_to_s2_search
_ROOT_DIR = os.path.dirname(_PKG_DIR)  # .../student_resource
for _p in (_HERE, _PKG_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from s1_to_s2_search.elasticsearch.elasticsearch_search import search_s1_id
except ImportError:
    from elasticsearch_search import search_s1_id


def parse_gt_ids(raw, corpus):
    """GT cell -> ids with this bucket's prefix only (S2- / S3-)."""
    prefix = corpus.upper() + "-"
    if raw is None:
        raise ValueError("missing matched_entity_ids cell")
    raw = str(raw).strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    if any(p == "" for p in parts):
        raise ValueError(f"malformed id list: {raw!r}")
    return [p for p in parts if p.startswith(prefix)]


def load_gt_subset(gt_path, wanted):
    """Stream GT until all wanted S1 ids are found -> {s1_id: raw cell}.

    Never builds the full 2.2M-row map; vectorised ``isin`` per chunk with
    early exit once every requested id is found.
    """
    import pandas as pd
    wanted = set(wanted)
    found = {}
    n_scanned = 0
    for chunk in pd.read_csv(gt_path, sep="\t", encoding="utf-8",
                             chunksize=100_000, dtype=str, keep_default_na=False):
        n_scanned += len(chunk)
        hit = chunk[chunk["source1_entity_id"].isin(wanted - found.keys())]
        for _, row in hit.iterrows():
            found[str(row["source1_entity_id"]).strip()] = row["matched_entity_ids"]
        if len(found) == len(wanted):
            break
    return found, n_scanned


def check_one(s1_id, gt_raw, corpus, source, topk, index):
    """Retrieve + match for one S1. Returns result dict (never raises)."""
    try:
        gt_ids = parse_gt_ids(gt_raw, corpus)
    except ValueError as exc:
        return {"s1_id": s1_id, "status": f"malformed GT: {exc}", "hit": 0}
    if gt_raw is None or (isinstance(gt_raw, float)):
        return {"s1_id": s1_id, "status": "S1 id not in ground truth", "hit": 0}
    if not gt_ids:
        return {"s1_id": s1_id, "status": f"no {corpus.upper()} ground truth "
                "(singleton or other-bucket-only)", "hit": 0, "gt_ids": []}
    try:
        query, df, ms = search_s1_id(s1_id, source, topk, index, "lexical", corpus)
    except Exception as exc:
        return {"s1_id": s1_id, "status": f"retrieval error: {exc}",
                "hit": 0, "gt_ids": gt_ids}
    retrieved = ([str(x) for x in df["entity_id"].tolist()]
                 if df is not None and len(df) and "entity_id" in df.columns else [])
    gt_set = set(gt_ids)
    matches = [(i, r) for i, r in enumerate(retrieved, 1) if r in gt_set]
    return {"s1_id": s1_id, "status": "ok", "query": query,
            "gt_ids": gt_ids, "retrieved": retrieved,
            "matches": matches, "hit": 1 if matches else 0, "ms": ms}


def main():
    ap = argparse.ArgumentParser(description="Bucket-aware Top-K match checker")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--s1_id", help="Single S1 id")
    g.add_argument("--s1_ids", help="Comma-separated S1 ids")
    ap.add_argument("--corpus", default="s3", choices=["s2", "s3"],
                    help="Bucket to search AND match against (default: s3)")
    ap.add_argument("--source", default="train", choices=["train", "test"])
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--index", default=None)
    ap.add_argument("--ground_truth", default=None)
    args = ap.parse_args()

    default_gt = os.path.join(_ROOT_DIR, "dataset", args.source,
                              f"{args.source}_ground_truth.tsv")
    gt_path = args.ground_truth or default_gt
    if not os.path.isfile(gt_path):
        print(f"Ground-truth file not found: {gt_path}", file=sys.stderr)
        return 2

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ids = [args.s1_id] if args.s1_id else [s.strip() for s in args.s1_ids.split(",") if s.strip()]
    print(f"Loading ground truth for {len(ids)} id(s): {gt_path} ...", flush=True)
    gt_map, n_scanned = load_gt_subset(gt_path, ids)
    print(f"GT rows scanned: {n_scanned:,} | found {len(gt_map)}/{len(ids)} | "
          f"corpus={args.corpus} | topk={args.topk}\n", flush=True)

    n_hit = n_eval = 0
    for sid in ids:
        raw = gt_map.get(sid)
        if raw is None:
            print(f"{sid}: SKIP — id not in ground truth\n")
            continue
        r = check_one(sid, raw, args.corpus, args.source, args.topk, args.index)
        if r["status"] != "ok":
            print(f"{sid}: SKIP — {r['status']}\n")
            continue
        n_eval += 1
        n_hit += r["hit"]
        q = r["query"]
        print(f"{sid} [{q['country']}] {q['business_name']} | {q['business_address']}")
        print(f"  GT {args.corpus.upper()} ids ({len(r['gt_ids'])}): "
              f"{', '.join(r['gt_ids'])}")
        if r["matches"]:
            print(f"  MATCHES ({len(r['matches'])}/{len(r['retrieved'])} retrieved):")
            for rank, eid in r["matches"]:
                print(f"    rank {rank}: {eid}")
            missed = [g for g in r["gt_ids"] if g not in set(r["retrieved"])]
            if missed:
                print(f"  GT missed: {', '.join(missed)}")
            print("  => HIT\n")
        else:
            print(f"  MATCHES: none of {len(r['retrieved'])} retrieved "
                  f"({r['ms']:.0f} ms)")
            print("  => MISS\n")

    if len(ids) > 1:
        print(f"Summary: {n_hit}/{n_eval} HIT "
              f"({100.0 * n_hit / n_eval:.1f}% Recall@{args.topk})" if n_eval
              else "Summary: no evaluable queries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
