#!/usr/bin/env python3
"""Recall@K evaluation for the existing S1 -> S2 BM25 retrieval pipeline.

Measures whether each S1 query's known ground-truth S2 entity/entities
appear in the Top-K candidates returned by the existing ``search_top_k()``
API. Does NOT reimplement blocking / tokenization / BM25 / ranking.

Ground-truth schema (discovered in dataset/train/train_ground_truth.tsv):
    source1_entity_id \\t matched_entity_ids
    e.g. S1-965667 \\t S2-681193310,S2-743505751,S3-775321672,...

Only S2- IDs are used as ground truth here because the retrieval pipeline
searches the S2 bucket only. S1 rows with no S2 ground truth (singletons
with an empty list, or S3-only rows) are skipped and counted separately.

Usage:
    python verify_recall.py --ground_truth <path> --source train --topk 50
    python verify_recall.py --source train --topk 50 --limit 20 --output out.tsv

    # from the repo root (student_resource/):
    python s1_to_s2_search/verify_recall.py --source train --limit 5
"""

import argparse
import csv
import os
import sys
import time

# Make the retrieval package importable whether this script is run as
#   python verify_recall.py            (cwd = s1_to_s2_search/)
#   python s1_to_s2_search/verify_recall.py   (cwd = student_resource/)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)  # student_resource/
for _p in (_HERE, _PKG_PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from s1_to_s2_search import search_top_k
except ImportError:  # fallback when cwd is the package dir itself
    from api import search_top_k

K_LIST = (1, 5, 10, 20, 50)


def parse_gt_s2_ids(raw: str):
    """Split a matched_entity_ids cell, keep S2- IDs only.

    Returns (s2_ids, n_s3_ignored). Malformed cells raise ValueError.
    """
    if raw is None:
        raise ValueError("missing matched_entity_ids cell")
    raw = str(raw).strip()
    if not raw:
        return [], 0
    parts = [p.strip() for p in raw.split(",")]
    # A truly empty token inside a non-empty cell is malformed (e.g. "S2-1,,S2-2")
    if any(p == "" for p in parts):
        raise ValueError(f"malformed id list (empty token): {raw!r}")
    s2 = [p for p in parts if p.startswith("S2-")]
    n_s3 = len(parts) - len(s2)
    return s2, n_s3


def evaluate_one(s1_id, gt_s2_ids, source, topk, s2_limit, all_countries):
    """Run retrieval and score one query. Returns a result dict.

    Never raises for retrieval problems: they are captured in the dict
    with status != 'ok'.
    """
    try:
        query, results = search_top_k(
            s1_id, source=source, topk=topk,
            s2_limit=s2_limit, all_countries=all_countries,
            verbose=False, save=False,
        )
    except Exception as exc:  # e.g. S1 id not found in source files
        return {
            "s1_id": s1_id,
            "ground_truth_s2_ids": ",".join(gt_s2_ids),
            "hit": 0, "match_rank": "",
            "num_retrieved": 0, "retrieved_s2_ids": "",
            "query_business_name": "", "query_business_address": "",
            "query_country": "", "status": f"retrieval_error: {exc}",
        }

    try:
        if results is None or len(results) == 0 or "entity_id" not in results.columns:
            retrieved = []
        else:
            retrieved = [str(x) for x in results["entity_id"].tolist()]
    except Exception as exc:
        return {
            "s1_id": s1_id,
            "ground_truth_s2_ids": ",".join(gt_s2_ids),
            "hit": 0, "match_rank": "",
            "num_retrieved": 0, "retrieved_s2_ids": "",
            "query_business_name": str(query.get("business_name", "")),
            "query_business_address": str(query.get("business_address", "")),
            "query_country": str(query.get("country", "")),
            "status": f"result_parse_error: {exc}",
        }

    gt_set = set(gt_s2_ids)
    match_rank = ""
    for i, rid in enumerate(retrieved, start=1):
        if rid in gt_set:
            match_rank = i
            break
    return {
        "s1_id": s1_id,
        "ground_truth_s2_ids": ",".join(gt_s2_ids),
        "hit": 1 if match_rank != "" else 0,
        "match_rank": match_rank,
        "num_retrieved": len(retrieved),
        "retrieved_s2_ids": ",".join(retrieved),
        "query_business_name": str(query.get("business_name", "")),
        "query_business_address": str(query.get("business_address", "")),
        "query_country": str(query.get("country", "")),
        "status": "ok",
    }


def load_gt_rows(gt_path, limit=None):
    """Yield (s1_id, raw_matched) for each GT row. Streams; never full-loads S2."""
    import pandas as pd
    n_read = 0
    for chunk in pd.read_csv(gt_path, sep="\t", encoding="utf-8",
                             chunksize=50_000, dtype=str, keep_default_na=False):
        if "source1_entity_id" not in chunk.columns or "matched_entity_ids" not in chunk.columns:
            raise ValueError(
                f"Unexpected ground-truth header {list(chunk.columns)}; "
                "expected ['source1_entity_id', 'matched_entity_ids']")
        for _, row in chunk.iterrows():
            yield str(row["source1_entity_id"]).strip(), row["matched_entity_ids"]
            n_read += 1
            if limit is not None and n_read >= limit:
                return


def main():
    ap = argparse.ArgumentParser(description="Recall@K evaluation for S1->S2 BM25 retrieval")
    ap.add_argument("--ground_truth", default=None,
                    help="Path to ground-truth TSV (default: dataset/<source>/"
                         "<source>_ground_truth.tsv for train; test has no ground truth)")
    ap.add_argument("--source", default="train", choices=["train", "test"])
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N ground-truth rows "
                         "(limits S1 queries, NOT S2 rows searched)")
    ap.add_argument("--output", default=None,
                    help="Per-query TSV path (default: verify_results.tsv next to this script)")
    ap.add_argument("--s2_limit", type=int, default=None,
                    help="ADVANCED/DEBUG ONLY: scan only first N S2 rows. "
                         "Default (None) searches the FULL S2 bucket, which is "
                         "required for a valid recall measurement.")
    ap.add_argument("--all_countries", action="store_true",
                    help="Pass through to search_top_k (search all countries)")
    args = ap.parse_args()

    default_gt = os.path.join(_PKG_PARENT, "dataset", args.source,
                              f"{args.source}_ground_truth.tsv")
    gt_path = args.ground_truth or default_gt
    if not os.path.isfile(gt_path):
        print(f"Ground-truth file not found: {gt_path}", file=sys.stderr)
        if args.source == "test":
            print("Note: the test split ships without ground truth; "
                  "evaluate on train.", file=sys.stderr)
        return 2
    out_path = args.output or os.path.join(_HERE, "verify_results.tsv")
    if args.s2_limit is not None:
        print(f"WARNING: --s2_limit={args.s2_limit} performs a PARTIAL S2 scan; "
              f"recall will be underestimated. Omit it for a valid measurement.")

    topk = args.topk
    ks = [k for k in K_LIST if k <= topk]

    rows_out = []
    n_hits = 0
    n_eval = 0
    n_skipped_no_s2 = 0   # singletons + S3-only (no S2 ground truth)
    n_skipped_malformed = 0
    n_errors = 0
    hits_at = {k: 0 for k in ks}
    t0 = time.time()

    for idx, (s1_id, raw_matched) in enumerate(load_gt_rows(gt_path, args.limit), start=1):
        if not s1_id:
            n_skipped_malformed += 1
            print(f"[{idx}] SKIP: empty source1_entity_id")
            continue
        try:
            gt_s2_ids, _n_s3 = parse_gt_s2_ids(raw_matched)
        except ValueError as exc:
            n_skipped_malformed += 1
            print(f"[{idx}] SKIP {s1_id}: malformed ground truth ({exc})")
            continue
        if not gt_s2_ids:
            n_skipped_no_s2 += 1
            continue  # singleton or S3-only: no S2 target to retrieve

        q_t0 = time.time()
        res = evaluate_one(s1_id, gt_s2_ids, args.source, topk,
                           args.s2_limit, args.all_countries)
        dt = time.time() - q_t0
        rows_out.append(res)
        n_eval += 1
        if res["status"] != "ok":
            n_errors += 1
            print(f"[{idx}] ERROR {s1_id}: {res['status']} ({dt:.1f}s)")
            continue
        if res["hit"]:
            n_hits += 1
            for k in ks:
                if res["match_rank"] != "" and res["match_rank"] <= k:
                    hits_at[k] += 1
        print(f"[{idx}] {'HIT ' if res['hit'] else 'MISS'} {s1_id} "
              f"rank={res['match_rank'] or '-'} "
              f"retrieved={res['num_retrieved']} ({dt:.1f}s)")

    # Per-query TSV (easy failure inspection)
    cols = ["s1_id", "ground_truth_s2_ids", "hit", "match_rank", "num_retrieved",
            "retrieved_s2_ids", "query_business_name", "query_business_address",
            "query_country", "status"]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nPer-query results -> {out_path}")

    # Summary
    print("\n========================================")
    print("BM25 RETRIEVAL EVALUATION")
    print("========================================\n")
    print(f"Ground truth              : {gt_path}")
    print(f"Total queries evaluated   : {n_eval}")
    print(f"Top-K                     : {topk}\n")
    print(f"Hits                      : {n_hits}")
    print(f"Misses                    : {n_eval - n_hits - n_errors}")
    print(f"Retrieval errors          : {n_errors}")
    print(f"Skipped (no S2 GT)        : {n_skipped_no_s2} "
          f"(singletons + S3-only rows)")
    print(f"Skipped (malformed GT)    : {n_skipped_malformed}\n")
    if n_eval:
        for k in ks:
            print(f"Recall@{k:<4} : {100.0 * hits_at[k] / n_eval:6.2f}%")
        print(f"\nRecall@{topk} = {100.0 * n_hits / n_eval:.2f}%")
    else:
        print("No queries evaluated.")
    print(f"\nElapsed: {time.time() - t0:.1f}s  "
          f"({(time.time() - t0) / max(n_eval, 1):.1f}s/query)")
    print(f"Successful evaluations    : {n_eval - n_errors}")
    print(f"Skipped queries           : {n_skipped_no_s2 + n_skipped_malformed}")
    print(f"Retrieval errors          : {n_errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
