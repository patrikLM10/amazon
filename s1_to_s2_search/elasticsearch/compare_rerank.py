#!/usr/bin/env python3
"""ES-alone vs ES + custom-BM25-rescore, same queries, same Top-50.

For each eligible S1: ES Top-50 -> rank via ES order AND via existing
search_core.rank_candidates() over those same 50. Compares Recall@K.

Usage:
    python compare_rerank.py --source train --topk 50 --limit 60
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

try:
    from s1_to_s2_search.elasticsearch.elasticsearch_search import search_s1_id
    from s1_to_s2_search.search_core import rank_candidates
except ImportError:
    from elasticsearch_search import search_s1_id
    from search_core import rank_candidates

K_LIST = (1, 5, 10, 20, 50)


def parse_gt_s2_ids(raw):
    raw = str(raw).strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    if any(p == "" for p in parts):
        raise ValueError(f"malformed: {raw!r}")
    return [p for p in parts if p.startswith("S2-")]


def first_rank(retrieved_ids, gt_set):
    return next((i for i, r in enumerate(retrieved_ids, 1) if r in gt_set), "")


def main():
    ap = argparse.ArgumentParser(description="ES vs ES+custom-rescore comparison")
    ap.add_argument("--source", default="train", choices=["train", "test"])
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--index", default=None)
    ap.add_argument("--mode", default="lexical", choices=["lexical", "ngram"])
    args = ap.parse_args()

    import pandas as pd
    gt_path = os.path.join(_ROOT_DIR, "dataset", args.source,
                           f"{args.source}_ground_truth.tsv")
    ks = [k for k in K_LIST if k <= args.topk]
    es_hits = {k: 0 for k in ks}
    ce_hits = {k: 0 for k in ks}
    wins = ties = losses = n = n_skip = 0

    rows = pd.read_csv(gt_path, sep="\t", encoding="utf-8",
                       nrows=args.limit, dtype=str, keep_default_na=False)
    for idx, row in rows.iterrows():
        s1_id, raw = str(row["source1_entity_id"]).strip(), row["matched_entity_ids"]
        try:
            gt = parse_gt_s2_ids(raw)
        except ValueError:
            n_skip += 1
            continue
        if not gt:
            n_skip += 1
            continue
        try:
            query, df, _ = search_s1_id(s1_id, args.source, args.topk,
                                        args.index, args.mode)
            es_ids = [str(x) for x in df["entity_id"].tolist()] if len(df) else []
            re_df = rank_candidates(query, df, args.topk)
            cu_ids = ([str(x) for x in re_df["entity_id"].tolist()]
                      if re_df is not None and len(re_df) else [])
        except Exception as exc:
            print(f"[{idx}] ERROR {s1_id}: {exc}")
            n_skip += 1
            continue
        gt_set = set(gt)
        r_es, r_cu = first_rank(es_ids, gt_set), first_rank(cu_ids, gt_set)
        n += 1
        for k in ks:
            es_hits[k] += (r_es != "" and r_es <= k)
            ce_hits[k] += (r_cu != "" and r_cu <= k)
        if (r_cu or 999) < (r_es or 999):
            wins += 1
        elif (r_cu or 999) > (r_es or 999):
            losses += 1
        else:
            ties += 1
        flag = "UP " if (r_cu or 999) < (r_es or 999) else ("DN " if (r_cu or 999) > (r_es or 999) else "== ")
        print(f"[{flag}] {s1_id} ES_rank={r_es or '-'} custom_rank={r_cu or '-'}")

    print("\nMetric        ES-alone   ES+custom-rescore")
    print("------------------------------------------")
    for k in ks:
        print(f"Recall@{k:<4}  {100.0 * es_hits[k] / n:6.2f}%    {100.0 * ce_hits[k] / n:6.2f}%"
              if n else "")
    print(f"\nQueries where rescore helped / tied / hurt: {wins} / {ties} / {losses} (n={n}, skipped={n_skip})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
