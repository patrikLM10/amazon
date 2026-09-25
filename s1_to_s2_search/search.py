#!/usr/bin/env python3
"""CLI wrapper: python search.py --s1_id S1-925783039 --source train --topk 50."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api import search_top_k


def main():
    ap = argparse.ArgumentParser(description="S1 -> S2 top-K BM25 search")
    ap.add_argument("--s1_id", required=True)
    ap.add_argument("--source", default="train", choices=["train", "test"])
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--s2_limit", type=int, default=None)
    ap.add_argument("--all_countries", action="store_true")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--no_save", action="store_true", help="Don't write results_<S1ID>.tsv")
    args = ap.parse_args()

    query, ranked = search_top_k(
        args.s1_id, source=args.source, topk=args.topk, chunk=args.chunk,
        s2_limit=args.s2_limit, all_countries=args.all_countries,
        data_dir=args.data_dir, verbose=True, save=not args.no_save)

    if len(ranked) == 0:
        print("No candidates found. Try --all_countries or a larger --s2_limit.")
        return 0

    print(f"\n=== TOP {len(ranked)} FROM S2 (BM25 name x3 + address x1) ===")
    for i, r in ranked.iterrows():
        print(f"{i+1:2d}. {r['entity_id']} | score={r['bm25_score']:.2f} | [{r['country']}]")
        print(f"     name: {r['business_name']}")
        print(f"     addr: {r['business_address']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
