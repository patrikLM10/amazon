#!/usr/bin/env python3
"""Rerank saved ES Top-50 files with custom BM25 into a separate folder.

Reads result .txt files (QUERY line + ranked hit blocks, as written by
elasticsearch_search.py), rescores the candidates with the existing
search_core.rank_candidates(), and writes reordered copies preserving the
relative folder layout (s2/, s3/, ...).

No Elasticsearch access needed — pure Python rescoring of 50 docs per file.

Usage:
    python rerank_folder.py <input_folder> <output_folder>
    python rerank_folder.py s1_to_s2_search/S1-357837082_to_S1-557991692 \\
        s1_to_s2_search/S1-357837082_to_S1-557991692_reranked --topk 50
"""

import argparse
import glob
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_HERE)
_ROOT_DIR = os.path.dirname(_PKG_DIR)
for _p in (_HERE, _PKG_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

try:
    from s1_to_s2_search.search_core import rank_candidates
except ImportError:
    from search_core import rank_candidates

TICK = "\u2705"
QUERY_RE = re.compile(r"^QUERY\s+(S1-\S+?)\s*:\s*(.*?)\s*\|\s*(.*?)\s*\[(.*?)\]\s*$")
HIT_RE = re.compile(r"^\s*(\d+)\.\s+(S[23]-\S+?)\s*\|\s*score=([\d.]+).*?\[(.*?)\]")
NAME_RE = re.compile(r"^\s*name:\s*(.*)$")
ADDR_RE = re.compile(r"^\s*addr:\s*(.*)$")
FILE_RE = re.compile(r"(S1-\d+)")


def parse_result_file(path):
    """Returns (query_dict, candidates_df) or raises ValueError."""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("utf-16")
    lines = text.splitlines()

    m = next((QUERY_RE.match(l) for l in lines if QUERY_RE.match(l)), None)
    if m:
        query = {"entity_id": m.group(1), "business_name": m.group(2),
                 "business_address": m.group(3), "country": m.group(4)}
    else:
        m2 = FILE_RE.search(os.path.basename(path))
        if not m2:
            raise ValueError("no QUERY line and no S1-* filename")
        query = {"entity_id": m2.group(1), "business_name": "",
                 "business_address": "", "country": ""}

    rows, cur = [], None
    for l in lines:
        hm = HIT_RE.match(l.replace(" " + TICK, ""))
        if hm:
            if cur:
                rows.append(cur)
            cur = {"es_rank": int(hm.group(1)), "entity_id": hm.group(2),
                   "es_score": float(hm.group(3)), "country": hm.group(4),
                   "business_name": "", "business_address": ""}
            continue
        nm = NAME_RE.match(l)
        if nm and cur is not None:
            cur["business_name"] = nm.group(1).strip()
            continue
        am = ADDR_RE.match(l)
        if am and cur is not None:
            cur["business_address"] = am.group(1).strip()
    if cur:
        rows.append(cur)
    if not rows:
        raise ValueError("no hit blocks parsed")
    df = pd.DataFrame(rows, columns=["entity_id", "business_name",
                                     "business_address", "country",
                                     "es_rank", "es_score"])
    return query, df


def rerank_file(src, dst, topk):
    query, df = parse_result_file(src)
    re_df = rank_candidates(query, df, topk)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write(f"QUERY {query['entity_id']}: {query['business_name']} | "
                f"{query['business_address']} [{query['country']}]\n")
        f.write(f"custom BM25 rerank of {len(df)} ES candidates\n\n")
        for i, r in re_df.iterrows():
            f.write(f"{i+1:2d}. {r['entity_id']} | bm25={r['bm25_score']:.2f} | "
                    f"es_rank={int(r['es_rank'])} es_score={r['es_score']:.2f} "
                    f"| [{r['country']}]")
            if TICK in str(r.get('entity_id', '')):
                f.write(" " + TICK)
            f.write(f"\n     name: {r['business_name']}\n"
                    f"     addr: {r['business_address']}\n")
    top1_changed = (len(re_df) and str(re_df.iloc[0]["entity_id"]) !=
                    str(df.sort_values("es_rank").iloc[0]["entity_id"]))
    return query["entity_id"], len(re_df), bool(top1_changed)


def main():
    ap = argparse.ArgumentParser(description="Rerank saved Top-50 files with custom BM25")
    ap.add_argument("input", help="Folder of result .txt files (recursive)")
    ap.add_argument("output", help="Separate output folder (layout preserved)")
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input, "**", "*.txt"), recursive=True))
    if not paths:
        print(f"No .txt files under {args.input}", file=sys.stderr)
        return 2

    n_ok = n_top1 = n_fail = 0
    for src in paths:
        rel = os.path.relpath(src, args.input)
        try:
            sid, n, changed = rerank_file(src, os.path.join(args.output, rel), args.topk)
            n_ok += 1
            n_top1 += changed
            print(f"{sid}: {n} reranked -> {rel} {'[top-1 changed]' if changed else ''}")
        except Exception as exc:
            n_fail += 1
            print(f"{src}: ERROR {exc}")
    print(f"\nDone: {n_ok} files reranked into {args.output} | "
          f"top-1 changed in {n_top1} | failed: {n_fail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
