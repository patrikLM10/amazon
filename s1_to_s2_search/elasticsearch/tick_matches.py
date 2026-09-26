#!/usr/bin/env python3
"""Append a tick beside retrieved IDs that match ground truth (same bucket only).

Usage:
    python tick_matches.py s1_to_s2_search/S1-550488938.txt
    python tick_matches.py <path> --corpus s2 --source train

- S1 id is parsed from the file's QUERY line (fallback: filename S1-*.txt).
- Bucket is auto-detected from the first retrieved id (S2-/S3-); --corpus
  overrides it. Ground truth is filtered to that bucket only.
- File is edited in place (UTF-8). Re-running is safe: ticks are not duplicated.
"""

import argparse
import os
import re
import sys

# Script lives in s1_to_s2_search/elasticsearch/ -> repo root is two levels up.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TICK = "\u2705"  # ✅
HIT_RE = re.compile(r"^(\s*\d+\.\s+)(S[23]-\S+?)(\s+\|.*(?:score|bm25)=.*)$")
QUERY_RE = re.compile(r"^QUERY\s+(S1-\S+?)\s*:")
FILE_RE = re.compile(r"(S1-\d+)")


def find_gt_cell(gt_path, s1_id):
    """Stream GT until the S1 row is found. Returns cell or None."""
    import pandas as pd
    for chunk in pd.read_csv(gt_path, sep="\t", encoding="utf-8",
                             chunksize=100_000, dtype=str, keep_default_na=False):
        hit = chunk[chunk["source1_entity_id"] == s1_id]
        if len(hit):
            return hit.iloc[0]["matched_entity_ids"]
    return None


def tick_file(path, corpus=None, source="train"):
    """Tick GT matches in one result file. Returns a summary dict."""
    if not os.path.isfile(path):
        return {"ok": False, "message": f"File not found: {path}"}

    with open(path, "rb") as f:
        raw = f.read()
    encoding = "utf-8"
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text, encoding = raw.decode("utf-16"), "utf-16"
    lines = text.splitlines()

    m = next((QUERY_RE.match(l) for l in lines if QUERY_RE.match(l)), None)
    s1_id = m.group(1) if m else None
    if s1_id is None:
        m2 = FILE_RE.search(os.path.basename(path))
        s1_id = m2.group(1) if m2 else None
    if s1_id is None:
        return {"ok": False, "message": "Could not determine S1 id."}

    hit_ids = [HIT_RE.match(l).group(2) for l in lines if HIT_RE.match(l)]
    if not hit_ids:
        return {"ok": False, "s1_id": s1_id,
                "message": "No retrieved hit lines found."}

    corpus = corpus or ("s2" if hit_ids[0].startswith("S2-") else "s3")
    prefix = corpus.upper() + "-"

    gt_path = os.path.join(_ROOT_DIR, "dataset", source,
                           f"{source}_ground_truth.tsv")
    if not os.path.isfile(gt_path):
        return {"ok": False, "s1_id": s1_id,
                "message": f"Ground-truth file not found: {gt_path}"}
    cell = find_gt_cell(gt_path, s1_id)
    if cell is None:
        return {"ok": True, "s1_id": s1_id, "hit": 0, "added": 0,
                "already": 0, "message": f"{s1_id} not in ground truth; unchanged."}
    gt_ids = {p.strip() for p in str(cell).split(",") if p.strip().startswith(prefix)}
    if not gt_ids:
        return {"ok": True, "s1_id": s1_id, "hit": 0, "added": 0,
                "already": 0,
                "message": f"{s1_id} has no {corpus.upper()} ground truth; unchanged."}

    added, already = 0, 0
    for i, l in enumerate(lines):
        hm = HIT_RE.match(l)
        if hm and hm.group(2) in gt_ids:
            if TICK in l:
                already += 1
            else:
                lines[i] = l + " " + TICK
                added += 1

    with open(path, "w", encoding=encoding, newline="") as f:
        f.write("\n".join(lines) + "\n")

    hit = 1 if added + already else 0
    return {"ok": True, "s1_id": s1_id, "corpus": corpus, "hit": hit,
            "added": added, "already": already, "gt_ids": sorted(gt_ids),
            "message": f"{s1_id} | GT: {', '.join(sorted(gt_ids))} | "
                       f"added={added} already={already} => {'HIT' if hit else 'MISS'}"}


def main():
    ap = argparse.ArgumentParser(description="Tick GT matches in a result file")
    ap.add_argument("path", help="Result .txt file, e.g. s1_to_s2_search/S1-550488938.txt")
    ap.add_argument("--corpus", default=None, choices=["s2", "s3"],
                    help="Bucket to match (default: auto-detect from hits)")
    ap.add_argument("--source", default="train", choices=["train", "test"])
    args = ap.parse_args()

    r = tick_file(args.path, args.corpus, args.source)
    print(r["message"], file=sys.stderr if not r["ok"] else sys.stdout)
    return 0 if r["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
