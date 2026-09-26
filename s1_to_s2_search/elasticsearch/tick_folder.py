#!/usr/bin/env python3
"""BOILERPLATE — tick every result .txt in a folder with one command.

Usage:
    python tick_folder.py student_resource/s1_to_s2_search/output_txt
    python tick_folder.py <folder> --corpus s2 --source train

Ticks each *.txt via tick_matches.tick_file() (bucket auto-detected per file
unless --corpus is given), then prints a per-file line and a final tally.
"""

import argparse
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))  # .../elasticsearch
_PKG_DIR = os.path.dirname(_HERE)
_ROOT_DIR = os.path.dirname(_PKG_DIR)
for _p in (_HERE, _PKG_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from s1_to_s2_search.elasticsearch.tick_matches import tick_file
except ImportError:
    from tick_matches import tick_file


def main():
    ap = argparse.ArgumentParser(description="Tick all result files in a folder")
    ap.add_argument("folder", help="Folder of result .txt files")
    ap.add_argument("--corpus", default=None, choices=["s2", "s3"])
    ap.add_argument("--source", default="train", choices=["train", "test"])
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.folder, "**", "*.txt"), recursive=True))
    if not paths:
        print(f"No .txt files in {args.folder}", file=sys.stderr)
        return 2

    n_hit = n_eval = n_skip = 0
    for p in paths:
        r = tick_file(p, args.corpus, args.source)
        print(r["message"])
        if not r.get("ok") or "hit" not in r:
            n_skip += 1
        else:
            n_eval += 1
            n_hit += r["hit"]

    print(f"\nTally: {n_hit}/{n_eval} HIT "
          f"({100.0 * n_hit / n_eval:.1f}%) | skipped: {n_skip}" if n_eval
          else f"\nTally: no evaluable files | skipped: {n_skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
