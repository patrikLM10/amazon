#!/usr/bin/env python3
"""Stage-level profiler for the existing S1 -> S2 BM25 retrieval pipeline.

MEASUREMENT ONLY: performs the exact same operations as
search_core.stream_block() + search_core.rank_candidates()
(same read call, same country filter, same blob construction, same
regex, same 20k cutoff, same normalization/tokenization/BM25/sort),
but wraps each stage with time.perf_counter() timers.

No retrieval behavior is changed. No file in api.py / search_core.py /
bm25.py / search.py / verify_recall.py is modified.

Usage:
    python profile_search.py --s1_id S1-965667 --source train --topk 50
    python profile_search.py --s1_id S1-965667 --source train --topk 50 --runs 3

    # from the repo root (student_resource/):
    python s1_to_s2_search/profile_search.py --s1_id S1-965667 --source train
"""

import argparse
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)  # student_resource/
for _p in (_HERE, _PKG_PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

try:
    from s1_to_s2_search.bm25 import (
        blocking_terms,
        bm25_field_weighted,
        char_trigrams,
        compute_idf,
        doc_tokens,
        normalize,
        word_tokens,
    )
    from s1_to_s2_search.search_core import find_s1, resolve_paths
except ImportError:  # cwd is the package dir itself
    from bm25 import (
        blocking_terms,
        bm25_field_weighted,
        char_trigrams,
        compute_idf,
        doc_tokens,
        normalize,
        word_tokens,
    )
    from search_core import find_s1, resolve_paths


def profile_scan(s2_path, pattern, query_country, same_country_only,
                 chunksize, s2_limit):
    """Mirror of search_core.stream_block() with per-stage timers.

    Returns (cands_df, n_scanned, stats) where stats holds stage times
    and row counts. Operations are identical to stream_block().
    """
    t = {"csv": 0.0, "country": 0.0, "blob": 0.0, "regex": 0.0, "collect": 0.0}
    n_scanned = 0
    n_after_country = 0
    n_surviving = 0
    rx = re.compile(pattern, re.IGNORECASE) if pattern else None
    cands = []

    reader = pd.read_csv(s2_path, sep="\t", encoding="utf-8",
                         chunksize=chunksize, dtype=str, keep_default_na=False)
    while True:
        t0 = time.perf_counter()
        try:
            chunk = next(reader)
        except StopIteration:
            t["csv"] += time.perf_counter() - t0
            break
        t["csv"] += time.perf_counter() - t0

        if s2_limit is not None and n_scanned >= s2_limit:
            break
        if s2_limit is not None:
            chunk = chunk.iloc[:max(0, s2_limit - n_scanned)]
        n_scanned += len(chunk)

        if same_country_only:
            t0 = time.perf_counter()
            chunk = chunk[chunk["country"] == query_country]
            t["country"] += time.perf_counter() - t0
            if len(chunk) == 0:
                continue
        n_after_country += len(chunk)

        if rx is None:
            t0 = time.perf_counter()
            cands.append(chunk)
            t["collect"] += time.perf_counter() - t0
            n_surviving += len(chunk)
            continue

        t0 = time.perf_counter()
        blob = (chunk["business_name"].fillna("") + " " +
                chunk["business_address"].fillna("")).str.lower()
        t["blob"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        counts_all = blob.str.count(rx)
        mask = counts_all >= 2
        if int(mask.sum()) == 0:
            mask = counts_all >= 1
        hit = chunk[mask]
        t["regex"] += time.perf_counter() - t0
        n_surviving += len(hit)

        if len(hit):
            t0 = time.perf_counter()
            if len(hit) > 20_000:
                hit = hit.assign(_hits=counts_all[mask].values)
                hit = hit.nlargest(20_000, "_hits").drop(columns=["_hits"])
            cands.append(hit)
            t["collect"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    if cands:
        out = pd.concat(cands, ignore_index=True).drop_duplicates(subset=["entity_id"])
    else:
        out = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    t["collect"] += time.perf_counter() - t0

    stats = dict(t)
    stats["n_scanned"] = n_scanned
    stats["n_after_country"] = n_after_country
    stats["n_surviving"] = n_surviving
    stats["n_final"] = len(out)
    return out, n_scanned, stats


def profile_rank(query, cands, topk):
    """Mirror of search_core.rank_candidates() with per-stage timers."""
    t = {"tokenize": 0.0, "idf": 0.0, "scoring": 0.0, "sort": 0.0}

    qn = normalize(query["business_name"])
    qa = normalize(query["business_address"])
    q_name_toks = word_tokens(qn) + char_trigrams(qn)
    q_addr_toks = word_tokens(qa) + char_trigrams(qa)

    t0 = time.perf_counter()
    d_name_list, d_addr_list = [], []
    name_lens, addr_lens = [], []
    for _, r in cands.iterrows():
        nn = normalize(r["business_name"])
        aa = normalize(r["business_address"])
        nt = word_tokens(nn) + char_trigrams(nn)
        at = word_tokens(aa) + char_trigrams(aa)
        d_name_list.append(nt)
        d_addr_list.append(at)
        name_lens.append(len(nt) or 1)
        addr_lens.append(len(at) or 1)
    t["tokenize"] = time.perf_counter() - t0

    n = len(cands)
    if n == 0:
        return cands, t
    avg_n = sum(name_lens) / n
    avg_a = sum(addr_lens) / n

    t0 = time.perf_counter()
    idf_n = compute_idf(d_name_list, n)
    idf_a = compute_idf(d_addr_list, n)
    t["idf"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = [bm25_field_weighted(q_name_toks, q_addr_toks, nt, at, nl, al,
                                  avg_n, avg_a, idf_n, idf_a)
              for nt, at, nl, al in zip(d_name_list, d_addr_list, name_lens, addr_lens)]
    t["scoring"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = cands.copy()
    out["bm25_score"] = scores
    out = out.sort_values("bm25_score", ascending=False).head(topk).reset_index(drop=True)
    t["sort"] = time.perf_counter() - t0
    return out, t


def profile_once(s1_id, source, topk, chunk, s2_limit, all_countries, data_dir):
    """One full instrumented run through the real pipeline path."""
    s1_p, s2_p = resolve_paths(source, data_dir, None, None)
    run = {}

    t0 = time.perf_counter()
    query = find_s1(s1_p, s1_id)
    run["s1_lookup"] = time.perf_counter() - t0
    if query is None:
        raise ValueError(f"{s1_id} not found in {s1_p} (wrong --source?)")

    _, qn_norm, qa_norm = doc_tokens(query["business_name"], query["business_address"])
    pattern, _, _ = blocking_terms(qn_norm, qa_norm)

    same_only = not all_countries
    cands, _, s1 = profile_scan(s2_p, pattern, query["country"],
                                same_country_only=same_only,
                                chunksize=chunk, s2_limit=s2_limit)
    run.update({f"scan1_{k}": v for k, v in s1.items()})
    fallback_stats = None
    if same_only and len(cands) < topk * 2:
        cands2, _, s2 = profile_scan(s2_p, pattern, query["country"],
                                     same_country_only=False,
                                     chunksize=chunk, s2_limit=s2_limit)
        t0 = time.perf_counter()
        cands = pd.concat([cands, cands2], ignore_index=True
                          ).drop_duplicates(subset=["entity_id"])
        merge_t = time.perf_counter() - t0
        fallback_stats = dict(s2)
        fallback_stats["merge"] = merge_t

    run["fallback"] = fallback_stats is not None
    if fallback_stats:
        run.update({f"scan2_{k}": v for k, v in fallback_stats.items()})

    if len(cands):
        _, rt = profile_rank(query, cands, topk)
    else:
        rt = {"tokenize": 0.0, "idf": 0.0, "scoring": 0.0, "sort": 0.0}
    run.update({f"rank_{k}": v for k, v in rt.items()})
    run["n_cands"] = len(cands)
    return query, run


def scan_total(run, prefix):
    return (run.get(f"{prefix}_csv", 0.0) + run.get(f"{prefix}_country", 0.0) +
            run.get(f"{prefix}_blob", 0.0) + run.get(f"{prefix}_regex", 0.0) +
            run.get(f"{prefix}_collect", 0.0))


def print_run(i, run, topk):
    s1t = scan_total(run, "scan1")
    s2t = scan_total(run, "scan2") + run.get("scan2_merge", 0.0) if run["fallback"] else 0.0
    rt = run["rank_tokenize"] + run["rank_idf"] + run["rank_scoring"] + run["rank_sort"]
    total = run["s1_lookup"] + s1t + s2t + rt
    run["_total"] = total
    print(f"\n----- Run {i} -----")
    print(f"S2 rows scanned           : {run['scan1_n_scanned']:,}")
    print(f"Rows after country filter : {run['scan1_n_after_country']:,}")
    print(f"Rows surviving blocking   : {run['scan1_n_surviving']:,}")
    print(f"Final candidate count     : {run['n_cands']:,}")
    print(f"Fallback triggered        : {'yes' if run['fallback'] else 'no'}")
    if run["fallback"]:
        print(f"  fallback rows surviving : {run['scan2_n_surviving']:,}")
    print("TIMING")
    print(f"S1 lookup                : {run['s1_lookup']:8.2f} s")
    print(f"S2 CSV reading (scan1)   : {run['scan1_csv']:8.2f} s")
    print(f"Country filtering (scan1): {run['scan1_country']:8.2f} s")
    print(f"Blocking text (scan1)    : {run['scan1_blob']:8.2f} s")
    print(f"Regex/blocking (scan1)   : {run['scan1_regex']:8.2f} s")
    print(f"Candidate concat (scan1) : {run['scan1_collect']:8.2f} s")
    if run["fallback"]:
        print(f"Fallback scan (total)    : {s2t:8.2f} s "
              f"(csv={run['scan2_csv']:.2f} regex={run['scan2_regex']:.2f})")
    print(f"Candidate tokenization   : {run['rank_tokenize']:8.2f} s")
    print(f"IDF calculation          : {run['rank_idf']:8.2f} s")
    print(f"BM25 scoring             : {run['rank_scoring']:8.2f} s")
    print(f"Sorting / Top-{topk:<6}: {run['rank_sort']:8.2f} s")
    print(f"TOTAL                    : {total:8.2f} s")
    return run


def main():
    ap = argparse.ArgumentParser(description="Stage-level profiler (measurement only)")
    ap.add_argument("--s1_id", required=True)
    ap.add_argument("--source", default="train", choices=["train", "test"])
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--s2_limit", type=int, default=None,
                    help="DEBUG ONLY: partial S2 scan. Omit for true measurement.")
    ap.add_argument("--all_countries", action="store_true")
    args = ap.parse_args()

    if args.s2_limit is not None:
        print("WARNING: --s2_limit performs a PARTIAL scan; numbers are not "
              "representative of full-query cost.")

    # Warm-up is just the first measured run (cold cache); label runs accordingly.
    runs = []
    query = None
    for i in range(1, args.runs + 1):
        print(f"\n=== Run {i}/{args.runs} (Run 1 = cold cache) ===", flush=True)
        query, run = profile_once(args.s1_id, args.source, args.topk,
                                  args.chunk, args.s2_limit,
                                  args.all_countries, args.data_dir)
        print_run(i, run, args.topk)
        runs.append(run)

    keys = ["s1_lookup", "scan1_csv", "scan1_country", "scan1_blob",
            "scan1_regex", "scan1_collect", "rank_tokenize", "rank_idf",
            "rank_scoring", "rank_sort"]
    avg = {k: sum(r[k] for r in runs) / len(runs) for k in keys}
    fb = [r for r in runs if r["fallback"]]
    if fb:
        avg["scan2_total"] = sum(scan_total(r, "scan2") + r.get("scan2_merge", 0.0)
                                 for r in fb) / len(fb)
    avg_total = sum(r["_total"] for r in runs) / len(runs)

    print("\n========================================")
    print("S1 -> S2 BM25 PROFILE (average)")
    print("========================================\n")
    print(f"S1 ID                     : {args.s1_id}")
    print(f"Country                   : {query['country']}")
    print(f"Name                      : {query['business_name']}")
    print(f"Top-K                     : {args.topk}")
    print(f"Runs                      : {args.runs}")
    for i, r in enumerate(runs, 1):
        print(f"Run {i}                     : {r['_total']:.2f} s")
    print(f"Average                   : {avg_total:.2f} s")
    print(f"\nS2 rows scanned           : {runs[0]['scan1_n_scanned']:,}")
    print(f"Rows after country filter : {runs[0]['scan1_n_after_country']:,}")
    print(f"Rows surviving blocking   : {runs[0]['scan1_n_surviving']:,}")
    print(f"Final candidate count     : {runs[0]['n_cands']:,}")
    print(f"Fallback triggered        : {'yes' if any(r['fallback'] for r in runs) else 'no'}")
    if fb:
        print(f"Fallback scan (avg)       : {avg['scan2_total']:.2f} s")
    print("\n----------------------------------------")
    print("TIMING (average)")
    print("----------------------------------------\n")
    labels = [("S1 lookup", "s1_lookup"), ("S2 CSV reading", "scan1_csv"),
              ("Country filtering", "scan1_country"),
              ("Blocking text creation", "scan1_blob"),
              ("Regex/blocking matching", "scan1_regex"),
              ("Candidate concat/dedup", "scan1_collect"),
              ("Candidate tokenization", "rank_tokenize"),
              ("IDF calculation", "rank_idf"),
              ("BM25 scoring", "rank_scoring"),
              ("Sorting/Top-K", "rank_sort")]
    for label, k in labels:
        pct = 100.0 * avg[k] / avg_total if avg_total else 0.0
        print(f"{label:24}: {avg[k]:8.2f} s  ({pct:5.1f}%)")
    if fb:
        pct = 100.0 * avg["scan2_total"] / avg_total
        print(f"{'Fallback scan':24}: {avg['scan2_total']:8.2f} s  ({pct:5.1f}%)")
    other = avg_total - sum(avg[k] for k in keys) - (avg.get("scan2_total", 0.0) if fb else 0.0)
    print(f"{'Other':24}: {other:8.2f} s  ({100.0 * other / avg_total if avg_total else 0:5.1f}%)")
    print(f"\nTOTAL (average)           : {avg_total:.2f} s")
    print("========================================")

    # Factual interpretation (measurement only, no optimization applied)
    ranked = sorted(labels, key=lambda lk: -avg[lk[1]])
    print("\nInterpretation (measured, not guessed):")
    print(f"Largest cost is {ranked[0][0]} "
          f"({100.0 * avg[ranked[0][1]] / avg_total:.1f}% of runtime).")
    print(f"S2 CSV parsing accounts for {100.0 * avg['scan1_csv'] / avg_total:.1f}%.")
    print(f"BM25 scoring accounts for {100.0 * avg['rank_scoring'] / avg_total:.1f}%.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
