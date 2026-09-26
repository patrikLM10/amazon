#!/usr/bin/env python3
"""No-match ES score distribution study (measurement only, no threshold picked).

For 500 randomly sampled S1 entities with NO S2/S3 ground truth, run the EXACT
production Elasticsearch S2 + S3 retrieval queries (same builder, same boosts,
same country filter, same lexical mode) and characterize the top-1 score
distribution. This is the negative/no-match side; the positive side comes later.

Google Colab:
    !pip install elasticsearch tqdm pandas matplotlib numpy
    !python nomatch_score_study.py --es-host http://localhost:9200
    # --es-host must reach an ES node holding the s2_train / s3_train indexes.

Local:
    python nomatch_score_study.py
    python nomatch_score_study.py --sample-n 500 --seed 42 --topk 20
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))  # .../nomatch_study
_ES_DIR = os.path.dirname(_HERE)  # .../elasticsearch
_PKG_DIR = os.path.dirname(_ES_DIR)  # .../s1_to_s2_search
_ROOT_DIR = os.path.dirname(_PKG_DIR)  # .../student_resource
for _p in (_HERE, _ES_DIR, _PKG_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import elasticsearch_search as _esmod
    from s1_to_s2_search.elasticsearch import es_config as _escfg
except ImportError:
    import elasticsearch_search as _esmod
    import es_config as _escfg

THRESHOLDS = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
PCTILES = [1, 5, 10, 25, 50, 75, 90, 95, 97, 98, 99, 99.5]


def safe_score(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def top1_of(df):
    """(top1_id, top1_score, n) from a search_elasticsearch DataFrame."""
    if df is None or len(df) == 0 or "entity_id" not in df.columns:
        return "", np.nan, 0
    r = df.iloc[0]
    return str(r["entity_id"]), safe_score(r.get("_score")), len(df)


def main():
    ap = argparse.ArgumentParser(description="No-match ES score distribution study")
    ap.add_argument("--no-match-tsv", default=None)
    ap.add_argument("--source", default="train", choices=["train", "test"])
    ap.add_argument("--sample-n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--es-host", default=os.environ.get("ES_HOST", "http://localhost:9200"))
    ap.add_argument("--output", default=None)
    ap.add_argument("--plots", default=None, help="PNG prefix for histogram/boxplot")
    args = ap.parse_args()

    nm_path = args.no_match_tsv or os.path.join(
        _ROOT_DIR, "dataset", args.source, "s1_no_s2_s3_ground_truth_matches.tsv")
    nm = pd.read_csv(nm_path, sep="\t", encoding="utf-8",
                     dtype=str, keep_default_na=False)
    if "source1_entity_id" not in nm.columns:
        print(f"Unexpected no-match header {list(nm.columns)}", file=sys.stderr)
        return 2
    if "matched_entity_ids" in nm.columns:
        bad = (nm["matched_entity_ids"].str.strip() != "").sum()
        if bad:
            print(f"WARNING: {bad} rows have non-empty matched ids; dropping them.")
            nm = nm[nm["matched_entity_ids"].str.strip() == ""]
    print(f"No-match S1 rows loaded: {len(nm):,}")

    sample = nm.sample(n=min(args.sample_n, len(nm)), random_state=args.seed)
    sample_ids = [str(x).strip() for x in sample["source1_entity_id"].tolist()]
    print(f"Sampled: {len(sample_ids)} (seed={args.seed})")

    # Point the shared ES client at the requested host (default: local node).
    _esmod._es = _escfg.get_client(args.es_host)
    info = _esmod._es.info()
    ver = info.body["version"]["number"] if hasattr(info, "body") else info["version"]["number"]
    print(f"Elasticsearch {ver} at {args.es_host}")

    # Harness-only optimization (does NOT alter scoring): preload the sampled
    # S1 records in one streamed pass instead of one full S1 scan per query.
    # The ES query text sent per S1 is identical to search_s1_id().
    try:
        from s1_to_s2_search.search_core import resolve_paths
    except ImportError:
        from search_core import resolve_paths
    s1_p, _ = resolve_paths(args.source, None, None, None)
    want = set(sample_ids)
    s1_map = {}
    for chunk in pd.read_csv(s1_p, sep="\t", encoding="utf-8",
                             chunksize=200_000, dtype=str, keep_default_na=False):
        hit = chunk[chunk["entity_id"].isin(want - s1_map.keys())]
        for _, r in hit.iterrows():
            s1_map[str(r["entity_id"])] = {
                "entity_id": r["entity_id"], "business_name": r["business_name"],
                "business_address": r["business_address"], "country": r["country"]}
        if len(s1_map) == len(want):
            break
    print(f"S1 records preloaded: {len(s1_map)}/{len(sample_ids)}")

    idx_s2 = _escfg.default_index(args.source, "s2")
    idx_s3 = _escfg.default_index(args.source, "s3")
    rows, n_err = [], 0
    for sid in tqdm(sample_ids, desc="ES S2+S3 queries"):
        q = s1_map.get(sid)
        if q is None:
            rows.append({"s1_id": sid, "status": "s1_not_found"})
            n_err += 1
            continue
        try:
            df2, _ = _esmod.search_elasticsearch(
                q["business_name"], q["business_address"], q["country"],
                args.topk, idx_s2, "lexical")
            df3, _ = _esmod.search_elasticsearch(
                q["business_name"], q["business_address"], q["country"],
                args.topk, idx_s3, "lexical")
        except Exception as exc:
            rows.append({"s1_id": sid, "status": f"es_error: {exc}"})
            n_err += 1
            continue
        id2, sc2, n2 = top1_of(df2)
        id3, sc3, n3 = top1_of(df3)
        both = [s for s in (sc2, sc3) if not np.isnan(s)]
        rows.append({
            "s1_id": sid, "country": q["country"],
            "business_name": q["business_name"],
            "business_address": q["business_address"],
            "s2_top1_id": id2, "s2_top1_score": sc2, "n_s2": n2,
            "s3_top1_id": id3, "s3_top1_score": sc3, "n_s3": n3,
            "max_top1_score": max(both) if both else np.nan,
            "s2_top20": ",".join(
                f"{r['entity_id']}:{safe_score(r.get('_score')):.2f}"
                for _, r in df2.iterrows()) if n2 else "",
            "s3_top20": ",".join(
                f"{r['entity_id']}:{safe_score(r.get('_score')):.2f}"
                for _, r in df3.iterrows()) if n3 else "",
            "status": "ok",
        })

    res = pd.DataFrame(rows)
    out_path = args.output or os.path.join(_HERE, "nomatch_scores.tsv")
    res.to_csv(out_path, sep="\t", index=False, encoding="utf-8")
    print(f"\nSaved {len(res)} rows -> {out_path}")

    ok = res[res["status"] == "ok"].copy()
    mx = ok["max_top1_score"].to_numpy(dtype=float)
    has2 = ok["n_s2"].to_numpy() > 0
    has3 = ok["n_s3"].to_numpy() > 0
    vals = mx[~np.isnan(mx)]

    print("\n========================================")
    print("NO-MATCH ES SCORE DISTRIBUTION")
    print("========================================\n")
    print(f"Sampled S1s                 : {len(sample_ids)}")
    print(f"Evaluated OK                : {len(ok)} (errors: {n_err})")
    print(f"Returning S2 candidates     : {int(has2.sum())}")
    print(f"Returning S3 candidates     : {int(has3.sum())}")
    print(f"Returning either source     : {int((has2 | has3).sum())}")
    print(f"Returning neither           : {int((~has2 & ~has3).sum())}\n")
    if len(vals):
        print(f"max_top1_score (n={len(vals)}):")
        print(f"  min / max   : {np.nanmin(vals):.2f} / {np.nanmax(vals):.2f}")
        print(f"  mean / std  : {np.nanmean(vals):.2f} / {np.nanstd(vals, ddof=1):.2f}")
        print(f"  median      : {np.nanmedian(vals):.2f}")
        for p in PCTILES:
            print(f"  P{p:<6}: {np.nanpercentile(vals, p):.2f}")
        print("\nThreshold analysis (share of evaluated no-match S1s "
              "with max_top1 >= T):")
        print(f"  {'T':>5}  {'count':>6}  {'pct':>7}")
        for t in THRESHOLDS:
            c = int(np.nansum(mx >= t))
            print(f"  {t:>5}  {c:>6}  {100.0 * c / len(ok):7.2f}%")
    else:
        print("No scores retrieved at all.")

    if len(vals):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pre = args.plots or os.path.join(_HERE, "nomatch_scores")
        plt.figure()
        plt.hist(vals, bins=50)
        plt.xlabel("max_top1_score (no-match S1s)")
        plt.ylabel("count")
        plt.title(f"Histogram of max_top1_score (n={len(vals)})")
        plt.savefig(pre + "_hist.png", dpi=120, bbox_inches="tight")
        plt.figure()
        plt.boxplot(vals, orientation="vertical")
        plt.ylabel("max_top1_score (no-match S1s)")
        plt.title(f"Box plot of max_top1_score (n={len(vals)})")
        plt.savefig(pre + "_box.png", dpi=120, bbox_inches="tight")
        print(f"\nPlots -> {pre}_hist.png, {pre}_box.png")
    print("\nNOTE: no threshold is declared here. Await the known-positive "
          "distribution before judging overlap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
