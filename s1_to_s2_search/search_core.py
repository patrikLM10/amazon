"""Shared core: path resolution + blocking + BM25 ranking (no CLI, no API)."""

import os
import re

import pandas as pd

try:
    from .bm25 import bm25_field_weighted, char_trigrams, compute_idf, normalize, word_tokens
except ImportError:
    from bm25 import bm25_field_weighted, char_trigrams, compute_idf, normalize, word_tokens

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.join(os.path.dirname(_HERE), "dataset")


def resolve_paths(source="train", data_dir=None, s1_path=None, s2_path=None):
    prefix = "train" if source == "train" else "test"
    base = data_dir or os.path.join(_DATA_ROOT, source)
    s1_p = s1_path or os.path.join(base, f"{prefix}_source1.tsv")
    s2_p = s2_path or os.path.join(base, f"{prefix}_source2.tsv")
    return s1_p, s2_p


def find_s1(s1_path: str, s1_id: str):
    """Stream S1 file to find one row. Returns dict or None."""
    for chunk in pd.read_csv(s1_path, sep="\t", encoding="utf-8",
                             chunksize=200_000, dtype=str, keep_default_na=False):
        hit = chunk[chunk["entity_id"] == s1_id]
        if len(hit):
            r = hit.iloc[0]
            return {"entity_id": r["entity_id"], "business_name": r["business_name"],
                    "business_address": r["business_address"], "country": r["country"]}
    return None


def stream_block(s2_path: str, pattern: str, query_country: str,
                 same_country_only: bool, chunksize: int, s2_limit=None):
    """Stage-1 blocking: stream S2, return (blocked DataFrame, n_scanned)."""
    rx = re.compile(pattern, re.IGNORECASE) if pattern else None
    cands = []
    n_scanned = 0
    for chunk in pd.read_csv(s2_path, sep="\t", encoding="utf-8",
                             chunksize=chunksize, dtype=str, keep_default_na=False):
        if s2_limit is not None and n_scanned >= s2_limit:
            break
        if s2_limit is not None:
            chunk = chunk.iloc[:max(0, s2_limit - n_scanned)]
        n_scanned += len(chunk)

        if same_country_only:
            chunk = chunk[chunk["country"] == query_country]
            if len(chunk) == 0:
                continue

        if rx is None:
            cands.append(chunk)
            continue

        blob = (chunk["business_name"].fillna("") + " " +
                chunk["business_address"].fillna("")).str.lower()
        counts_all = blob.str.count(rx)
        mask = counts_all >= 2
        if int(mask.sum()) == 0:
            mask = counts_all >= 1
        hit = chunk[mask]
        if len(hit):
            if len(hit) > 20_000:
                hit = hit.assign(_hits=counts_all[mask].values)
                hit = hit.nlargest(20_000, "_hits").drop(columns=["_hits"])
            cands.append(hit)

    if cands:
        out = pd.concat(cands, ignore_index=True).drop_duplicates(subset=["entity_id"])
    else:
        out = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    return out, n_scanned


def rank_candidates(query, cands: pd.DataFrame, topk: int):
    """Stage-2 BM25F ranking. Returns DataFrame with bm25_score, sorted desc."""
    qn = normalize(query["business_name"])
    qa = normalize(query["business_address"])
    q_name_toks = word_tokens(qn) + char_trigrams(qn)
    q_addr_toks = word_tokens(qa) + char_trigrams(qa)

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

    n = len(cands)
    if n == 0:
        return cands
    avg_n = sum(name_lens) / n
    avg_a = sum(addr_lens) / n
    idf_n = compute_idf(d_name_list, n)
    idf_a = compute_idf(d_addr_list, n)

    scores = [bm25_field_weighted(q_name_toks, q_addr_toks, nt, at, nl, al,
                                  avg_n, avg_a, idf_n, idf_a)
              for nt, at, nl, al in zip(d_name_list, d_addr_list, name_lens, addr_lens)]
    out = cands.copy()
    out["bm25_score"] = scores
    return out.sort_values("bm25_score", ascending=False).head(topk).reset_index(drop=True)
