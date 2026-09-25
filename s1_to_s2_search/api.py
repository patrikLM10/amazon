"""High-level API for S1 -> S2 BM25 search.

Use from any other script:

    from s1_to_s2_search import search_top_k
    query, results = search_top_k("S1-925783039", source="train", topk=50)

    # or search by raw text (no S1 id needed):
    from s1_to_s2_search import search_by_text
    query, results = search_by_text("Orelee's Barbershop",
                                    "1795 Westchester Drive, High Point, NC",
                                    country="US", source="train", topk=50)

Returns:
    query   : dict(entity_id, business_name, business_address, country)
    results : pandas DataFrame(entity_id, business_name, business_address,
                               country, bm25_score) sorted desc.
"""

import os

import pandas as pd

try:  # package import: from s1_to_s2_search import ...
    from .bm25 import blocking_terms, bm25_field_weighted, char_trigrams, compute_idf, doc_tokens, normalize, word_tokens
except ImportError:  # direct script run inside the folder
    from bm25 import blocking_terms, bm25_field_weighted, char_trigrams, compute_idf, doc_tokens, normalize, word_tokens

try:
    from .search_core import find_s1, rank_candidates, resolve_paths, stream_block
except ImportError:
    from search_core import find_s1, rank_candidates, resolve_paths, stream_block

__all__ = ["search_top_k", "search_by_text"]


def _run_query(query, s2_path, topk=50, chunk=200_000, s2_limit=None,
               all_countries=False, verbose=True):
    _, qn_norm, qa_norm = doc_tokens(query["business_name"], query["business_address"])
    pattern, _, _ = blocking_terms(qn_norm, qa_norm)

    same_only = not all_countries
    cands, scanned = stream_block(s2_path, pattern, query["country"],
                                  same_country_only=same_only,
                                  chunksize=chunk, s2_limit=s2_limit)
    if verbose:
        print(f"scanned S2 rows: {scanned}, blocked candidates: {len(cands)}")

    if same_only and len(cands) < topk * 2:
        if verbose:
            print("too few same-country candidates, rescanning other countries ...")
        cands2, scanned2 = stream_block(s2_path, pattern, query["country"],
                                        same_country_only=False,
                                        chunksize=chunk, s2_limit=s2_limit)
        cands = pd.concat([cands, cands2], ignore_index=True).drop_duplicates(subset=["entity_id"])
        scanned = max(scanned, scanned2)
        if verbose:
            print(f"after fallback: {len(cands)} candidates")

    if len(cands) == 0:
        return query, cands, scanned

    ranked = rank_candidates(query, cands, topk)
    return query, ranked, scanned


def search_top_k(s1_id, source="train", topk=50, chunk=200_000,
                 s2_limit=None, all_countries=False, data_dir=None,
                 s1_path=None, s2_path=None, verbose=True, save=False):
    """Find top-K S2 records similar to one S1 id.

    Args:
        s1_id: e.g. "S1-925783039".
        source: "train" | "test" (used to locate dataset files unless
            s1_path/s2_path given).
        topk: number of S2 hits to return.
        chunk: S2 streaming chunksize.
        s2_limit: scan only first N S2 rows (fast demo) or None for full scan.
        all_countries: True to skip same-country-first blocking.
        data_dir, s1_path, s2_path: path overrides.
        verbose: print progress.
        save: if True, write results_<S1ID>.tsv next to this file.

    Returns:
        (query_dict, results_df). Raises ValueError if s1_id not found.
    """
    s1_p, s2_p = resolve_paths(source, data_dir, s1_path, s2_path)
    query = find_s1(s1_p, s1_id)
    if query is None:
        raise ValueError(f"{s1_id} not found in {s1_p} (wrong --source?)")
    if verbose:
        print(f"QUERY {query['entity_id']}: {query['business_name']} | "
              f"{query['business_address']} [{query['country']}]")
    query, ranked, _ = _run_query(query, s2_p, topk, chunk, s2_limit, all_countries, verbose)
    if save:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_{s1_id}.tsv")
        ranked.to_csv(out, sep="\t", index=False, encoding="utf-8")
        if verbose:
            print(f"Saved {len(ranked)} rows -> {out}")
    return query, ranked


def search_by_text(name, address="", country="US", source="train", topk=50,
                   chunk=200_000, s2_limit=None, all_countries=False,
                   data_dir=None, s2_path=None, verbose=True):
    """Same as search_top_k but query is raw text, no S1 id needed."""
    _, s2_p = resolve_paths(source, data_dir, None, s2_path)
    query = {"entity_id": "TEXT-QUERY", "business_name": name,
             "business_address": address, "country": country}
    query, ranked, _ = _run_query(query, s2_p, topk, chunk, s2_limit, all_countries, verbose)
    return query, ranked
