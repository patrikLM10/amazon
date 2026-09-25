# Elasticsearch Retrieval Baseline — S1 → S2 Business Entity Resolution

**Author:** Jishnu Pal · **Date:** September 2026 · **Status:** Implemented, measured, reproducible

---

## 1. Objective

Establish an independent, indexed retrieval baseline for the S1 → S2 entity-resolution
task, so the existing custom Python BM25 pipeline can be compared against an
industry-standard engine on identical data, identical queries, and identical
ground-truth rules. The goal of this document is the baseline itself — not to
replace the current system yet.

**Question under test:** can Elasticsearch retrieve the correct S2 entity, at what
recall, and at what latency — and how does that compare with custom BM25?

---

## 2. Architecture

```
                        OFFLINE (once)
                        ──────────────
train_source2.tsv (5.03M rows)
        │  streaming bulk index (2000 docs/batch)
        ▼
Elasticsearch 8.18.1 (local node)
  inverted index + text analysis  ◄── mapping defined once in es_config.py
        │
        │               PER QUERY
        │               ─────────
S1 id ──► S1 lookup ──► query JSON ──► Elasticsearch ──► Top-K S2 + scores
(name/address/country)  (country filter   (filter → BM25
                         + boosted match)   over postings)
                                                  │
                                                  ▼
                                   verify_elasticsearch.py
                                   (GT membership → HIT/MISS + rank
                                    → Recall@K + latency stats)
```

The custom BM25 files (`api.py`, `search_core.py`, `bm25.py`, `search.py`,
`verify_recall.py`) were **not modified**. All Elasticsearch code lives in
`s1_to_s2_search/elasticsearch/` (`es_config.py`, `elasticsearch_index.py`,
`elasticsearch_search.py`, `verify_elasticsearch.py`).

---

## 3. Environment

| Item | Value |
|---|---|
| Engine | Official Elasticsearch **8.18.1** (zip distribution, bundled JDK — no system Java needed) |
| Client | `elasticsearch==8.18.1` (Python) |
| Topology | Single node, `localhost:9200`, security disabled (local dev only), 2 GB heap |
| Host | Windows 11 laptop, 24 GB RAM |
| Index name | `s2_train` |

No Docker/WSL was available in this environment, so the native Windows
distribution was used. Nothing about the retrieval logic depends on that choice.

---

## 4. Index design

**Document** (one per S2 row, `_id = entity_id`):

```json
{"entity_id": "S2-…", "business_name": "…", "business_address": "…", "country": "US"}
```

**Mapping** (`es_config.INDEX_SETTINGS`, 2 shards / 0 replicas):

| Field | Type | Analysis |
|---|---|---|
| `entity_id` | `keyword` | exact |
| `business_name` | `text` (`er_text`) + `.ngram` subfield | see below |
| `business_address` | `text` (`er_text`) + `.ngram` subfield | see below |
| `country` | `keyword` | exact (filterable, not scored) |

**Analyzers:**
- `er_text` — standard tokenizer → lowercase → accent folding. The lexical
  baseline (Experiment A). Roughly mirrors the custom pipeline's normalization
  using ES-native components rather than copying its Python tokenizer.
- `er_ngram` — same, plus 3–4 character n-grams, exposed as `.ngram`
  subfields. Indexed once so the n-gram variant (Experiment B) can run with
  **no reindexing** (built, not yet evaluated).

**Integrity audit:** a streaming check confirmed all 5,034,616 S2 `entity_id`s
are unique (zero duplicates/empties), and the indexed doc count matches the TSV
row count exactly — so no silent bulk-overwrite occurred.

---

## 5. Indexing benchmark (measured)

| Metric | Value |
|---|---|
| S2 documents indexed | **5,034,616** |
| Indexing time | **339.5 s** |
| Throughput | **≈14,831 docs/s** |
| Bulk failures | **0** |

```
python s1_to_s2_search/elasticsearch/elasticsearch_index.py --source train --recreate
```

---

## 6. Retrieval logic

For each S1 (name, address, country), `build_query()` issues **one** ES request —
no TSV scan, no Python-side candidate handling:

```json
{"size": 50, "query": {"bool": {
  "filter": [{"term": {"country": "<S1 country>"}}],
  "must": {"bool": {"should": [
    {"match": {"business_name":    {"query": "<name>",    "boost": 3.0}}},
    {"match": {"business_address": {"query": "<address>", "boost": 1.0}}}
  ], "minimum_should_match": 1}}}}
```

Execution inside the engine: the `country` keyword filter narrows the space via
the index → query text is analyzed with the same analyzer as the field →
posting lists are walked → **Lucene BM25** scores each candidate (TF × IDF ×
length-norm, × field boosts) → a heap retains only the Top-K. Per-query work is
proportional to matching postings, never to corpus size. The 3:1 name/address
emphasis mirrors the custom system as a lexical analogue — it does not claim to
reproduce its score.

Each call returns `(entity_id, business_name, business_address, country, _score)`
in rank order plus the HTTP round-trip latency.

```
python s1_to_s2_search/elasticsearch/elasticsearch_search.py --s1_id S1-965667 --topk 50
```

---

## 7. Evaluation methodology (fairness)

- **Same ground truth**: `dataset/train/train_ground_truth.tsv`
  (`source1_entity_id`, `matched_entity_ids`).
- **Same rules as `verify_recall.py`**: only `S2-*` IDs count; S3-only,
  singleton, and malformed rows are skipped with separate counters; HIT = any
  valid GT S2 ID in Top-K; rank = position of first match.
- **Same query population**: GT file order, `--limit N` takes the first N rows
  for both systems.
- **No tuning on the eval set**: one reasonable baseline config, reported as-is.
- Per-query TSV (`s1_id, country, ground_truth_s2_ids, hit, match_rank,
  num_retrieved, retrieved_s2_ids, query_*, latency_ms, status`) makes every
  success and failure inspectable; single-query errors never abort a run.

```
python s1_to_s2_search/elasticsearch/verify_elasticsearch.py --source train --topk 50 --limit 100
```

---

## 8. Results (measured)

### 8.1 Elasticsearch lexical baseline — first 100 GT rows → 86 evaluated

| Metric | Value |
|---|---|
| Recall@1 | 80.23% |
| Recall@5 | 91.86% |
| Recall@10 | 91.86% |
| Recall@20 | 93.02% |
| **Recall@50** | **94.19% (81/86)** |
| Latency mean / median / p95 | 73.6 ms / 54.9 ms / 169.7 ms |
| Retrieval errors | 0 (14 skipped: no S2 GT) |

### 8.2 Head-to-head on identical queries (S1-965667, S1-55344266, S1-343815751)

| Metric | Custom BM25 | Elasticsearch |
|---|---|---|
| Recall@50 (n=3) | 100% (3/3, all rank 1) | 100% (3/3, all rank 1) |
| Latency per query | ≈136 s | <0.5 s |

### 8.3 Reading the comparison honestly

- **Quality**: tied on the 3 shared queries; ES holds 94.2% Recall@50 on n=86.
  A full-sample custom-BM25 run (≈86 queries × ~136 s ≈ 3.5 h) was not executed
  in this pass, so no broad quality claim is made beyond what was measured.
- **Latency**: differs by roughly three orders of magnitude (~136 s vs ~74 ms
  mean). The custom cost is dominated by re-scanning 5M rows and re-tokenizing
  ~500k Python-side candidates per query (see `profile_search.py` measurements);
  the ES cost is one index lookup.
- Note: end-to-end harness time per query includes a ~5 s S1 TSV lookup that is
  identical for both evaluators and is **excluded** from the latency figures above.

---

## 9. Limitations & next steps

1. Custom BM25 at n=86 is pending (long runtime) — needed before any final
   quality verdict.
2. Experiment B (`.ngram` fields, typo-oriented) is indexed but unevaluated.
3. Single-machine, single-node numbers; production latency would differ.
4. Failure analysis on the 5 ES misses is the natural next step before any
   reranker design, since Recall@K here is the reranker's ceiling.

---

## 10. Reproducibility

```bash
# 1. start the engine (once per boot)
elasticsearch.bat -E discovery.type=single-node -E xpack.security.enabled=false

# 2. build the index (once per dataset)
python s1_to_s2_search/elasticsearch/elasticsearch_index.py --source train --recreate

# 3. single query
python s1_to_s2_search/elasticsearch/elasticsearch_search.py --s1_id S1-965667 --topk 50

# 4. evaluate (library use also available)
python s1_to_s2_search/elasticsearch/verify_elasticsearch.py --source train --topk 50 --limit 100
from s1_to_s2_search.elasticsearch import search_s1_id
```

All commands run from `student_resource/`; scripts also work when invoked from
inside the `elasticsearch/` folder.
