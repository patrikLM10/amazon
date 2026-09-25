#!/usr/bin/env python3
"""BOILERPLATE 1 — Elasticsearch retrieval basics. Copy this file anywhere.

Covers the four calls you will actually reuse:
  A. one S1 id  -> Top-K from S2
  B. raw text    -> Top-K from S2 (no S1 id needed)
  C. one S1 id  -> Top-K from S3  (index S3 first, see below)
  D. many S1 ids -> one DataFrame (batch loop)

Prereqs: Elasticsearch running on localhost:9200 with index ``s2_train`` built:
    python s1_to_s2_search/elasticsearch/elasticsearch_index.py --source train --recreate
For C (S3) also build:
    python s1_to_s2_search/elasticsearch/elasticsearch_index.py --source train --corpus s3 --recreate

Run:  python example_1_retrieval.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))  # .../elasticsearch
_PKG_DIR = os.path.dirname(_HERE)  # .../s1_to_s2_search
_ROOT_DIR = os.path.dirname(_PKG_DIR)  # .../student_resource
for _p in (_HERE, _PKG_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from s1_to_s2_search.elasticsearch import search_elasticsearch, search_s1_id

# ---------------------------------------------------------------- A. S1 id -> S2
query, hits, ms = search_s1_id("S1-965667", source="train", topk=5)
print(f"A. {query['entity_id']} -> {len(hits)} hits in {ms:.0f} ms")
print(hits[["entity_id", "_score"]].to_string(index=False), "\n")

# ---------------------------------------------------------------- B. raw text -> S2
hits, ms = search_elasticsearch(
    "Maure Williams Colombier", "85 Wayne Avenue, Ticonderoga, NY",
    country="US", topk=5)
print(f"B. text query -> {len(hits)} hits in {ms:.0f} ms")
print(hits[["entity_id", "_score"]].to_string(index=False), "\n")

# ---------------------------------------------------------------- C. S1 id -> S3
# query, hits, ms = search_s1_id("S1-965667", source="train", topk=5, corpus="s3")
# print(hits[["entity_id", "country", "_score"]].to_string(index=False))

# ---------------------------------------------------------------- D. batch of ids -> one DataFrame
ids = ["S1-965667", "S1-55344266"]
rows = []
for sid in ids:
    q, h, _ = search_s1_id(sid, source="train", topk=5)
    h = h.copy()
    h["s1_id"] = sid
    rows.append(h)
import pandas as pd
batch = pd.concat(rows, ignore_index=True)
print(f"D. batch: {len(batch)} rows for {len(ids)} queries")
print(batch[["s1_id", "entity_id", "_score"]].to_string(index=False))
