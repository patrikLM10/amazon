"""Shared Elasticsearch configuration: connection, index mapping, query builder.

No retrieval behavior of the existing custom BM25 files is touched.
"""

import os

ES_HOST = "http://localhost:9200"
INDEX_TRAIN = "s2_train"
INDEX_TEST = "s2_test"

# Experiment A (lexical baseline) searches business_name / business_address.
# Experiment B (ngram variant) searches business_name.ngram / business_address.ngram.
# Both field families are indexed once so A/B need no reindexing.
INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 2,
        "number_of_replicas": 0,
        "refresh_interval": "30s",
        "analysis": {
            "analyzer": {
                "er_text": {
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding"],
                },
                "er_ngram": {
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding", "er_ngram_3_4"],
                },
            },
            "filter": {
                "er_ngram_3_4": {"type": "ngram", "min_gram": 3, "max_gram": 4},
            },
        },
    },
    "mappings": {
        "properties": {
            "entity_id": {"type": "keyword"},
            "business_name": {
                "type": "text",
                "analyzer": "er_text",
                "fields": {"ngram": {"type": "text", "analyzer": "er_ngram"}},
            },
            "business_address": {
                "type": "text",
                "analyzer": "er_text",
                "fields": {"ngram": {"type": "text", "analyzer": "er_ngram"}},
            },
            "country": {"type": "keyword"},
        }
    },
}

# Mirrors the custom system's name x3 / address x1 emphasis (lexical analogue,
# not a reproduction of the custom score).
NAME_BOOST = 3.0
ADDRESS_BOOST = 1.0


def build_query(business_name, business_address, country, topk=50, mode="lexical"):
    """Country-filtered multi-field lexical query, name-boosted."""
    suffix = "" if mode == "lexical" else ".ngram"
    return {
        "size": topk,
        "query": {
            "bool": {
                "filter": [{"term": {"country": country}}],
                "must": {
                    "bool": {
                        "should": [
                            {"match": {f"business_name{suffix}": {
                                "query": business_name or "", "boost": NAME_BOOST}}},
                            {"match": {f"business_address{suffix}": {
                                "query": business_address or "", "boost": ADDRESS_BOOST}}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
            }
        },
    }


def _pip_module(fullname):
    """Import from the installed ``elasticsearch`` pip package.

    This subfolder is itself named ``elasticsearch``, so once its parent
    directory is on ``sys.path`` a plain ``import elasticsearch`` would
    resolve to this folder instead of the pip client. Temporarily hide the
    shadowing path entries while importing.
    """
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    hidden = {os.path.abspath(here), os.path.abspath(parent)}
    saved = sys.path
    sys.path = [p for p in saved
                if not p or os.path.abspath(p) not in hidden]
    try:
        return __import__(fullname, fromlist=["*"])
    finally:
        sys.path = saved


def get_client(host=ES_HOST, timeout=60):
    es_mod = _pip_module("elasticsearch")
    return es_mod.Elasticsearch(host, request_timeout=timeout)


def streaming_bulk_iter(*args, **kwargs):
    helpers = _pip_module("elasticsearch.helpers")
    return helpers.streaming_bulk(*args, **kwargs)


CORPUSES = ("s2", "s3")


def default_index(source="train", corpus="s2"):
    """Index-name convention: <corpus>_<source>, e.g. s3_train."""
    if corpus not in CORPUSES:
        raise ValueError(f"corpus must be one of {CORPUSES}")
    return f"{corpus}_{source}"
