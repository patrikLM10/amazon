"""Elasticsearch S1 -> S2 retrieval baseline (separate from custom BM25)."""

from .elasticsearch_search import search_elasticsearch, search_s1_id
from .es_config import (
    CORPUSES,
    INDEX_TEST,
    INDEX_TRAIN,
    build_query,
    default_index,
    get_client,
)

__all__ = [
    "search_elasticsearch",
    "search_s1_id",
    "build_query",
    "get_client",
    "default_index",
    "CORPUSES",
    "INDEX_TRAIN",
    "INDEX_TEST",
]
