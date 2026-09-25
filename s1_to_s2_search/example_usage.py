"""Example: use the search pipeline inside your own script.

Run:  python example_usage.py
"""

import os
import sys

# Make parent (student_resource/) importable so `import s1_to_s2_search` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if os.path.basename(os.getcwd()) == "s1_to_s2_search"
                else os.getcwd())

from s1_to_s2_search import search_by_text, search_top_k

# 1) By S1 id (you only pass the candidate id)
query, results = search_top_k(
    "S1-151608539",
    source="train",   # or "test"
    topk=50,
    # s2_limit=500_000,  # uncomment for fast demo (partial S2 scan)
    verbose=True,
    save=True,       # True to also write results_<S1ID>.tsv
)

print("\nQUERY:", query)
print(results[["entity_id", "business_name", "business_address", "country", "bm25_score"]]
      .head(10).to_string(index=False))

# 2) By raw text (no id needed)
# q2, r2 = search_by_text("Orelee's Barbershop",
#                         "1795 Westchester Drive, High Point, NC",
#                         country="US", source="train", topk=10, verbose=False)
