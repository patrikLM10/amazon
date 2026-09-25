# S1 -> S2 Top-K Search (BM25)

Standalone demo: given one S1 id, find top-50 similar records in S2.

## Run

```bash
cd student_resource/s1_to_s2_search
python search.py --s1_id S1-925783039 --source train --topk 50
python search.py --s1_id S1-XXXX --source test --topk 50
```

Options:
- `--s2_limit 1000000` : scan only first 1M S2 rows (fast demo, less recall)
- `--all_countries` : skip same-country-first blocking, search everything
- `--chunk 200000` : streaming chunksize for the 5M-row S2 file

## Pipeline

1. Lookup query in `*_source1.tsv` by `entity_id`.
2. **Blocking**: trigram + long-word regex (`str.contains`, vectorised) streams S2 in chunks. Same-country first, fallback to others.
3. **Ranking**: exact Okapi BM25, `3 x name + 1 x address`, over words + char-trigrams. IDF computed on blocked set.
4. Output: console table + `results_<S1ID>.tsv` with `entity_id, business_name, business_address, country, bm25_score`.

Needs only `pandas` + `numpy`.

## Use in another script

```python
from s1_to_s2_search import search_top_k, search_by_text

# just pass the S1 id
query, results = search_top_k("S1-925783039", source="train", topk=50)
# query: dict(entity_id, business_name, business_address, country)
# results: DataFrame(entity_id, business_name, business_address, country, bm25_score)

# or query by raw text (no id)
query, results = search_by_text("Orelee's Barbershop",
                                "1795 Westchester Drive, High Point, NC",
                                country="US", source="train", topk=50)
```

See `example_usage.py`. Full signature:
`search_top_k(s1_id, source="train", topk=50, chunk=200_000, s2_limit=None, all_countries=False, data_dir=None, s1_path=None, s2_path=None, verbose=True, save=False)`.
