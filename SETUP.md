# Setup & Run Guide — S1 → S2/S3 Entity Resolution (Elasticsearch baseline)

> Run every command from the **repo root** (the folder containing `s1_to_s2_search/`
> and `dataset/`). Estimated total setup time: ~30 min (mostly downloads + indexing).

## 0. What you need

| Requirement | Notes |
|---|---|
| Python 3.10+ | `python --version` |
| 8 GB+ RAM, ~15 GB free disk | ES heap uses 2 GB; indexes ~3 GB |
| Internet | ES download (~470 MB), pip packages |
| Dataset files (NOT in git) | `dataset/train/` + `dataset/test/` TSVs from the challenge bundle |

## 1. Clone and Python environment

```bash
git clone https://github.com/patrikLM10/amazon.git
cd amazon
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r s1_to_s2_search/requirements.txt -r s1_to_s2_search/elasticsearch/requirements-es.txt
```

## 2. Place the dataset

The repo ships **without** `dataset/` (too large for GitHub). Copy the challenge
files so the tree looks like this:

```
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

The no-match list used by the threshold study is derived, not distributed —
regenerate it with:

```bash
python -c "import pandas as pd; df=pd.read_csv('dataset/train/train_ground_truth.tsv',sep='\t',dtype=str,keep_default_na=False); df[df['matched_entity_ids'].str.strip()=='' ].to_csv('dataset/train/s1_no_s2_s3_ground_truth_matches.tsv',sep='\t',index=False)"
```

## 3. Install Elasticsearch 8.18.1 (pinned — must match the client)

Download **8.18.1** for your OS from
https://www.elastic.co/downloads/past-releases/elasticsearch-8-18-1
(or `https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.18.1-<os>.zip|.tar.gz`):

| OS | File |
|---|---|
| Windows x86_64 | `elasticsearch-8.18.1-windows-x86_64.zip` |
| Linux x86_64 | `elasticsearch-8.18.1-linux-x86_64.tar.gz` |
| macOS ARM64 | `elasticsearch-8.18.1-darwin-aarch64.tar.gz` |

Unzip anywhere (e.g. next to the repo, **not** inside it).

## 4. Start the server (once per boot)

```bash
# Windows (PowerShell):
$env:ES_JAVA_OPTS="-Xms2g -Xmx2g"
Start-Process ".\elasticsearch-8.18.1\bin\elasticsearch.bat" -ArgumentList "-E discovery.type=single-node","-E xpack.security.enabled=false" -WindowStyle Hidden

# Linux/macOS:
ES_JAVA_OPTS="-Xms2g -Xmx2g" ./elasticsearch-8.18.1/bin/elasticsearch -E discovery.type=single-node -E xpack.security.enabled=false
```

Wait ~45 s, then verify (expect `200` and version `8.18.1`):

```bash
curl http://localhost:9200
```

## 5. Build the indexes (once per dataset, ~6 min each)

```bash
python s1_to_s2_search/elasticsearch/elasticsearch_index.py --source train --recreate
python s1_to_s2_search/elasticsearch/elasticsearch_index.py --source train --corpus s3 --recreate
curl http://localhost:9200/_cat/indices/s2_train,s3_train?v
# expect s2_train ≈ 5,034,616 docs, s3_train ≈ 5,285,603 docs, both green
```

## 6. Run searches

```bash
# one S1 -> top-50 from S3 (omit --corpus for S2)
python s1_to_s2_search/elasticsearch/elasticsearch_search.py --s1_id S1-965667 --corpus s3 --topk 50

# batch: many ids -> one .txt per id, both buckets
New-Item -ItemType Directory -Force -Path "s1_to_s2_search/output_s2","s1_to_s2_search/output_s3" | Out-Null
@("S1-965667","S1-55344266") | ForEach-Object {
  python -X utf8 s1_to_s2_search/elasticsearch/elasticsearch_search.py --s1_id $_ --corpus s2 --topk 50 > "s1_to_s2_search/output_s2/$_.txt"
  python -X utf8 s1_to_s2_search/elasticsearch/elasticsearch_search.py --s1_id $_ --corpus s3 --topk 50 > "s1_to_s2_search/output_s3/$_.txt"
}
```

### Worked example: 20 ids, both buckets, then tick-verify against ground truth

```powershell
$baseDir = "s1_to_s2_search/S1-357837082_to_S1-557991692"
New-Item -ItemType Directory -Force -Path "$baseDir/s2", "$baseDir/s3" | Out-Null

@("S1-357837082", "S1-292603366", "S1-108246457", "S1-727766958", "S1-719571188", "S1-646393119", "S1-723966103", "S1-76693096", "S1-967724835", "S1-288276341", "S1-887125153", "S1-304029686", "S1-166609944", "S1-178800424", "S1-924630389", "S1-81771195", "S1-403692077", "S1-745867296", "S1-671241040", "S1-557991692") | ForEach-Object {
  python -X utf8 s1_to_s2_search/elasticsearch/elasticsearch_search.py --s1_id $_ --corpus s2 --topk 50 > "$baseDir/s2/$_.txt"
  python -X utf8 s1_to_s2_search/elasticsearch/elasticsearch_search.py --s1_id $_ --corpus s3 --topk 50 > "$baseDir/s3/$_.txt"
}
```

Tick-verify every file against ground truth (each bucket matched to its own
ground-truth IDs; ✅ is appended to hit lines, files edited in place):

```bash
python s1_to_s2_search/elasticsearch/tick_folder.py "s1_to_s2_search/S1-357837082_to_S1-557991692/s2" --corpus s2
python s1_to_s2_search/elasticsearch/tick_folder.py "s1_to_s2_search/S1-357837082_to_S1-557991692/s3" --corpus s3
# each run ends with a tally, e.g. "Tally: 12/13 HIT (92.3%)"
# single file instead: tick_matches.py <path> [--corpus s2|s3]
```

## 7. Evaluate / tick / study

```bash
# tick ✅ on ground-truth hits in a result file / folder (bucket auto-detected)
python s1_to_s2_search/elasticsearch/tick_matches.py s1_to_s2_search/output_s3/S1-965667.txt
python s1_to_s2_search/elasticsearch/tick_folder.py s1_to_s2_search/output_s3 --corpus s3

# recall + latency eval (first 20 GT rows)
python s1_to_s2_search/elasticsearch/verify_elasticsearch.py --source train --topk 50 --limit 20

# no-match score distribution study (500 queries, ~5 min)
python s1_to_s2_search/elasticsearch/nomatch_study/nomatch_score_study.py

# copy-paste boilerplates
python s1_to_s2_search/elasticsearch/example_1_retrieval.py
pip install sentence-transformers  # only for this one:
python s1_to_s2_search/elasticsearch/example_2_rerank.py --s1_id S1-965667 --topk 20 --final-k 5
```

## 8. Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `Failed to connect to localhost:9200` | Server not running → step 4. |
| `No such file .../s1_to_s2_search/...` doubled path | Run from repo root, don't repeat folder names. |
| `security_exception / missing authentication` | Restarted without `-E xpack.security.enabled=false` → stop node, start with the exact step-4 command. |
| ` port 9200 already in use` | Old node still running → reuse it (check `curl`), or kill the `java`/`elasticsearch` process first. |
| `elasticsearch-py version conflict` | Client major must equal server major (8.x ↔ 8.x) → `pip install "elasticsearch==8.18.1"`. |
| `ModuleNotFoundError: s1_to_s2_search` | Wrong working directory → `cd` to repo root. Do not `pip install` the project; scripts fix `sys.path` themselves. |
| `S2 file not found` / `Ground-truth file not found` | `dataset/` misplaced → step 2 (exact tree). |
| First query slow (~0.5–1 s), rest ~50 ms | Normal (connection warm-up). |
| Out of memory during indexing | Close other apps or lower heap to `-Xms1g -Xmx1g` (slower indexing). |
