# Image Search

Semantic image search that runs entirely on your own machine. Search a local
image collection **by meaning** (type "a church", get churches) and click any
result to have a local vision-language model describe it — no external APIs, no
cloud inference.

![demo](demo.gif)

## What it does

- **Text-to-image search** — a text query and the images live in the same CLIP
  embedding space, so results are ranked by semantic similarity rather than
  filenames or tags.
- **On-demand captioning** — clicking a result runs a local VLM (LLaVA) on that
  single image and caches the caption in Postgres, so repeat clicks are instant.
- **Fully local** — embeddings, vector search, and captioning all run on the
  local machine; nothing is sent to a third-party service.

## How it works

The system is built as four layers:

| Layer | Responsibility | Tech |
|-------|----------------|------|
| Ingestion | Walk an image folder, embed each image, store it | `open_clip`, PyTorch |
| Storage | Hold vectors + metadata together, indexed for fast search | PostgreSQL + pgvector (HNSW) |
| Retrieval | Embed the query, find nearest neighbours | pgvector cosine (`<=>`) |
| Serving | Search API + web UI + on-demand VLM captioning | FastAPI, LLaVA via Ollama |

Images and text are both encoded with **CLIP (ViT-B-32)** into 512-dimensional
vectors. Those vectors are stored in a pgvector column with an **HNSW index** for
approximate nearest-neighbour search. A query is embedded the same way and matched
against the stored vectors by cosine distance. Captions are generated only when a
user clicks an image, keeping ingestion fast.

## Stack

- **Embeddings:** CLIP (ViT-B-32) via `open-clip-torch`
- **Vector store:** PostgreSQL 17 + pgvector (HNSW, cosine)
- **VLM:** LLaVA (7B) served locally through Ollama
- **API / UI:** FastAPI + a single-file frontend
- **Dataset:** [Imagenette](https://github.com/fastai/imagenette) (a 10-class
  subset of ImageNet; 100 train images per class are ingested)

## Results

Measured on an Apple M2 (16 GB) with `python eval.py`; full per-class table and
hardware details in [`results/eval.md`](results/eval.md).

**Text-to-image retrieval.** One query per class (`"a photo of a {class}."`)
against 1000 images (10 classes × 100), with class labels as ground truth:

| P@1 | P@5 | P@10 | R-precision |
|-----|-----|------|-------------|
| 1.00 | 1.00 | 1.00 | 0.96 |

P@k is saturated, so R-precision (precision over the top 100, i.e. the whole
class) is the informative number; the weakest classes are chain saw (0.87) and
cassette player (0.92).

**HNSW vs exact search.** 100 text queries, k = 10, default index settings:

| | p50 | p95 |
|--|--|--|
| Exact (seq scan) | 5.8 ms | 7.3 ms |
| HNSW | 3.4 ms | 4.7 ms |
| Round trip only (`SELECT 1`) | 0.11 ms | 0.29 ms |

Latency varies by about 0.5 ms between runs. HNSW recall@10 against exact
search is 1.000. With only 1000 vectors this is expected and says little about
HNSW at scale; see limitations.

### Reproducing

With the stack running and images ingested (see [Setup](#setup)):

```bash
pip install -r requirements-dev.txt
python eval.py          # writes results/eval.json and results/eval.md
python -m pytest tests  # metric unit tests, no DB or model needed
```

`eval.py` runs on the host so queries use MPS on Apple Silicon. It forces each
search path with `SET LOCAL` and checks the plan with `EXPLAIN`, because the
planner may pick either plan on its own. Images are embedded in the container
on CPU and queries on the host; the measured difference is negligible
(max |1 − cosine| ≈ 1e-7).

## Design decisions

- **pgvector instead of a dedicated vector DB.** Vectors, labels and cached
  captions live in one Postgres table, so search, filtering and the caption
  cache are plain SQL with no second service to run. A dedicated vector DB
  would pay off at much larger scale or with heavy filtered ANN workloads.
- **HNSW instead of IVFFlat.** HNSW needs no training step, so the index is
  created on the empty table in `init.sql` and maintained on insert; IVFFlat
  needs data before building its lists. The cost is slower inserts and more
  memory, irrelevant at this size. An HNSW scan returns at most
  `hnsw.ef_search` rows, so `/search` raises it to `k` per query.
- **Captions on demand.** CLIP embeds the image itself, so captions are not
  needed for search, only for display. LLaVA-7B takes ~4 s per image on a warm
  M2, so captioning all 1000 images up front would take over an hour. Captions are
  generated on click and cached in Postgres, and search works without Ollama.

## Setup

### Requirements

- Docker with Compose v2
- [Ollama](https://ollama.com) on the host — optional, only needed for captions

### Quickstart

```bash
cp .env.example .env

# dataset (Imagenette, 160px)
curl -L -o imagenette.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
mkdir -p data && tar -xzf imagenette.tgz -C data

docker compose up -d --build                 # Postgres + pgvector, API
docker compose run --rm api python ingest.py  # embed images into Postgres
```

Open **http://localhost:8000** and search by meaning.

`init.sql` creates the schema and the HNSW index when the database volume is
first created. To start from an empty database: `docker compose down -v`.

The database is exposed on host port **5433** so it does not clash with a
locally installed Postgres.

### Captions (optional)

Ollama stays on the host so it can use the host GPU / Apple Silicon. The API
container reaches it at `host.docker.internal:11434` (set via `OLLAMA_HOST` in
`docker-compose.yml`).

```bash
ollama pull llava:7b
ollama serve        # skip if the Ollama app is already running
```

On Linux, Ollama listens on loopback only by default; start it with
`OLLAMA_HOST=0.0.0.0 ollama serve` so the container can reach it.

Search works without Ollama; only the click-to-describe feature needs it.

### Running on the host (without the API container)

Useful for development, and on Apple Silicon it runs CLIP on MPS instead of CPU.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

docker compose up -d db
python ingest.py
uvicorn app:app --reload
```

The host scripts read `DATABASE_URL` from `.env`.

## Project structure

```
.
├── common.py           # CLIP model + DB connection shared by all scripts
├── ingest.py           # ingestion pipeline: folder -> embeddings -> Postgres
├── app.py              # FastAPI server + web UI + on-demand captioning
├── eval.py             # retrieval quality + HNSW vs exact benchmark
├── metrics.py          # metric functions (unit-tested in tests/)
├── results/            # eval output (JSON + markdown)
├── init.sql            # schema + HNSW index, run on first DB start
├── docker-compose.yml  # db (pgvector) + api
├── Dockerfile          # API image (CPU-only torch)
├── requirements.txt
└── data/               # Imagenette images (downloaded, not committed)
```

## Limitations

- **Small corpus.** 1000 vectors is too few to show the HNSW recall/latency
  trade-off: recall is 1.000 and the planner switches to a seq scan from
  k = 28. The latency numbers show relative cost, not behaviour at scale.
- **Easy dataset.** Imagenette's 10 classes are visually distinct and close to
  CLIP's training distribution; the scores do not transfer to fine-grained or
  domain-specific collections.
- **Narrow eval.** One prompt per class and class-level relevance only. Known
  CLIP weaknesses (counting, spatial relations, negation, fine-grained
  distinctions) are not tested here.
- **Always returns something.** Semantic search has no "no result"; an
  out-of-domain query returns the least-bad match.
- **Caption quality** is bounded by LLaVA-7B, which can hallucinate background
  details.
- Code is copied into the image, so after editing it run
  `docker compose up -d --build` before `docker compose run`.
