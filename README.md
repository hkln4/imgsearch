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
  subset of ImageNet)

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
├── init.sql            # schema + HNSW index, run on first DB start
├── docker-compose.yml  # db (pgvector) + api
├── Dockerfile          # API image (CPU-only torch)
├── requirements.txt
└── data/               # Imagenette images (downloaded, not committed)
```

## Notes & limitations

- Search only returns images that exist in the collection — a semantic search
  always returns *something*, so out-of-domain queries return the least-bad match
  rather than "no result".
- Ingestion currently loads a subset of Imagenette (configurable in `ingest.py`
  via `CATEGORIES_LIMIT` / `PER_CATEGORY_LIMIT`).
- Captioning quality is bounded by the small LLaVA-7B model; it identifies main
  subjects reliably but can hallucinate background details.