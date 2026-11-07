# Evaluation results

Run 2026-11-07T14:48:41+00:00 on Apple M2, 16.0 GB RAM; queries embedded on `mps`; Postgres 17.11 (Debian 17.11-1.pgdg12+2) + pgvector 0.8.6 in Docker (8 CPUs, 8217858048 bytes).

Corpus: 1000 images, 10 classes, 512-dim CLIP ViT-B-32 (laion2b_s34b_b79k).

## Text-to-image retrieval (exact search, prompt `a photo of a {}.`)

| Class | P@1 | P@5 | P@10 | R-precision |
|---|---|---|---|---|
| tench | 1.00 | 1.00 | 1.00 | 0.99 |
| English springer | 1.00 | 1.00 | 1.00 | 0.98 |
| cassette player | 1.00 | 1.00 | 1.00 | 0.92 |
| chain saw | 1.00 | 1.00 | 1.00 | 0.87 |
| church | 1.00 | 1.00 | 1.00 | 0.98 |
| French horn | 1.00 | 1.00 | 1.00 | 0.98 |
| garbage truck | 1.00 | 1.00 | 1.00 | 0.99 |
| gas pump | 1.00 | 1.00 | 1.00 | 0.94 |
| golf ball | 1.00 | 1.00 | 1.00 | 0.96 |
| parachute | 1.00 | 1.00 | 1.00 | 0.97 |
| **Mean** | **1.00** | **1.00** | **1.00** | **0.96** |

## HNSW vs exact search (k=10, 100 text queries, ef_search=40)

HNSW recall@10 vs exact: mean 1.000, min 1.000.
Index options: defaults (m=16, ef_construction=64). Planner uses the index without forcing: True.

| Query | p50 (ms) | p95 (ms) | samples |
|---|---|---|---|
| exact (seq scan) | 5.76 | 7.31 | 500 |
| HNSW (index scan) | 3.44 | 4.74 | 500 |
| round-trip baseline (`SELECT 1`) | 0.11 | 0.29 | 500 |

Latency is client-side wall time from the host, so it includes the host-to-Docker round trip shown by the baseline row.

Embedding drift (container CPU vs host mps, 20 images): max |1 - cosine| 1.4e-07, max per-component difference 1.8e-06.
