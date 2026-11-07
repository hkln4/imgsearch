"""Evaluate text->image retrieval quality and HNSW vs exact search.

Run on the host, not in the API container, so query embeddings use MPS on
Apple Silicon and the recorded hardware is the machine the numbers came from.

    python eval.py   ->  results/eval.json, results/eval.md
"""
import json
import os
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from common import CLIP_MODEL, CLIP_PRETRAINED, connect, device, embed_image, embed_text
from metrics import overlap_at_k, percentile, precision_at_k, r_precision

SEED = 0
KS = (1, 5, 10)
ANN_K = 10
LATENCY_REPEATS = 5
WARMUP_QUERIES = 10
BASELINE_ROUNDTRIPS = 500
DRIFT_SAMPLE = 20
OUT_DIR = Path("results")

CLASS_NAMES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}
QUALITY_TEMPLATE = "a photo of a {}."
# extra phrasings so the index benchmark has 100 queries instead of 10
INDEX_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a close-up photo of a {}.",
    "a photo of the large {}.",
    "a photo of a small {}.",
    "a bright photo of a {}.",
    "a dark photo of a {}.",
    "a cropped photo of a {}.",
    "a low resolution photo of a {}.",
    "a good photo of a {}.",
]

# same query as app.py /search
SEARCH_SQL = """SELECT id, path, category, embedding <=> %s::vector AS dist
                FROM images ORDER BY dist LIMIT %s"""

# The planner picks a seq scan on a table this small, so each path is forced
# explicitly; otherwise both measurements could silently use the same plan.
MODES = {
    "exact": "SET LOCAL enable_indexscan = off",
    "hnsw": "SET LOCAL enable_seqscan = off",
}


def search(conn, mode, emb, k):
    with conn.transaction():
        conn.execute(MODES[mode])
        start = time.perf_counter()
        rows = conn.execute(SEARCH_SQL, (emb, k)).fetchall()
        elapsed = time.perf_counter() - start
    return [r[0] for r in rows], elapsed


def plan_uses_index(conn, emb, mode=None):
    with conn.transaction():
        if mode:
            conn.execute(MODES[mode])
        plan = conn.execute("EXPLAIN " + SEARCH_SQL, (emb, ANN_K)).fetchall()
    return any("images_embedding_hnsw" in line[0] for line in plan)


def evaluate_quality(conn, labels):
    relevant = {wnid: {i for i, c in labels.items() if c == wnid} for wnid in CLASS_NAMES}
    per_class = {}
    for wnid, name in CLASS_NAMES.items():
        ids, _ = search(conn, "exact", embed_text(QUALITY_TEMPLATE.format(name)), len(relevant[wnid]))
        row = {f"p@{k}": precision_at_k(ids, relevant[wnid], k) for k in KS}
        row["r_precision"] = r_precision(ids, relevant[wnid])
        row["n_relevant"] = len(relevant[wnid])
        per_class[name] = row
    metric_names = [f"p@{k}" for k in KS] + ["r_precision"]
    mean = {m: sum(r[m] for r in per_class.values()) / len(per_class) for m in metric_names}
    return {"prompt_template": QUALITY_TEMPLATE, "search": "exact", "per_class": per_class, "mean": mean}


def evaluate_index(conn):
    queries = [embed_text(t.format(n)) for t in INDEX_TEMPLATES for n in CLASS_NAMES.values()]

    assert plan_uses_index(conn, queries[0], "hnsw"), "hnsw mode did not use the HNSW index"
    assert not plan_uses_index(conn, queries[0], "exact"), "exact mode used the HNSW index"

    for emb in queries[:WARMUP_QUERIES]:
        for mode in MODES:
            search(conn, mode, emb, ANN_K)

    overlaps = []
    for emb in queries:
        exact_ids, _ = search(conn, "exact", emb, ANN_K)
        hnsw_ids, _ = search(conn, "hnsw", emb, ANN_K)
        overlaps.append(overlap_at_k(hnsw_ids, exact_ids, ANN_K))

    # interleave modes in a shuffled order so neither benefits from running second
    jobs = [(mode, emb) for _ in range(LATENCY_REPEATS) for emb in queries for mode in MODES]
    random.shuffle(jobs)
    latencies = {mode: [] for mode in MODES}
    for mode, emb in jobs:
        latencies[mode].append(search(conn, mode, emb, ANN_K)[1] * 1000)

    baseline = []
    for _ in range(BASELINE_ROUNDTRIPS):
        start = time.perf_counter()
        conn.execute("SELECT 1").fetchone()
        baseline.append((time.perf_counter() - start) * 1000)

    def summary(ms):
        return {"p50_ms": percentile(ms, 50), "p95_ms": percentile(ms, 95), "n": len(ms)}

    return {
        "k": ANN_K,
        "n_queries": len(queries),
        "ef_search": int(conn.execute("SHOW hnsw.ef_search").fetchone()[0]),
        "index_options": conn.execute(
            "SELECT reloptions FROM pg_class WHERE relname = 'images_embedding_hnsw'"
        ).fetchone()[0] or "defaults (m=16, ef_construction=64)",
        "planner_uses_index_by_default": plan_uses_index(conn, queries[0]),
        "hnsw_recall_vs_exact": {"mean": sum(overlaps) / len(overlaps), "min": min(overlaps)},
        "latency": {mode: summary(ms) for mode, ms in latencies.items()},
        "roundtrip_baseline": summary(baseline),
    }


def embedding_drift(conn):
    """Stored (container, CPU) vs fresh (host) image embeddings of the same files."""
    rows = conn.execute("SELECT path, embedding FROM images ORDER BY id").fetchall()
    cos_gaps, diffs = [], []
    for path, stored in random.sample(rows, DRIFT_SAMPLE):
        fresh, stored = embed_image(path), stored.to_list()
        cos_gaps.append(abs(1 - sum(a * b for a, b in zip(fresh, stored))))
        diffs.append(max(abs(a - b) for a, b in zip(fresh, stored)))
    return {"n": DRIFT_SAMPLE, "max_abs_1_minus_cosine": max(cos_gaps), "max_abs_component_diff": max(diffs)}


def hardware(conn):
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    if platform.system() == "Darwin":
        cpu = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        ram = int(sh(["sysctl", "-n", "hw.memsize"]) or 0)
    else:
        cpu = platform.processor()
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return {
        "platform": platform.platform(),
        "cpu": cpu,
        "ram_gb": round(ram / 2**30, 1),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "query_embedding_device": device,
        "postgres": conn.execute("SHOW server_version").fetchone()[0],
        "pgvector": conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0],
        # the db runs inside Docker, so its resources can differ from the host's
        "docker_vm": sh(["docker", "info", "--format", "{{.NCPU}} CPUs, {{.MemTotal}} bytes"]),
    }


def to_markdown(r):
    hw, q, ix = r["hardware"], r["quality"], r["index"]
    lines = [
        "# Evaluation results",
        "",
        f"Run {r['timestamp']} on {hw['cpu']}, {hw['ram_gb']} GB RAM; "
        f"queries embedded on `{hw['query_embedding_device']}`; "
        f"Postgres {hw['postgres']} + pgvector {hw['pgvector']} in Docker ({hw['docker_vm']}).",
        "",
        f"Corpus: {r['corpus']['images']} images, {r['corpus']['classes']} classes, "
        f"{r['corpus']['dims']}-dim CLIP {r['model']}.",
        "",
        f"## Text-to-image retrieval (exact search, prompt `{q['prompt_template']}`)",
        "",
        "| Class | P@1 | P@5 | P@10 | R-precision |",
        "|---|---|---|---|---|",
    ]
    for name, m in q["per_class"].items():
        lines.append(f"| {name} | {m['p@1']:.2f} | {m['p@5']:.2f} | {m['p@10']:.2f} | {m['r_precision']:.2f} |")
    m = q["mean"]
    lines.append(f"| **Mean** | **{m['p@1']:.2f}** | **{m['p@5']:.2f}** | **{m['p@10']:.2f}** | **{m['r_precision']:.2f}** |")
    rec = ix["hnsw_recall_vs_exact"]
    lines += [
        "",
        f"## HNSW vs exact search (k={ix['k']}, {ix['n_queries']} text queries, ef_search={ix['ef_search']})",
        "",
        f"HNSW recall@{ix['k']} vs exact: mean {rec['mean']:.3f}, min {rec['min']:.3f}.",
        f"Index options: {ix['index_options']}. "
        f"Planner uses the index without forcing: {ix['planner_uses_index_by_default']}.",
        "",
        "| Query | p50 (ms) | p95 (ms) | samples |",
        "|---|---|---|---|",
    ]
    for label, s in [("exact (seq scan)", ix["latency"]["exact"]),
                     ("HNSW (index scan)", ix["latency"]["hnsw"]),
                     ("round-trip baseline (`SELECT 1`)", ix["roundtrip_baseline"])]:
        lines.append(f"| {label} | {s['p50_ms']:.2f} | {s['p95_ms']:.2f} | {s['n']} |")
    d = r["embedding_drift"]
    lines += [
        "",
        "Latency is client-side wall time from the host, so it includes the "
        "host-to-Docker round trip shown by the baseline row.",
        "",
        f"Embedding drift (container CPU vs host {hw['query_embedding_device']}, {d['n']} images): "
        f"max |1 - cosine| {d['max_abs_1_minus_cosine']:.1e}, "
        f"max per-component difference {d['max_abs_component_diff']:.1e}.",
        "",
    ]
    return "\n".join(lines)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    conn = connect()
    conn.autocommit = True  # each search runs in its own transaction so SET LOCAL cannot leak

    labels = dict(conn.execute("SELECT id, category FROM images").fetchall())
    missing = set(CLASS_NAMES) - set(labels.values())
    if missing:
        raise SystemExit(f"classes not ingested: {sorted(missing)}; run ingest.py first")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "model": f"{CLIP_MODEL} ({CLIP_PRETRAINED})",
        "corpus": {"images": len(labels), "classes": len(set(labels.values())), "dims": 512},
        "hardware": hardware(conn),
        "quality": evaluate_quality(conn, labels),
        "index": evaluate_index(conn),
        "embedding_drift": embedding_drift(conn),
    }
    conn.close()

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "eval.json").write_text(json.dumps(results, indent=2) + "\n")
    (OUT_DIR / "eval.md").write_text(to_markdown(results))
    print(to_markdown(results))


if __name__ == "__main__":
    main()
