-- Runs once, when the Postgres data volume is first created.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS images (
    id         serial PRIMARY KEY,
    path       text UNIQUE NOT NULL,
    category   text,
    caption    text,
    created_at timestamptz DEFAULT now(),
    embedding  vector(512)
);

-- Built up front: pgvector maintains HNSW incrementally on insert, so
-- there is no separate post-ingest indexing step.
CREATE INDEX IF NOT EXISTS images_embedding_hnsw
    ON images USING hnsw (embedding vector_cosine_ops);
