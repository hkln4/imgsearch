import open_clip
import torch
import psycopg
from pgvector.psycopg import register_vector

DB_DSN = "host=localhost dbname=imgsearch user=postgres password=devpass"

#img categories
LABELS = {
    "n01440764": "tench (fish)",
    "n02102040": "english springer (dog)",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
}

device = "mps" if torch.backends.mps.is_available() else "cpu"
model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

def embed_text(query):
    tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb /= emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).cpu().tolist()

# db 
conn = psycopg.connect(DB_DSN)
register_vector(conn)

def search(query, k=5):
    emb = embed_text(query)
    rows = conn.execute(
        """SELECT path, category, embedding <=> %s::vector AS dist
           FROM images
           ORDER BY dist
           LIMIT %s""",
        (emb, k),
    ).fetchall()

    print(f"\n=== query: '{query}' ===")
    for path, category, dist in rows:
        label = LABELS.get(category, category)
        filename = path.split("/")[-1]
        print(f"  {float(dist):.4f}  [{label:24s}]  {filename}")

# test searches
for q in ["a church", "a dog", "a chainsaw", "a fish", "a cassette player"]:
    search(q)

conn.close()