import open_clip, torch, psycopg
from pgvector.psycopg import register_vector
from PIL import Image

# --- model ---
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()

def embed_image(path):
    x = preprocess(Image.open(path)).unsqueeze(0)
    with torch.no_grad():
        e = model.encode_image(x)
        e /= e.norm(dim=-1, keepdim=True)
    return e.squeeze(0).tolist()

def embed_text(q):
    t = tokenizer([q])
    with torch.no_grad():
        e = model.encode_text(t)
        e /= e.norm(dim=-1, keepdim=True)
    return e.squeeze(0).tolist()

# --- db ---
conn = psycopg.connect("host=localhost dbname=imgsearch user=postgres password=devpass")
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(conn)   # python listesini vector tipine bağlar
conn.execute("""
CREATE TABLE IF NOT EXISTS images (
  id serial PRIMARY KEY,
  path text,
  embedding vector(512)
)""")
conn.commit()

# --- ingest (şimdilik tek resim) ---
for p in ["test.jpg"]:
    conn.execute("INSERT INTO images (path, embedding) VALUES (%s, %s)",
                 (p, embed_image(p)))
conn.commit()

# --- search (metinle) ---
q = embed_text("a cat")   # kendi resmine uygun bir sorgu yaz
rows = conn.execute(
    "SELECT path, embedding <=> %s::vector AS dist FROM images ORDER BY dist LIMIT 5",
    (q,)
).fetchall()

for path, dist in rows:
    print(path, round(float(dist), 4))   # dist küçük = daha yakın


print("\n--- sort test ---")
for q in ["a car", "an SUV", "a cat", "a boat", "a plate of food"]:
    e = embed_text(q)
    dist = conn.execute(
        "SELECT embedding <=> %s::vector AS dist FROM images ORDER BY dist LIMIT 1",
        (e,)
    ).fetchone()[0]
    print(f"{q:18s} {float(dist):.4f}")