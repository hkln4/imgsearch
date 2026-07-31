import open_clip
import torch
import psycopg
from pathlib import Path
from pgvector.psycopg import register_vector
from PIL import Image

DATA_DIR = "data/imagenette2-160/train"
CATEGORIES_LIMIT = 5      # only ingest first 5 categories
PER_CATEGORY_LIMIT = 100  # max images per category
DB_DSN = "host=localhost dbname=imgsearch user=postgres password=devpass"

# model
device = "mps" if torch.backends.mps.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model = model.to(device).eval()

def embed_image(path):
    image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(image)
        emb /= emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).cpu().tolist()

# image paths
root = Path(DATA_DIR)
categories = sorted([d for d in root.iterdir() if d.is_dir()])[:CATEGORIES_LIMIT]

jobs = []
for cat in categories:
    images = sorted(cat.glob("*.JPEG"))[:PER_CATEGORY_LIMIT]
    for img_path in images:
        jobs.append((str(img_path), cat.name))

print(f"{len(jobs)} images to ingest from {len(categories)} categories")

# ingest
conn = psycopg.connect(DB_DSN)
register_vector(conn)

for i, (img_path, category) in enumerate(jobs, 1):
    emb = embed_image(img_path)
    conn.execute(
        """INSERT INTO images (path, category, embedding)
           VALUES (%s, %s, %s)
           ON CONFLICT (path) DO NOTHING""",
        (img_path, category, emb),
    )
    if i % 50 == 0:
        conn.commit()
        print(f"  {i}/{len(jobs)} done")

conn.commit()
conn.close()
print("ingestion complete")