from pathlib import Path

from common import connect, embed_image

DATA_DIR = "data/imagenette2-160/train"
CATEGORIES_LIMIT = 10     # all 10 Imagenette classes (eval ground truth)
PER_CATEGORY_LIMIT = 100  # max images per category

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
conn = connect()

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