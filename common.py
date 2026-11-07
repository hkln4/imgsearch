import os
import threading

import open_clip
import psycopg
import torch
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from PIL import Image

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# --- model (one definition shared by ingest, api and eval) ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL, pretrained=CLIP_PRETRAINED
)
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

# PyTorch inference on MPS is not thread-safe: concurrent calls from FastAPI's
# thread pool abort the process (SIGABRT). One model call at a time; a CLIP
# forward pass takes milliseconds, so this does not bottleneck the API.
_model_lock = threading.Lock()

def embed_text(query):
    tokens = tokenizer([query])
    with _model_lock, torch.no_grad():
        emb = model.encode_text(tokens.to(device))
        emb /= emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0).cpu().tolist()

def embed_image(path):
    image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)  # CPU work stays outside the lock
    with _model_lock, torch.no_grad():
        emb = model.encode_image(image.to(device))
        emb /= emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0).cpu().tolist()

def connect():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn
