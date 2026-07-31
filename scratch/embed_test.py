import open_clip, torch
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()

img = preprocess(Image.open("test.jpg")).unsqueeze(0)
text = tokenizer(["a photo of a cat", "a photo of a car"])

with torch.no_grad():
    img_emb = model.encode_image(img)
    txt_emb = model.encode_text(text)
    img_emb /= img_emb.norm(dim=-1, keepdim=True)   # normalizasyon (Konsept 1)
    txt_emb /= txt_emb.norm(dim=-1, keepdim=True)
    sims = (img_emb @ txt_emb.T).squeeze(0)          # cosine benzerlik

print("boyut:", img_emb.shape[-1])          # 512 beklenir
print("benzerlikler:", sims.tolist())       # resme uyan metin daha yüksek çıkmalı