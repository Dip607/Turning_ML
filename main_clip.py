import os
import torch
import clip
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


ORIGINAL_DIR = "data/originals"
COPY_DIR = "data/copies"
SIMILARITY_THRESHOLD = 0.92
HASH_THRESHOLD = 10

device = "cuda" if torch.cuda.is_available() else "cpu"


model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()


def load_image(path):
    image = Image.open(path).convert("RGB")
    return preprocess(image).unsqueeze(0).to(device)

def get_embedding(img_tensor):
    with torch.no_grad():
        embedding = model.encode_image(img_tensor)
        embedding = embedding / embedding.norm(dim=1, keepdim=True)
    return embedding.squeeze().cpu().numpy()

def get_phash(path):
    image = Image.open(path).convert("RGB")
    return imagehash.phash(image)


original_embeddings = {}
original_hashes = {}

print("Processing original images...")

for img_name in tqdm(os.listdir(ORIGINAL_DIR)):
    img_path = os.path.join(ORIGINAL_DIR, img_name)
    img_tensor = load_image(img_path)

    original_embeddings[img_name] = get_embedding(img_tensor)
    original_hashes[img_name] = get_phash(img_path)


y_true = []
y_pred = []

print("\nChecking duplicate images...")

for copy_img in tqdm(os.listdir(COPY_DIR)):
    copy_path = os.path.join(COPY_DIR, copy_img)
    copy_tensor = load_image(copy_path)

    copy_embedding = get_embedding(copy_tensor)
    copy_hash = get_phash(copy_path)

    is_duplicate = False

    for orig_img in original_embeddings:
        
        if (copy_hash - original_hashes[orig_img]) > HASH_THRESHOLD:
            continue

        
        similarity = np.dot(copy_embedding, original_embeddings[orig_img])
        if similarity >= SIMILARITY_THRESHOLD:
            is_duplicate = True
            break

    y_true.append(1)
    y_pred.append(1 if is_duplicate else 0)


precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n===== RESULTS =====")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
