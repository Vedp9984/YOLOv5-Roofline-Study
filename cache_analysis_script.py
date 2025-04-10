
import sys
import torch
import pickle

# Load model and images
with open('model_cache.pkl', 'rb') as f:
    model = pickle.load(f)

with open('images_cache.pkl', 'rb') as f:
    images = pickle.load(f)

# Run inference
for img in images:
    with torch.no_grad():
        _ = model(img)
