import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image: Image.Image) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()

def get_text_embedding(text: str) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()
