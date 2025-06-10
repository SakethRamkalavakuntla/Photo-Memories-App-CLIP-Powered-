import gradio as gr
import numpy as np
from PIL import Image
import faiss
import os
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel

# Suppress OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Get image embedding
def get_image_embedding(image: Image.Image) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()

# Get text embedding
def get_text_embedding(text: str) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()

# Build FAISS index
def build_index(images: list) -> tuple:
    index = faiss.IndexFlatL2(512)
    embeddings = []
    for image in images:
        emb = get_image_embedding(image)
        embeddings.append(emb[0])
    embeddings_np = np.stack(embeddings)
    index.add(embeddings_np)
    return index, embeddings_np

# Generate slideshow video
def generate_video(images: list, indices: list, output_path: str):
    video_frames = []
    max_w, max_h = 0, 0

    for i in indices:
        img = np.array(images[i])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video_frames.extend([img] * 100)  # 5 seconds per image @ 20 fps

        h, w, _ = img.shape
        max_w = max(max_w, w)
        max_h = max(max_h, h)

    if not video_frames:
        print("❌ No video frames created.")
        return False

    final_frames = []
    for frame in video_frames:
        h, w, _ = frame.shape
        top = (max_h - h) // 2
        bottom = max_h - h - top
        left = (max_w - w) // 2
        right = max_w - w - left
        padded = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        final_frames.append(padded)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (max_w, max_h))
    for frame in final_frames:
        out.write(frame)
    out.release()
    return True

# Main function
def create_memory_video(image_files, text_query):
    if not image_files or not text_query.strip():
        return None

    try:
        pil_images = [Image.open(file.name).convert("RGB") for file in image_files if file is not None]

        if len(pil_images) == 0:
            print("❌ No valid images found.")
            return None

        index, _ = build_index(pil_images)
        query_emb = get_text_embedding(text_query)
        _, I = index.search(query_emb, min(3, len(pil_images)))

        video_path = "memories_output.mp4"
        success = generate_video(pil_images, I[0], video_path)
        return video_path if success else None

    except Exception as e:
        print("❌ Error in video generation:", str(e))
        return None

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📸 Photo Memories App with CLIP")
    gr.Markdown("Upload multiple images and describe the theme to generate a video memory!")

    with gr.Row():
        image_input = gr.File(file_types=["image"], label="Upload Images", file_count="multiple")
        text_input = gr.Textbox(label="Enter memory theme (e.g., beach, family, night)", placeholder="e.g. sunset at the beach")

    btn = gr.Button("🎞️ Generate Slideshow")
    output_video = gr.Video(label="Your Memory Slideshow")

    btn.click(fn=create_memory_video, inputs=[image_input, text_input], outputs=output_video)

# Launch app
demo.launch()
