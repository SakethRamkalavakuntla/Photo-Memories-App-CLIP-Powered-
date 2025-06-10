import gradio as gr
from PIL import Image
from faiss_utils import build_faiss_index
from embedding_utils import get_text_embedding
from video_utils import generate_slideshow
import os

# Suppress OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def create_memory_video(files, text_query):
    if not files or not text_query.strip():
        return None

    pil_images = []
    for file in files:
        try:
            img = Image.open(file.name).convert("RGB")
            pil_images.append(img)
        except Exception as e:
            print(f"⚠️ Skipping {file.name}: {e}")

    if not pil_images:
        return None

    index, _ = build_faiss_index(pil_images)
    query_emb = get_text_embedding(text_query)
    _, I = index.search(query_emb, min(3, len(pil_images)))

    video_path = "memories_output.mp4"
    generate_slideshow(pil_images, I[0], output_path=video_path)
    return video_path

# with gr.Blocks() as demo:
#     gr.Markdown("## 📸 CLIP-Powered Photo Memory Generator")
#     with gr.Row():
#         image_input = gr.File(file_types=["image"], label="Upload Images", file_count="multiple")
#         text_input = gr.Textbox(label="Memory Theme", placeholder="e.g. trip to the mountains")
#     generate_button = gr.Button("🎞️ Generate Slideshow")
#     output = gr.File(label="🎬 Download Video")

#     generate_button.click(fn=create_memory_video, inputs=[image_input, text_input], outputs=output)
# Gradio UI
with gr.Blocks(css=".gradio-container {max-width: 800px; margin: auto;}") as demo:
    gr.Markdown("""
    # 📸 CLIP Photo Memories Generator
    Upload your favorite images and describe a memory or moment.
    The app will find the most relevant ones and turn them into a short video 🎞️
    """)
    
    with gr.Group():
        with gr.Column():
            image_input = gr.File(
                file_types=["image"], 
                label="🖼️ Upload multiple images", 
                file_count="multiple"
            )
            text_input = gr.Textbox(
                label="📝 Describe the theme (e.g., beach sunset, birthday party)",
                placeholder="Enter memory description here..."
            )
            btn = gr.Button("✨ Generate Memory Slideshow")
    
    gr.Markdown("## 📽️ Generated Slideshow")
    output_video = gr.Video(label="Your Memory Video")

    btn.click(fn=create_memory_video, inputs=[image_input, text_input], outputs=output_video)

demo.launch()
