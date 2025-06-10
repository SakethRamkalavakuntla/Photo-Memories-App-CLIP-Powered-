# 📸 Photo Memories App (CLIP-Powered)

Turn your favorite images into a **personalized video memory** using AI. Just upload your photos, describe the moment, and the app will generate a short video slideshow of your best-matching pictures.

---

## 🧠 Powered By

- **[CLIP](https://github.com/openai/CLIP)** – Understands the relationship between images and text
- **[FAISS](https://github.com/facebookresearch/faiss)** – Finds the most semantically relevant images
- **[OpenCV](https://opencv.org/)** – Creates the video slideshow
- **[Gradio](https://www.gradio.app/)** – Simple, modern, and responsive web interface

---

## 🖼️ Gradio Interface

Gradio makes it easy to use machine learning models with just a browser. This app offers:

- 📁 **Image Upload Field** – Drag and drop or select multiple images
- ✍️ **Text Input** – Type a memory or theme (e.g., “sunset at the beach”)
- 🎞️ **Generate Button** – Launches the slideshow generation process
- 📺 **Video Output** – Displays the generated 15-second memory video

Gradio runs locally by default but can be **shared publicly** via a shareable link for easy demos or collaborations.

---

## 🚀 How It Works

1. **Upload Images**: Drop in your photos (e.g., beach trip, birthday, etc.)
2. **Type a Theme**: Like "happy memories" or "adventure"
3. **Behind the Scenes**:
   - Images → CLIP embeddings
   - Prompt → CLIP text embedding
   - FAISS finds top 3 matching images
   - OpenCV turns those into a 15-second MP4
4. **View & Download** the personalized video in-browser!

---

## 🏗️ Deployment on AWS EC2
- Uses Gunicorn as the WSGI HTTP server.
- Configured Nginx as a reverse proxy to forward requests to Gunicorn.
- Runs Gunicorn inside a Python virtual environment for dependency isolation.
- Set up a systemd service for automatic start, restart, and management of the app.
- AWS security group configured to allow HTTP (port 80) and the app port (e.g., 5000 or 8000) as needed.
