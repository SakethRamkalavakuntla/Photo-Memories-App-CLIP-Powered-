# 📸 Photo Memories App (CLIP-Powered)

Turn your favorite images into a **personalized video memory** using AI. Just upload your photos, describe the moment, and the app will generate a short video slideshow of your best-matching pictures.

---

## 🧠 Powered By

- **CLIP** – Understands the relationship between images and text
- **FAISS** – Finds the most semantically relevant images
- **OpenCV** – Creates the video slideshow
- **Gradio** – Simple and interactive web interface

---

## 🚀 How It Works

1. **Upload Multiple Images**  
   Drop in your photos (vacation, parties, events, etc.)

2. **Enter a Memory Prompt**  
   Type a theme like `"sunset at the beach"` or `"happy family moments"`

3. **Let the App Work**  
   - All images are converted into **CLIP embeddings**
   - Your text is also converted into an embedding
   - The app uses **FAISS** to find the **3 closest images** to your prompt
   - These images are stitched together into a **video slideshow**

4. **Watch/Download Your Video**  
   A 15-second clip (5 seconds per image) is generated and shown in the app!

---
