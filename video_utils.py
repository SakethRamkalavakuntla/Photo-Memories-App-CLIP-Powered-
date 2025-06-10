import numpy as np
import cv2

def generate_slideshow(images, indices, output_path="memories_output.mp4", fps=20, duration_sec=5):
    video_frames = []
    max_w, max_h = 0, 0

    for i in indices:
        img = np.array(images[i])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video_frames.extend([img] * (fps * duration_sec))

        h, w, _ = img.shape
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    final_frames = []
    for frame in video_frames:
        h, w, _ = frame.shape
        top = (max_h - h) // 2
        bottom = max_h - h - top
        left = (max_w - w) // 2
        right = max_w - w - left
        padded = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        final_frames.append(padded)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (max_w, max_h))
    for frame in final_frames:
        out.write(frame)
    out.release()

    return output_path
