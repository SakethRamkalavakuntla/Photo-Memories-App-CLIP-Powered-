import faiss
import numpy as np
from PIL import Image
from embedding_utils import get_image_embedding

def build_faiss_index(images: list) -> tuple:
    index = faiss.IndexFlatL2(512)
    embeddings = []

    for image in images:
        emb = get_image_embedding(image)
        embeddings.append(emb[0])

    embeddings_np = np.stack(embeddings)
    index.add(embeddings_np)
    return index, embeddings_np
