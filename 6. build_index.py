
import json
import numpy as np
from sentence_transformers import SentenceTransformer


# 1. Load chunks
chunks = []

with open("chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

print(f"Loaded {len(chunks)} chunks")


# 2. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded")


# 3. Extract text
texts = [chunk["text"] for chunk in chunks]


# 4. Create embeddings
embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

embeddings = np.array(embeddings)

print(f"Created embeddings with shape: {embeddings.shape}")


# 5. Save embeddings
np.save("embeddings.npy", embeddings)


# 6. Save chunk information
with open("chunk_metadata.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)


print("Embeddings saved to embeddings.npy")
print("Chunk metadata saved to chunk_metadata.json")
print("Index building completed successfully.")
