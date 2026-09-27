
import json
import numpy as np
from sentence_transformers import SentenceTransformer


# 1. Load saved embeddings
embeddings = np.load("embeddings.npy")

# 2. Load chunk information
with open("chunk_metadata.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)


# 3. Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 4. Get query from user
query = input("Enter your query: ")


# 5. Convert query into an embedding
query_embedding = model.encode(
    [query],
    normalize_embeddings=True
)[0]


# 6. Calculate similarity
similarities = np.dot(embeddings, query_embedding)


# 7. Get top 3 results
top_indices = np.argsort(similarities)[::-1][:3]


# 8. Display results
print("\nTop 3 Retrieved Chunks")
print("=" * 60)

for rank, index in enumerate(top_indices, start=1):

    chunk = chunks[index]

    print(f"\nRank {rank}")
    print(f"Similarity: {similarities[index]:.4f}")
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Source: {chunk['source']}")
    print(f"Text: {chunk['text']}")
