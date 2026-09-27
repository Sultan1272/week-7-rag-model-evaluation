
import json
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


# ============================================================
# 1. Load embeddings and chunks
# ============================================================

embeddings = np.load("embeddings.npy")

with open("chunk_metadata.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)


# ============================================================
# 2. Load embedding model
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# 3. Load fine-tuned model
# ============================================================

BASE_MODEL = "google/flan-t5-small"
ADAPTER_PATH = "saved_model"

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32, # Specify data type, can be bfloat16 for better memory on some GPUs
    device_map="auto", # Automatically maps the model to available devices (GPU/CPU)
    tie_word_embeddings=False # Added to silence the tied weights warning
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()


# ============================================================
# 4. Retrieve top-k chunks
# ============================================================

def retrieve(question, top_k=3):

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    normalized_embeddings = embeddings / (
        np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        ) + 1e-10
    )

    scores = normalized_embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "chunk_id": chunks[index]["chunk_id"],
            "source": chunks[index]["source"],
            "text": chunks[index]["text"],
            "score": float(scores[index])
        })

    return results


# ============================================================
# 5. Build grounded prompt
# ============================================================

def build_prompt(question, retrieved_chunks):

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"Source: {chunk['source']}\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a technical support assistant.

Use ONLY the information provided in the context
to answer the question.

Do not use outside knowledge.
Do not invent information.

If the context does not contain enough information
to answer the question, say:

"The information is not available in the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


# ============================================================
# 6. Generate answer
# ============================================================

def generate_answer(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=128
        )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer.strip()


# ============================================================
# 7. Complete RAG pipeline
# ============================================================

def rag_pipeline(question, top_k=3):

    retrieved_chunks = retrieve(
        question,
        top_k
    )

    prompt = build_prompt(
        question,
        retrieved_chunks
    )

    answer = generate_answer(prompt)

    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "answer": answer
    }


# ============================================================
# 8. Test
# ============================================================

if __name__ == "__main__":

    question = input(
        "Enter your question: "
    )

    result = rag_pipeline(question)

    print("\n" + "=" * 60)
    print("QUESTION")
    print("=" * 60)

    print(result["question"])

    print("\n" + "=" * 60)
    print("RETRIEVED CHUNKS")
    print("=" * 60)

    for chunk in result["retrieved_chunks"]:

        print(
            f"\nChunk ID: {chunk['chunk_id']}"
        )

        print(
            f"Source: {chunk['source']}"
        )

        print(
            f"Score: {chunk['score']:.4f}"
        )

        print(
            f"Text: {chunk['text']}"
        )

    print("\n" + "=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)

    print(result["answer"])
