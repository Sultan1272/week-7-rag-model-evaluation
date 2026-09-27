import os
import json

# Get the folder where this Python script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Knowledge base folder
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")

# Output file
OUTPUT_FILE = os.path.join(BASE_DIR, "chunks.jsonl")


def chunk_document(text):
    """
    Split a document into paragraph-based chunks.

    Each paragraph becomes one chunk because the source
    documents are short and each paragraph contains
    a complete piece of information.
    """
    paragraphs = text.split("\n\n")

    chunks = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if paragraph:
            chunks.append(paragraph)

    return chunks


def main():
    all_chunks = []
    chunk_id = 0
    processed_documents = 0

    # Read every .txt file from the knowledge base
    for filename in sorted(os.listdir(KNOWLEDGE_BASE_DIR)):

        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()

        chunks = chunk_document(text)

        processed_documents += 1

        for chunk in chunks:
            all_chunks.append({
                "chunk_id": chunk_id,
                "source": filename,
                "text": chunk
            })

            chunk_id += 1

    # Write chunks to JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        for chunk in all_chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Processed documents: {processed_documents}")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()