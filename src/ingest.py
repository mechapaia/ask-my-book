# src/ingest.py
# Reads the textbook PDF, splits it into chunks, embeds them with a free
# local model, and stores them in a local Chroma vector database.

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# --- Config ---
PDF_PATH = "data/raw/Introduction-to-Data-Science-1758198551._print.pdf"
DB_PATH = "chroma_db"
COLLECTION_NAME = "textbook"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


def load_pages(pdf_path):
    """Return a list of (page_number, text) for each page that has text."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append((i + 1, text))  # 1-based page numbers
    doc.close()
    return pages


def chunk_pages(pages):
    """Split each page into overlapping chunks, keeping the page number."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    for page_num, text in pages:
        for chunk in splitter.split_text(text):
            chunks.append({"text": chunk, "page": page_num})
    return chunks


def main():
    print(f"Loading PDF: {PDF_PATH}")
    pages = load_pages(PDF_PATH)
    print(f"  Pages with text: {len(pages)}")

    chunks = chunk_pages(pages)
    print(f"  Total chunks: {len(chunks)}")

    print(f"Loading embedding model: {EMBED_MODEL} (first run downloads it)")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in chunks]
    print("Embedding chunks... (this can take a few minutes)")
    embeddings = model.encode(
        texts, show_progress_bar=True, normalize_embeddings=True
    )

    print(f"Storing in Chroma at: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)
    # Start fresh each run so re-ingesting doesn't create duplicates
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=texts,
        embeddings=[e.tolist() for e in embeddings],
        metadatas=[{"page": c["page"]} for c in chunks],
    )

    print(f"Done. Stored {collection.count()} chunks.")


if __name__ == "__main__":
    main()
