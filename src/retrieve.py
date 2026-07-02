# src/retrieve.py
# Ask a question and get back the most relevant textbook chunks + page numbers.

import sys
from sentence_transformers import SentenceTransformer
import chromadb

DB_PATH = "chroma_db"
COLLECTION_NAME = "textbook"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5

# BGE retrieval works best when the QUERY (not the passages) gets this prefix.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def retrieve(question, top_k=TOP_K):
    model = SentenceTransformer(EMBED_MODEL)
    query_vec = model.encode(
        QUERY_PREFIX + question, normalize_embeddings=True
    ).tolist()

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    return collection.query(query_embeddings=[query_vec], n_results=top_k)


def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("Ask a question about the textbook: ").strip()

    results = retrieve(question)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    print(f"\nQuestion: {question}\n")
    print(f"Top {len(docs)} matching chunks:\n" + "=" * 60)
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        preview = doc.replace("\n", " ").strip()
        if len(preview) > 300:
            preview = preview[:300] + "..."
        print(f"\n[{i}] page {meta['page']}  (distance {dist:.3f})")
        print(preview)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
