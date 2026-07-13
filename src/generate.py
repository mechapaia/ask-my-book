# src/generate.py
# Retrieve relevant textbook passages, then use an LLM (Groq) to answer the
# question using ONLY those passages, citing page numbers.

import os
import sys

from dotenv import load_dotenv
from groq import Groq

from retrieve import retrieve  # reuse the retrieval we already built

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are a helpful study assistant. Answer the user's question using ONLY "
    "the provided textbook excerpts. Cite the page number(s) you used in square "
    "brackets, like [p. 204]. If the excerpts do not contain the answer, reply "
    '"I don\'t know based on this textbook." Do not use outside knowledge.'
)


def build_context(chunks, pages):
    """Format retrieved chunks with their page numbers for the prompt."""
    return "\n\n".join(f"[p. {page}] {chunk}" for chunk, page in zip(chunks, pages))


def answer(question, top_k=5):
    results = retrieve(question, top_k=top_k)
    chunks = results["documents"][0]
    pages = [m["page"] for m in results["metadatas"][0]]

    user_prompt = (
        f"Textbook excerpts:\n\n{build_context(chunks, pages)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the excerpts above, and cite page numbers."
    )

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content, sorted(set(pages))


def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("Ask a question about the textbook: ").strip()

    text, pages = answer(question)
    print(f"\nQuestion: {question}\n")
    print("Answer:\n" + text)
    print(f"\nRetrieved from pages: {pages}")


if __name__ == "__main__":
    main()
