# tests/test_rag.py
# Programmatic tests for the AskMyBook RAG components.

import os
import sys
import pytest

# Add src/ to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ingest import load_pages, chunk_pages
from retrieve import retrieve
from generate import build_context, answer

PDF_PATH = "data/raw/Introduction-to-Data-Science-1758198551._print.pdf"

def test_load_pages():
    """Verify that PyMuPDF correctly loads the PDF pages and returns text."""
    assert os.path.exists(PDF_PATH), f"Textbook PDF not found at {PDF_PATH}"
    pages = load_pages(PDF_PATH)
    
    assert isinstance(pages, list)
    assert len(pages) > 0
    # Verify tuple structure: (page_number, text)
    page_num, text = pages[0]
    assert isinstance(page_num, int)
    assert isinstance(text, str)
    assert len(text) > 0

def test_chunk_pages():
    """Verify chunking logic splits text and retains page mapping."""
    dummy_pages = [
        (1, "This is page one text. It is relatively short but should be chunked properly."),
        (2, "This is page two text. We want to test if page numbers map correctly to chunks.")
    ]
    chunks = chunk_pages(dummy_pages)
    
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "text" in chunk
        assert "page" in chunk
        assert chunk["page"] in [1, 2]
        assert isinstance(chunk["text"], str)

def test_build_context():
    """Verify formatting of retrieved chunks with page numbers."""
    chunks = ["first chunk text", "second chunk text"]
    pages = [10, 20]
    formatted = build_context(chunks, pages)
    
    assert "[p. 10] first chunk text" in formatted
    assert "[p. 20] second chunk text" in formatted

def test_retrieval():
    """Verify that retrieval queries ChromaDB and returns structured matching chunks."""
    question = "What is data science?"
    results = retrieve(question, top_k=3)
    
    assert "documents" in results
    assert "metadatas" in results
    assert "distances" in results
    
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    
    assert len(documents) == 3
    assert len(metadatas) == 3
    assert len(distances) == 3
    assert "page" in metadatas[0]

def test_generation_qa():
    """Integration test: Verify end-to-end question answering utilizing Groq API."""
    # Only run if GROQ_API_KEY is configured
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("Skipping generation test because GROQ_API_KEY is not set.")
        
    question = "What is the CRISP-DM lifecycle?"
    ans_text, pages = answer(question, top_k=3)
    
    assert isinstance(ans_text, str)
    assert len(ans_text) > 0
    assert isinstance(pages, list)
    assert len(pages) > 0
    
    # Check that guardrail rules are followed or standard formatting is present
    # Answer should contain citations or be in-scope
    assert "[p." in ans_text or "I don't know" in ans_text
