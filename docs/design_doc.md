# AskMyBook — Design Doc (1 page)

- **Author:** `<Your Name>`
- **Segment:** Foundations of Applied Machine Learning
- **Problem statement:** I2 — Document Q&A (RAG)
- **Date:** 3 July 2026 (Week 1)

---

## 1. Problem

Finding and understanding specific things inside a long textbook is slow:
flipping through hundreds of pages, or keyword-searching for terms that don't
match how the book phrases them. General chatbots answer instantly but invent
facts and give no source you can check.

**AskMyBook** lets a student ask a question in plain English about one textbook
and get an answer that is (a) **grounded** in the book's actual passages and
(b) **cited** with page numbers, so every claim is verifiable.

## 2. Scope

- **In scope:** one text-based PDF as the corpus; semantic retrieval of relevant
  passages; LLM-generated answers with page citations; a simple web UI.
- **Out of scope (for this internship):** OCR of scanned PDFs, multi-user
  accounts, fine-tuning a model, real-time document updates.
- **Corpus for development/testing:** *Principles of Data Science* (open textbook).

## 3. Architecture

```
Ingestion (run once):
  PDF ──▶ PyMuPDF (text + page numbers) ──▶ chunker (~900 chars, overlap)
      ──▶ BGE embeddings (local) ──▶ ChromaDB (vector store)

Query (per question):
  question ──▶ BGE embedding ──▶ ChromaDB top-k search
           ──▶ relevant chunks + page numbers
           ──▶ LLM (Groq) ──▶ answer with inline page citations
```

## 4. Tech stack

| Component | Choice | Why |
|---|---|---|
| PDF parsing | PyMuPDF | Fast; extracts text **and** page numbers (needed for citations) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Simple, sensible overlapping chunks |
| Embeddings | `BAAI/bge-small-en-v1.5` (local) | Free, CPU-friendly, strong retrieval quality |
| Vector DB | ChromaDB | Zero-infra local store with metadata filtering |
| Generation | Groq API (Llama 3.x, free tier) | Free, fast, and works when deployed |
| UI | Streamlit | Simple to build; free hosting on Streamlit Cloud |

**Key decision:** generation runs on the **Groq API** rather than a local model
so the same code works both on a laptop and on the free (CPU-only) deployment
host. Embeddings stay local because BGE runs fine on CPU. _(Full ADRs in Week 2.)_

## 5. Evaluation plan

- Build a set of **20+ test questions** with known answers from the textbook.
- Rate each answer on **correctness**, **citation precision** (are the cited
  pages actually the source?), and **completeness**.
- Add **guardrails**: the system says "I don't know" when retrieved evidence is
  weak, instead of guessing. _(Eval + guardrails land in Week 3.)_

## 6. Milestones

| Week | Deliverable |
|---|---|
| 1 (now) | Data layer: ingestion + semantic retrieval working ✅ |
| 2 | Groq generation with citations + Streamlit UI + hybrid (BM25) retrieval |
| 3 | Mini-extension ("Compare Two Documents"), eval set, guardrails, 1 test |
| 4 | Deploy to Streamlit Cloud, reflection, 3rd-year roadmap |

## 7. Risks & open questions

- **Chunk size / overlap** may need tuning for answer quality — will validate
  against the eval set.
- **Groq free-tier rate limits** could throttle heavy testing — mitigate by
  caching and, if needed, falling back to a local Ollama model for dev.
- **Retrieval quality on tables/figures** — PyMuPDF extracts text well but not
  layout; figure-heavy questions may retrieve poorly (documented limitation).
