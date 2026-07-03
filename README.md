# AskMyBook 📖🔍

**Ask questions about your textbook and get answers grounded in the actual pages — a Retrieval-Augmented Generation (RAG) Document Q&A tool.**

- **Segment:** Foundations of Applied Machine Learning
- **Problem statement:** I2 — Document Q&A (RAG)
- **Author:** `<Your Name>`
- **Status:** 🚧 Week 1 — data layer working (ingestion + retrieval). Generation, web UI, and deployment coming in Weeks 2–4.

---

## Demo

> _Coming in Week 4:_ Loom walkthrough + live deployed URL will be linked here.
>
> For now, see [What works today](#what-works-today) for a terminal demo of retrieval.

---

## Problem statement

Students constantly need to find and understand specific things inside long
textbooks, but flipping through hundreds of pages (or `Ctrl+F`-ing for keywords
that don't quite match) is slow and misses anything phrased differently from
your question. Generic chatbots, meanwhile, will happily make up an answer with
no source you can check.

**AskMyBook** solves this for a single textbook: you ask a question in plain
English, it finds the most relevant passages **by meaning** (not just keywords),
and — once generation is added — answers using only those passages, **citing the
page numbers** so you can verify every claim. The test corpus is the open
textbook *Principles of Data Science*.

---

## Architecture

```
                        INGESTION (offline, run once)
  ┌────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
  │ Textbook   │──▶│ PyMuPDF      │──▶│ Chunker       │──▶│ BGE embedder │──┐
  │ PDF        │   │ (text+pages) │   │ (~900 chars)  │   │ (local)      │  │
  └────────────┘   └──────────────┘   └───────────────┘   └──────────────┘  │
                                                                            ▼
                                                                 ┌────────────────┐
                                                                 │  ChromaDB      │
                                                                 │ (vector store) │
                                                                 └────────────────┘
                        QUERY (per question)                              ▲
  ┌────────────┐   ┌──────────────┐   ┌───────────────┐                   │
  │ Question   │──▶│ BGE embedder │──▶│ Top-k search  │───────────────────┘
  └────────────┘   └──────────────┘   └───────────────┘
                                              │
                                              ▼
                             ┌──────────────────────────────┐
                             │ Relevant chunks + page numbers│
                             │  → (Week 2) LLM answer w/      │
                             │     inline citations (Groq)    │
                             └──────────────────────────────┘
```

_A polished architecture diagram (PNG) will be added for Milestone 1._

---

## Tech stack

| Component        | Choice                                   | Why |
|------------------|------------------------------------------|-----|
| Language         | Python 3.12                              | Best RAG ecosystem |
| PDF parsing      | PyMuPDF (`fitz`)                         | Fast, extracts text **and** page numbers (needed for citations) |
| Chunking         | LangChain `RecursiveCharacterTextSplitter` | Sensible overlapping chunks |
| Embeddings       | `sentence-transformers` + `BAAI/bge-small-en-v1.5` | Free, local, CPU-friendly, strong quality |
| Vector database  | ChromaDB (persistent, local)             | Zero infra, built-in metadata filtering |
| Generation _(Week 2)_ | Groq API (Llama 3.x, free tier)     | Free, fast, and works when deployed |
| Web UI _(Week 2)_ | Streamlit                               | Simple, deployable on a free tier |
| Deployment _(Week 4)_ | Streamlit Community Cloud            | Free hosting connected to GitHub |

---

## Quickstart

### Prerequisites
- Python 3.11+
- Git
- A text-based (not scanned) PDF to use as your corpus

### Install
```bash
git clone https://github.com/mechapaia/ask-my-book.git
cd ask-my-book
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
# source venv/bin/activate
pip install -r requirements.txt
```

### Run
1. Place your PDF at `data/raw/` and update `PDF_PATH` at the top of
   `src/ingest.py` if the filename differs.
2. Build the vector database (first run downloads the embedding model, ~130 MB):
   ```bash
   python src/ingest.py
   ```
3. Ask a question:
   ```bash
   python src/retrieve.py "What is overfitting in machine learning?"
   ```
   You'll get the most relevant passages, each with its page number.

### Test
> _Coming in Week 3:_ a small test suite (`pytest`). For now, verification is
> manual — see [What works today](#what-works-today).

---

## Data sources

- **Corpus:** *Principles of Data Science* (open textbook, PDF).
- The PDF itself is **not committed** to the repo (it's git-ignored to keep the
  repo lightweight). Download it separately and drop it into `data/raw/`.

---

## ADRs (Architecture Decision Records)

> _Coming in Week 2:_ ADRs documenting the key choices (embedding model, vector
> DB, and generation model) will live in [`docs/adr/`](docs/adr/).

---

## Mini-extension (planned, Week 3)

**"Compare Two Documents":** upload two PDFs (e.g. two versions of a syllabus or
two papers) and ask *"what's different between these on topic X?"*. The system
retrieves from both and answers with a comparison.

**Why:** multi-document reasoning is a clear step up from single-document RAG.

---

## Known limitations

- **Retrieval only, for now** — the LLM answer layer (generation) is not wired up
  yet; the tool currently returns relevant passages, not a synthesized answer.
- **Dense retrieval only** — keyword (BM25) hybrid retrieval is planned for Week 2.
- **Text PDFs only** — scanned/image PDFs won't work without OCR.
- **Single corpus** — one textbook at a time until the mini-extension lands.

---

## What I learned this week

- **Embeddings turn meaning into geometry** — similar ideas end up as nearby
  vectors, which is what lets "search by meaning" beat keyword search.
- **Chunking matters** — you retrieve chunks, not whole pages, so chunk size and
  overlap directly affect answer quality.
- **`.gitignore` is a safety tool, not just tidiness** — it's what keeps API keys
  (`.env`) and large files (the PDF, the vector DB) out of a public repo.
- _(add 1–2 of your own here — e.g. something that surprised you about ChromaDB
  or PyMuPDF)_

---

## What I'd do in 3rd year

> _Coming in Week 4:_ see `docs/roadmap_3rd_year.md`. The short version: grow this
> from a single-textbook tool into a multi-source, evaluated, deployed system —
> the seed of a 3rd-year "enterprise RAG" project.

---

## License + Acknowledgements

- **License:** MIT (see [LICENSE](LICENSE)).
- **Corpus:** *Principles of Data Science* (open educational resource).
- Built as the Foundations of Applied ML project for the 2nd-year internship.

---

## What works today

Running `python src/retrieve.py "how does linear regression work"` against the
ingested textbook (1,887 chunks) returns the top matching passages with page
numbers — e.g. results from pages 204, 206, and 303, all genuinely about linear
regression. This confirms the ingestion → embedding → retrieval pipeline is
working end to end.
