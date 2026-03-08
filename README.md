# ResearchNarrative

**RAG-Based Research Storyline Generation Engine**

## Overview

ResearchNarrative transforms academic literature search from a retrieval task into a narrative generation task. Instead of returning ranked lists of papers, it generates structured storylines that explain how a research field evolved, which approaches competed, and what the current frontier looks like.

## Architecture

```
Topic Query
    │
    ▼
┌───────────────────┐
│  Paper Ingestion   │  arXiv API + Semantic Scholar API
│  (src/api/)        │  → Unified paper schema
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Embedding         │  SPECTER2 / Sentence Transformers
│  (src/embeddings/) │  → 768-dim document vectors
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  FAISS Index       │  Inner-product similarity search
│  (src/embeddings/) │  → Nearest-neighbor retrieval
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Thread Discovery  │  UMAP dimensionality reduction
│  (src/clustering/) │  HDBSCAN clustering
│                    │  LLM-assisted labeling
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  RAG Narrative     │  Structured prompt templates
│  (src/narrative/)  │  Citation-grounded generation
│                    │  Multi-section output
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Dashboard         │  Streamlit interactive UI
│  (app.py)          │  Plotly visualizations
│                    │  Paper browser + search
└───────────────────┘
```

## Setup

### 1. Clone & Install

```bash
git clone https://github.gatech.edu/bchamarthi6/CS6235_ResearchNarrative.git
cd CS6235_ResearchNarrative
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` with your keys:

- **SEMANTIC_SCHOLAR_API_KEY** — Get from [Semantic Scholar API](https://api.semanticscholar.org/)
- **OPENAI_API_KEY** — Required for LLM-based narrative generation and cluster labeling
- **OPENALEX_EMAIL** — Optional, for faster OpenAlex rate limits

> The system works without API keys (arXiv is free, S2 has public access), but narratives will use template-based fallback instead of LLM generation.

### 3. Run the Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Usage

1. **Enter a research topic** in the sidebar (e.g., "transformer architectures in NLP")
2. **Configure settings** — year range, max papers, data sources
3. **Click "Analyze Topic"** — the pipeline will:
   - Fetch papers from arXiv and Semantic Scholar
   - Generate SPECTER embeddings
   - Cluster papers into research threads
   - Generate a structured narrative
4. **Explore results** across five tabs:
   - **Narrative** — Full research storyline with citations
   - **Clusters** — UMAP visualization and thread details
   - **Timeline** — Publication and citation trends over time
   - **Papers** — Browsable, filterable paper list
   - **Search** — Semantic similarity search

## Project Structure

```
CS6235_ResearchNarrative/
├── app.py                        # Streamlit dashboard
├── requirements.txt              # Python dependencies
├── .env.example                  # API key template
├── src/
│   ├── config.py                 # Central configuration
│   ├── pipeline.py               # End-to-end pipeline orchestrator
│   ├── models/
│   │   └── paper.py              # Unified paper data model
│   ├── api/
│   │   ├── arxiv_client.py       # arXiv API client
│   │   ├── semantic_scholar_client.py  # S2 API client
│   │   └── ingestion.py          # Multi-source ingestion pipeline
│   ├── embeddings/
│   │   ├── embedder.py           # SPECTER embedding pipeline
│   │   └── vector_store.py       # FAISS vector store
│   ├── clustering/
│   │   └── thread_discovery.py   # HDBSCAN clustering + labeling
│   ├── narrative/
│   │   └── generator.py          # RAG narrative generation
│   └── visualization/            # (reserved for CP3)
├── data/
│   ├── papers/                   # Cached paper collections (JSON)
│   ├── embeddings/               # Cached embedding arrays
│   ├── faiss_index/              # Persisted FAISS indices
│   └── cache/                    # API response cache
└── CS6235_ResearchNarrative_ckpt1.pdf
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data Sources | arXiv API, Semantic Scholar API |
| Embeddings | SPECTER2 (allenai/specter2) via Sentence Transformers |
| Vector Store | FAISS (IndexFlatIP with cosine similarity) |
| Clustering | UMAP + HDBSCAN |
| Narrative Gen | OpenAI GPT-4o-mini via structured prompts |
| Dashboard | Streamlit + Plotly |
| Language | Python 3.10+ |

## Checkpoint Status

- **CP1 (Proposal)** — Completed
- **CP2 (Data Pipeline & Embedding System)** — Completed
  - Paper ingestion from arXiv + Semantic Scholar with unified schema
  - SPECTER embedding pipeline with caching
  - FAISS vector store for similarity search
  - HDBSCAN clustering for thread discovery
  - LLM-assisted cluster labeling
  - RAG narrative generation (with fallback)
  - Streamlit dashboard with 5 interactive tabs

## Author

Bindhumadhavavarma Chamarthi
Georgia Institute of Technology
CS 6365: IEC — Spring 2026
