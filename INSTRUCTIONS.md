# INSTRUCTIONS.md — AI Workflow Guide

This file enables a seamless AI workflow where an LLM (Claude, GPT, Gemini) can ingest this file to understand how to build, run, and test the ResearchNarrative project.

---

## Project Overview

**ResearchNarrative** is a RAG-based Research Storyline Generation Engine that transforms academic literature search into structured narrative generation. Given a research topic, it retrieves papers from multiple APIs, generates semantic embeddings, clusters papers into research threads, and produces citation-grounded narratives displayed in an interactive Streamlit dashboard.

## Tech Stack

- **Language**: Python 3.10+
- **APIs**: arXiv (XML/Atom), Semantic Scholar (REST/JSON)
- **Embeddings**: SPECTER2 (`allenai/specter2`) via Sentence Transformers, with `peft` adapters
- **Vector Store**: FAISS (`IndexFlatIP`, cosine similarity via inner product)
- **Clustering**: UMAP + HDBSCAN
- **LLM**: Azure OpenAI GPT-4o (also supports standard OpenAI)
- **Dashboard**: Streamlit + Plotly
- **Key Libraries**: `sentence-transformers`, `faiss-cpu`, `hdbscan`, `umap-learn`, `openai`, `xmltodict`, `pandas`, `numpy`

## Repository Structure

```
CS6235_ResearchNarrative/
├── app.py                              # Streamlit dashboard entry point (6 tabs)
├── requirements.txt                    # Python dependencies
├── .env.example                        # API key template
├── INSTRUCTIONS.md                     # This file
├── README.md                           # Project documentation
├── src/
│   ├── config.py                       # Central configuration (env vars, constants)
│   ├── pipeline.py                     # End-to-end pipeline orchestrator
│   ├── models/
│   │   └── paper.py                    # Unified Paper dataclass + PaperCollection
│   ├── api/
│   │   ├── arxiv_client.py             # arXiv API client (rate-limited, paginated)
│   │   ├── semantic_scholar_client.py  # S2 API client (retry, citation extraction)
│   │   └── ingestion.py               # Multi-source ingestion orchestrator
│   ├── embeddings/
│   │   ├── embedder.py                 # SPECTER2 embedding pipeline
│   │   └── vector_store.py             # FAISS index management
│   ├── clustering/
│   │   └── thread_discovery.py         # UMAP + HDBSCAN + LLM labeling
│   ├── citation/
│   │   ├── graph.py                    # NetworkX citation graph construction
│   │   ├── influence.py                # PageRank, HITS, bridge, pioneer, burst scoring
│   │   └── competition.py             # Competition detection + dominance tracking
│   ├── narrative/
│   │   └── generator.py                # RAG narrative generation (citation-augmented)
│   ├── evaluation/
│   │   └── metrics.py                  # Pipeline quality evaluation (silhouette, DB, etc.)
│   └── utils/
│       └── llm.py                      # LLM client factory (Azure/OpenAI)
├── data/
│   ├── papers/                         # Cached paper collections (JSON)
│   ├── embeddings/                     # Cached embedding arrays (numpy)
│   ├── faiss_index/                    # Persisted FAISS indices
│   └── cache/                          # API response cache
└── reports/                            # Checkpoint reports
```

## Build & Setup

### Prerequisites
- Python 3.10 or higher
- pip
- (Optional) Azure OpenAI account with a GPT-4o deployment
- (Optional) Semantic Scholar API key

### Step-by-step Setup

```bash
# 1. Clone the repository
git clone https://github.gatech.edu/bchamarthi6/CS6235_ResearchNarrative.git
cd CS6235_ResearchNarrative

# 2. Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your API keys (see Configuration section below)
```

### Configuration (.env)

```env
# Option A: Standard OpenAI
OPENAI_API_KEY=sk-...

# Option B: Azure OpenAI (recommended if you have Azure credits)
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2025-01-01-preview

# Optional: Semantic Scholar API key (works without, just slower)
SEMANTIC_SCHOLAR_API_KEY=your_s2_key

# Optional: OpenAlex polite pool email
OPENALEX_EMAIL=your_email@example.com
```

**Minimum viable configuration**: The system works with NO API keys at all. arXiv is free, Semantic Scholar allows unauthenticated access (slower rate limits), and narratives fall back to template-based generation without an LLM key.

## Run

```bash
# Start the Streamlit dashboard
streamlit run app.py
```

This opens the dashboard at `http://localhost:8501`.

### Using the Dashboard

1. Enter a research topic in the sidebar (e.g., "transformer architectures in NLP")
2. Optionally adjust: max papers, year range, min citations, data sources
3. Click "Analyze Topic"
4. The progress card shows live pipeline status:
   - Paper retrieval from arXiv and Semantic Scholar
   - SPECTER embedding generation
   - FAISS index building
   - UMAP + HDBSCAN clustering
   - LLM-based cluster labeling
   - RAG narrative generation
5. Explore results across 7 tabs: Narrative, Clusters, Citation Analysis, Evaluation, Timeline, Papers, Search

### Programmatic Usage

```python
from src.pipeline import ResearchNarrativePipeline

pipeline = ResearchNarrativePipeline()
results = pipeline.run(
    topic="attention mechanisms in NLP",
    max_papers=100,
    start_year=2017,
    end_year=2025,
)

# Access results
papers = results["papers"]                       # List[Paper]
clusters = results["clusters"]                   # dict[int, List[Paper]]
narrative = results["narrative"]                 # str (markdown)
citation_graph = results["citation_graph"]       # CitationGraph
influence_scores = results["influence_scores"]   # dict[str, dict[str, float]]
competition = results["competition_analysis"]    # dict with competition_pairs, etc.
evaluation = results["evaluation"]               # dict with retrieval, clustering, etc.
```

## Test

### Quick Smoke Test

```bash
# Verify all modules compile
python -c "import py_compile; import os; [print(f'OK: {os.path.join(r,f)}') for r,_,files in os.walk('src') for f in files if f.endswith('.py') and py_compile.compile(os.path.join(r,f), doraise=True)]"

# Verify app compiles
python -c "import py_compile; py_compile.compile('app.py', doraise=True); print('OK: app.py')"
```

### Test API Connectivity

```python
# Test arXiv
from src.api.arxiv_client import ArxivClient
arxiv = ArxivClient()
papers = arxiv.search("machine learning", max_results=3)
print(f"arXiv: {len(papers)} papers")

# Test Semantic Scholar
from src.api.semantic_scholar_client import SemanticScholarClient
s2 = SemanticScholarClient()
papers = s2.search("machine learning", max_results=3)
print(f"S2: {len(papers)} papers")
```

### Test LLM Connection

```python
from src.utils.llm import get_llm_client, get_model_name
from src.config import HAS_LLM

print(f"LLM available: {HAS_LLM}")
if HAS_LLM:
    client = get_llm_client()
    resp = client.chat.completions.create(
        model=get_model_name(),
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10,
    )
    print(f"LLM response: {resp.choices[0].message.content}")
```

### Test Full Pipeline (Small)

```python
from src.pipeline import ResearchNarrativePipeline

pipeline = ResearchNarrativePipeline()
results = pipeline.run(
    topic="neural architecture search",
    max_papers=20,
    sources=["arxiv"],  # arXiv only for speed
)
print(f"Papers: {len(results['papers'])}")
print(f"Clusters: {len([c for c in results['clusters'] if c != -1])}")
print(f"Narrative length: {len(results['narrative'])} chars")
```

## Pipeline Architecture

```
User Query (topic string)
        │
        ▼
┌─────────────────────────┐
│ PaperIngestionPipeline   │
│  ├─ ArxivClient.search() │    arXiv Atom API → XML parsing
│  ├─ S2Client.search()    │    S2 Graph API → JSON parsing
│  └─ Dedup + merge        │    Cross-source ID matching
└──────────┬──────────────┘
           │ PaperCollection (JSON cached)
           ▼
┌─────────────────────────┐
│ PaperEmbedder            │
│  └─ SPECTER2 / MiniLM   │    title+abstract → 768/384-dim vectors
└──────────┬──────────────┘
           │ numpy array (cached)
           ▼
┌─────────────────────────┐
│ FAISSVectorStore         │
│  └─ IndexFlatIP          │    Cosine similarity search
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ ThreadDiscovery          │
│  ├─ UMAP (10D + 2D)     │    Dimensionality reduction
│  ├─ HDBSCAN              │    Density clustering
│  └─ LLM labeling         │    GPT-4o cluster names
└──────────┬──────────────┘
           │ clusters dict
           ▼
┌─────────────────────────┐
│ Citation Graph Analysis  │
│  ├─ CitationGraph        │    NetworkX DiGraph from refs/cited_by
│  ├─ InfluenceScorer      │    PageRank, HITS, bridge, pioneer, burst
│  └─ CompetitionDetector  │    Cross-cluster competition + dominance
└──────────┬──────────────┘
           │ influence scores + competition data
           ▼
┌─────────────────────────┐
│ NarrativeGenerator       │
│  └─ GPT-4o with RAG     │    Citation-augmented prompts + grounding
│     prompts              │    → Multi-section markdown narrative
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ PipelineEvaluator        │
│  ├─ Silhouette score     │    Clustering quality assessment
│  ├─ Citation coverage    │    Graph connectivity metrics
│  └─ Narrative quality    │    Citation density & accuracy
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Streamlit Dashboard      │    7 tabs: Narrative, Clusters, Citation
│  └─ Plotly + custom CSS  │    Analysis, Evaluation, Timeline,
│                          │    Papers, Search
└─────────────────────────┘
```

## Key Configuration Constants (src/config.py)

| Constant | Default | Purpose |
|---|---|---|
| `ARXIV_RATE_LIMIT` | 3.0s | Seconds between arXiv API calls |
| `S2_RATE_LIMIT` | 1.0s | S2 rate limit (with API key) |
| `S2_RATE_LIMIT_NO_KEY` | 5.0s | S2 rate limit (without key) |
| `EMBEDDING_MODEL` | `allenai/specter2` | Primary embedding model |
| `EMBEDDING_BATCH_SIZE` | 32 | Papers per embedding batch |
| `HDBSCAN_MIN_CLUSTER_SIZE` | 5 | Minimum papers per cluster |
| `UMAP_N_COMPONENTS` | 10 | UMAP output dimensions |
| `LLM_TEMPERATURE` | 0.3 | LLM generation temperature |
| `DEFAULT_MAX_PAPERS` | 200 | Default paper retrieval limit |

## Troubleshooting

| Issue | Solution |
|---|---|
| `peft` import error for SPECTER2 | `pip install peft` — required for SPECTER2's PEFT adapters |
| S2 429 rate limit errors | System retries 10 times with 5s delay; reduce `max_papers` if persistent |
| Azure OpenAI `DeploymentNotFound` | Create a model deployment in Azure OpenAI Studio; update `AZURE_OPENAI_DEPLOYMENT` in `.env` |
| FAISS dimension mismatch | Delete `data/embeddings/` and `data/faiss_index/` contents; re-run |
| Stale results on new topic | Fixed in current version; pipeline resets state on each run |
| Dark mode text invisible | Fixed in current version; uses gradient/light-gray colors |
