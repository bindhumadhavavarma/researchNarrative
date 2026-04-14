"""Central configuration for the ResearchNarrative pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
PAPERS_DIR = DATA_DIR / "papers"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
FAISS_DIR = DATA_DIR / "faiss_index"

for d in [CACHE_DIR, PAPERS_DIR, EMBEDDINGS_DIR, FAISS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- API Keys ---
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")

# --- Azure OpenAI (set these if using Azure instead of standard OpenAI) ---
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

# True if Azure credentials are configured
USE_AZURE_OPENAI = bool(AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT)
# At least one LLM backend available
HAS_LLM = bool(OPENAI_API_KEY) or USE_AZURE_OPENAI

# --- API Rate Limits ---
ARXIV_RATE_LIMIT = 3.0          # seconds between requests
S2_RATE_LIMIT = 1.0             # seconds between requests (with key)
S2_RATE_LIMIT_NO_KEY = 5.0      # seconds between requests (no key, avoid 429s)

# --- Embedding ---
EMBEDDING_MODEL = "allenai/specter2"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DIM = 768

# --- Clustering ---
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 3
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 10
UMAP_METRIC = "cosine"

# --- Narrative Generation ---
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.3
MAX_NARRATIVE_PAPERS = 50

# --- Dashboard ---
STREAMLIT_PAGE_TITLE = "ResearchNarrative"
STREAMLIT_PAGE_ICON = ""
DEFAULT_MAX_PAPERS = 200
