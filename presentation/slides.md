# ResearchNarrative — Presentation Slides
## RAG-Based Research Storyline Generation Engine

**Bindhumadhavavarma Chamarthi** | GT ID: 904091250 | Group 9  
CS 6235: IEC — Spring 2026 | Georgia Institute of Technology

---

---

## SLIDE 1: Title Slide

**Title:** ResearchNarrative: A RAG-Based Research Storyline Generation Engine

**Subtitle:** Transforming Academic Literature Search into Structured Research Narratives

**Presenter:** Bindhumadhavavarma Chamarthi  
**Course:** CS 6235 — IEC | Spring 2026  
**Georgia Institute of Technology**

---

---

## SLIDE 2: The Problem — Why Literature Review is Broken

**Heading:** The Research Discovery Problem

**Key Points:**

- You start researching a new topic — say "Retrieval-Augmented Generation in Healthcare"
- Google Scholar returns **thousands** of papers as a ranked list
- But you don't need a list — you need **answers:**
  - What is the **main storyline** of this field?
  - Which ideas are **competing** with each other?
  - What approaches **won**, and why?
  - What is the **current frontier** and open problems?
- Manually synthesizing these insights takes **weeks** of reading
- Existing tools (Google Scholar, Semantic Scholar, Connected Papers) help you **find** papers — but none of them **tell you the story**

**Bottom line:** Literature search gives you **ingredients**, but researchers need the **recipe**

---

---

## SLIDE 3: What Exists — Literature Review & Gaps

**Heading:** Existing Tools & What's Missing

**What's Available:**

| Tool | What it Does | What it Lacks |
|------|-------------|---------------|
| Google Scholar | Keyword search, citation counts | No clustering, no narrative, no storylines |
| Semantic Scholar | Semantic search, citation graphs | Shows individual papers, not research threads |
| Connected Papers | Visual citation graph | No temporal analysis, no competition detection |
| Elicit / Consensus | AI-powered Q&A over papers | Single answers, not structured narratives |
| Survey papers | Expert-written storylines | Take months to write, immediately outdated |

**The Gap We Address:**

- No existing tool combines **paper retrieval + thread discovery + citation analysis + narrative generation** into one pipeline
- No tool produces a **citation-grounded storyline** that traces how a field evolved
- No tool detects **competing research threads** and tracks which approach dominated over time
- Our contribution: an **end-to-end system** that does all of this automatically in under 2 minutes

---

---

## SLIDE 4: Pipeline Overview (All 7 Steps on One Slide)

**Heading:** The ResearchNarrative Pipeline — 7 Steps from Topic to Story

**General Points:**
- End-to-end system: user enters a topic → gets a citation-grounded research narrative in ~90 seconds
- Each step feeds the next: raw papers → vectors → clusters → graph → narrative → quality score
- Fully automated — no manual intervention required after entering the topic

**Step Titles for the Overview Diagram:**
1. Paper Ingestion — arXiv + Semantic Scholar
2. Semantic Embedding — SPECTER2 → 768-dim vectors
3. Similarity Index — FAISS nearest-neighbor search
4. Thread Discovery — UMAP + HDBSCAN clustering
5. Citation Graph — PageRank, HITS, competition detection
6. RAG Narrative — GPT-4o with citation verification
7. Automated Evaluation — Silhouette, accuracy, grading

---

---

## SLIDE 4a: Step 1 — Paper Ingestion

**Heading:** Step 1: Paper Ingestion

- Queries two academic APIs simultaneously: **arXiv** (XML/Atom feed) and **Semantic Scholar** (REST/JSON)
- S2 Batch API fetches up to **500 papers per request** — 36x faster than individual calls
- Cross-source **deduplication**: papers found in both APIs are merged (arXiv ID + S2 ID matching)
- Unified **Paper schema**: title, abstract, authors, year, citations, references — all normalized into one data model
- Collects **100-300 papers** per topic with full metadata (citation counts, fields of study, venue)

---

---

## SLIDE 4b: Step 2 — Semantic Embedding

**Heading:** Step 2: Semantic Embedding

- Uses **SPECTER2** (allenai/specter2), a model specifically trained on academic papers using citation signal
- Each paper's **title + abstract** is encoded into a **768-dimensional** dense vector
- Captures **semantic meaning** — papers about the same concept cluster together even with different keywords
- Processed in **batches of 32** for GPU efficiency; embeddings are **cached to disk** for reuse
- Fallback to **MiniLM-L6** if SPECTER2 fails to load (384 dimensions, still effective)

---

---

## SLIDE 4c: Step 3 — Similarity Index

**Heading:** Step 3: FAISS Similarity Index

- **FAISS IndexFlatIP** (Facebook AI Similarity Search) builds an inner-product index over all paper vectors
- Enables **sub-millisecond** nearest-neighbor search across the entire collection
- Supports the **Semantic Search** tab — users type a question, get the most relevant papers ranked by similarity
- Index is **persisted to disk** — reload without recomputing embeddings
- Cosine similarity via inner product on L2-normalized vectors — mathematically equivalent, computationally faster

---

---

## SLIDE 4d: Step 4 — Thread Discovery

**Heading:** Step 4: Thread Discovery (UMAP + HDBSCAN)

- **UMAP** reduces 768 dimensions → 10 dimensions while preserving local neighborhood structure
- **HDBSCAN** (density-based clustering) groups papers into research threads — automatically determines the number of clusters
- Unlike K-means, HDBSCAN naturally handles **noise** — papers that don't belong to any thread stay unclustered
- **LLM labeling**: GPT-4o reads the top-8 papers per cluster and generates a concise **3-6 word label** (e.g., "RAG in Clinical Decision Support")
- Typically finds **4-8 threads** per topic; also generates a **2D UMAP projection** for the scatter plot visualization

---

---

## SLIDE 4e: Step 5 — Citation Graph Analysis

**Heading:** Step 5: Citation Graph Analysis

- Builds a **NetworkX directed graph** from paper references and citations — edges only between papers in our collection
- **PageRank**: identifies structurally important papers (most cited by other important papers)
- **HITS (Hub/Authority)**: finds authoritative sources and hub papers that connect sub-areas
- **Bridge scoring**: detects papers that cite or are cited across **multiple different clusters** — community connectors
- **Competition detection**: measures cross-citation asymmetry between clusters — identifies **rival threads** and tracks **dominance over time** (which thread published more, which got more citations per year)

---

---

## SLIDE 4f: Step 6 — RAG Narrative Generation

**Heading:** Step 6: RAG Narrative Generation

- GPT-4o generates a **5-section narrative**: Origins, Major Threads, Competing Approaches, Evolution, Current Frontier
- All retrieved papers are provided as **context with influence scores** — the LLM is fully grounded, not hallucinating
- **Citation verification**: regex extracts every [Author et al., Year] citation → matched against our paper database → accuracy badge (High/Medium/Low)
- **Thread deep-dives**: a second LLM call generates focused narratives for each individual research thread
- Result: **2,000-4,000 word** narrative with **clickable citations** linking to Semantic Scholar, exportable as Markdown, HTML, or Full Report

---

---

## SLIDE 4g: Step 7 — Automated Evaluation

**Heading:** Step 7: Automated Evaluation

- **Clustering quality**: silhouette score (cosine) + Davies-Bouldin index — measures how well-separated the threads are
- **Narrative quality**: citation accuracy (hallucination check), citation density per paragraph, paper coverage
- **Citation graph**: enrichment coverage, graph connectivity, average in-degree
- **Retrieval quality**: abstract coverage, metadata completeness, source diversity, year span
- Weighted composite → **letter grade** (A+ through D) with **actionable recommendations** for improvement

---

---

## SLIDE 5: Evaluation — How We Measure Quality

**Heading:** Automated Evaluation — 4 Dimensions → Letter Grade

| Dimension (Weight) | Key Metrics |
|---------------------|-------------|
| Retrieval (20%) | Abstract coverage, metadata completeness, source diversity |
| Clustering (25%) | Silhouette score, Davies-Bouldin index, noise ratio |
| Citation Graph (20%) | Graph connectivity, enrichment coverage, avg in-degree |
| Narrative (35%) | Citation accuracy, citation density, paper coverage |

- Weighted composite score → **letter grade** (A+ through D) with **actionable recommendations**
- Narrative weighted highest because it's the primary user-facing output
- Citation accuracy directly measures **hallucination prevention** — % of LLM citations matching real papers

---

---

## SLIDE 6: Reproducibility — Packaged for Anyone to Run

**Heading:** Reproducibility & Project Structure

**Modular Architecture:**
```
CS6235_ResearchNarrative/
├── app.py                    # Dashboard (1,176 lines)
├── requirements.txt          # All dependencies pinned
├── .env.example              # API key template
├── INSTRUCTIONS.md           # Step-by-step guide for reproduction
├── README.md                 # Project documentation
├── src/
│   ├── api/                  # Data ingestion (arXiv + S2)
│   ├── embeddings/           # SPECTER2 + FAISS
│   ├── clustering/           # UMAP + HDBSCAN
│   ├── citation/             # Graph analysis
│   ├── narrative/            # RAG generation
│   ├── evaluation/           # Quality metrics
│   └── pipeline.py           # Orchestrator
└── reports/                  # All checkpoint reports
```

**How to Reproduce (5 commands):**
```bash
git clone <repo>
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # Add API keys
streamlit run app.py          # Open browser
```

**Key Reproducibility Features:**
- `INSTRUCTIONS.md`: detailed guide with setup, usage, testing, troubleshooting
- `requirements.txt`: all 25+ dependencies with version pins
- Works with **zero API keys** (arXiv is free, S2 has public access, narrative falls back to templates)
- Cached results: papers, embeddings, FAISS indices are persisted — rerun without re-fetching
- Pipeline is fully deterministic given same input (UMAP random_state=42)

**Codebase Stats:**
- 16 Python modules, ~4,472 lines of code
- 7 interactive dashboard tabs
- 3 export formats (Markdown, HTML, Full Report)

---

---

## SLIDE 7: AI Workflow — Building with Cursor + Claude

**Heading:** AI-Assisted Development Workflow

**Our SDLC with Cursor + Claude:**

1. **Design phase** (Human-led): Defined architecture, chose tech stack, designed data models — Claude helped brainstorm but human made all decisions
2. **Implementation phase** (AI-accelerated): Claude generated initial module implementations from specifications — code was reviewed and modified by human before integration
3. **Integration & Debugging** (Collaborative): Runtime errors surfaced during testing → Claude analyzed stack traces and proposed fixes → human verified and applied
4. **Iteration** (Feedback-driven): User testing revealed UX issues (e.g., progress card not auto-scrolling, S2 enrichment too slow) → discussed with Claude → implemented fixes together

**What Worked Well:**
- **Rapid prototyping**: 16 modules implemented across 5 checkpoints in weeks, not months
- **Consistent code quality**: uniform error handling, logging, type hints across all modules
- **Boilerplate elimination**: API client patterns, data serialization, CSS styling generated quickly
- **Debugging assistance**: Claude could analyze full stack traces and suggest targeted fixes

**Challenges & How We Fixed Them:**

| Challenge | What Happened | How We Fixed It |
|-----------|---------------|-----------------|
| API rate limits | S2 returned 429s constantly during enrichment | Implemented retry with backoff; later switched to batch API (36x faster) |
| Non-retryable errors | 404 errors caused 10 retries × 5 sec = 50 sec wasted per missing paper | Added non-retryable status code set {400, 403, 404, 405, 410} — skip immediately |
| Slow narrative generation | 7+ LLM calls per run (chunked + per-thread) took 3-5 minutes | Consolidated to 2 calls: 1 for narrative, 1 for all thread deep-dives |
| Stale results on new topic | Pipeline didn't reset state between runs — showed old results | Added explicit state reset at the start of each `run()` call |
| Dark mode UI issues | Text was invisible (black on black) in Streamlit dark mode | Switched to gradient backgrounds and light-gray text colors in CSS |

**Limitations of AI-Assisted Development:**
- Complex architectural decisions (chunking strategy, evaluation weights) required human judgment
- Domain-specific knowledge (academic API quirks, embedding model selection) needed human input
- AI sometimes generated plausible-looking but subtly wrong code — manual testing caught these
- The batch API optimization came from human observation of a performance bottleneck, not AI suggestion

**Bottom Line:** AI accelerated implementation ~3-4x, but every design decision and final validation was human-driven. The human-AI collaboration worked best when the human focused on "what" and "why" while the AI focused on "how."

---

---

## SLIDE 8: Demo / Screenshots

**Heading:** Live System Walkthrough

*[This slide is for the live demo or screenshots]*

**Dashboard Tabs to Show:**
1. **Narrative tab**: Full research storyline with clickable citations, citation accuracy badge, thread deep-dives, and export buttons
2. **Clusters tab**: UMAP scatter plot showing paper clusters, thread details
3. **Citation Analysis tab**: Influence scores table, radar chart, Sankey diagram, dominance timeline
4. **Evaluation tab**: Letter grade, component score gauges, radar chart, recommendations

**Key Points to Highlight During Demo:**
- Enter a topic → pipeline runs in ~90 seconds → full narrative with citations
- Click any citation → goes to the paper on Semantic Scholar
- Citation accuracy badge shows hallucination prevention
- Evaluation tab gives quality assurance with actionable recommendations
- Export as HTML → print-ready research report

---

---

## SLIDE 9: Conclusion & Key Takeaways

**Heading:** Summary

**What We Built:**
- An end-to-end system that transforms a research topic query into a structured, citation-grounded narrative
- 7-step pipeline: Retrieve → Embed → Index → Cluster → Analyze → Narrate → Evaluate
- Interactive dashboard with 7 tabs for exploring results from multiple angles

**Key Technical Contributions:**
- Multi-signal influence scoring (PageRank + HITS + Bridge + Pioneer + Burst)
- Competition detection between research threads via cross-citation asymmetry
- Citation verification system preventing LLM hallucinations
- Automated 4-dimensional evaluation framework with letter grading

**Skills Learned (IEC Context):**
- RAG pipeline architecture: bridging information retrieval and generative AI
- Citation graph analysis: PageRank, HITS in practice
- Unsupervised clustering evaluation: silhouette scores, Davies-Bouldin
- API engineering: batch processing, rate limiting, cross-source deduplication
- Full-stack AI application development with evaluation

**Thank you!**

GitHub: https://github.gatech.edu/bchamarthi6/CS6235_ResearchNarrative.git

---
