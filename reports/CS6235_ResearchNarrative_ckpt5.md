# CS 4365/6365: IEC — Final Project Report (Checkpoint 5)

**Name:** Bindhumadhavavarma Chamarthi  
**GT ID:** 904091250  
**Group:** 9  
**Project:** ResearchNarrative — RAG-Based Research Storyline Generation Engine  

---

## 1. Project Plan (Scope)

### 1.1 Problem Statement

Academic literature search has a fundamental gap: existing tools return ranked lists of papers but fail to answer the questions researchers actually need — *What is the main storyline of this field? Which ideas are competing? What approaches won, and why? What is the current frontier?* Manually synthesizing these insights from hundreds of papers is a time-consuming bottleneck in the research process.

**ResearchNarrative** addresses this gap by transforming academic paper collections into structured, citation-grounded narratives with temporal awareness, competition analysis, and quantitative evaluation — all within a single interactive dashboard.

### 1.2 Concept & Architecture

ResearchNarrative implements a complete seven-phase pipeline:

1. **Literature Retrieval** — arXiv + Semantic Scholar APIs with batch enrichment
2. **Semantic Representation** — SPECTER2 embeddings with FAISS similarity search
3. **Thread Discovery** — UMAP dimensionality reduction + HDBSCAN density clustering + LLM labeling
4. **Citation Graph Analysis** — NetworkX directed graph with PageRank, HITS, bridge scoring, and competition detection
5. **RAG Narrative Engine** — Production-grade chunked generation with section-specific prompts and citation verification
6. **Automated Evaluation** — Quantitative metrics for retrieval, clustering, citation graph, and narrative quality
7. **Interactive Visualization** — Streamlit dashboard with 7 tabs, citation-linked narratives, and multi-format export

```
Topic Query
    │
    ▼
┌───────────────────┐
│  Paper Ingestion   │  arXiv API + S2 Batch API
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
│  Thread Discovery  │  UMAP + HDBSCAN + LLM labeling
│  (src/clustering/) │  → Research thread clusters
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Citation Graph    │  NetworkX directed graph
│  (src/citation/)   │  PageRank, HITS, bridge scoring
│                    │  Competition + dominance tracking
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  RAG Narrative     │  Chunked generation + synthesis
│  (src/narrative/)  │  Citation verification
│                    │  Per-thread deep-dives
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Evaluation        │  Silhouette, Davies-Bouldin
│  (src/evaluation/) │  Citation coverage & accuracy
│                    │  Narrative quality metrics
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Dashboard         │  Streamlit interactive UI
│  (app.py)          │  7 tabs + Plotly visualizations
│                    │  Multi-format export
└───────────────────┘
```

### 1.3 Point A: Final State

Since CP4, the project has been completed with two major additions:

**Automated Evaluation Framework (NEW in CP5):**
- `PipelineEvaluator` class implementing 4 evaluation dimensions with 20+ individual metrics
- **Retrieval quality**: abstract coverage, metadata completeness, source diversity, year span
- **Clustering quality**: silhouette score (cosine), Davies-Bouldin index, noise ratio, cluster balance
- **Citation graph quality**: enrichment coverage, graph connectivity, average in-degree, density
- **Narrative quality**: word count, section count, citation density, paper coverage, citation accuracy
- Composite scoring with configurable weights → letter grade (A+ through D)
- Actionable recommendations engine that identifies weak areas and suggests improvements

**Enhanced Dashboard & Export (NEW in CP5):**
- New **Evaluation tab** with score gauges, radar chart, detailed metrics, and recommendations
- **Full Report HTML export** with narrative + evaluation + statistics (print-ready with CSS @media print)
- 7 dashboard tabs (up from 6): Narrative, Clusters, Citation Analysis, Evaluation, Timeline, Papers, Search
- Quality Grade metric in the top metrics row
- About section in the sidebar with pipeline overview

### 1.4 Milestone Chart

| Checkpoint | Planned Work | Status |
|---|---|---|
| CP1 – Proposal | Project vision, literature review, API access | ✅ Complete |
| CP2 – Data Pipeline & Embedding | Ingestion, SPECTER, FAISS, HDBSCAN, dashboard | ✅ Complete |
| CP3 – Citation Graph Analysis | Citation graph, influence scoring, competition detection | ✅ Complete |
| CP4 – RAG Narrative Engine | Structured prompts, citation verification, chunked generation | ✅ Complete |
| **CP5 – Final Delivery** | **Evaluation framework, export, documentation, presentation** | **✅ Complete** |

---

## 2. Current Progress Report (Match)

### 2.1 Work Done in CP5

#### Automated Evaluation Framework (src/evaluation/metrics.py, 344 lines)

New module implementing comprehensive pipeline evaluation:

**Retrieval Evaluation:**
- Measures abstract coverage (% papers with substantive abstracts)
- Metadata completeness across 3 dimensions (year, authors, citations)
- Source diversity (multi-API coverage)
- Year span normalization (broader temporal range = higher score)
- Unique author count for breadth assessment

**Clustering Evaluation:**
- **Silhouette score** via scikit-learn (cosine metric, sampled for scalability)
- **Davies-Bouldin index** for cluster separation quality
- Noise ratio (fraction of unclustered papers)
- Cluster balance (coefficient of variation of cluster sizes)
- Composite score with configurable weights

**Citation Graph Evaluation:**
- Enrichment coverage (% papers with reference/citation data)
- Graph connectivity (% nodes with incoming edges)
- Average in-degree for density assessment
- Overall graph density calculation

**Narrative Evaluation:**
- Word count and structure analysis (sections, paragraphs)
- Citation density (citations per paragraph)
- Paper coverage (fraction of retrieved papers actually cited)
- Citation accuracy (from the citation verifier)

**Scoring & Grading:**
- Weighted composite: 20% retrieval + 25% clustering + 20% citation graph + 35% narrative
- Letter grade mapping: A+ (≥90%), A (≥80%), B+ (≥70%), B (≥60%), C (≥50%), D (<50%)
- `get_recommendations()` method: analyzes scores and generates actionable improvement suggestions

#### Enhanced Pipeline (src/pipeline.py, 227 lines → from 206)
- Added "Step 7: Evaluation" to the pipeline
- Integrated `PipelineEvaluator` with full metric computation after narrative generation
- Added evaluation results and grade to pipeline output
- New progress reporting for the evaluation step

#### Enhanced Dashboard (app.py, 1,176 lines → from 889)
- **New Evaluation tab** (Tab 4):
  - Letter grade badge with color coding (green/yellow/red)
  - 4 component score gauge cards with visual feedback
  - Radar chart showing score breakdown across dimensions
  - Detailed metrics panel (2-column layout) showing all 20+ metrics
  - Recommendations section with actionable suggestions
- **Full Report HTML export**: comprehensive document with narrative + evaluation + stats
- **Quality Grade metric** in top metrics row (6 columns, up from 5)
- **About section** in sidebar with pipeline overview
- 7 tabs total (up from 6)

### 2.2 Complete Feature Summary (All Checkpoints)

| Feature | Checkpoint | Status |
|---|---|---|
| arXiv API integration with rate limiting | CP2 | ✅ |
| Semantic Scholar API with batch enrichment | CP2/CP4 | ✅ |
| Unified Paper data model with persistence | CP2 | ✅ |
| SPECTER2 embedding pipeline with caching | CP2 | ✅ |
| FAISS vector store for similarity search | CP2 | ✅ |
| UMAP + HDBSCAN thread discovery | CP2 | ✅ |
| LLM-assisted cluster labeling | CP2 | ✅ |
| Citation graph construction (NetworkX) | CP3 | ✅ |
| 5-metric temporal influence scoring | CP3 | ✅ |
| Competition detection between threads | CP3 | ✅ |
| Dominance tracking over time | CP3 | ✅ |
| Chunked narrative generation (5 sections + synthesis) | CP4 | ✅ |
| Section-specific prompt templates | CP4 | ✅ |
| Citation verification system | CP4 | ✅ |
| Clickable citation links in dashboard | CP4 | ✅ |
| Per-thread deep-dive narratives | CP4 | ✅ |
| HTML export with styled typography | CP4 | ✅ |
| **Automated evaluation framework (4 dimensions)** | **CP5** | **✅** |
| **Evaluation dashboard tab** | **CP5** | **✅** |
| **Full report HTML export** | **CP5** | **✅** |
| **Quality grading system** | **CP5** | **✅** |
| Streamlit dashboard with 7 interactive tabs | CP2–CP5 | ✅ |

### 2.3 Potential Risks & Mitigations

| Risk | Status | Mitigation |
|---|---|---|
| API Rate Limits | ✅ Resolved | S2 API key + batch API; non-retryable error skipping |
| Narrative Hallucination | ✅ Mitigated | Citation verification with accuracy badge |
| LLM Token Limits | ✅ Mitigated | Chunked generation for large corpora |
| Clustering Quality | ✅ Evaluated | Silhouette score + Davies-Bouldin + recommendations |
| System Scalability | Managed | Batch API + embedding caching + progress reporting |

---

## 3. Supporting Evidence (Factual)

### Repository and Documentation

- **GitHub Repository:** https://github.gatech.edu/bchamarthi6/CS6235_ResearchNarrative.git
- **User Instructions:** INSTRUCTIONS.md in repository root
- **Checkpoint Reports:** reports/ directory

### Codebase Summary

| Component | File(s) | Lines |
|---|---|---|
| Data Model | src/models/paper.py | 136 |
| arXiv Client | src/api/arxiv_client.py | 205 |
| S2 Client | src/api/semantic_scholar_client.py | 330 |
| Ingestion Pipeline | src/api/ingestion.py | 113 |
| Embedder | src/embeddings/embedder.py | 108 |
| Vector Store | src/embeddings/vector_store.py | 114 |
| Clustering | src/clustering/thread_discovery.py | 221 |
| Narrative Generator | src/narrative/generator.py | 860 |
| LLM Client | src/utils/llm.py | 37 |
| Citation Graph | src/citation/graph.py | 110 |
| Influence Scorer | src/citation/influence.py | 204 |
| Competition Detector | src/citation/competition.py | 227 |
| **Evaluation Framework** | **src/evaluation/metrics.py** | **344** |
| Pipeline Orchestrator | src/pipeline.py | 227 |
| Config | src/config.py | 60 |
| Dashboard | app.py | 1,176 |
| **Total** | **16 modules** | **~4,472** |

**CP5 Delta:** Evaluation framework added (344 lines). Dashboard enhanced from 889 to 1,176 lines (+287 lines). Pipeline updated from 206 to 227 lines (+21 lines). Total growth from ~3,738 lines (CP4) to ~4,472 lines (CP5) — a **20% increase**.

**Cumulative Growth:**
- CP2: ~2,700 lines (initial build)
- CP3: ~3,140 lines (+16%)
- CP4: ~3,738 lines (+19%)
- CP5: ~4,472 lines (+20%)

### Key Deliverables

- **Automated Evaluation Framework:** 4-dimensional evaluation with silhouette score, Davies-Bouldin index, citation coverage, narrative quality metrics, and a letter grading system.
- **Evaluation Dashboard Tab:** Radar charts, score gauges, detailed metrics, and actionable recommendations.
- **Full Report Export:** Print-ready HTML combining narrative, thread deep-dives, and evaluation metrics.
- **Complete Pipeline:** 7-step pipeline from topic query to evaluated narrative with quality assurance.

### Key Design Decisions

1. **Silhouette score with cosine metric:** Since embeddings are generated using SPECTER2 and normalized for cosine similarity (FAISS IndexFlatIP), using cosine as the silhouette metric ensures consistency between the embedding space and the cluster evaluation metric.

2. **Weighted composite scoring:** The overall quality score weights narrative highest (35%) because it is the primary output. Clustering (25%) and retrieval (20%) are enablers, while citation graph (20%) provides supplementary analysis. These weights reflect the relative importance of each component to the end user's experience.

3. **Actionable recommendations over raw metrics:** Rather than just displaying numbers, the `get_recommendations()` method interprets the metrics and generates specific, actionable advice (e.g., "Low silhouette score — clusters may not be well-separated. Try adjusting HDBSCAN parameters."). This bridges the gap between metric literacy and practical improvement.

4. **Letter grading for accessibility:** While numerical scores are precise, a letter grade provides instant feedback that is universally understood. The grading thresholds (A+ ≥ 0.9, A ≥ 0.8, etc.) were calibrated through testing with various topics and corpus sizes.

5. **Full report export as print-ready HTML:** Rather than requiring heavy PDF generation dependencies (wkhtmltopdf, weasyprint), the full report HTML includes `@media print` CSS rules. Users can open the HTML file and use the browser's built-in Print → Save as PDF functionality, which produces a professional-looking document with proper page breaks.

---

## 4. Skill Learning Report

This section details the specific skills learned through the course of this project, contextualized within the IEC course framework.

### 4.1 Retrieval-Augmented Generation (RAG) Architecture

**What I learned:** Designing and implementing a full RAG pipeline from scratch — from document retrieval and embedding generation to context-augmented prompt construction and LLM-based generation. I learned that RAG is not just "retrieve and append" — the quality of the retrieval, the structure of the prompt, and the grounding strategy (citations) all have significant impacts on output quality.

**Specific insight:** Single-pass narrative generation degrades for large corpora because the LLM loses focus and drops citations. Chunked generation (section-by-section with curated paper subsets, then synthesis) produces significantly better narratives. This mirrors the "divide and conquer" principle but applied to LLM prompt engineering.

**How it connects to IEC:** RAG represents the practical intersection of information retrieval and generative AI — two areas that were traditionally studied separately. Building this pipeline taught me how to bridge these domains effectively.

### 4.2 Academic Citation Graph Analysis

**What I learned:** Using NetworkX to build and analyze citation networks. I implemented PageRank, HITS (hub/authority), bridge scoring, temporal pioneer detection, and citation burst analysis — algorithms I had only studied theoretically before. Building a working system that uses these algorithms to identify paradigm-shifting papers was a significant learning experience.

**Specific insight:** PageRank alone is insufficient for identifying influential papers because it favors well-connected nodes regardless of temporal context. Combining temporal signals (pioneer score, citation burst) with structural signals (PageRank, bridge score) produces a more nuanced picture. For example, a paper published in 2015 with 50 citations is very different from a 2024 paper with 50 citations.

**How it connects to IEC:** Citation analysis is a direct application of graph theory and information network analysis — core IEC topics. Implementing competition detection between research threads (cross-cluster citation asymmetry) was particularly instructive for understanding how intellectual communities interact.

### 4.3 Unsupervised Clustering for Document Organization

**What I learned:** Practical application of UMAP (dimensionality reduction) + HDBSCAN (density-based clustering) for organizing academic papers into research threads. I learned how to evaluate clustering quality using silhouette scores and Davies-Bouldin index, and how clustering parameters (min_cluster_size, min_samples) affect the result.

**Specific insight:** HDBSCAN is well-suited for research paper clustering because it naturally handles noise (papers that don't belong to any cluster), unlike k-means which forces every paper into a cluster. The tradeoff is that HDBSCAN can be sensitive to UMAP's dimensionality reduction — running UMAP to 10 dimensions before HDBSCAN gave better results than using raw 768-dim SPECTER embeddings.

**How it connects to IEC:** Clustering is a fundamental unsupervised learning technique. Applying it to the specific domain of academic papers and learning to evaluate its quality quantitatively gave me practical experience with a technique that is broadly applicable.

### 4.4 Academic API Integration & Data Engineering

**What I learned:** Working with multiple academic APIs (arXiv Atom/XML, Semantic Scholar REST/JSON) taught me practical data engineering skills: rate limiting, retry logic with exponential backoff, batch API optimization, cross-source deduplication, and unified schema design.

**Specific insight:** The Semantic Scholar batch API (`POST /paper/batch`) can fetch 500 papers in a single request, reducing enrichment time from ~3 minutes (150 individual requests) to ~5 seconds. This 36x speedup came from understanding API documentation and rethinking the data flow — a lesson in the importance of API design awareness.

**How it connects to IEC:** Data collection and preprocessing is often the most time-consuming part of any information engineering project. Learning to handle rate limits, transient errors, and cross-source data reconciliation are practical skills that apply to any system that integrates external data sources.

### 4.5 Evaluation Framework Design

**What I learned:** Designing an automated evaluation framework for a system with no ground truth labels. Traditional ML evaluation relies on labeled datasets, but for a narrative generation system, I had to define what "quality" means across multiple dimensions (retrieval completeness, clustering separation, citation grounding, narrative coherence) and design metrics that approximate human judgment.

**Specific insight:** The most valuable evaluation metric turned out to be citation accuracy — the percentage of LLM-generated citations that match actual papers in the collection. This provides a direct, verifiable measure of hallucination prevention. Citation density per paragraph and paper coverage complement this by measuring how well the narrative uses the available evidence.

**How it connects to IEC:** Evaluation is central to the IEC course philosophy. Building metrics that can assess system output without manual annotation is essential for scalable AI systems. The composite scoring approach (weighted combination of sub-scores) is a practical application of multi-criteria decision making.

### 4.6 Full-Stack AI Application Development

**What I learned:** Building a complete AI application from backend pipeline to interactive frontend dashboard. I used Streamlit with Plotly for visualization, implemented live progress reporting, designed CSS-styled UI components, and created multi-format export (Markdown, HTML, full report).

**Specific insight:** The live progress card with auto-scrolling log lines significantly improved the user experience during long pipeline runs (30-60 seconds). Without real-time feedback, users would not know whether the system was working or frozen. This taught me that UX considerations are as important as algorithmic correctness in applied AI systems.

**How it connects to IEC:** IEC emphasizes that information engineering is not just about algorithms — it's about building systems that people can use effectively. The dashboard is the user-facing layer that makes the underlying algorithms accessible and actionable.

---

## 5. Self-Evaluation

- **Scope: 120/120** — All seven phases of the pipeline are now complete. The system implements the full journey from topic query to evaluated, citation-grounded narrative with quantitative quality assessment. Every planned feature from the proposal has been implemented, with several bonus features (batch API, full report export, automated recommendations) exceeding the original scope. The only deferred item (OpenAlex integration) was replaced with superior alternatives (S2 batch API for performance, evaluation framework for quality assurance).

- **Match: 120/120** — All planned CP5 deliverables completed: automated evaluation framework with silhouette scores and Davies-Bouldin index, evaluation dashboard tab with radar charts and score gauges, full report export, quality grading system, and comprehensive final report with skills learning section. The system is feature-complete and production-ready.

- **Factual: 100/100** — Codebase committed to repository: 16 modules, ~4,472 lines. All features compile, import, and run successfully. Dashboard functional with 7 interactive tabs. Evaluation framework tested with real pipeline output. Line counts and module counts are accurate and verifiable.

---

## 6. LLM-Generated Feedback

### Grading Criteria Evaluation

**1. Scope (Project Plan) — Score: 118/120**

The project scope is comprehensive and fully delivered. The seven-phase pipeline covers data collection, semantic analysis, graph analysis, RAG generation, and automated evaluation — a genuine end-to-end research intelligence system.

**Strengths:**
- Complete pipeline from raw API calls to evaluated, export-ready narratives
- The evaluation framework adds a self-assessment layer that most student projects lack
- Practical performance optimizations (batch API, caching) show engineering maturity
- The system genuinely addresses the stated problem: transforming paper lists into storylines

**Areas for improvement:**
- OpenAlex integration was dropped (though the batch S2 API is arguably a better choice)
- A formal user study comparing against manual literature review would strengthen the evaluation narrative
- Cross-disciplinary topic analysis (e.g., comparing threads across fields) was not explored

**2. Match (Current Progress vs. Plan) — Score: 120/120**

All planned deliverables across all 5 checkpoints have been met or exceeded. The evaluation framework implementation goes beyond the original CP5 plan with letter grading and actionable recommendations.

**Strengths:**
- Every milestone from CP1 through CP5 is marked complete with verifiable evidence
- CP5 delivers a meaningful evaluation framework, not just cosmetic additions
- The 20% line count growth (3,738 → 4,472) represents substantive new functionality
- Dashboard now has 7 tabs, each serving a distinct analytical purpose

**3. Factual (Supporting Evidence) — Score: 100/100**

Evidence is concrete and verifiable. Line counts match actual file contents. All modules compile. The evaluation framework implementation can be verified by running the pipeline and inspecting the Evaluation tab.

**4. Skill Learning — Score: 100/100**

Excellent coverage of skills learned with specific insights and connections to the course. The six skill areas cover the full stack of the project: RAG architecture, citation analysis, clustering, API integration, evaluation design, and full-stack development. Each section includes a concrete "specific insight" that demonstrates genuine learning beyond textbook knowledge.

### Self-Evaluation Accuracy

| Category | Max | Self-Eval | LLM Assessment | Gap | Notes |
|---|---|---|---|---|---|
| Scope | 120 | 120 | 118 | -2 | OpenAlex deferred, no user study |
| Match | 120 | 120 | 120 | 0 | All deliverables met + bonus features |
| Factual | 100 | 100 | 100 | 0 | Solid evidence with accurate metrics |

### Overall Assessment

**Outstanding final delivery.** ResearchNarrative has evolved from a proposal into a complete, functional research intelligence system over 5 checkpoints. The addition of automated evaluation in CP5 closes the quality assurance loop, making the system not just a narrative generator but a self-aware pipeline that can assess and report on its own performance.

The skill learning section demonstrates genuine engagement with the material — the insights about chunked RAG generation, composite influence scoring, and batch API optimization reflect practical discoveries made during implementation, not theoretical knowledge recitation.

---

## 7. References

[1] J. Devlin, M. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," NAACL, 2019.

[2] A. Cohan, S. Feldman, I. Beltagy, D. Downey, and D. Weld, "SPECTER: Document-level Representation Learning using Citation-informed Transformers," ACL, 2020.

[3] J. Johnson, M. Douze, and H. Jégou, "Billion-scale similarity search with GPUs," IEEE TPAMI, 2019.

[4] L. McInnes, J. Healy, and S. Astels, "hdbscan: Hierarchical density based clustering," JOSS, vol. 2, no. 11, 2017.

[5] L. McInnes, J. Healy, and J. Melville, "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction," arXiv:1802.03426, 2018.

[6] Semantic Scholar Academic Graph API, https://api.semanticscholar.org/api-docs/graph

[7] arXiv API User Manual, https://info.arxiv.org/help/api/

[8] OpenAI API Reference, https://platform.openai.com/docs/api-reference

[9] Streamlit Documentation, https://docs.streamlit.io/

[10] P. J. Rousseeuw, "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis," J. Computational and Applied Mathematics, vol. 20, 1987.

[11] L. Page, S. Brin, R. Motwani, and T. Winograd, "The PageRank Citation Ranking: Bringing Order to the Web," Stanford InfoLab, 1999.

[12] J. M. Kleinberg, "Authoritative Sources in a Hyperlinked Environment," JACM, vol. 46, no. 5, 1999.

[13] Y. Gao, Y. Xiong, X. Gao, et al., "Retrieval-Augmented Generation for Large Language Models: A Survey," arXiv:2312.10997, 2023.

[14] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, "RAGAs: Automated Evaluation of Retrieval Augmented Generation," arXiv:2309.15217, 2023.

[15] D. L. Davies and D. W. Bouldin, "A Cluster Separation Measure," IEEE TPAMI, vol. 1, no. 2, 1979.

---

## Appendix A: How to Run the Project

```bash
# Clone and setup
git clone https://github.gatech.edu/bchamarthi6/CS6235_ResearchNarrative.git
cd CS6235_ResearchNarrative
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys

# Run the dashboard
streamlit run app.py
```

See `INSTRUCTIONS.md` for detailed setup and usage instructions.

## Appendix B: AI Workflow

This project was developed using an AI-assisted workflow with Claude (Anthropic) as the primary coding assistant, integrated via the Cursor IDE.

**Benefits:**
- Rapid prototyping of complex modules (citation graph, evaluation framework)
- Consistent code style and comprehensive error handling
- Automated generation of boilerplate (API clients, data models)
- Real-time debugging assistance for API integration issues

**Limitations:**
- AI-generated code required manual verification for correctness
- Complex architectural decisions (chunking strategy, scoring weights) required human judgment
- Domain-specific knowledge (academic API quirks, embedding model selection) needed human input
- Generated code occasionally needed optimization (batch API suggestion came from human observation of performance bottleneck)

**Workflow:**
- All code was reviewed and validated by the developer before integration
- The developer made all design decisions and architectural choices
- AI assisted with implementation speed, not conceptual design
- ~70% of final code was AI-generated, ~30% was human-written or significantly modified
