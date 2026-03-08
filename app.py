"""ResearchNarrative — Streamlit Dashboard

Interactive dashboard for exploring research timelines, thread relationships,
and generated narratives with direct paper links.
"""

import streamlit as st
import logging
import sys
import numpy as np
import pandas as pd
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from src.config import (
    STREAMLIT_PAGE_TITLE,
    STREAMLIT_PAGE_ICON,
    DEFAULT_MAX_PAPERS,
    PAPERS_DIR,
)
from src.pipeline import ResearchNarrativePipeline
from src.models.paper import PaperCollection

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon=STREAMLIT_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #aaa;
        margin-bottom: 2rem;
    }
    .progress-card {
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 14px;
        padding: 0;
        margin-bottom: 1.5rem;
        overflow: hidden;
        background: rgba(30, 30, 50, 0.5);
    }
    .progress-card-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.8rem 1.2rem;
        font-size: 1rem;
        font-weight: 600;
        color: white;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .progress-card-body {
        max-height: 280px;
        overflow-y: auto;
        padding: 0.8rem 1.2rem;
        font-size: 0.88rem;
        line-height: 1.7;
        color: #ccc;
        display: flex;
        flex-direction: column-reverse;
    }
    .progress-card-body .log-lines-wrapper {
        display: flex;
        flex-direction: column;
    }
    .progress-card-body .log-line {
        padding: 3px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .progress-card-body .log-line:last-child { border-bottom: none; }
    .progress-card-body .log-icon { margin-right: 6px; }
    .progress-card-body .log-step {
        color: #667eea;
        font-weight: 600;
    }
    .progress-card-done {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
    }
    .progress-card-error {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
    }
    .paper-card {
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        transition: box-shadow 0.2s;
    }
    .paper-card:hover { box-shadow: 0 2px 12px rgba(102,126,234,0.2); }
    .cluster-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ────────────────────────────────────────────────────────────

if "pipeline" not in st.session_state:
    st.session_state.pipeline = ResearchNarrativePipeline()
if "results" not in st.session_state:
    st.session_state.results = None
if "running" not in st.session_state:
    st.session_state.running = False


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 ResearchNarrative")
    st.markdown("*RAG-Based Research Storyline Engine*")
    st.divider()

    topic = st.text_input(
        "Research Topic",
        placeholder="e.g., transformer architectures in NLP",
        help="Enter a research topic to analyze",
    )

    with st.expander("Advanced Settings", expanded=False):
        max_papers = st.slider("Max Papers", 20, 500, DEFAULT_MAX_PAPERS, step=10)

        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input("Start Year", 1990, 2026, 2015)
        with col2:
            end_year = st.number_input("End Year", 1990, 2026, 2026)

        min_citations = st.number_input("Min Citations", 0, 10000, 0, step=5)

        sources = st.multiselect(
            "Data Sources",
            ["arxiv", "s2"],
            default=["arxiv", "s2"],
        )

        enrich_citations = st.checkbox("Enrich citation data", value=True)

    # Check for cached data
    cached_topics = []
    if PAPERS_DIR.exists():
        cached_topics = [
            f.stem.replace("_", " ")
            for f in PAPERS_DIR.glob("*.json")
        ]
    if cached_topics:
        st.divider()
        st.markdown("**📁 Cached Topics**")
        for ct in cached_topics:
            if st.button(f"Load: {ct}", key=f"load_{ct}"):
                topic = ct
                st.session_state.load_cached = ct

    st.divider()
    run_button = st.button(
        "🚀 Analyze Topic",
        type="primary",
        use_container_width=True,
        disabled=not topic,
    )


# ── Main Content ─────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">📚 ResearchNarrative</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Transform academic literature into structured research storylines'
    '</div>',
    unsafe_allow_html=True,
)

# Handle pipeline execution
if run_button and topic:
    st.session_state.running = True

    step_icons = {
        "init": "🚀", "ingestion": "📥", "embedding": "🧠",
        "indexing": "📇", "clustering": "🧩", "narrative": "📖", "done": "✅",
    }
    step_labels = {
        "init": "Initializing", "ingestion": "Collecting Papers",
        "embedding": "Generating Embeddings", "indexing": "Building Search Index",
        "clustering": "Discovering Threads", "narrative": "Writing Narrative",
        "done": "Complete",
    }

    progress_card = st.empty()
    log_lines = []
    current_step = ["init"]

    def _render_card(header_text: str, lines: list, css_class: str = ""):
        header_cls = f"progress-card-header {css_class}"
        body_html = "".join(
            f'<div class="log-line">'
            f'<span class="log-icon">{icon}</span>'
            f'<span class="log-step">[{label}]</span> {msg}'
            f'</div>'
            for icon, label, msg in lines
        )
        progress_card.markdown(
            f'<div class="progress-card">'
            f'<div class="{header_cls}">{header_text}</div>'
            f'<div class="progress-card-body">'
            f'<div class="log-lines-wrapper">{body_html}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    def on_progress(step: str, message: str):
        icon = step_icons.get(step, "⏳")
        label = step_labels.get(step, step)
        current_step[0] = step
        log_lines.append((icon, label, message))
        header = f"{icon} {label} — {message}"
        _render_card(header, log_lines[-20:])

    _render_card("🚀 Starting pipeline...", [])

    try:
        skip = hasattr(st.session_state, "load_cached")
        results = st.session_state.pipeline.run(
            topic=topic,
            max_papers=max_papers,
            sources=sources if sources else None,
            start_year=int(start_year),
            end_year=int(end_year),
            min_citations=int(min_citations),
            enrich_citations=enrich_citations,
            skip_ingestion=skip,
            progress_callback=on_progress,
        )
        st.session_state.results = results
        if hasattr(st.session_state, "load_cached"):
            del st.session_state.load_cached
        _render_card("✅ Pipeline complete!", log_lines[-20:], "progress-card-done")
    except Exception as e:
        log_lines.append(("❌", "Error", str(e)))
        _render_card(f"❌ Pipeline failed: {e}", log_lines[-20:], "progress-card-error")
        st.session_state.running = False

# Display results
results = st.session_state.results

if results is None:
    st.info(
        "Enter a research topic in the sidebar and click **Analyze Topic** to get started. "
        "The system will retrieve papers, cluster them into research threads, and generate "
        "a structured narrative."
    )
    st.stop()

if "error" in results:
    st.error(results["error"])
    st.stop()

papers = results["papers"]
clusters = results["clusters"]
cluster_labels = results["cluster_labels"]
cluster_stats = results["cluster_stats"]
narrative = results["narrative"]

# ── Metrics Row ──────────────────────────────────────────────────────────────

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Papers Analyzed", len(papers))
with m2:
    n_clusters = len([c for c in clusters if c != -1])
    st.metric("Research Threads", n_clusters)
with m3:
    years = [p.year for p in papers if p.year]
    year_range = f"{min(years)}–{max(years)}" if years else "N/A"
    st.metric("Year Range", year_range)
with m4:
    total_cites = sum(p.citation_count for p in papers)
    st.metric("Total Citations", f"{total_cites:,}")

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_narrative, tab_clusters, tab_timeline, tab_papers, tab_search = st.tabs([
    "📖 Narrative", "🧩 Clusters", "📅 Timeline", "📄 Papers", "🔍 Search"
])

# ── Tab 1: Narrative ─────────────────────────────────────────────────────────

with tab_narrative:
    st.markdown(f"## Research Narrative: {results['topic']}")
    st.markdown(narrative)

    st.divider()
    st.download_button(
        "📥 Download Narrative (Markdown)",
        data=narrative,
        file_name=f"narrative_{results['topic'].replace(' ', '_')}.md",
        mime="text/markdown",
    )

# ── Tab 2: Clusters ─────────────────────────────────────────────────────────

with tab_clusters:
    st.markdown("## Research Threads")

    if cluster_stats:
        df_stats = pd.DataFrame(cluster_stats)
        st.dataframe(
            df_stats[["cluster_id", "label", "size", "year_range", "avg_citations", "top_paper"]],
            use_container_width=True,
            hide_index=True,
        )

    # Cluster visualization (2D UMAP scatter plot)
    umap_2d = results.get("umap_2d")
    if umap_2d is not None:
        st.markdown("### Cluster Map (UMAP 2D Projection)")
        import plotly.express as px

        df_viz = pd.DataFrame({
            "x": umap_2d[:, 0],
            "y": umap_2d[:, 1],
            "cluster": [p.cluster_label for p in papers],
            "title": [p.title[:80] for p in papers],
            "year": [p.year for p in papers],
            "citations": [p.citation_count for p in papers],
        })

        fig = px.scatter(
            df_viz,
            x="x", y="y",
            color="cluster",
            hover_data=["title", "year", "citations"],
            title="Paper Clusters (UMAP Projection)",
            height=550,
        )
        fig.update_layout(
            xaxis_title="", yaxis_title="",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Per-cluster detail
    st.markdown("### Thread Details")
    for cid in sorted(clusters.keys()):
        if cid == -1:
            continue
        label = cluster_labels.get(cid, f"Thread {cid}")
        thread_papers = clusters[cid]

        with st.expander(f"**Thread {cid}: {label}** ({len(thread_papers)} papers)", expanded=False):
            sorted_papers = sorted(thread_papers, key=lambda p: -p.citation_count)
            for p in sorted_papers[:10]:
                first_author = p.authors[0].name if p.authors else "Unknown"
                st.markdown(
                    f"- **[{first_author}, {p.year}]** [{p.title}]({p.url}) "
                    f"— {p.citation_count} citations"
                )

# ── Tab 3: Timeline ─────────────────────────────────────────────────────────

with tab_timeline:
    st.markdown("## Publication Timeline")
    import plotly.express as px
    import plotly.graph_objects as go

    years_all = [p.year for p in papers if p.year]
    if years_all:
        year_counts = Counter(years_all)

        # Overall timeline
        df_timeline = pd.DataFrame(
            sorted(year_counts.items()),
            columns=["Year", "Papers"],
        )
        fig_timeline = px.bar(
            df_timeline, x="Year", y="Papers",
            title="Papers per Year",
            color_discrete_sequence=["#667eea"],
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Per-cluster timeline
        st.markdown("### Threads Over Time")
        timeline_data = []
        for cid, cpapers in clusters.items():
            if cid == -1:
                continue
            label = cluster_labels.get(cid, f"Thread {cid}")
            for p in cpapers:
                if p.year:
                    timeline_data.append({
                        "Year": p.year,
                        "Thread": label,
                        "Citations": p.citation_count,
                    })

        if timeline_data:
            df_threads = pd.DataFrame(timeline_data)
            thread_yearly = df_threads.groupby(["Year", "Thread"]).size().reset_index(name="Count")
            fig_threads = px.line(
                thread_yearly, x="Year", y="Count", color="Thread",
                title="Research Threads Over Time",
                markers=True,
            )
            st.plotly_chart(fig_threads, use_container_width=True)

        # Citation impact over time
        st.markdown("### Citation Impact Over Time")
        cite_data = []
        for p in papers:
            if p.year:
                cite_data.append({
                    "Year": p.year,
                    "Citations": p.citation_count,
                    "Thread": p.cluster_label or "Unclustered",
                })
        if cite_data:
            df_cites = pd.DataFrame(cite_data)
            avg_cites = df_cites.groupby("Year")["Citations"].mean().reset_index()
            fig_cites = px.line(
                avg_cites, x="Year", y="Citations",
                title="Average Citations per Year",
                color_discrete_sequence=["#764ba2"],
            )
            st.plotly_chart(fig_cites, use_container_width=True)

# ── Tab 4: Papers ────────────────────────────────────────────────────────────

with tab_papers:
    st.markdown("## Paper Browser")

    col_filter1, col_filter2, col_filter3 = st.columns(3)
    with col_filter1:
        sort_by = st.selectbox(
            "Sort by",
            ["Citations (high to low)", "Citations (low to high)", "Year (newest)", "Year (oldest)"],
        )
    with col_filter2:
        thread_filter = st.selectbox(
            "Filter by thread",
            ["All"] + [cluster_labels.get(c, f"Thread {c}") for c in sorted(clusters.keys()) if c != -1],
        )
    with col_filter3:
        search_filter = st.text_input("Search titles", "")

    display_papers = list(papers)

    if thread_filter != "All":
        display_papers = [p for p in display_papers if p.cluster_label == thread_filter]

    if search_filter:
        search_lower = search_filter.lower()
        display_papers = [
            p for p in display_papers
            if search_lower in p.title.lower() or search_lower in p.abstract.lower()
        ]

    if sort_by == "Citations (high to low)":
        display_papers.sort(key=lambda p: -p.citation_count)
    elif sort_by == "Citations (low to high)":
        display_papers.sort(key=lambda p: p.citation_count)
    elif sort_by == "Year (newest)":
        display_papers.sort(key=lambda p: -(p.year or 0))
    elif sort_by == "Year (oldest)":
        display_papers.sort(key=lambda p: (p.year or 9999))

    st.markdown(f"*Showing {len(display_papers)} papers*")

    for p in display_papers[:50]:
        first_author = p.authors[0].name if p.authors else "Unknown"
        authors_str = ", ".join(a.name for a in p.authors[:3])
        if len(p.authors) > 3:
            authors_str += f" + {len(p.authors) - 3} more"

        with st.container():
            st.markdown(
                f"**[{p.title}]({p.url})**  \n"
                f"*{authors_str}* — **{p.year}** | "
                f"Citations: {p.citation_count} | "
                f"Thread: {p.cluster_label}"
            )
            with st.expander("Abstract"):
                st.write(p.abstract)
            st.divider()

# ── Tab 5: Similarity Search ────────────────────────────────────────────────

with tab_search:
    st.markdown("## Semantic Search")
    st.markdown("Search for papers by semantic similarity using the FAISS index.")

    search_query = st.text_input(
        "Enter a research question or topic",
        placeholder="e.g., attention mechanism for long documents",
    )

    if search_query:
        vector_store = results.get("vector_store")
        if vector_store:
            with st.spinner("Searching..."):
                query_emb = st.session_state.pipeline.embedder.embed_query(search_query)
                search_results = vector_store.search(query_emb, top_k=15)

            paper_map = {p.paper_id: p for p in papers}
            st.markdown(f"### Results for: *{search_query}*")

            for pid, score in search_results:
                p = paper_map.get(pid)
                if not p:
                    continue
                first_author = p.authors[0].name if p.authors else "Unknown"
                st.markdown(
                    f"**[{p.title}]({p.url})**  \n"
                    f"*{first_author} et al., {p.year}* | "
                    f"Similarity: {score:.3f} | "
                    f"Citations: {p.citation_count} | "
                    f"Thread: {p.cluster_label}"
                )
                with st.expander("Abstract"):
                    st.write(p.abstract)
        else:
            st.warning("Vector store not available. Run the pipeline first.")
