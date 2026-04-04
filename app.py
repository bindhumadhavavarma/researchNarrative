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
        "indexing": "📇", "clustering": "🧩", "citation_graph": "🔗",
        "narrative": "📖", "done": "✅",
    }
    step_labels = {
        "init": "Initializing", "ingestion": "Collecting Papers",
        "embedding": "Generating Embeddings", "indexing": "Building Search Index",
        "clustering": "Discovering Threads", "citation_graph": "Citation Graph Analysis",
        "narrative": "Writing Narrative", "done": "Complete",
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

m1, m2, m3, m4, m5 = st.columns(5)
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
with m5:
    cg = results.get("citation_graph")
    n_edges = cg.graph.number_of_edges() if cg else 0
    st.metric("Citation Links", n_edges)

st.divider()

def _build_html_export(topic, narrative, verification, thread_narratives, cluster_labels):
    """Build a styled HTML export of the narrative."""
    import markdown
    try:
        narrative_html = markdown.markdown(narrative, extensions=["extra", "smarty"])
    except Exception:
        narrative_html = f"<pre>{narrative}</pre>"

    thread_sections = ""
    if thread_narratives:
        thread_sections = "<h2>Thread Deep-Dives</h2>"
        for cid in sorted(thread_narratives.keys()):
            label = cluster_labels.get(cid, f"Thread {cid}")
            try:
                t_html = markdown.markdown(thread_narratives[cid], extensions=["extra", "smarty"])
            except Exception:
                t_html = f"<pre>{thread_narratives[cid]}</pre>"
            thread_sections += f"<h3>Thread {cid}: {label}</h3>{t_html}"

    ver_section = ""
    if verification:
        stats = verification["stats"]
        ver_section = (
            f'<div class="verification-badge">'
            f'Citation Accuracy: {stats["verified_count"]}/{stats["total"]} '
            f'({stats["accuracy"]:.0%})'
            f'</div>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Research Narrative: {topic}</title>
<style>
body {{ font-family: 'Georgia', serif; max-width: 800px; margin: 2rem auto;
       padding: 0 1rem; line-height: 1.8; color: #333; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #667eea; padding-bottom: 0.5rem; }}
h2 {{ color: #34495e; margin-top: 2rem; }}
h3 {{ color: #667eea; }}
a {{ color: #667eea; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.verification-badge {{ background: #f0f0f0; padding: 8px 16px; border-radius: 8px;
                       font-size: 0.9rem; margin-bottom: 1.5rem; display: inline-block; }}
.footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #ddd;
           font-size: 0.85rem; color: #888; }}
</style>
</head>
<body>
<h1>Research Narrative: {topic}</h1>
{ver_section}
{narrative_html}
<hr>
{thread_sections}
<div class="footer">
Generated by ResearchNarrative — RAG-Based Research Storyline Generation Engine
</div>
</body>
</html>"""


# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_narrative, tab_clusters, tab_citations, tab_timeline, tab_papers, tab_search = st.tabs([
    "📖 Narrative", "🧩 Clusters", "🔗 Citation Analysis", "📅 Timeline", "📄 Papers", "🔍 Search"
])

# ── Tab 1: Narrative ─────────────────────────────────────────────────────────

with tab_narrative:
    st.markdown(f"## Research Narrative: {results['topic']}")

    # Citation verification badge
    verification = results.get("citation_verification")
    if verification:
        stats = verification["stats"]
        if stats["total"] > 0:
            accuracy_pct = stats["accuracy"] * 100
            if accuracy_pct >= 90:
                badge_color = "#2ecc71"
                badge_label = "High"
            elif accuracy_pct >= 70:
                badge_color = "#f39c12"
                badge_label = "Medium"
            else:
                badge_color = "#e74c3c"
                badge_label = "Low"
            st.markdown(
                f'<div style="display:inline-flex;align-items:center;gap:8px;margin-bottom:1rem;">'
                f'<span style="background:{badge_color};color:white;padding:3px 10px;'
                f'border-radius:12px;font-size:0.8rem;font-weight:600;">'
                f'✓ Citation Accuracy: {badge_label} ({accuracy_pct:.0f}%)</span>'
                f'<span style="color:#aaa;font-size:0.8rem;">'
                f'{stats["verified_count"]}/{stats["total"]} citations verified against paper database</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Display narrative with clickable citation links
    pipeline = st.session_state.pipeline
    if pipeline.narrative_gen.verifier:
        linked_narrative = pipeline.narrative_gen.verifier.add_paper_links(narrative)
        st.markdown(linked_narrative)
    else:
        st.markdown(narrative)

    # Unverified citations warning
    if verification and verification["unverified"]:
        with st.expander(f"⚠️ {len(verification['unverified'])} unverified citations"):
            st.markdown(
                "These citations in the narrative could not be matched to papers "
                "in the retrieved collection. They may reference papers outside the "
                "search scope or use slightly different author name formats."
            )
            for cite in verification["unverified"]:
                st.markdown(f"- {cite}")

    st.divider()

    # Thread deep-dives
    thread_narratives = results.get("thread_narratives", {})
    if thread_narratives:
        st.markdown("### Thread Deep-Dives")
        for cid in sorted(thread_narratives.keys()):
            label = cluster_labels.get(cid, f"Thread {cid}")
            with st.expander(f"📖 Thread {cid}: {label}"):
                thread_text = thread_narratives[cid]
                if pipeline.narrative_gen.verifier:
                    thread_text = pipeline.narrative_gen.verifier.add_paper_links(thread_text)
                st.markdown(thread_text)
        st.divider()

    # Download options
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "📥 Download Narrative (Markdown)",
            data=narrative,
            file_name=f"narrative_{results['topic'].replace(' ', '_')}.md",
            mime="text/markdown",
        )
    with dl_col2:
        html_narrative = _build_html_export(results['topic'], narrative, verification, thread_narratives, cluster_labels)
        st.download_button(
            "📥 Download Narrative (HTML)",
            data=html_narrative,
            file_name=f"narrative_{results['topic'].replace(' ', '_')}.html",
            mime="text/html",
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

# ── Tab 3: Citation Analysis ─────────────────────────────────────────────────

with tab_citations:
    st.markdown("## Citation Graph Analysis")
    import plotly.express as px
    import plotly.graph_objects as go

    citation_graph = results.get("citation_graph")
    influence_scores = results.get("influence_scores")
    competition_analysis = results.get("competition_analysis")

    if citation_graph:
        cg_m1, cg_m2, cg_m3 = st.columns(3)
        with cg_m1:
            st.metric("Graph Nodes", citation_graph.graph.number_of_nodes())
        with cg_m2:
            st.metric("Internal Citations", citation_graph.graph.number_of_edges())
        with cg_m3:
            density = 0
            if citation_graph.graph.number_of_nodes() > 1:
                n = citation_graph.graph.number_of_nodes()
                density = citation_graph.graph.number_of_edges() / (n * (n - 1))
            st.metric("Graph Density", f"{density:.4f}")

    # Influence scores
    if influence_scores:
        st.markdown("### Most Influential Papers")
        st.markdown(
            "*Composite score combines PageRank (structural importance), "
            "HITS authority, bridge score (cross-cluster connections), "
            "temporal pioneer (early entrant), and citation burst.*"
        )

        ranked_papers = sorted(
            [(pid, s) for pid, s in influence_scores.items()],
            key=lambda x: x[1].get("composite", 0),
            reverse=True,
        )[:20]

        inf_data = []
        paper_map = {p.paper_id: p for p in papers}
        for pid, scores in ranked_papers:
            p = paper_map.get(pid)
            if not p:
                continue
            first_author = p.authors[0].name if p.authors else "Unknown"
            inf_data.append({
                "Paper": f"{first_author} et al., {p.year}",
                "Title": p.title[:60],
                "Composite": round(scores["composite"], 4),
                "PageRank": round(scores["pagerank"], 4),
                "Authority": round(scores["authority"], 4),
                "Bridge": round(scores["bridge"], 4),
                "Pioneer": round(scores["temporal_pioneer"], 4),
                "Burst": round(scores["citation_burst"], 4),
                "Thread": p.cluster_label,
            })

        if inf_data:
            df_inf = pd.DataFrame(inf_data)
            st.dataframe(df_inf, use_container_width=True, hide_index=True)

            # Radar chart for top-5 influential papers
            st.markdown("### Influence Profile — Top 5 Papers")
            top5 = inf_data[:5]
            categories = ["PageRank", "Authority", "Bridge", "Pioneer", "Burst"]

            fig_radar = go.Figure()
            for row in top5:
                values = [row[c] for c in categories]
                values.append(values[0])
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=row["Paper"],
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=480,
                title="Influence Dimensions",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # Competition analysis
    if competition_analysis:
        comp_pairs = competition_analysis.get("competition_pairs", [])
        complementary_pairs = competition_analysis.get("complementary_pairs", [])

        if comp_pairs:
            st.markdown("### Competing Research Threads")
            st.markdown(
                "*Competing threads cite each other frequently with balanced "
                "cross-citation flow, indicating alternative approaches.*"
            )
            comp_data = []
            for cp in comp_pairs:
                comp_data.append({
                    "Thread A": cp["label_a"],
                    "Thread B": cp["label_b"],
                    "A → B": cp["a_cites_b"],
                    "B → A": cp["b_cites_a"],
                    "Total Cross-Citations": cp["total_cross_citations"],
                    "Asymmetry": cp["asymmetry"],
                })
            df_comp = pd.DataFrame(comp_data)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

            # Sankey diagram of cross-citations
            if len(comp_pairs) > 0:
                st.markdown("### Cross-Citation Flow")
                all_labels_set = set()
                for cp in comp_pairs:
                    all_labels_set.add(cp["label_a"])
                    all_labels_set.add(cp["label_b"])
                all_labels_list = sorted(all_labels_set)
                label_idx = {l: i for i, l in enumerate(all_labels_list)}

                sankey_source = []
                sankey_target = []
                sankey_value = []

                for cp in comp_pairs:
                    if cp["a_cites_b"] > 0:
                        sankey_source.append(label_idx[cp["label_a"]])
                        sankey_target.append(label_idx[cp["label_b"]])
                        sankey_value.append(cp["a_cites_b"])
                    if cp["b_cites_a"] > 0:
                        sankey_source.append(label_idx[cp["label_b"]])
                        sankey_target.append(label_idx[cp["label_a"]])
                        sankey_value.append(cp["b_cites_a"])

                if sankey_value:
                    fig_sankey = go.Figure(data=[go.Sankey(
                        node=dict(
                            pad=15, thickness=20, line=dict(color="black", width=0.5),
                            label=all_labels_list,
                        ),
                        link=dict(
                            source=sankey_source, target=sankey_target, value=sankey_value,
                        ),
                    )])
                    fig_sankey.update_layout(
                        title_text="Citation Flow Between Competing Threads",
                        height=400,
                    )
                    st.plotly_chart(fig_sankey, use_container_width=True)

        if complementary_pairs:
            st.markdown("### Complementary Threads")
            st.markdown(
                "*One thread builds heavily on another (asymmetric citation flow).*"
            )
            for cp in complementary_pairs:
                st.markdown(
                    f"- **{cp['foundation_label']}** → **{cp['builder_label']}** "
                    f"({cp['builder_to_foundation']} citations toward foundation, "
                    f"asymmetry: {cp['asymmetry']})"
                )

        # Dominance timeline
        dominance = competition_analysis.get("dominance_timeline", {})
        if dominance:
            st.markdown("### Thread Dominance Over Time")
            st.markdown("*Share of papers and citations per year by research thread.*")

            dom_data = []
            for year, entries in sorted(dominance.items()):
                for entry in entries:
                    dom_data.append({
                        "Year": int(year),
                        "Thread": entry["label"],
                        "Paper Share": entry["paper_share"],
                        "Citation Share": entry["citation_share"],
                        "Papers": entry["papers"],
                    })

            if dom_data:
                df_dom = pd.DataFrame(dom_data)

                fig_dom = px.area(
                    df_dom, x="Year", y="Paper Share", color="Thread",
                    title="Thread Dominance by Paper Share",
                    groupnorm="fraction",
                    height=420,
                )
                st.plotly_chart(fig_dom, use_container_width=True)

                fig_cite_dom = px.area(
                    df_dom, x="Year", y="Citation Share", color="Thread",
                    title="Thread Dominance by Citation Share",
                    groupnorm="fraction",
                    height=420,
                )
                st.plotly_chart(fig_cite_dom, use_container_width=True)

    if not citation_graph and not influence_scores and not competition_analysis:
        st.info("Citation graph analysis data is not available. Run the pipeline to see results.")

# ── Tab 4: Timeline ─────────────────────────────────────────────────────────

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
