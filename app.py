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
    .metric-card {
        text-align: center;
        padding: 0.8rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        background: rgba(30, 30, 50, 0.3);
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #aaa;
        margin-top: 2px;
    }
    .citation-node-card {
        border: 1px solid rgba(102,126,234,0.2);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        background: rgba(30, 30, 50, 0.3);
    }
    .score-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
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


# ── Helper: Full Portal HTML Export ──────────────────────────────────────────

def _build_portal_html(results, cluster_labels):
    """Build a self-contained HTML page mirroring the full portal with interactive Plotly charts."""
    import markdown
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio

    BRIGHT_COLORS = ['#667eea', '#f093fb', '#2ecc71', '#f39c12', '#e74c3c',
                     '#3498db', '#9b59b6', '#1abc9c', '#e67e22', '#00d2d3']

    pio.templates.default = 'plotly_dark'

    PLOTLY_DARK_TEMPLATE = dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        colorway=BRIGHT_COLORS,
    )

    def _fig_html(fig, height=None):
        if height:
            fig.update_layout(height=height)
        fig.update_layout(**PLOTLY_DARK_TEMPLATE)
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.08)', linecolor='rgba(255,255,255,0.15)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.08)', linecolor='rgba(255,255,255,0.15)')
        return fig.to_html(include_plotlyjs=False, full_html=False, config={"displayModeBar": False})

    topic = results["topic"]
    narrative = results["narrative"]
    verification = results.get("citation_verification")
    thread_narratives = results.get("thread_narratives", {})
    eval_data = results.get("evaluation", {})
    papers = results["papers"]
    clusters = results["clusters"]
    influence_scores = results.get("influence_scores", {})
    competition_analysis = results.get("competition_analysis", {})
    umap_2d = results.get("umap_2d")
    citation_graph = results.get("citation_graph")

    pipeline_ref = st.session_state.pipeline
    verifier = pipeline_ref.narrative_gen.verifier
    linked_narrative = verifier.add_paper_links(narrative) if verifier else narrative

    try:
        narrative_html = markdown.markdown(linked_narrative, extensions=["extra", "smarty"])
    except Exception:
        narrative_html = f"<pre>{linked_narrative}</pre>"

    # Thread narratives
    thread_sections = ""
    if thread_narratives:
        for cid in sorted(thread_narratives.keys()):
            label = cluster_labels.get(cid, f"Thread {cid}")
            linked_thread = verifier.add_paper_links(thread_narratives[cid]) if verifier else thread_narratives[cid]
            try:
                t_html = markdown.markdown(linked_thread, extensions=["extra", "smarty"])
            except Exception:
                t_html = f"<pre>{linked_thread}</pre>"
            thread_sections += (
                f'<details class="thread-card"><summary><strong>Thread {cid}: {label}</strong></summary>'
                f'{t_html}</details>'
            )

    # Verification badge
    ver_section = ""
    if verification:
        stats = verification["stats"]
        if stats["total"] > 0:
            acc = stats["accuracy"] * 100
            color = "#2ecc71" if acc >= 90 else "#f39c12" if acc >= 70 else "#e74c3c"
            level = "High" if acc >= 90 else "Medium" if acc >= 70 else "Low"
            ver_section = (
                f'<div style="margin-bottom:1rem;">'
                f'<span class="badge" style="background:{color};">'
                f'Narrative Citations: {stats["verified_count"]}/{stats["total"]} grounded ({acc:.0f}%)</span>'
                f'<span class="badge-sub">'
                f'{stats["verified_count"]} citations used in generating narrative matched to retrieved papers</span></div>'
            )

    # ── Cluster tab: UMAP chart + thread cards ──
    real_clusters = {cid: ps for cid, ps in clusters.items() if cid != -1}

    umap_chart_html = ""
    if umap_2d is not None:
        df_viz = pd.DataFrame({
            "x": umap_2d[:, 0], "y": umap_2d[:, 1],
            "cluster": [p.cluster_label for p in papers],
            "title": [p.title[:80] for p in papers],
            "year": [p.year for p in papers],
            "citations": [p.citation_count for p in papers],
        })
        fig_umap = px.scatter(
            df_viz, x="x", y="y", color="cluster",
            hover_data=["title", "year", "citations"],
            title="Paper Clusters (UMAP Projection)", height=550,
            color_discrete_sequence=BRIGHT_COLORS,
        )
        fig_umap.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.5, color='rgba(255,255,255,0.3)')))
        fig_umap.update_layout(xaxis_title="", yaxis_title="",
                               xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
        umap_chart_html = _fig_html(fig_umap, 550)

    # Cluster stats table
    cluster_stats = results.get("cluster_stats", [])
    cluster_table = ""
    if cluster_stats:
        rows = ""
        for cs in cluster_stats:
            rows += (f'<tr><td>{cs["cluster_id"]}</td><td>{cs["label"]}</td>'
                     f'<td>{cs["size"]}</td><td>{cs["year_range"]}</td>'
                     f'<td>{cs["avg_citations"]}</td><td>{cs["top_paper"][:60]}</td></tr>')
        cluster_table = (
            '<table class="data-table"><thead><tr><th>ID</th><th>Label</th><th>Size</th>'
            '<th>Years</th><th>Avg Cites</th><th>Top Paper</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>'
        )

    cluster_details = ""
    for cid in sorted(real_clusters.keys()):
        label = cluster_labels.get(cid, f"Thread {cid}")
        cpapers = real_clusters[cid]
        years_c = [p.year for p in cpapers if p.year]
        yr = f"{min(years_c)}-{max(years_c)}" if years_c else "N/A"
        top = sorted(cpapers, key=lambda p: -p.citation_count)[:5]
        paper_list = "".join(
            f'<li><a href="{p.url}">{(p.authors[0].name if p.authors else "Unknown")} et al., {p.year}</a> — '
            f'{p.title} ({p.citation_count} cites)</li>'
            for p in top
        )
        cluster_details += (
            f'<details class="thread-card"><summary><strong>Thread {cid}: {label}</strong> '
            f'({len(cpapers)} papers · {yr})</summary><ul>{paper_list}</ul></details>'
        )

    # ── Citation Analysis tab: charts ──
    paper_map = {p.paper_id: p for p in papers}

    # Influence table
    influential_table = ""
    inf_bar_chart = ""
    inf_radar_chart = ""
    if influence_scores:
        ranked = sorted(influence_scores.items(), key=lambda x: x[1].get("composite", 0), reverse=True)[:20]
        inf_data = []
        for pid, scores in ranked:
            p = paper_map.get(pid)
            if not p:
                continue
            fa = p.authors[0].name if p.authors else "Unknown"
            inf_data.append({
                "Paper": f"{fa} et al., {p.year}", "Title": p.title[:60],
                "Composite": round(scores["composite"], 4),
                "PageRank": round(scores["pagerank"], 4),
                "Authority": round(scores["authority"], 4),
                "Bridge": round(scores["bridge"], 4),
                "Pioneer": round(scores["temporal_pioneer"], 4),
                "Burst": round(scores["citation_burst"], 4),
                "Thread": p.cluster_label, "url": p.url or "",
            })

        if inf_data:
            rows = "".join(
                f'<tr><td><a href="{r["url"]}">{r["Paper"]}</a></td><td>{r["Title"]}</td>'
                f'<td>{r["Composite"]}</td><td>{r["PageRank"]}</td><td>{r["Authority"]}</td>'
                f'<td>{r["Bridge"]}</td><td>{r["Pioneer"]}</td><td>{r["Burst"]}</td>'
                f'<td>{r["Thread"]}</td></tr>'
                for r in inf_data
            )
            influential_table = (
                '<table class="data-table"><thead><tr><th>Paper</th><th>Title</th>'
                '<th>Composite</th><th>PageRank</th><th>Authority</th><th>Bridge</th>'
                '<th>Pioneer</th><th>Burst</th><th>Thread</th></tr></thead>'
                f'<tbody>{rows}</tbody></table>'
            )

            # Stacked bar chart
            top10 = inf_data[:10]
            categories = ["PageRank", "Authority", "Bridge", "Pioneer", "Burst"]
            colors = {"PageRank": "#667eea", "Authority": "#764ba2", "Bridge": "#2ecc71",
                      "Pioneer": "#f39c12", "Burst": "#e74c3c"}
            fig_bar = go.Figure()
            for cat in categories:
                fig_bar.add_trace(go.Bar(
                    y=[r["Paper"] for r in reversed(top10)],
                    x=[r[cat] for r in reversed(top10)],
                    name=cat, orientation='h', marker_color=colors[cat],
                ))
            fig_bar.update_layout(barmode='stack', title="Influence Components (Top 10)",
                                  xaxis_title="Score", margin=dict(l=200),
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            inf_bar_chart = _fig_html(fig_bar, 420)

            # Radar chart top 5
            top5 = inf_data[:5]
            fig_radar = go.Figure()
            for row in top5:
                vals = [row[c] for c in categories] + [row[categories[0]]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=categories + [categories[0]], fill='toself', name=row["Paper"],
                ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.2)', linecolor='rgba(255,255,255,0.3)'),
                    angularaxis=dict(gridcolor='rgba(255,255,255,0.2)', linecolor='rgba(255,255,255,0.3)'),
                    bgcolor='rgba(0,0,0,0)',
                ),
                showlegend=True, title="Influence Dimensions")
            inf_radar_chart = _fig_html(fig_radar, 480)

    # Competition: Sankey
    sankey_html = ""
    comp_table = ""
    if competition_analysis:
        comp_pairs = competition_analysis.get("competition_pairs", [])
        if comp_pairs:
            comp_rows = "".join(
                f'<tr><td>{cp["label_a"]}</td><td>{cp["label_b"]}</td>'
                f'<td>{cp["a_cites_b"]}</td><td>{cp["b_cites_a"]}</td>'
                f'<td>{cp["total_cross_citations"]}</td><td>{cp["asymmetry"]}</td></tr>'
                for cp in comp_pairs
            )
            comp_table = (
                '<h3>Competing Research Threads</h3>'
                '<table class="data-table"><thead><tr><th>Thread A</th><th>Thread B</th>'
                '<th>A→B</th><th>B→A</th><th>Total</th><th>Asymmetry</th></tr></thead>'
                f'<tbody>{comp_rows}</tbody></table>'
            )

            all_labels_set = set()
            for cp in comp_pairs:
                all_labels_set.add(cp["label_a"])
                all_labels_set.add(cp["label_b"])
            all_labels_list = sorted(all_labels_set)
            label_idx = {l: i for i, l in enumerate(all_labels_list)}
            s_src, s_tgt, s_val = [], [], []
            for cp in comp_pairs:
                if cp["a_cites_b"] > 0:
                    s_src.append(label_idx[cp["label_a"]]); s_tgt.append(label_idx[cp["label_b"]]); s_val.append(cp["a_cites_b"])
                if cp["b_cites_a"] > 0:
                    s_src.append(label_idx[cp["label_b"]]); s_tgt.append(label_idx[cp["label_a"]]); s_val.append(cp["b_cites_a"])
            if s_val:
                node_colors = ['#667eea', '#f093fb', '#2ecc71', '#f39c12', '#e74c3c',
                               '#3498db', '#9b59b6', '#1abc9c', '#e67e22', '#00d2d3']
                fig_sankey = go.Figure(data=[go.Sankey(
                    node=dict(pad=15, thickness=20, label=all_labels_list,
                              color=[node_colors[i % len(node_colors)] for i in range(len(all_labels_list))]),
                    link=dict(source=s_src, target=s_tgt, value=s_val,
                              color=['rgba(102,126,234,0.4)'] * len(s_val)),
                )])
                fig_sankey.update_layout(title_text="Citation Flow Between Competing Threads")
                sankey_html = _fig_html(fig_sankey, 400)

        # Dominance timeline charts
        dominance = competition_analysis.get("dominance_timeline", {})
        dominance_charts = ""
        if dominance:
            dom_data = []
            for year, entries in sorted(dominance.items()):
                for entry in entries:
                    dom_data.append({"Year": int(year), "Thread": entry["label"],
                                     "Paper Share": entry["paper_share"],
                                     "Citation Share": entry["citation_share"]})
            if dom_data:
                df_dom = pd.DataFrame(dom_data)
                fig_dom = px.area(df_dom, x="Year", y="Paper Share", color="Thread",
                                  title="Thread Dominance by Paper Share", groupnorm="fraction",
                                  color_discrete_sequence=BRIGHT_COLORS)
                fig_cite = px.area(df_dom, x="Year", y="Citation Share", color="Thread",
                                   title="Thread Dominance by Citation Share", groupnorm="fraction",
                                   color_discrete_sequence=BRIGHT_COLORS)
                dominance_charts = '<h3>Thread Dominance Over Time</h3>' + _fig_html(fig_dom, 400) + _fig_html(fig_cite, 400)
    else:
        dominance_charts = ""

    # ── Evaluation tab ──
    eval_section = ""
    if eval_data:
        overall = eval_data.get("overall", {})
        component_scores = overall.get("component_scores", {})
        grade = pipeline_ref.evaluator.get_grade()
        grade_colors = {"A+": "#2ecc71", "A": "#27ae60", "B+": "#f1c40f",
                        "B": "#f39c12", "C": "#e67e22", "D": "#e74c3c"}
        g_color = grade_colors.get(grade, "#95a5a6")

        component_rows = ""
        label_map = {"retrieval": "Retrieval", "clustering": "Clustering",
                     "citation_graph": "Citation Graph", "narrative": "Narrative"}
        for k, lbl in label_map.items():
            s = component_scores.get(k, 0)
            bar_color = "#2ecc71" if s >= 0.7 else "#f39c12" if s >= 0.5 else "#e74c3c"
            component_rows += (
                f'<tr><td>{lbl}</td><td>'
                f'<div class="score-bar-bg"><div class="score-bar" '
                f'style="width:{s*100:.0f}%;background:{bar_color};"></div></div>'
                f'</td><td style="text-align:right;font-weight:600;">{s:.0%}</td></tr>'
            )

        # Radar
        r_cats = list(label_map.keys())
        r_labels = [label_map[k] for k in r_cats]
        r_vals = [component_scores.get(k, 0) for k in r_cats]
        fig_eval = go.Figure(data=[go.Scatterpolar(
            r=r_vals + [r_vals[0]], theta=r_labels + [r_labels[0]],
            fill='toself', line=dict(color='#667eea'), fillcolor='rgba(102,126,234,0.2)',
        )])
        fig_eval.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.2)', linecolor='rgba(255,255,255,0.3)'),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.2)', linecolor='rgba(255,255,255,0.3)'),
                bgcolor='rgba(0,0,0,0)',
            ),
            showlegend=False, title="Component Score Radar")
        eval_radar = _fig_html(fig_eval, 400)

        # Detail metrics
        r_det = eval_data.get("retrieval", {}).get("details", {})
        c_det = eval_data.get("clustering", {}).get("details", {})
        cg_det = eval_data.get("citation_graph", {}).get("details", {})
        n_det = eval_data.get("narrative", {}).get("details", {})

        detail_html = (
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:2rem;margin-top:1rem;">'
            '<div><h4>Retrieval</h4><ul>'
            f'<li>Papers: <strong>{r_det.get("total_papers",0)}</strong></li>'
            f'<li>Abstract coverage: <strong>{r_det.get("abstract_coverage",0):.0%}</strong></li>'
            f'<li>Metadata completeness: <strong>{r_det.get("metadata_completeness",0):.0%}</strong></li>'
            f'<li>Year span: <strong>{r_det.get("year_span",0)} years</strong></li>'
            f'<li>Unique authors: <strong>{r_det.get("unique_authors",0)}</strong></li></ul>'
            f'<h4>Citation Graph</h4><ul>'
            f'<li>Nodes: <strong>{cg_det.get("nodes",0)}</strong>, Edges: <strong>{cg_det.get("edges",0)}</strong></li>'
            f'<li>Density: <strong>{cg_det.get("density",0):.6f}</strong></li>'
            f'<li>Connectivity: <strong>{cg_det.get("connectivity",0):.0%}</strong></li>'
            f'<li>Avg in-degree: <strong>{cg_det.get("avg_in_degree",0):.2f}</strong></li></ul></div>'
            '<div><h4>Clustering</h4><ul>'
            f'<li>Clusters: <strong>{c_det.get("n_clusters",0)}</strong></li>'
            f'<li>Silhouette: <strong>{c_det.get("silhouette_score","N/A")}</strong></li>'
            f'<li>Noise ratio: <strong>{c_det.get("noise_ratio",0):.0%}</strong></li>'
            f'<li>Balance: <strong>{c_det.get("balance_score",0):.0%}</strong></li></ul>'
            f'<h4>Narrative</h4><ul>'
            f'<li>Word count: <strong>{n_det.get("word_count",0):,}</strong></li>'
            f'<li>Sections: <strong>{n_det.get("n_sections",0)}</strong></li>'
            f'<li>Citations: <strong>{n_det.get("n_citations",0)}</strong> ({n_det.get("unique_citations",0)} unique)</li>'
            f'<li>Density: <strong>{n_det.get("citation_density",0):.1f}</strong> per paragraph</li>'
            f'<li>Coverage: <strong>{n_det.get("paper_coverage",0):.0%}</strong></li>'
            f'<li>Accuracy: <strong>{n_det.get("citation_accuracy",0):.0%}</strong></li></ul></div></div>'
        )

        recs_html = ""
        recs = pipeline_ref.evaluator.get_recommendations()
        if recs:
            recs_html = '<h3>Recommendations</h3><ul>' + "".join(f'<li>{r}</li>' for r in recs) + '</ul>'

        eval_section = (
            f'<div class="eval-grade" style="border-color:{g_color};">'
            f'<span class="grade-letter" style="color:{g_color};">{grade}</span>'
            f'<span class="grade-score">Overall: {overall.get("score", 0):.0%}</span></div>'
            f'<h3>Component Scores</h3>'
            f'<table class="score-table"><thead><tr><th>Component</th><th>Score</th><th></th></tr></thead>'
            f'<tbody>{component_rows}</tbody></table>'
            f'{eval_radar}{detail_html}{recs_html}'
        )

    # ── Timeline tab: Plotly charts ──
    years_all = [p.year for p in papers if p.year]
    timeline_charts = ""
    if years_all:
        year_counts = Counter(years_all)
        df_tl = pd.DataFrame(sorted(year_counts.items()), columns=["Year", "Papers"])
        fig_tl = px.bar(df_tl, x="Year", y="Papers", title="Papers per Year",
                        color_discrete_sequence=["#667eea"])
        timeline_charts += _fig_html(fig_tl, 400)

        tl_data = []
        for cid, cpapers in clusters.items():
            if cid == -1:
                continue
            label = cluster_labels.get(cid, f"Thread {cid}")
            for p in cpapers:
                if p.year:
                    tl_data.append({"Year": p.year, "Thread": label})
        if tl_data:
            df_th = pd.DataFrame(tl_data)
            thread_yearly = df_th.groupby(["Year", "Thread"]).size().reset_index(name="Count")
            fig_th = px.line(thread_yearly, x="Year", y="Count", color="Thread",
                             title="Research Threads Over Time", markers=True,
                             color_discrete_sequence=BRIGHT_COLORS)
            fig_th.update_traces(line=dict(width=2.5))
            timeline_charts += _fig_html(fig_th, 400)

        cite_data = [{"Year": p.year, "Citations": p.citation_count} for p in papers if p.year]
        if cite_data:
            df_c = pd.DataFrame(cite_data)
            avg_c = df_c.groupby("Year")["Citations"].mean().reset_index()
            fig_c = px.line(avg_c, x="Year", y="Citations", title="Average Citations per Year",
                            color_discrete_sequence=["#764ba2"])
            timeline_charts += _fig_html(fig_c, 400)

    # ── Stats ──
    n_clusters = len(real_clusters)
    year_range = f"{min(years_all)}-{max(years_all)}" if years_all else "N/A"
    total_cites = sum(p.citation_count for p in papers)
    cg = results.get("citation_graph")
    n_edges = cg.graph.number_of_edges() if cg else 0

    # ── Paper browser ──
    papers_table_rows = ""
    sorted_papers = sorted(papers, key=lambda p: -p.citation_count)[:100]
    for p in sorted_papers:
        authors_str = ", ".join(a.name for a in p.authors[:3])
        if len(p.authors) > 3:
            authors_str += f" +{len(p.authors)-3}"
        papers_table_rows += (
            f'<tr><td><a href="{p.url}">{p.title}</a></td>'
            f'<td>{authors_str}</td><td>{p.year}</td>'
            f'<td>{p.citation_count}</td><td>{p.cluster_label}</td></tr>'
        )

    quality_pct = eval_data.get('overall', {}).get('score', 0)

    pio.templates.default = 'plotly'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Research Narrative: {topic}</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
:root {{
    --primary: #667eea;
    --primary-dark: #764ba2;
    --bg: #0e1117;
    --surface: #1a1d23;
    --surface-light: #262730;
    --text: #e0e0e0;
    --text-dim: #aaa;
    --border: rgba(255,255,255,0.1);
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.7;
    padding: 2rem; max-width: 1200px; margin: 0 auto;
}}
h1 {{
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}}
h2 {{ color: var(--text); margin: 2rem 0 1rem; padding-bottom: 0.5rem;
      border-bottom: 2px solid var(--primary); font-size: 1.4rem; }}
h3 {{ color: var(--primary); margin: 1.5rem 0 0.8rem; font-size: 1.15rem; }}
h4 {{ color: var(--text); margin: 1rem 0 0.5rem; font-size: 1.05rem; }}
a {{ color: var(--primary); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
p {{ margin-bottom: 1rem; }}
.subtitle {{ color: var(--text-dim); font-size: 1.05rem; margin-bottom: 2rem; }}
.metrics-row {{ display: grid; grid-template-columns: repeat(6,1fr); gap: 1rem; margin: 1.5rem 0; }}
.metric-box {{ text-align:center; padding:1rem; background:var(--surface);
               border:1px solid var(--border); border-radius:10px; }}
.metric-box .val {{ font-size:1.6rem; font-weight:700; color:var(--primary); }}
.metric-box .lbl {{ font-size:0.8rem; color:var(--text-dim); margin-top:4px; }}
.tabs {{ display:flex; gap:0; border-bottom:2px solid var(--border); margin:2rem 0 0; position:sticky; top:0; background:var(--bg); z-index:10; }}
.tab {{ padding:0.6rem 1.2rem; cursor:pointer; font-weight:600; font-size:0.9rem;
        color:var(--text-dim); border-bottom:3px solid transparent; transition:all 0.2s;
        background:none; border-top:none; border-left:none; border-right:none; }}
.tab:hover {{ color:var(--text); }}
.tab.active {{ color:var(--primary); border-bottom-color:var(--primary); }}
.tab-content {{ display:none; padding:1.5rem 0; }}
.tab-content.active {{ display:block; }}
.thread-card {{ background:var(--surface); border:1px solid var(--border);
                border-radius:10px; padding:1.2rem; margin-bottom:1rem; }}
.thread-card summary {{ cursor:pointer; padding:0.3rem 0; }}
.badge {{ display:inline-block; padding:3px 12px; border-radius:12px; font-size:0.8rem;
          font-weight:600; color:white; margin-right:8px; }}
.badge-sub {{ color:var(--text-dim); font-size:0.8rem; }}
.data-table {{ width:100%; border-collapse:collapse; margin:1rem 0; font-size:0.85rem; }}
.data-table th {{ background:var(--surface-light); padding:8px 12px; text-align:left;
                  font-weight:600; border-bottom:2px solid var(--border); position:sticky; top:40px; }}
.data-table td {{ padding:8px 12px; border-bottom:1px solid var(--border); }}
.data-table tr:hover td {{ background:rgba(102,126,234,0.05); }}
.eval-grade {{ text-align:center; margin:1.5rem auto; padding:1.5rem;
               border:3px solid; border-radius:16px; max-width:200px; }}
.grade-letter {{ display:block; font-size:3rem; font-weight:700; }}
.grade-score {{ display:block; font-size:0.9rem; color:var(--text-dim); margin-top:4px; }}
.score-table {{ width:100%; border-collapse:collapse; margin:1rem 0; }}
.score-table th,.score-table td {{ padding:8px 12px; text-align:left; }}
.score-table th {{ border-bottom:2px solid var(--border); font-weight:600; }}
.score-table td {{ border-bottom:1px solid var(--border); }}
.score-bar-bg {{ background:var(--surface-light); border-radius:6px; height:10px; width:100%; }}
.score-bar {{ height:10px; border-radius:6px; }}
.narrative-body {{ font-size:1rem; line-height:1.85; }}
.narrative-body h2 {{ margin-top:2.5rem; }}
.narrative-body p {{ margin-bottom:1.2rem; }}
.footer {{ margin-top:3rem; padding-top:1rem; border-top:1px solid var(--border);
           font-size:0.8rem; color:var(--text-dim); text-align:center; }}
ul {{ margin-left:1.2rem; margin-bottom:1rem; }}
li {{ margin-bottom:0.3rem; }}
.js-plotly-plot .plotly .modebar {{ display:none !important; }}
@media print {{
    body {{ background:white; color:#333; }}
    .tabs {{ display:none; }}
    .tab-content {{ display:block !important; page-break-before:always; }}
    .metric-box {{ border-color:#ddd; }}
    .metric-box .val {{ color:#333; }}
}}
</style>
</head>
<body>
<h1>ResearchNarrative</h1>
<p class="subtitle">Research narrative for: <strong>{topic}</strong></p>

<div class="metrics-row">
    <div class="metric-box"><div class="val">{len(papers)}</div><div class="lbl">Papers Analyzed</div></div>
    <div class="metric-box"><div class="val">{n_clusters}</div><div class="lbl">Research Threads</div></div>
    <div class="metric-box"><div class="val">{year_range}</div><div class="lbl">Year Range</div></div>
    <div class="metric-box"><div class="val">{total_cites:,}</div><div class="lbl">Total Citations</div></div>
    <div class="metric-box"><div class="val">{n_edges}</div><div class="lbl">Internal Citations</div></div>
    <div class="metric-box"><div class="val">{quality_pct:.0%}</div><div class="lbl">Quality Score</div></div>
</div>

<div class="tabs">
    <button class="tab active" onclick="showTab('narrative',this)">Narrative</button>
    <button class="tab" onclick="showTab('clusters',this)">Clusters</button>
    <button class="tab" onclick="showTab('citations',this)">Citation Analysis</button>
    <button class="tab" onclick="showTab('timeline',this)">Timeline</button>
    <button class="tab" onclick="showTab('papers',this)">Papers</button>
    <button class="tab" onclick="showTab('evaluation',this)">Evaluation</button>
</div>

<div id="narrative" class="tab-content active">
    <h2>Research Narrative: {topic}</h2>
    {ver_section}
    <div class="narrative-body">{narrative_html}</div>
    {"<h3>Thread Deep-Dives</h3>" + thread_sections if thread_sections else ""}
</div>

<div id="clusters" class="tab-content">
    <h2>Research Threads</h2>
    {cluster_table}
    {umap_chart_html}
    <h3>Thread Details</h3>
    {cluster_details}
</div>

<div id="citations" class="tab-content">
    <h2>Citation Graph Analysis</h2>
    <h3>Most Influential Papers</h3>
    {influential_table}
    {inf_bar_chart}
    {inf_radar_chart}
    {comp_table}
    {sankey_html}
    {dominance_charts}
</div>

<div id="timeline" class="tab-content">
    <h2>Publication Timeline</h2>
    {timeline_charts}
</div>

<div id="papers" class="tab-content">
    <h2>Paper Browser ({len(papers)} papers)</h2>
    <table class="data-table">
        <thead><tr><th>Title</th><th>Authors</th><th>Year</th><th>Citations</th><th>Thread</th></tr></thead>
        <tbody>{papers_table_rows}</tbody>
    </table>
</div>

<div id="evaluation" class="tab-content">
    <h2>Pipeline Quality Evaluation</h2>
    {eval_section}
</div>

<div class="footer">
    Generated by <strong>ResearchNarrative</strong> — RAG-Based Research Storyline Generation Engine<br>
    CS 6235: IEC — Georgia Institute of Technology
</div>

<script>
function showTab(name, btn) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    document.getElementById(name).classList.add('active');
    btn.classList.add('active');
    window.dispatchEvent(new Event('resize'));
}}
</script>
</body>
</html>"""


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ResearchNarrative")
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
            ["arxiv", "s2", "google_scholar"],
            default=["arxiv", "s2"],
            format_func=lambda x: {"arxiv": "arXiv", "s2": "Semantic Scholar", "google_scholar": "Google Scholar"}.get(x, x),
        )

        enrich_citations = st.checkbox("Enrich citation data", value=True)

    run_button = st.button(
        "Analyze Topic",
        type="primary",
        use_container_width=True,
        disabled=not topic,
    )

    has_results = st.session_state.results is not None and "error" not in st.session_state.results
    if has_results:
        _portal_html = _build_portal_html(
            st.session_state.results,
            st.session_state.results.get("cluster_labels", {}),
        )
        st.download_button(
            "Download Full Report (HTML)",
            data=_portal_html,
            file_name=f"research_narrative_{st.session_state.results['topic'].replace(' ', '_')}.html",
            mime="text/html",
            use_container_width=True,
        )
    else:
        st.button(
            "Download Full Report (HTML)",
            use_container_width=True,
            disabled=True,
        )

    # Cached topics
    cached_topics = []
    if PAPERS_DIR.exists():
        cached_topics = [
            f.stem.replace("_", " ")
            for f in PAPERS_DIR.glob("*.json")
        ]
    if cached_topics:
        st.divider()
        st.markdown("**Cached Topics**")
        for ct in cached_topics:
            if st.button(f"Load: {ct}", key=f"load_{ct}"):
                topic = ct
                st.session_state.load_cached = ct

    st.divider()

    with st.expander("About", expanded=False):
        st.markdown(
            "**ResearchNarrative** transforms academic paper search into "
            "structured research storylines.\n\n"
            "**Pipeline:** arXiv + S2 → SPECTER2 → FAISS → HDBSCAN → "
            "Citation Graph → RAG Narrative\n\n"
            "**Tabs:** Narrative, Clusters, Citation Analysis, "
            "Evaluation, Timeline, Papers, Search\n\n"
            "CS 6235 · Georgia Tech · Spring 2026"
        )


# ── Main Content ─────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">ResearchNarrative</div>', unsafe_allow_html=True)
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
        "init": "", "ingestion": "", "embedding": "",
        "indexing": "", "clustering": "", "citation_graph": "",
        "narrative": "", "evaluation": "", "done": "",
    }
    step_labels = {
        "init": "Initializing", "ingestion": "Collecting Papers",
        "embedding": "Generating Embeddings", "indexing": "Building Search Index",
        "clustering": "Discovering Threads", "citation_graph": "Citation Graph Analysis",
        "narrative": "Writing Narrative", "evaluation": "Evaluating Quality",
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
        icon = step_icons.get(step, "")
        label = step_labels.get(step, step)
        current_step[0] = step
        log_lines.append((icon, label, message))
        header = f"{label} — {message}"
        _render_card(header, log_lines[-20:])

    _render_card("Starting pipeline...", [])

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
        _render_card("Pipeline complete!", log_lines[-20:], "progress-card-done")
        st.rerun()
    except Exception as e:
        log_lines.append(("", "Error", str(e)))
        _render_card(f"Pipeline failed: {e}", log_lines[-20:], "progress-card-error")
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

m1, m2, m3, m4, m5, m6 = st.columns(6)
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
    st.metric("Internal Citations", n_edges)
with m6:
    eval_results = results.get("evaluation", {})
    overall_score = eval_results.get("overall", {}).get("score", 0)
    pipeline_obj = st.session_state.pipeline
    grade = pipeline_obj.evaluator.get_grade() if eval_results else "—"
    st.metric("Quality Grade", grade)

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_narrative, tab_clusters, tab_citations, tab_timeline, tab_papers, tab_search, tab_eval = st.tabs([
    "Narrative", "Clusters", "Citation Analysis",
    "Timeline", "Papers", "Search", "Evaluation",
])

# ── Tab 1: Narrative ─────────────────────────────────────────────────────────

with tab_narrative:
    st.markdown(f"## Research Narrative: {results['topic']}")

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
                f'Narrative Citations: {stats["verified_count"]}/{stats["total"]} grounded ({accuracy_pct:.0f}%)</span>'
                f'<span style="color:#aaa;font-size:0.8rem;">'
                f'{stats["verified_count"]} citations used in generating narrative matched to retrieved papers</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    pipeline = st.session_state.pipeline
    if pipeline.narrative_gen.verifier:
        linked_narrative = pipeline.narrative_gen.verifier.add_paper_links(narrative)
        st.markdown(linked_narrative)
    else:
        st.markdown(narrative)

    if verification and verification["unverified"]:
        with st.expander(f"{len(verification['unverified'])} unverified citations"):
            st.markdown(
                "These citations in the narrative could not be matched to papers "
                "in the retrieved collection. They may reference papers outside the "
                "search scope or use slightly different author name formats."
            )
            for cite in verification["unverified"]:
                st.markdown(f"- {cite}")

    st.divider()

    thread_narratives = results.get("thread_narratives", {})
    if thread_narratives:
        st.markdown("### Thread Deep-Dives")
        for cid in sorted(thread_narratives.keys()):
            label = cluster_labels.get(cid, f"Thread {cid}")
            with st.expander(f"Thread {cid}: {label}"):
                thread_text = thread_narratives[cid]
                if pipeline.narrative_gen.verifier:
                    thread_text = pipeline.narrative_gen.verifier.add_paper_links(thread_text)
                st.markdown(thread_text)

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
        cg_m1, cg_m2, cg_m3, cg_m4 = st.columns(4)
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
        with cg_m4:
            in_degs = [citation_graph.graph.in_degree(n) for n in citation_graph.graph.nodes]
            connected = sum(1 for d in in_degs if d > 0)
            connectivity = connected / max(len(in_degs), 1)
            st.metric("Connectivity", f"{connectivity:.0%}")

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

            # Horizontal bar chart of top papers by composite score
            st.markdown("### Influence Score Breakdown — Top 10")
            top10 = inf_data[:10]
            categories = ["PageRank", "Authority", "Bridge", "Pioneer", "Burst"]
            colors = {"PageRank": "#667eea", "Authority": "#764ba2", "Bridge": "#2ecc71",
                      "Pioneer": "#f39c12", "Burst": "#e74c3c"}

            fig_bar = go.Figure()
            for cat in categories:
                fig_bar.add_trace(go.Bar(
                    y=[r["Paper"] for r in reversed(top10)],
                    x=[r[cat] for r in reversed(top10)],
                    name=cat,
                    orientation='h',
                    marker_color=colors[cat],
                ))
            fig_bar.update_layout(
                barmode='stack',
                height=420,
                title="Influence Components (stacked)",
                xaxis_title="Score",
                yaxis_title="",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=200),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Radar chart for top-5
            st.markdown("### Influence Profile — Top 5 Papers")
            top5 = inf_data[:5]

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

    # In-degree distribution
    if citation_graph and citation_graph.graph.number_of_edges() > 0:
        st.markdown("### Citation Distribution")
        col_dist1, col_dist2 = st.columns(2)

        with col_dist1:
            in_degrees = [citation_graph.graph.in_degree(n) for n in citation_graph.graph.nodes]
            df_indeg = pd.DataFrame({"In-Degree": in_degrees})
            fig_indeg = px.histogram(
                df_indeg, x="In-Degree",
                title="In-Degree Distribution (papers cited by others)",
                color_discrete_sequence=["#667eea"],
                nbins=min(30, max(in_degrees) + 1) if max(in_degrees) > 0 else 10,
            )
            fig_indeg.update_layout(height=350)
            st.plotly_chart(fig_indeg, use_container_width=True)

        with col_dist2:
            out_degrees = [citation_graph.graph.out_degree(n) for n in citation_graph.graph.nodes]
            df_outdeg = pd.DataFrame({"Out-Degree": out_degrees})
            fig_outdeg = px.histogram(
                df_outdeg, x="Out-Degree",
                title="Out-Degree Distribution (papers that cite others)",
                color_discrete_sequence=["#764ba2"],
                nbins=min(30, max(out_degrees) + 1) if max(out_degrees) > 0 else 10,
            )
            fig_outdeg.update_layout(height=350)
            st.plotly_chart(fig_outdeg, use_container_width=True)

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
                    "Total": cp["total_cross_citations"],
                    "Asymmetry": cp["asymmetry"],
                })
            df_comp = pd.DataFrame(comp_data)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

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

# ── Tab 4: Evaluation ────────────────────────────────────────────────────────

with tab_eval:
    st.markdown("## Pipeline Quality Evaluation")
    import plotly.graph_objects as go

    eval_data = results.get("evaluation", {})
    if eval_data:
        overall = eval_data.get("overall", {})
        overall_score = overall.get("score", 0)
        component_scores = overall.get("component_scores", {})
        pipeline_ref = st.session_state.pipeline
        grade = pipeline_ref.evaluator.get_grade()

        grade_colors = {
            "A+": "#2ecc71", "A": "#27ae60", "B+": "#f1c40f",
            "B": "#f39c12", "C": "#e67e22", "D": "#e74c3c",
        }
        badge_color = grade_colors.get(grade, "#95a5a6")
        st.markdown(
            f'<div style="text-align:center;margin:1rem 0 2rem 0;">'
            f'<span style="background:{badge_color};color:white;padding:12px 32px;'
            f'border-radius:16px;font-size:2rem;font-weight:700;">'
            f'Grade: {grade}</span>'
            f'<div style="color:#aaa;margin-top:8px;font-size:1rem;">'
            f'Overall Score: {overall_score:.1%}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Component score gauges
        st.markdown("### Component Scores")
        gauge_cols = st.columns(4)
        component_labels = {
            "retrieval": ("Retrieval", "Paper collection quality"),
            "clustering": ("Clustering", "Thread separation quality"),
            "citation_graph": ("Citation Graph", "Graph connectivity"),
            "narrative": ("Narrative", "Generated text quality"),
        }
        for i, (key, (label, desc)) in enumerate(component_labels.items()):
            with gauge_cols[i]:
                score = component_scores.get(key, 0)
                color = "#2ecc71" if score >= 0.7 else "#f39c12" if score >= 0.5 else "#e74c3c"
                st.markdown(
                    f'<div style="text-align:center;padding:1rem;border:1px solid rgba(255,255,255,0.1);'
                    f'border-radius:12px;">'
                    f'<div style="font-size:0.9rem;color:#aaa;">{label}</div>'
                    f'<div style="font-size:2.2rem;font-weight:700;color:{color};">'
                    f'{score:.0%}</div>'
                    f'<div style="font-size:0.75rem;color:#666;">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.divider()

        # Radar chart
        st.markdown("### Score Breakdown")
        radar_categories = list(component_labels.keys())
        radar_labels = [component_labels[k][0] for k in radar_categories]
        radar_values = [component_scores.get(k, 0) for k in radar_categories]
        radar_values_closed = radar_values + [radar_values[0]]
        radar_labels_closed = radar_labels + [radar_labels[0]]

        fig_eval_radar = go.Figure(data=[go.Scatterpolar(
            r=radar_values_closed,
            theta=radar_labels_closed,
            fill='toself',
            line=dict(color='#667eea'),
            fillcolor='rgba(102, 126, 234, 0.2)',
        )])
        fig_eval_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            height=400,
            title="Component Score Radar",
        )
        st.plotly_chart(fig_eval_radar, use_container_width=True)

        # Detailed metrics
        st.markdown("### Detailed Metrics")

        detail_cols = st.columns(2)

        with detail_cols[0]:
            r_details = eval_data.get("retrieval", {}).get("details", {})
            st.markdown("**Retrieval Quality**")
            st.markdown(
                f"- Papers retrieved: **{r_details.get('total_papers', 0)}**\n"
                f"- Abstract coverage: **{r_details.get('abstract_coverage', 0):.0%}**\n"
                f"- Metadata completeness: **{r_details.get('metadata_completeness', 0):.0%}**\n"
                f"- Year span: **{r_details.get('year_span', 0)} years**\n"
                f"- Unique authors: **{r_details.get('unique_authors', 0)}**\n"
                f"- Sources: {r_details.get('source_distribution', {})}"
            )

            st.markdown("")
            cg_details = eval_data.get("citation_graph", {}).get("details", {})
            st.markdown("**Citation Graph**")
            st.markdown(
                f"- Nodes: **{cg_details.get('nodes', 0)}**, Edges: **{cg_details.get('edges', 0)}**\n"
                f"- Density: **{cg_details.get('density', 0):.6f}**\n"
                f"- Enrichment coverage: **{cg_details.get('enrichment_coverage', 0):.0%}**\n"
                f"- Connectivity: **{cg_details.get('connectivity', 0):.0%}**\n"
                f"- Avg in-degree: **{cg_details.get('avg_in_degree', 0):.2f}**"
            )

        with detail_cols[1]:
            c_details = eval_data.get("clustering", {}).get("details", {})
            st.markdown("**Clustering Quality**")
            sil = c_details.get("silhouette_score")
            db = c_details.get("davies_bouldin_index")
            st.markdown(
                f"- Clusters: **{c_details.get('n_clusters', 0)}**\n"
                f"- Silhouette score: **{sil:.4f}**\n" if sil is not None else
                f"- Clusters: **{c_details.get('n_clusters', 0)}**\n"
                f"- Silhouette score: *N/A*\n"
            )
            st.markdown(
                f"- Davies-Bouldin index: **{db:.4f}**" if db is not None else
                "- Davies-Bouldin index: *N/A*"
            )
            st.markdown(
                f"- Noise ratio: **{c_details.get('noise_ratio', 0):.0%}**\n"
                f"- Cluster balance: **{c_details.get('balance_score', 0):.0%}**\n"
                f"- Cluster sizes: {c_details.get('cluster_sizes', [])}"
            )

            st.markdown("")
            n_details = eval_data.get("narrative", {}).get("details", {})
            st.markdown("**Narrative Quality**")
            st.markdown(
                f"- Word count: **{n_details.get('word_count', 0):,}**\n"
                f"- Sections: **{n_details.get('n_sections', 0)}**\n"
                f"- Paragraphs: **{n_details.get('n_paragraphs', 0)}**\n"
                f"- Citations: **{n_details.get('n_citations', 0)}** "
                f"({n_details.get('unique_citations', 0)} unique)\n"
                f"- Citation density: **{n_details.get('citation_density', 0):.1f}** per paragraph\n"
                f"- Paper coverage: **{n_details.get('paper_coverage', 0):.0%}**\n"
                f"- Citation accuracy: **{n_details.get('citation_accuracy', 0):.0%}**"
            )

        st.divider()

        recs = pipeline_ref.evaluator.get_recommendations()
        st.markdown("### Recommendations")
        for rec in recs:
            st.markdown(f"- {rec}")

    else:
        st.info("Evaluation data is not available. Run the pipeline to see results.")

# ── Tab 5: Timeline ─────────────────────────────────────────────────────────

with tab_timeline:
    st.markdown("## Publication Timeline")
    import plotly.express as px
    import plotly.graph_objects as go

    years_all = [p.year for p in papers if p.year]
    if years_all:
        year_counts = Counter(years_all)

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

# ── Tab 6: Papers ────────────────────────────────────────────────────────────

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

# ── Tab 7: Similarity Search ────────────────────────────────────────────────

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
