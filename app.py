import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import glob
import json
import hashlib
import requests
from datetime import date

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Creatio · LLM Brand Visibility",
    page_icon="📡",
    layout="wide",
)

BRAND         = "creatio"
DATA_DIR      = "data"
CLUSTERS_FILE = "clusters.csv"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

LLM_COLORS = {
    "ChatGPT": "#10A37F",
    "Gemini":  "#4285F4",
    "Copilot": "#7B61FF",
}

# ── styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 1.5rem; max-width: 1400px; }
.metric-card {
    background: #0d0d10; border: 1px solid #1e1e24;
    border-radius: 12px; padding: 1.1rem 1.3rem 1rem;
}
.metric-label {
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #555; margin-bottom: 0.35rem;
}
.metric-value {
    font-size: 2.1rem; font-weight: 600;
    color: #efefef; font-family: 'DM Mono', monospace; line-height: 1;
}
.metric-sub { font-size: 0.78rem; margin-top: 0.35rem; }
.up   { color: #4ade80; }
.down { color: #f87171; }
.neu  { color: #555; }
.section-title {
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: #444; margin: 1.8rem 0 0.9rem;
    padding-bottom: 0.35rem; border-bottom: 1px solid #1a1a20;
}
div[data-testid="stSidebar"] {
    background: #09090c; border-right: 1px solid #161618;
}
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    font-size: 0.8rem; font-weight: 500;
    letter-spacing: 0.04em; padding: 6px 16px;
    border-radius: 6px; background: transparent;
    border: 1px solid #1e1e24; color: #666;
}
.stTabs [aria-selected="true"] {
    background: #1a1a24 !important; color: #eee !important;
    border-color: #333 !important;
}
.warn {
    background:#1a1200; border:1px solid #3a2e00;
    border-radius:8px; padding:0.7rem 1rem;
    color:#cca600; font-size:0.82rem;
}
.info-box {
    background:#0d1a0d; border:1px solid #1a3a1a;
    border-radius:8px; padding:0.7rem 1rem;
    color:#7aba7a; font-size:0.82rem;
}
</style>
""", unsafe_allow_html=True)

# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_clusters() -> pd.DataFrame:
    df = pd.read_csv(CLUSTERS_FILE)
    df["Keyword_lower"] = df["Prompt"].str.strip().str.lower()
    return df[["Keyword_lower", "Tags"]]


def mentions_brand(cell) -> bool:
    return False if pd.isna(cell) else BRAND in str(cell).lower()


def file_hash(raw_bytes: bytes) -> str:
    return hashlib.md5(raw_bytes).hexdigest()[:8]


@st.cache_data
def process_export(raw_bytes: bytes) -> tuple:
    import io
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-16")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding="latin-1")
    df.columns = df.columns.str.strip()
    df["Keyword_lower"] = df["Keyword"].str.strip().str.lower()
    df["mentioned"] = df["Mentions"].apply(mentions_brand)
    clusters = load_clusters()
    merged = df.merge(clusters, on="Keyword_lower", how="left")
    unmatched = sorted(merged[merged["Tags"].isna()]["Keyword"].unique().tolist())
    return merged[merged["Tags"].notna()].copy(), unmatched


def compute_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (country, model, tag), grp in df.groupby(["Country", "Model", "Tags"]):
        uniq = grp.drop_duplicates("Keyword_lower")
        total = len(uniq)
        mentioned = int(uniq["mentioned"].sum())
        rows.append({
            "Country": country, "Model": model, "Tags": tag,
            "total_prompts": total, "mentioned_prompts": mentioned,
            "coverage_pct": round(mentioned / total * 100, 1) if total else 0.0,
        })
    return pd.DataFrame(rows)


def load_snapshots() -> list:
    """Returns list of (label, df) sorted newest first."""
    files = sorted(glob.glob(f"{DATA_DIR}/snapshot_*.csv"), reverse=True)
    result = []
    for f in files:
        label = os.path.basename(f).replace("snapshot_", "").replace(".csv", "")
        result.append((label, pd.read_csv(f)))
    return result


def load_hashes() -> set:
    path = f"{DATA_DIR}/.hashes"
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return set(f.read().splitlines())


def save_hash(h: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(f"{DATA_DIR}/.hashes", "a") as f:
        f.write(h + "\n")


def save_snapshot(df: pd.DataFrame, label: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = f"{DATA_DIR}/snapshot_{label}.csv"
    df.to_csv(path, index=False)
    return path


def metric_card(label: str, value: float, prev=None):
    if prev is not None and not pd.isna(prev):
        diff = value - prev
        sign = "▲" if diff > 0 else ("▼" if diff < 0 else "→")
        cls  = "up" if diff > 0 else ("down" if diff < 0 else "neu")
        sub  = f'<div class="metric-sub {cls}">{sign} {diff:+.1f}% vs previous</div>'
    else:
        sub = '<div class="metric-sub neu">no previous period</div>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value:.1f}%</div>
        {sub}
    </div>""", unsafe_allow_html=True)


# ── load saved snapshots ──────────────────────────────────────────────────────

snapshots   = load_snapshots()
snap_labels = [s[0] for s in snapshots]

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📡 LLM Visibility")
    st.markdown("---")

    # ── data source ──
    st.markdown("**Data source**")
    data_mode = st.radio(
        "data_mode", ["Upload new export", "Use saved snapshot"],
        label_visibility="collapsed",
    )

    uploaded      = None
    current_snap  = None
    snapshot_label = str(date.today())

    if data_mode == "Upload new export":
        uploaded = st.file_uploader("Ahrefs export CSV", type=["csv"])
        snapshot_label = st.text_input("Snapshot label", value=str(date.today()))
        process_btn = st.button(
            "▶ Save & show report",
            use_container_width=True,
            type="primary",
            disabled=(uploaded is None),
        )
    else:
        process_btn = False
        if snap_labels:
            chosen = st.selectbox("Select snapshot", snap_labels)
            current_snap = next(s[1] for s in snapshots if s[0] == chosen)
        else:
            st.info("No saved snapshots yet. Upload a file first.")

    st.markdown("---")

    # ── compare with ──
    st.markdown("**Compare with**")
    compare_snap = None
    compare_options = ["— none —"] + snap_labels

    # If using saved snapshot, exclude the currently selected one from compare list
    if data_mode == "Use saved snapshot" and snap_labels:
        compare_options = ["— none —"] + [l for l in snap_labels if l != chosen]

    if len(compare_options) > 1:
        cmp_choice = st.selectbox("Period", compare_options, label_visibility="collapsed")
        if cmp_choice != "— none —":
            compare_snap = next(s[1] for s in snapshots if s[0] == cmp_choice)
    else:
        st.caption("Save at least 2 snapshots to enable comparison.")

    st.markdown("---")
    anthropic_key = st.text_input("Anthropic API key", type="password",
                                  help="Required for AI Insights tab")
    st.caption("Creatio · Brand Visibility Tracker")

# ── resolve active data ───────────────────────────────────────────────────────

merged   = None
coverage = None

if data_mode == "Upload new export":
    if uploaded is None:
        # Show latest snapshot if available, otherwise landing screen
        st.markdown("# LLM Brand Visibility")
        if snapshots:
            st.info("No file uploaded. Showing latest saved snapshot.")
            _, latest_df = snapshots[0]
            sel = st.selectbox("Country", sorted(latest_df["Country"].unique()))
            flt = latest_df[latest_df["Country"] == sel]
            pivot = flt.pivot_table(
                index="Tags", columns="Model", values="coverage_pct", aggfunc="first"
            ).reset_index()
            pct_cols = [c for c in pivot.columns if c != "Tags"]
            st.dataframe(
                pivot.style
                    .format({c: "{:.1f}%" for c in pct_cols})
                    .background_gradient(cmap="Greens", subset=pct_cols, vmin=0, vmax=100),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("Upload an Ahrefs Brand Radar CSV to get started.")
        st.stop()

    # Check for duplicate file
    raw_bytes = uploaded.read()
    h = file_hash(raw_bytes)
    existing_hashes = load_hashes()

    if h in existing_hashes:
        st.error(
            "⛔ This file has already been imported. "
            "Please upload a different export or use 'Use saved snapshot' to view existing data."
        )
        st.stop()

    merged, unmatched = process_export(raw_bytes)

    if unmatched:
        with st.expander(f"⚠️ {len(unmatched)} prompts not found in clusters.csv"):
            st.caption("Add these to clusters.csv and push to GitHub.")
            for kw in unmatched:
                st.markdown(f"- `{kw}`")

    if process_btn:
        with st.spinner("Processing…"):
            cov_save = compute_coverage(merged)
            save_snapshot(cov_save, snapshot_label)
            save_hash(h)
        st.success(f"Snapshot `{snapshot_label}` saved. Push the `data/` folder to GitHub.")
        st.cache_data.clear()
        snapshots   = load_snapshots()
        snap_labels = [s[0] for s in snapshots]

    coverage = compute_coverage(merged)

else:
    # Use saved snapshot
    if current_snap is None:
        st.markdown("# LLM Brand Visibility")
        st.info("No saved snapshots yet. Upload a file first.")
        st.stop()
    coverage = current_snap

# ── global filters ────────────────────────────────────────────────────────────

st.markdown("# LLM Brand Visibility")
countries = sorted(coverage["Country"].unique())
sel_country = st.selectbox("Country", countries)

filtered     = coverage[coverage["Country"] == sel_country]
cmp_filtered = (compare_snap[compare_snap["Country"] == sel_country]
                if compare_snap is not None else None)
llms = sorted(filtered["Model"].unique())

# ── tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_compare, tab_responses, tab_insights = st.tabs([
    "📊 Overview", "🔄 Period Comparison", "💬 LLM Responses", "🤖 AI Insights"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_overview:

    st.markdown('<div class="section-title">Avg. Share of Voice</div>',
                unsafe_allow_html=True)
    cols = st.columns(len(llms))
    for col, llm in zip(cols, llms):
        with col:
            avg = filtered[filtered["Model"] == llm]["coverage_pct"].mean()
            prev_avg = (cmp_filtered[cmp_filtered["Model"] == llm]["coverage_pct"].mean()
                        if cmp_filtered is not None else None)
            metric_card(llm, avg, prev_avg)

    st.markdown('<div class="section-title">Coverage by cluster</div>',
                unsafe_allow_html=True)
    pivot = filtered.pivot_table(
        index="Tags", columns="Model", values="coverage_pct", aggfunc="first"
    ).fillna(0).reset_index()

    fig = go.Figure()
    for llm in [c for c in pivot.columns if c != "Tags"]:
        fig.add_trace(go.Bar(
            name=llm, x=pivot["Tags"], y=pivot[llm],
            marker_color=LLM_COLORS.get(llm, "#888"), marker_line_width=0,
        ))
    fig.update_layout(
        barmode="group", plot_bgcolor="#09090c", paper_bgcolor="#09090c",
        font=dict(family="DM Sans", color="#888", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="#161618"),
        yaxis=dict(gridcolor="#161618", ticksuffix="%", range=[0, 110]),
        margin=dict(l=0, r=0, t=30, b=0), height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Coverage table</div>',
                unsafe_allow_html=True)
    pct_cols = [c for c in pivot.columns if c != "Tags"]
    st.dataframe(
        pivot.style
            .format({c: "{:.1f}%" for c in pct_cols})
            .background_gradient(cmap="Greens", subset=pct_cols, vmin=0, vmax=100),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "⬇ Download snapshot CSV",
        data=coverage.to_csv(index=False).encode(),
        file_name=f"snapshot_{snapshot_label}.csv",
        mime="text/csv",
    )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PERIOD COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab_compare:

    if cmp_filtered is None:
        st.info("Select a snapshot to compare with in the sidebar.")
    else:
        cur_piv = filtered.pivot_table(
            index="Tags", columns="Model", values="coverage_pct", aggfunc="first"
        ).fillna(0)
        prv_piv = cmp_filtered.pivot_table(
            index="Tags", columns="Model", values="coverage_pct", aggfunc="first"
        ).fillna(0)
        delta_piv = (cur_piv - prv_piv).round(1)

        st.markdown('<div class="section-title">Coverage change per LLM</div>',
                    unsafe_allow_html=True)

        for llm in llms:
            if llm not in delta_piv.columns:
                continue
            d = delta_piv[[llm]].reset_index()
            d.columns = ["Tags", "delta"]
            d = d.sort_values("delta")
            bar_colors = [LLM_COLORS.get(llm, "#888") if v >= 0 else "#f87171"
                          for v in d["delta"]]
            fig2 = go.Figure(go.Bar(
                x=d["delta"], y=d["Tags"], orientation="h",
                marker_color=bar_colors, marker_line_width=0,
                text=[f"{v:+.1f}%" for v in d["delta"]],
                textposition="outside",
            ))
            fig2.update_layout(
                title=dict(text=llm, font=dict(size=13, color="#aaa")),
                plot_bgcolor="#09090c", paper_bgcolor="#09090c",
                font=dict(family="DM Sans", color="#888", size=11),
                xaxis=dict(gridcolor="#161618", ticksuffix="%",
                           zeroline=True, zerolinecolor="#333"),
                yaxis=dict(gridcolor="#161618"),
                margin=dict(l=0, r=70, t=35, b=0), height=270,
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-title">Delta table</div>',
                    unsafe_allow_html=True)
        delta_display = delta_piv.reset_index()
        delta_cols = [c for c in delta_display.columns if c != "Tags"]

        def _style_delta(val):
            if not isinstance(val, (int, float)) or pd.isna(val): return ""
            return "color: #4ade80" if val > 0 else ("color: #f87171" if val < 0 else "color:#555")

        def _fmt_delta(v):
            if pd.isna(v): return "—"
            return f"+{v:.1f}%" if v > 0 else f"{v:.1f}%"

        st.dataframe(
            delta_display.style
                .format({c: _fmt_delta for c in delta_cols})
                .map(_style_delta, subset=delta_cols),
            use_container_width=True, hide_index=True,
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — LLM RESPONSES
# ════════════════════════════════════════════════════════════════════════════
with tab_responses:

    if merged is None:
        st.info("LLM Responses are only available when uploading a new export file. "
                "Switch to 'Upload new export' in the sidebar.")
    else:
        st.markdown('<div class="section-title">Response explorer</div>',
                    unsafe_allow_html=True)

        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            r_country = st.selectbox("Country", countries, key="r_country")
        with rc2:
            r_cluster = st.selectbox(
                "Cluster", ["All"] + sorted(merged["Tags"].dropna().unique()), key="r_cluster"
            )
        with rc3:
            r_llm = st.selectbox(
                "LLM", ["All"] + sorted(merged["Model"].unique()), key="r_llm"
            )
        with rc4:
            r_mention = st.selectbox(
                "Creatio mentioned", ["All", "Yes", "No"], key="r_mention"
            )

        rdf = merged[merged["Country"] == r_country].copy()
        if r_cluster != "All": rdf = rdf[rdf["Tags"] == r_cluster]
        if r_llm     != "All": rdf = rdf[rdf["Model"] == r_llm]
        if r_mention == "Yes": rdf = rdf[rdf["mentioned"] == True]
        elif r_mention == "No": rdf = rdf[rdf["mentioned"] == False]

        rdf_dedup = rdf.drop_duplicates(subset=["Keyword", "Model"]).reset_index(drop=True)
        st.caption(f"{len(rdf_dedup)} responses")

        for _, row in rdf_dedup.iterrows():
            badge = "✅ Creatio mentioned" if row["mentioned"] else "❌ Not mentioned"
            with st.expander(
                f"**{row['Keyword']}** · {row['Model']} · {row.get('Tags','—')} · {badge}"
            ):
                ca, cb = st.columns([3, 1])
                with ca:
                    st.markdown("**LLM Response**")
                    st.markdown(str(row.get("Response", "—")))
                with cb:
                    st.markdown("**Brands mentioned**")
                    for m in str(row.get("Mentions", "")).split("\n"):
                        m = m.strip()
                        if m:
                            st.markdown(f"🟢 **{m}**" if m.lower() == BRAND else f"- {m}")
                    urls = str(row.get("Link URL", ""))
                    if urls.strip():
                        st.markdown("**Sources**")
                        for link in urls.split("\n"):
                            link = link.strip()
                            if link.startswith("http"):
                                st.markdown(f"[↗ {link[:45]}…]({link})")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tab_insights:

    st.markdown('<div class="section-title">AI-powered analysis</div>',
                unsafe_allow_html=True)

    if not anthropic_key:
        st.markdown(
            '<div class="warn">Paste your Anthropic API key in the sidebar to enable AI Insights.</div>',
            unsafe_allow_html=True,
        )
    else:
        # Custom question
        custom_question = st.text_area(
            "Your question (optional)",
            placeholder=(
                "e.g. Why is our visibility in AI Agents so low compared to FinServ? "
                "What should we do to improve coverage in CRM T-1 for ChatGPT?"
            ),
            height=90,
            help="Leave blank for a full general analysis, or ask a specific question about the data.",
        )

        if st.button("🤖 Generate insights", type="primary"):
            cov_json = filtered[["Model", "Tags", "coverage_pct"]].to_dict(orient="records")

            delta_summary = ""
            if cmp_filtered is not None:
                cur_p = filtered.pivot_table(
                    index="Tags", columns="Model",
                    values="coverage_pct", aggfunc="first"
                ).fillna(0)
                prv_p = cmp_filtered.pivot_table(
                    index="Tags", columns="Model",
                    values="coverage_pct", aggfunc="first"
                ).fillna(0)
                delta_summary = (
                    f"\n\nCHANGE vs previous period (percentage points):\n"
                    f"{(cur_p - prv_p).round(1).to_string()}"
                )

            if custom_question.strip():
                task = (
                    f"Answer this specific question from the marketing team:\n"
                    f"\"{custom_question.strip()}\"\n\n"
                    f"Base your answer strictly on the data provided. Be specific and reference actual numbers."
                )
            else:
                task = """Provide a structured analysis:
1. **Overall performance** — how is Creatio performing across LLMs?
2. **Strongest clusters** — where does Creatio appear most consistently and why?
3. **Blind spots** — where is visibility low or zero?
4. **LLM differences** — notable differences between ChatGPT, Gemini, Copilot?
5. **Key changes** (if delta data available) — what improved, what declined?
6. **Recommended actions** — 2-3 concrete priorities for the marketing team."""

            prompt = f"""You are a senior digital marketing analyst specializing in AI/LLM brand visibility for B2B SaaS companies.

Analyze the following brand visibility data for Creatio (a no-code CRM and workflow automation platform) across LLM models.

COVERAGE DATA (country: {sel_country}):
{json.dumps(cov_json, indent=2)}{delta_summary}

{task}

Write in clear business English. Be specific and reference actual numbers from the data."""

            with st.spinner("Analyzing with Claude…"):
                try:
                    resp = requests.post(
                        ANTHROPIC_URL,
                        headers={
                            "x-api-key": anthropic_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                        json={
                            "model": "claude-sonnet-4-20250514",
                            "max_tokens": 1500,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                        timeout=60,
                    )
                    resp.raise_for_status()
                    st.session_state["last_insight"] = resp.json()["content"][0]["text"]
                except requests.exceptions.HTTPError as e:
                    st.error(f"API error {e.response.status_code}: {e.response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

        if "last_insight" in st.session_state:
            st.markdown(st.session_state["last_insight"])
            st.download_button(
                "⬇ Download insights",
                data=st.session_state["last_insight"],
                file_name=f"insights_{snapshot_label}_{sel_country}.txt",
                mime="text/plain",
            )
