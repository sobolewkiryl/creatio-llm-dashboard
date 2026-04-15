import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import requests

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Creatio · LLM Brand Visibility",
    page_icon="📡",
    layout="wide",
)

BRAND         = "creatio"
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
.gained {
    background:#052e16; border:1px solid #14532d;
    border-radius:8px; padding:0.6rem 1rem;
    margin-bottom:0.3rem; color:#4ade80; font-size:0.85rem;
}
.lost {
    background:#2d0a0a; border:1px solid #7f1d1d;
    border-radius:8px; padding:0.6rem 1rem;
    margin-bottom:0.3rem; color:#f87171; font-size:0.85rem;
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


def compute_prompts(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Country", "Model", "Keyword", "Tags", "mentioned"]
    if "Volume" in df.columns:
        cols.append("Volume")
    return (
        df[cols]
        .drop_duplicates(subset=["Country", "Model", "Keyword"])
        .reset_index(drop=True)
    )


def metric_card(label: str, value: float, prev=None):
    if prev is not None and not pd.isna(prev):
        diff = value - prev
        sign = "▲" if diff > 0 else ("▼" if diff < 0 else "→")
        cls  = "up" if diff > 0 else ("down" if diff < 0 else "neu")
        sub  = f'<div class="metric-sub {cls}">{sign} {diff:+.1f}% vs previous</div>'
    else:
        sub = '<div class="metric-sub neu">no comparison period</div>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value:.1f}%</div>
        {sub}
    </div>""", unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📡 LLM Visibility")
    st.markdown("---")

    st.markdown("**Current period**")
    uploaded_cur = st.file_uploader("Ahrefs export CSV", type=["csv"], key="cur")

    st.markdown("---")
    st.markdown("**Previous period** *(optional)*")
    uploaded_prv = st.file_uploader("Ahrefs export CSV", type=["csv"], key="prv")

    st.markdown("---")
    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        anthropic_key = st.text_input("Anthropic API key", type="password")

    st.caption("Creatio · Brand Visibility Tracker")

# ── landing ───────────────────────────────────────────────────────────────────

if uploaded_cur is None:
    st.markdown("# LLM Brand Visibility")
    st.info("Upload the current period Ahrefs export in the sidebar to get started. "
            "Optionally upload a previous period export to enable comparison.")
    st.stop()

# ── process ───────────────────────────────────────────────────────────────────

cur_bytes = uploaded_cur.read()
cur_merged, cur_unmatched = process_export(cur_bytes)

prv_merged   = None
prv_coverage = None
prv_prompts  = None

if uploaded_prv is not None:
    prv_bytes = uploaded_prv.read()
    prv_merged, _ = process_export(prv_bytes)
    prv_coverage = compute_coverage(prv_merged)
    prv_prompts  = compute_prompts(prv_merged)

if cur_unmatched:
    with st.expander(f"⚠️ {len(cur_unmatched)} prompts not found in clusters.csv"):
        st.caption("Add these to clusters.csv and push to GitHub.")
        for kw in cur_unmatched:
            st.markdown(f"- `{kw}`")

cur_coverage = compute_coverage(cur_merged)
cur_prompts  = compute_prompts(cur_merged)

# ── global filters ────────────────────────────────────────────────────────────

st.markdown("# LLM Brand Visibility")
countries = sorted(cur_coverage["Country"].unique())
sel_country = st.selectbox("Country", countries)

filtered     = cur_coverage[cur_coverage["Country"] == sel_country]
cmp_filtered = (prv_coverage[prv_coverage["Country"] == sel_country]
                if prv_coverage is not None else None)
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
        "⬇ Download coverage CSV",
        data=cur_coverage.to_csv(index=False).encode(),
        file_name=f"coverage_{sel_country}.csv",
        mime="text/csv",
    )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PERIOD COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab_compare:

    if cmp_filtered is None:
        st.info("Upload a previous period export in the sidebar to enable comparison.")
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

        # ── prompt-level diff ──
        st.markdown('<div class="section-title">Prompt-level changes</div>',
                    unsafe_allow_html=True)

        cur_prm_f = cur_prompts[cur_prompts["Country"] == sel_country]
        prv_prm_f = prv_prompts[prv_prompts["Country"] == sel_country]

        merge_keys = ["Country", "Model", "Keyword", "Tags"]
        prv_cols = merge_keys + ["mentioned"]
        if "Volume" in prv_prm_f.columns:
            prv_cols.append("Volume")
        diff = cur_prm_f.merge(
            prv_prm_f[prv_cols],
            on=merge_keys, how="outer", suffixes=("_cur", "_prv"),
        )
        diff["mentioned_cur"] = diff["mentioned_cur"].fillna(False)
        diff["mentioned_prv"] = diff["mentioned_prv"].fillna(False)

        gained = diff[(diff["mentioned_cur"] == True)  & (diff["mentioned_prv"] == False)]
        lost   = diff[(diff["mentioned_cur"] == False) & (diff["mentioned_prv"] == True)]

        sel_llm_diff = st.selectbox("Filter by LLM", ["All"] + llms, key="diff_llm")
        if sel_llm_diff != "All":
            gained = gained[gained["Model"] == sel_llm_diff]
            lost   = lost[lost["Model"] == sel_llm_diff]

        col_g, col_l = st.columns(2)
        with col_g:
            st.markdown(f"### ✅ Gained ({len(gained)})")
            st.caption("Creatio appeared — wasn't mentioned before")
            if gained.empty:
                st.caption("No new appearances.")
            else:
                for _, row in gained.sort_values("Volume_cur" if "Volume_cur" in gained.columns else "Tags", ascending=False if "Volume_cur" in gained.columns else True).iterrows():
                    vol = f' · vol: {int(row["Volume_cur"]):,}' if "Volume_cur" in row and pd.notna(row["Volume_cur"]) else ""
                    st.markdown(
                        f'<div class="gained"><strong>{row["Keyword"]}</strong>'
                        f'<br><span style="opacity:.7">{row["Tags"]} · {row["Model"]}{vol}</span></div>',
                        unsafe_allow_html=True,
                    )
        with col_l:
            st.markdown(f"### ❌ Lost ({len(lost)})")
            st.caption("Creatio disappeared — was mentioned before")
            if lost.empty:
                st.caption("No lost appearances.")
            else:
                for _, row in lost.sort_values("Volume_cur" if "Volume_cur" in lost.columns else "Tags", ascending=False if "Volume_cur" in lost.columns else True).iterrows():
                    vol = f' · vol: {int(row["Volume_cur"]):,}' if "Volume_cur" in row and pd.notna(row["Volume_cur"]) else ""
                    st.markdown(
                        f'<div class="lost"><strong>{row["Keyword"]}</strong>'
                        f'<br><span style="opacity:.7">{row["Tags"]} · {row["Model"]}{vol}</span></div>',
                        unsafe_allow_html=True,
                    )

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — LLM RESPONSES
# ════════════════════════════════════════════════════════════════════════════
with tab_responses:

    st.markdown('<div class="section-title">Response explorer</div>',
                unsafe_allow_html=True)

    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        r_country = st.selectbox("Country", countries, key="r_country")
    with rc2:
        r_cluster = st.selectbox(
            "Cluster", ["All"] + sorted(cur_merged["Tags"].dropna().unique()), key="r_cluster"
        )
    with rc3:
        r_llm = st.selectbox(
            "LLM", ["All"] + sorted(cur_merged["Model"].unique()), key="r_llm"
        )
    with rc4:
        r_mention = st.selectbox(
            "Creatio mentioned", ["All", "Yes", "No"], key="r_mention"
        )

    rdf = cur_merged[cur_merged["Country"] == r_country].copy()
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
        custom_question = st.text_area(
            "Your question (optional)",
            placeholder=(
                "e.g. Why is our visibility in AI Agents so low? "
                "What should we prioritize to improve CRM T-1 coverage in ChatGPT?"
            ),
            height=90,
        )

        if st.button("🤖 Generate insights", type="primary"):
            cov_json = filtered[["Model", "Tags", "coverage_pct"]].to_dict(orient="records")

            delta_summary = ""
            prompt_diff_summary = ""

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
                    f"\n\nCOVERAGE CHANGE vs previous period (percentage points):\n"
                    f"{(cur_p - prv_p).round(1).to_string()}"
                )

                if prv_prompts is not None:
                    cur_pf = cur_prompts[cur_prompts["Country"] == sel_country]
                    prv_pf = prv_prompts[prv_prompts["Country"] == sel_country]
                    mk = ["Country", "Model", "Keyword", "Tags"]
                    prv_c = mk + ["mentioned"]
                    if "Volume" in prv_pf.columns:
                        prv_c.append("Volume")
                    diff_ai = cur_pf.merge(prv_pf[prv_c], on=mk, how="outer", suffixes=("_cur", "_prv"))
                    diff_ai["mentioned_cur"] = diff_ai["mentioned_cur"].fillna(False)
                    diff_ai["mentioned_prv"] = diff_ai["mentioned_prv"].fillna(False)
                    gained_ai = diff_ai[(diff_ai["mentioned_cur"]==True) & (diff_ai["mentioned_prv"]==False)].copy()
                    lost_ai   = diff_ai[(diff_ai["mentioned_cur"]==False) & (diff_ai["mentioned_prv"]==True)].copy()
                    vol_col = "Volume_cur" if "Volume_cur" in diff_ai.columns else ("Volume" if "Volume" in diff_ai.columns else None)

                    def fmt_prompts(df, vc):
                        rows = []
                        for _, r in df.iterrows():
                            vol_str = f" (vol: {int(r[vc]):,})" if vc and pd.notna(r.get(vc)) else ""
                            rows.append(f"  - {r['Keyword']}{vol_str} [{r['Tags']} · {r['Model']}]")
                        return "\n".join(rows) if rows else "  none"

                    gained_vol = int(gained_ai[vol_col].fillna(0).sum()) if vol_col else "n/a"
                    lost_vol   = int(lost_ai[vol_col].fillna(0).sum()) if vol_col else "n/a"
                    net_vol    = (gained_vol - lost_vol) if isinstance(gained_vol, int) else "n/a"

                    prompt_diff_summary = f"""

PROMPT-LEVEL CHANGES:
Gained ({len(gained_ai)} prompts, total volume: {gained_vol if isinstance(gained_vol,int) else gained_vol:,}):
{fmt_prompts(gained_ai, vol_col)}

Lost ({len(lost_ai)} prompts, total volume: {lost_vol if isinstance(lost_vol,int) else lost_vol:,}):
{fmt_prompts(lost_ai, vol_col)}

Net volume change: {f"{net_vol:+,}" if isinstance(net_vol, int) else net_vol}"""

            if custom_question.strip():
                task = (
                    f'Answer this specific question:\n"{custom_question.strip()}"\n\n'
                    f"Base your answer strictly on the data above. Reference actual prompt names, volumes, and cluster names."
                )
            else:
                task = """Provide a structured analysis:
1. **Overall performance** — how is Creatio performing across LLMs?
2. **Strongest clusters** — where does Creatio appear most consistently?
3. **Blind spots** — where is visibility low or zero?
4. **LLM differences** — notable differences between ChatGPT, Gemini, Copilot?
5. **Key changes** (if delta data available) — what improved, what declined? Reference specific prompts and volumes.
6. **Recommended actions** — 2-3 concrete priorities for the marketing team."""

            prompt = f"""You are a senior digital marketing analyst specializing in AI/LLM brand visibility for B2B SaaS.

Analyze brand visibility data for Creatio (no-code CRM and workflow automation platform).

COVERAGE DATA (country: {sel_country}):
{json.dumps(cov_json, indent=2)}{delta_summary}{prompt_diff_summary}

{task}

Write in clear business English. Reference actual prompt names and volume numbers where available."""

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
                file_name=f"insights_{sel_country}.txt",
                mime="text/plain",
            )
