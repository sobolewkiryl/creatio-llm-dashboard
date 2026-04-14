import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import glob
from datetime import date

# ── config ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Creatio · LLM Brand Visibility",
    page_icon="📡",
    layout="wide",
)

BRAND = "creatio"
DATA_DIR = "data"
CLUSTERS_FILE = "clusters.csv"

LLM_COLORS = {
    "ChatGPT": "#10A37F",
    "Gemini":  "#4285F4",
    "Copilot": "#7B61FF",
}

# ── styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.block-container { padding-top: 2rem; }

.metric-card {
    background: #0f0f11;
    border: 1px solid #222;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.5rem;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 600;
    color: #f0f0f0;
    font-family: 'DM Mono', monospace;
    line-height: 1;
}
.metric-delta-pos { color: #4ade80; font-size: 0.85rem; margin-top: 0.3rem; }
.metric-delta-neg { color: #f87171; font-size: 0.85rem; margin-top: 0.3rem; }
.metric-delta-neu { color: #888; font-size: 0.85rem; margin-top: 0.3rem; }

.section-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #555;
    margin: 2rem 0 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e1e1e;
}

.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
}
.pill-up   { background: #052e16; color: #4ade80; }
.pill-down { background: #2d0a0a; color: #f87171; }
.pill-neu  { background: #1a1a1a; color: #888; }

div[data-testid="stSidebar"] {
    background: #0a0a0c;
    border-right: 1px solid #1a1a1a;
}
</style>
""", unsafe_allow_html=True)

# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_clusters():
    df = pd.read_csv(CLUSTERS_FILE)
    df["Keyword_lower"] = df["Prompt"].str.strip().str.lower()
    return df[["Keyword_lower", "Tags"]]


def mentions_brand(cell):
    if pd.isna(cell):
        return False
    return BRAND in str(cell).lower()


@st.cache_data
def process_export(raw_bytes: bytes) -> pd.DataFrame:
    import io
    df = pd.read_csv(io.BytesIO(raw_bytes))
    df.columns = df.columns.str.strip()
    df["Keyword_lower"] = df["Keyword"].str.strip().str.lower()
    df["mentioned"] = df["Mentions"].apply(mentions_brand)

    clusters = load_clusters()
    merged = df.merge(clusters, on="Keyword_lower", how="left")
    merged = merged[merged["Tags"].notna()].copy()
    return merged


def compute_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (country, model, tag), grp in df.groupby(["Country", "Model", "Tags"]):
        uniq = grp.drop_duplicates("Keyword_lower")
        total = len(uniq)
        mentioned = uniq["mentioned"].sum()
        rows.append({
            "Country": country,
            "Model": model,
            "Tags": tag,
            "total_prompts": total,
            "mentioned_prompts": int(mentioned),
            "coverage_pct": round(mentioned / total * 100, 1) if total else 0,
        })
    return pd.DataFrame(rows)


def load_snapshots() -> list[tuple[str, pd.DataFrame]]:
    files = sorted(glob.glob(f"{DATA_DIR}/snapshot_*.csv"), reverse=True)
    return [(os.path.basename(f), pd.read_csv(f)) for f in files]


def save_snapshot(df: pd.DataFrame, label: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = f"{DATA_DIR}/snapshot_{label}.csv"
    df.to_csv(path, index=False)
    return path


def delta_card(label, current, previous=None):
    if previous is not None and not pd.isna(previous):
        diff = current - previous
        if diff > 0:
            delta_html = f'<div class="metric-delta-pos">▲ +{diff:.1f} pp vs poprzednio</div>'
        elif diff < 0:
            delta_html = f'<div class="metric-delta-neg">▼ {diff:.1f} pp vs poprzednio</div>'
        else:
            delta_html = f'<div class="metric-delta-neu">→ bez zmian</div>'
    else:
        delta_html = '<div class="metric-delta-neu">brak poprzedniego okresu</div>'

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{current:.1f}%</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📡 LLM Visibility")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Wgraj eksport z Ahrefs",
        type=["csv"],
        help="Plik CSV z Ahrefs Brand Radar"
    )

    snapshot_label = st.text_input(
        "Etykieta snapshotu",
        value=str(date.today()),
        help="np. 2026-04-14"
    )

    process_btn = st.button("▶ Generuj raport", use_container_width=True, type="primary")

    st.markdown("---")

    snapshots = load_snapshots()
    snapshot_names = [s[0] for s in snapshots]

    compare_label = None
    if len(snapshots) >= 1:
        compare_with = st.selectbox(
            "Porównaj z poprzednim snapshotem",
            options=["(brak)"] + snapshot_names,
        )
        compare_label = compare_with if compare_with != "(brak)" else None

    st.markdown("---")
    st.caption("Creatio · Brand Visibility Tracker")

# ── unmatched warning helper ──────────────────────────────────────────────────

def show_unmatched(raw_bytes):
    import io
    df = pd.read_csv(io.BytesIO(raw_bytes))
    df.columns = df.columns.str.strip()
    df["Keyword_lower"] = df["Keyword"].str.strip().str.lower()
    clusters = load_clusters()
    merged = df.merge(clusters, on="Keyword_lower", how="left")
    unmatched = merged[merged["Tags"].isna()]["Keyword"].unique()
    if len(unmatched):
        with st.expander(f"⚠️ {len(unmatched)} promptów bez klastra"):
            st.caption("Te prompty są w eksporcie, ale nie ma ich w clusters.csv. Dodaj je do pliku i wgraj ponownie na GitHub.")
            for kw in sorted(unmatched):
                st.markdown(f"- `{kw}`")

# ── main ──────────────────────────────────────────────────────────────────────

st.markdown("# LLM Brand Visibility")
st.markdown('<div class="section-title">Creatio · Brand Radar</div>', unsafe_allow_html=True)

if not uploaded:
    st.info("Wgraj plik CSV z Ahrefs Brand Radar w panelu bocznym, żeby zobaczyć raport.")

    if snapshots:
        st.markdown('<div class="section-title">Poprzednie snapshoty</div>', unsafe_allow_html=True)
        latest_name, latest_df = snapshots[0]
        st.caption(f"Ostatni snapshot: **{latest_name}**")

        countries = sorted(latest_df["Country"].unique())
        selected_country = st.selectbox("Kraj", countries)

        filtered = latest_df[latest_df["Country"] == selected_country]

        pivot = filtered.pivot_table(
            index="Tags", columns="Model", values="coverage_pct", aggfunc="first"
        ).reset_index()

        st.dataframe(
            pivot.style.background_gradient(cmap="Greens", subset=[c for c in pivot.columns if c != "Tags"]),
            use_container_width=True,
            hide_index=True,
        )
    st.stop()

# ── process ───────────────────────────────────────────────────────────────────

raw_bytes = uploaded.read()
show_unmatched(raw_bytes)

if process_btn:
    with st.spinner("Przetwarzam dane..."):
        merged = process_export(raw_bytes)
        coverage = compute_coverage(merged)
        save_snapshot(coverage, snapshot_label)
    st.success(f"Snapshot `{snapshot_label}` zapisany!")
    st.cache_data.clear()
    snapshots = load_snapshots()

merged = process_export(raw_bytes)
coverage = compute_coverage(merged)

# ── filters ───────────────────────────────────────────────────────────────────

countries = sorted(coverage["Country"].unique())
col_f1, col_f2 = st.columns([2, 4])
with col_f1:
    selected_country = st.selectbox("Kraj", countries)

filtered = coverage[coverage["Country"] == selected_country]

# ── load comparison snapshot ──────────────────────────────────────────────────

compare_df = None
if compare_label:
    match = [s for s in snapshots if s[0] == compare_label]
    if match:
        compare_df = match[0][1]
        compare_df = compare_df[compare_df["Country"] == selected_country]

# ── avg share of voice cards ──────────────────────────────────────────────────

st.markdown('<div class="section-title">Avg. Share of Voice</div>', unsafe_allow_html=True)

llms = sorted(filtered["Model"].unique())
cols = st.columns(len(llms))

for col, llm in zip(cols, llms):
    with col:
        avg = filtered[filtered["Model"] == llm]["coverage_pct"].mean()
        prev_avg = None
        if compare_df is not None:
            prev_avg = compare_df[compare_df["Model"] == llm]["coverage_pct"].mean()
        delta_card(llm, avg, prev_avg)

# ── coverage heatmap ──────────────────────────────────────────────────────────

st.markdown('<div class="section-title">Coverage per klaster</div>', unsafe_allow_html=True)

pivot = filtered.pivot_table(
    index="Tags", columns="Model", values="coverage_pct", aggfunc="first"
).fillna(0).reset_index()

# Bar chart per cluster
clusters_order = pivot.sort_values(
    by=[c for c in pivot.columns if c != "Tags"],
    ascending=False
)["Tags"].tolist()

fig = go.Figure()
for llm in [c for c in pivot.columns if c != "Tags"]:
    color = LLM_COLORS.get(llm, "#888")
    fig.add_trace(go.Bar(
        name=llm,
        x=pivot["Tags"],
        y=pivot[llm],
        marker_color=color,
        marker_line_width=0,
    ))

fig.update_layout(
    barmode="group",
    plot_bgcolor="#0a0a0c",
    paper_bgcolor="#0a0a0c",
    font=dict(family="DM Sans", color="#aaa", size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(gridcolor="#1a1a1a", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#1a1a1a", ticksuffix="%", range=[0, 110]),
    margin=dict(l=0, r=0, t=30, b=0),
    height=340,
)
st.plotly_chart(fig, use_container_width=True)

# ── table with delta ──────────────────────────────────────────────────────────

st.markdown('<div class="section-title">Tabela coverage</div>', unsafe_allow_html=True)

if compare_df is not None:
    compare_pivot = compare_df.pivot_table(
        index="Tags", columns="Model", values="coverage_pct", aggfunc="first"
    ).fillna(0).reset_index()

    for llm in [c for c in pivot.columns if c != "Tags"]:
        prev_col = compare_pivot.set_index("Tags")[llm] if llm in compare_pivot.columns else None
        if prev_col is not None:
            pivot[f"{llm} Δ"] = (
                pivot.set_index("Tags")[llm] - prev_col
            ).round(1).values

display_cols = ["Tags"]
for llm in [c for c in filtered["Model"].unique()]:
    display_cols.append(llm)
    delta_col = f"{llm} Δ"
    if delta_col in pivot.columns:
        display_cols.append(delta_col)

pivot_display = pivot[[c for c in display_cols if c in pivot.columns]]

def style_delta(val):
    if pd.isna(val):
        return ""
    if isinstance(val, float) and abs(val) < 100:
        if val > 0:
            return "color: #4ade80"
        elif val < 0:
            return "color: #f87171"
    return ""

pct_cols = [c for c in pivot_display.columns if "Δ" not in c and c != "Tags"]
delta_cols = [c for c in pivot_display.columns if "Δ" in c]

styled = pivot_display.style \
    .format({c: "{:.1f}%" for c in pct_cols}) \
    .format({c: lambda v: f"+{v:.1f} pp" if v > 0 else f"{v:.1f} pp" if not pd.isna(v) else "-" for c in delta_cols}) \
    .applymap(style_delta, subset=delta_cols) \
    .background_gradient(cmap="Greens", subset=pct_cols, vmin=0, vmax=100)

st.dataframe(styled, use_container_width=True, hide_index=True)

# ── prompt detail ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-title">Szczegóły promptów</div>', unsafe_allow_html=True)

col_d1, col_d2, col_d3 = st.columns(3)
with col_d1:
    sel_cluster = st.selectbox("Klaster", ["Wszystkie"] + sorted(merged["Tags"].dropna().unique()))
with col_d2:
    sel_llm = st.selectbox("LLM", ["Wszystkie"] + sorted(merged["Model"].unique()))
with col_d3:
    sel_mention = st.selectbox("Wzmianka Creatio", ["Wszystkie", "Tak", "Nie"])

detail = merged[merged["Country"] == selected_country].copy()
if sel_cluster != "Wszystkie":
    detail = detail[detail["Tags"] == sel_cluster]
if sel_llm != "Wszystkie":
    detail = detail[detail["Model"] == sel_llm]
if sel_mention == "Tak":
    detail = detail[detail["mentioned"] == True]
elif sel_mention == "Nie":
    detail = detail[detail["mentioned"] == False]

detail_display = detail[["Tags", "Keyword", "Model", "mentioned"]].drop_duplicates().sort_values(["Tags", "Model", "Keyword"])
detail_display = detail_display.rename(columns={
    "Tags": "Klaster", "Keyword": "Prompt", "Model": "LLM", "mentioned": "Creatio wspomniane"
})

st.dataframe(detail_display, use_container_width=True, hide_index=True, height=350)

# ── download ──────────────────────────────────────────────────────────────────

st.markdown("---")
csv_out = coverage.to_csv(index=False).encode()
st.download_button(
    "⬇ Pobierz snapshot CSV",
    data=csv_out,
    file_name=f"snapshot_{snapshot_label}.csv",
    mime="text/csv",
)
