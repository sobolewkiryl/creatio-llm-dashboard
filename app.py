import io
import json
from urllib.parse import urlparse

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Brand Visibility",
    page_icon="📡",
    layout="wide",
)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

LLM_COLORS = {
    "ChatGPT": "#10A37F",
    "Gemini": "#4285F4",
    "Copilot": "#7B61FF",
    "Perplexity": "#20B8CD",
    "AI Overview": "#F4B400",
    "AI Mode": "#DB4437",
}
LOSS_COLOR = "#E5534B"

SOURCE_TYPE_CITED = "Cited"
SOURCE_TYPE_FOUND = "Found, not cited"

# ── styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; max-width: 1400px; }
.metric-card {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 12px; padding: 1.1rem 1.3rem 1rem;
}
.metric-label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; opacity: 0.6; margin-bottom: 0.35rem;
}
.metric-value { font-size: 2.1rem; font-weight: 600; line-height: 1; }
.metric-sub { font-size: 0.8rem; margin-top: 0.4rem; }
.up   { color: #1a9e55; }
.down { color: #d33f37; }
.neu  { opacity: 0.55; }
.section-title {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; opacity: 0.55;
    margin: 1.8rem 0 0.9rem; padding-bottom: 0.35rem;
    border-bottom: 1px solid rgba(128,128,128,0.25);
}
.gained, .lost {
    border-radius: 8px; padding: 0.6rem 1rem;
    margin-bottom: 0.35rem; font-size: 0.86rem;
}
.gained { background: rgba(26,158,85,0.10); border: 1px solid rgba(26,158,85,0.35); }
.lost   { background: rgba(211,63,55,0.10); border: 1px solid rgba(211,63,55,0.35); }
.meta { opacity: 0.7; }
</style>
""", unsafe_allow_html=True)


# ── parsing helpers ───────────────────────────────────────────────────────────

def split_lines(cell) -> list:
    """Ahrefs puts multiple values in one cell separated by newlines."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    return [p.strip() for p in str(cell).split("\n") if p.strip() and p.strip().lower() != "nan"]


def split_tags(cell) -> list:
    """A prompt can carry several tags, e.g. 'UK, CAFM software'."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return ["Untagged"]
    tags = [t.strip() for t in str(cell).split(",") if t.strip()]
    return tags or ["Untagged"]


def brand_aliases(brand_input: str) -> list:
    return [b.strip().lower() for b in brand_input.split(",") if b.strip()]


def mentions_brand(cell, aliases: list) -> bool:
    names = [m.lower() for m in split_lines(cell)]
    return any(a == n for a in aliases for n in names)


def read_csv_any(raw_bytes: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
        except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError):
            continue
    raise ValueError("Could not read the CSV file.")


REQUIRED_COLS = {"Country", "Keyword", "Model", "Mentions"}


@st.cache_data
def process_export(raw_bytes: bytes, brand_input: str) -> pd.DataFrame:
    """One row per Country / Model / Keyword, with a brand-mention flag."""
    df = read_csv_any(raw_bytes)
    df.columns = df.columns.str.strip().str.lstrip("﻿")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in export: {', '.join(sorted(missing))}")

    aliases = brand_aliases(brand_input)
    df["Keyword"] = df["Keyword"].astype(str).str.strip()
    df["Country"] = df["Country"].astype(str).str.strip()
    df["Model"] = df["Model"].astype(str).str.strip()
    df["mentioned"] = df["Mentions"].apply(lambda c: mentions_brand(c, aliases))
    df["tag_list"] = df["Tags"].apply(split_tags) if "Tags" in df.columns else [["Untagged"]] * len(df)
    df["Tags"] = df["tag_list"].apply(", ".join)
    if "Volume" not in df.columns:
        df["Volume"] = pd.NA
    return df.reset_index(drop=True)


def compute_prompts(df: pd.DataFrame) -> pd.DataFrame:
    """Unique prompt per Country / Model / Keyword."""
    return (
        df.groupby(["Country", "Model", "Keyword"], as_index=False)
        .agg(mentioned=("mentioned", "max"),
             Volume=("Volume", "first"),
             Tags=("Tags", "first"),
             tag_list=("tag_list", "first"))
    )


def compute_coverage(prompts: pd.DataFrame) -> pd.DataFrame:
    """Coverage per tag. A prompt with several tags counts in each of them."""
    exploded = prompts.explode("tag_list").rename(columns={"tag_list": "Tag"})
    out = (
        exploded.groupby(["Country", "Model", "Tag"], as_index=False)
        .agg(total_prompts=("Keyword", "nunique"),
             mentioned_prompts=("mentioned", "sum"))
    )
    out["coverage_pct"] = (out["mentioned_prompts"] / out["total_prompts"] * 100).round(1)
    return out


def overall_coverage(prompts: pd.DataFrame) -> pd.Series:
    """Share of all prompts that mention the brand, per LLM."""
    g = prompts.groupby("Model")["mentioned"]
    return (g.sum() / g.count() * 100).round(1)


def extract_sources(df: pd.DataFrame) -> pd.DataFrame:
    """One row per source URL. Supports old (Link URL) and new (Cited pages) exports."""
    cols = ["Country", "Model", "Keyword", "Tags", "tag_list", "URL", "Title", "Domain", "Type"]
    rows = []
    for _, row in df.iterrows():
        base = {k: row[k] for k in ["Country", "Model", "Keyword", "Tags", "tag_list"]}
        groups = []
        if "Cited pages" in df.columns:
            groups.append((split_lines(row.get("Cited pages")), [], SOURCE_TYPE_CITED))
        if "Found but not cited" in df.columns:
            groups.append((split_lines(row.get("Found but not cited")), [], SOURCE_TYPE_FOUND))
        if "Link URL" in df.columns:
            groups.append((split_lines(row.get("Link URL")),
                           split_lines(row.get("Link Title")), SOURCE_TYPE_CITED))
        for urls, titles, src_type in groups:
            for i, url in enumerate(urls):
                if not url.startswith("http"):
                    continue
                rows.append({
                    **base,
                    "URL": url,
                    "Title": titles[i] if i < len(titles) else "",
                    "Domain": urlparse(url).netloc.replace("www.", ""),
                    "Type": src_type,
                })
    return pd.DataFrame(rows, columns=cols)


def has_tag(tag_lists: pd.Series, tag: str) -> pd.Series:
    return tag_lists.apply(lambda tl: tag in tl)


def fmt_pct_delta(v) -> str:
    if pd.isna(v):
        return "—"
    return f"+{v:.1f}%" if v > 0 else f"{v:.1f}%"


def fmt_volume(v) -> str:
    return f"{int(v):,}" if pd.notna(v) else "n/a"


def metric_card(label: str, value: float, prev=None):
    if prev is not None and not pd.isna(prev):
        diff = value - prev
        sign = "▲" if diff > 0 else ("▼" if diff < 0 else "→")
        cls = "up" if diff > 0 else ("down" if diff < 0 else "neu")
        sub = f'<div class="metric-sub {cls}">{sign} {diff:+.1f}% vs previous</div>'
    else:
        sub = '<div class="metric-sub neu">no comparison period</div>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value:.1f}%</div>
        {sub}
    </div>""", unsafe_allow_html=True)


def chart_layout(fig, height, **extra):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        margin=dict(l=0, r=0, t=30, b=0), height=height, **extra,
    )
    return fig


def prompt_diff(cur_p: pd.DataFrame, prv_p: pd.DataFrame) -> pd.DataFrame:
    keys = ["Country", "Model", "Keyword"]
    diff = cur_p.merge(
        prv_p[keys + ["mentioned", "Volume", "Tags", "tag_list"]],
        on=keys, how="outer", suffixes=("_cur", "_prv"),
    )
    diff["mentioned_cur"] = diff["mentioned_cur"].fillna(False).astype(bool)
    diff["mentioned_prv"] = diff["mentioned_prv"].fillna(False).astype(bool)
    diff["Volume"] = diff["Volume_cur"].combine_first(diff["Volume_prv"])
    diff["Tags"] = diff["Tags_cur"].combine_first(diff["Tags_prv"])
    diff["tag_list"] = diff["tag_list_cur"].combine_first(diff["tag_list_prv"])
    diff["status"] = "unchanged"
    diff.loc[diff["mentioned_cur"] & ~diff["mentioned_prv"], "status"] = "gained"
    diff.loc[~diff["mentioned_cur"] & diff["mentioned_prv"], "status"] = "lost"
    return diff


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📡 LLM Visibility")

    brand_input = st.text_input(
        "Brand name",
        value=st.secrets.get("BRAND", "Singu"),
        help="As it appears in the Mentions column. Separate aliases with commas.",
    )
    brand_label = brand_input.split(",")[0].strip() or "Brand"

    st.markdown("---")
    st.markdown("**Current period**")
    uploaded_cur = st.file_uploader("Ahrefs export CSV", type=["csv"], key="cur")

    st.markdown("**Previous period** *(optional)*")
    uploaded_prv = st.file_uploader("Ahrefs export CSV", type=["csv"], key="prv")

    st.markdown("---")
    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        anthropic_key = st.text_input("Anthropic API key", type="password")

# ── landing ───────────────────────────────────────────────────────────────────

st.markdown("# LLM Brand Visibility")

if uploaded_cur is None:
    st.info("Upload the current period Ahrefs Brand Radar export in the sidebar. "
            "Add a previous period export to unlock comparison.")
    st.stop()

if not brand_aliases(brand_input):
    st.warning("Enter a brand name in the sidebar.")
    st.stop()

# ── process ───────────────────────────────────────────────────────────────────

try:
    cur_raw = process_export(uploaded_cur.getvalue(), brand_input)
    prv_raw = (process_export(uploaded_prv.getvalue(), brand_input)
               if uploaded_prv is not None else None)
except ValueError as e:
    st.error(str(e))
    st.stop()

if uploaded_prv is not None and uploaded_prv.getvalue() == uploaded_cur.getvalue():
    st.warning("Both uploads are the same file — the comparison will show no changes.")

if not cur_raw["mentioned"].any():
    st.warning(f"'{brand_label}' doesn't appear in any Mentions in the current export. "
               "Check the brand name in the sidebar.")

countries = sorted(cur_raw["Country"].unique())
sel_country = st.selectbox("Country", countries)

cur_df = cur_raw[cur_raw["Country"] == sel_country]
prv_df = prv_raw[prv_raw["Country"] == sel_country] if prv_raw is not None else None

cur_prompts = compute_prompts(cur_df)
prv_prompts = compute_prompts(prv_df) if prv_df is not None and not prv_df.empty else None

cur_cov = compute_coverage(cur_prompts)
prv_cov = compute_coverage(prv_prompts) if prv_prompts is not None else None

llms = sorted(cur_prompts["Model"].unique())
all_tags = sorted({t for tl in cur_prompts["tag_list"] for t in tl})

if prv_raw is not None and prv_prompts is None:
    st.warning(f"The previous period file has no data for country '{sel_country}'.")

cur_overall = overall_coverage(cur_prompts)
prv_overall = overall_coverage(prv_prompts) if prv_prompts is not None else None

# ── tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_compare, tab_responses, tab_sources, tab_insights = st.tabs([
    "📊 Overview", "🔄 Period Comparison", "💬 LLM Responses", "🔗 Sources", "🤖 AI Insights",
])

# ════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown('<div class="section-title">Overall coverage (share of prompts mentioning the brand)</div>',
                unsafe_allow_html=True)
    cols = st.columns(max(len(llms), 1))
    for col, llm in zip(cols, llms):
        with col:
            prev = prv_overall.get(llm) if prv_overall is not None else None
            metric_card(llm, cur_overall[llm], prev)

    st.caption(f"{cur_prompts['Keyword'].nunique()} tracked prompts · "
               f"{int(cur_prompts['mentioned'].sum())} of {len(cur_prompts)} responses mention {brand_label}")

    st.markdown('<div class="section-title">Coverage by tag</div>', unsafe_allow_html=True)
    pivot = cur_cov.pivot_table(index="Tag", columns="Model",
                                values="coverage_pct", aggfunc="first").fillna(0)

    fig = go.Figure()
    for llm in pivot.columns:
        fig.add_trace(go.Bar(name=llm, x=pivot.index, y=pivot[llm],
                             marker_color=LLM_COLORS.get(llm, "#888"), marker_line_width=0))
    chart_layout(fig, 340, barmode="group",
                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                 yaxis=dict(ticksuffix="%", range=[0, 105], gridcolor="rgba(128,128,128,0.2)"))
    st.plotly_chart(fig, width="stretch")

    counts = cur_cov.groupby("Tag")["total_prompts"].max()
    table = pivot.copy()
    table.insert(0, "Prompts", counts)
    pct_cols = list(pivot.columns)
    st.dataframe(
        table.reset_index().style
        .format({c: "{:.1f}%" for c in pct_cols})
        .background_gradient(cmap="Greens", subset=pct_cols, vmin=0, vmax=100),
        width="stretch", hide_index=True,
    )
    st.caption("A prompt with several tags counts in each of its tags.")

# ════════════════════════════════════════════════════════════════════════════
# PERIOD COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab_compare:
    if prv_prompts is None:
        st.info("Upload a previous period export in the sidebar to enable comparison.")
    else:
        cur_piv = cur_cov.pivot_table(index="Tag", columns="Model", values="coverage_pct", aggfunc="first")
        prv_piv = prv_cov.pivot_table(index="Tag", columns="Model", values="coverage_pct", aggfunc="first")
        idx = cur_piv.index.union(prv_piv.index)
        cols_ = cur_piv.columns.union(prv_piv.columns)
        delta = (cur_piv.reindex(index=idx, columns=cols_).fillna(0)
                 - prv_piv.reindex(index=idx, columns=cols_).fillna(0)).round(1)

        st.markdown('<div class="section-title">Coverage change per LLM</div>', unsafe_allow_html=True)
        for llm in llms:
            if llm not in delta.columns:
                continue
            d = delta[llm].sort_values()
            fig2 = go.Figure(go.Bar(
                x=d.values, y=d.index, orientation="h",
                marker_color=[LLM_COLORS.get(llm, "#888") if v >= 0 else LOSS_COLOR for v in d.values],
                marker_line_width=0,
                text=[fmt_pct_delta(v) for v in d.values], textposition="outside",
            ))
            chart_layout(fig2, 60 + 34 * len(d),
                         title=dict(text=llm, font=dict(size=13)),
                         xaxis=dict(ticksuffix="%", zeroline=True, gridcolor="rgba(128,128,128,0.2)"))
            fig2.update_layout(margin=dict(l=0, r=70, t=35, b=0))
            st.plotly_chart(fig2, width="stretch")

        st.markdown('<div class="section-title">Delta table</div>', unsafe_allow_html=True)

        def _style_delta(val):
            if not isinstance(val, (int, float)) or pd.isna(val):
                return ""
            return "color: #1a9e55" if val > 0 else ("color: #d33f37" if val < 0 else "")

        st.dataframe(
            delta.reset_index().style
            .format({c: fmt_pct_delta for c in delta.columns})
            .map(_style_delta, subset=list(delta.columns)),
            width="stretch", hide_index=True,
        )

        # prompt-level changes
        st.markdown('<div class="section-title">Prompt-level changes</div>', unsafe_allow_html=True)
        diff = prompt_diff(cur_prompts, prv_prompts)

        fc1, fc2 = st.columns(2)
        with fc1:
            d_llm = st.selectbox("LLM", ["All"] + llms, key="diff_llm")
        with fc2:
            d_tag = st.selectbox("Tag", ["All"] + all_tags, key="diff_tag")
        if d_llm != "All":
            diff = diff[diff["Model"] == d_llm]
        if d_tag != "All":
            diff = diff[has_tag(diff["tag_list"], d_tag)]

        gained = diff[diff["status"] == "gained"].sort_values("Volume", ascending=False)
        lost = diff[diff["status"] == "lost"].sort_values("Volume", ascending=False)

        g_vol = gained["Volume"].fillna(0).sum()
        l_vol = lost["Volume"].fillna(0).sum()
        st.caption(f"Volume gained: {int(g_vol):,} · volume lost: {int(l_vol):,} · "
                   f"net: {int(g_vol - l_vol):+,}")

        col_g, col_l = st.columns(2)
        for col, data, css, title, empty in [
            (col_g, gained, "gained", f"✅ Gained ({len(gained)})", "No new appearances."),
            (col_l, lost, "lost", f"❌ Lost ({len(lost)})", "No lost appearances."),
        ]:
            with col:
                st.markdown(f"### {title}")
                if data.empty:
                    st.caption(empty)
                for _, r in data.iterrows():
                    st.markdown(
                        f'<div class="{css}"><strong>{r["Keyword"]}</strong><br>'
                        f'<span class="meta">{r["Tags"]} · {r["Model"]} · vol: {fmt_volume(r["Volume"])}</span></div>',
                        unsafe_allow_html=True,
                    )

# ════════════════════════════════════════════════════════════════════════════
# LLM RESPONSES
# ════════════════════════════════════════════════════════════════════════════
with tab_responses:
    st.markdown('<div class="section-title">Response explorer</div>', unsafe_allow_html=True)

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        r_tag = st.selectbox("Tag", ["All"] + all_tags, key="r_tag")
    with rc2:
        r_llm = st.selectbox("LLM", ["All"] + llms, key="r_llm")
    with rc3:
        r_mention = st.selectbox(f"{brand_label} mentioned", ["All", "Yes", "No"], key="r_mention")

    rdf = cur_df.copy()
    if r_tag != "All":
        rdf = rdf[has_tag(rdf["tag_list"], r_tag)]
    if r_llm != "All":
        rdf = rdf[rdf["Model"] == r_llm]
    if r_mention == "Yes":
        rdf = rdf[rdf["mentioned"]]
    elif r_mention == "No":
        rdf = rdf[~rdf["mentioned"]]

    rdf = rdf.drop_duplicates(subset=["Keyword", "Model"])
    st.caption(f"{len(rdf)} responses")

    aliases = brand_aliases(brand_input)
    for _, row in rdf.iterrows():
        badge = f"✅ {brand_label} mentioned" if row["mentioned"] else "❌ Not mentioned"
        with st.expander(f"**{row['Keyword']}** · {row['Model']} · {row['Tags']} · {badge}"):
            ca, cb = st.columns([3, 1])
            with ca:
                st.markdown("**LLM response**")
                st.markdown(str(row.get("Response", "—")))
            with cb:
                meta = [f"{c}: {row[c]}" for c in ["Intent", "Funnel stage", "Volume"]
                        if c in row.index and pd.notna(row[c])]
                if meta:
                    st.caption(" · ".join(str(m) for m in meta))
                st.markdown("**Brands mentioned**")
                for m in split_lines(row.get("Mentions")):
                    st.markdown(f"🟢 **{m}**" if m.lower() in aliases else f"- {m}")
                srcs = extract_sources(pd.DataFrame([row]))
                for src_type in [SOURCE_TYPE_CITED, SOURCE_TYPE_FOUND]:
                    part = srcs[srcs["Type"] == src_type]
                    if not part.empty:
                        st.markdown(f"**{src_type}**")
                        for _, s in part.iterrows():
                            st.markdown(f"[↗ {s['Title'] or s['Domain']}]({s['URL']})")

# ════════════════════════════════════════════════════════════════════════════
# SOURCES
# ════════════════════════════════════════════════════════════════════════════
with tab_sources:
    st.markdown('<div class="section-title">Source URLs</div>', unsafe_allow_html=True)

    cur_src = extract_sources(cur_df)
    prv_src = extract_sources(prv_df) if prv_df is not None else None

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        s_llm = st.selectbox("LLM", ["All"] + llms, key="s_llm")
    with sc2:
        s_tag = st.selectbox("Tag", ["All"] + all_tags, key="s_tag")
    with sc3:
        type_options = ["All"] + [t for t in [SOURCE_TYPE_CITED, SOURCE_TYPE_FOUND]
                                  if t in set(cur_src["Type"])]
        s_type = st.selectbox("Source type", type_options, key="s_type",
                              help="Cited = linked in the answer. Found, not cited = "
                                   "the LLM read the page but didn't link it.")

    def filter_sources(df):
        if df is None:
            return None
        if s_llm != "All":
            df = df[df["Model"] == s_llm]
        if s_tag != "All":
            df = df[has_tag(df["tag_list"], s_tag)]
        if s_type != "All":
            df = df[df["Type"] == s_type]
        return df

    def summarise_urls(df):
        """One row per URL with every prompt and LLM that uses it."""
        if df.empty:
            return pd.DataFrame(columns=["URL", "Domain", "Type", "LLMs", "Tags", "Prompts"])
        return (
            df.groupby("URL")
            .agg(Domain=("Domain", "first"),
                 Type=("Type", lambda s: ", ".join(sorted(set(s)))),
                 LLMs=("Model", lambda s: ", ".join(sorted(set(s)))),
                 Tags=("tag_list", lambda s: ", ".join(sorted({t for tl in s for t in tl}))),
                 Prompts=("Keyword", lambda s: " | ".join(sorted(set(s)))))
            .reset_index()
            .sort_values(["Domain", "URL"])
        )

    link_cfg = {"URL": st.column_config.LinkColumn("URL")}
    cur_f = filter_sources(cur_src)
    prv_f = filter_sources(prv_src)

    if prv_f is None:
        st.info("Upload a previous period export to see new and lost sources.")
    else:
        new_urls = set(cur_f["URL"]) - set(prv_f["URL"])
        lost_urls = set(prv_f["URL"]) - set(cur_f["URL"])

        new_tbl = summarise_urls(cur_f[cur_f["URL"].isin(new_urls)])
        lost_tbl = summarise_urls(prv_f[prv_f["URL"].isin(lost_urls)])

        st.markdown(f"### 🆕 New sources ({len(new_tbl)})")
        st.caption("Used by LLMs in the current period, not in the previous one")
        if new_tbl.empty:
            st.caption("No new sources.")
        else:
            st.dataframe(new_tbl, width="stretch", hide_index=True, column_config=link_cfg)
            st.download_button("⬇ Download new sources", new_tbl.to_csv(index=False).encode(),
                               f"new_sources_{sel_country}.csv", "text/csv")

        st.markdown(f"### ❌ Lost sources ({len(lost_tbl)})")
        st.caption("Used in the previous period, no longer used")
        if lost_tbl.empty:
            st.caption("No lost sources.")
        else:
            st.dataframe(lost_tbl, width="stretch", hide_index=True, column_config=link_cfg)
            st.download_button("⬇ Download lost sources", lost_tbl.to_csv(index=False).encode(),
                               f"lost_sources_{sel_country}.csv", "text/csv")

    st.markdown('<div class="section-title">All sources in current period</div>', unsafe_allow_html=True)
    all_tbl = summarise_urls(cur_f)
    if all_tbl.empty:
        st.caption("No sources for the selected filters.")
    else:
        st.dataframe(all_tbl, width="stretch", hide_index=True, column_config=link_cfg)
        st.download_button("⬇ Download all sources", all_tbl.to_csv(index=False).encode(),
                           f"sources_{sel_country}.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════
# AI INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown('<div class="section-title">AI-powered analysis</div>', unsafe_allow_html=True)

    if not anthropic_key:
        st.warning("Add an Anthropic API key (Streamlit Secrets or the sidebar) to enable AI Insights.")
    else:
        custom_question = st.text_area(
            "Your question (optional)",
            placeholder="e.g. What is the sum of gained vs lost volume? Which tags should we prioritise?",
            height=90,
        )

        if st.button("🤖 Generate insights", type="primary"):
            prompt_lines = [
                f"- {r.Keyword} [{r.Tags} · {r.Model}] vol: {fmt_volume(r.Volume)} · "
                f"{'mentioned' if r.mentioned else 'not mentioned'}"
                for r in cur_prompts.itertuples()
            ]
            context = (
                f"OVERALL COVERAGE per LLM (country: {sel_country}):\n{cur_overall.to_string()}\n\n"
                f"COVERAGE BY TAG (%):\n{cur_cov.pivot_table(index='Tag', columns='Model', values='coverage_pct').to_string()}\n\n"
                f"ALL PROMPTS (current period):\n" + "\n".join(prompt_lines)
            )

            if prv_prompts is not None:
                diff_ai = prompt_diff(cur_prompts, prv_prompts)
                g = diff_ai[diff_ai["status"] == "gained"]
                l = diff_ai[diff_ai["status"] == "lost"]

                def _lines(d):
                    return "\n".join(f"  - {r.Keyword} [{r.Tags} · {r.Model}] vol: {fmt_volume(r.Volume)}"
                                     for r in d.itertuples()) or "  none"

                gv, lv = int(g["Volume"].fillna(0).sum()), int(l["Volume"].fillna(0).sum())
                context += (
                    f"\n\nOVERALL COVERAGE previous period:\n{prv_overall.to_string()}"
                    f"\n\nPROMPT-LEVEL CHANGES vs previous period:\n"
                    f"Gained ({len(g)}, total volume {gv:,}):\n{_lines(g)}\n"
                    f"Lost ({len(l)}, total volume {lv:,}):\n{_lines(l)}\n"
                    f"Net volume change: {gv - lv:+,}"
                )

            if custom_question.strip():
                task = (f'Answer this question:\n"{custom_question.strip()}"\n\n'
                        "Base the answer only on the data above. Name specific prompts, tags and volumes.")
            else:
                task = (
                    "Provide a structured analysis:\n"
                    "1. Overall performance across LLMs\n"
                    "2. Strongest tags\n"
                    "3. Blind spots (low or zero visibility)\n"
                    "4. Differences between LLMs\n"
                    "5. Key changes vs the previous period, if available, with specific prompts and volumes\n"
                    "6. 2-3 concrete priorities for the marketing team"
                )

            prompt = (
                "You are a senior digital marketing analyst specialising in brand visibility in LLM answers.\n\n"
                f"Brand analysed: {brand_label}\n\n{context}\n\n{task}\n\n"
                "Write in clear business English and reference actual numbers."
            )

            with st.spinner("Analysing…"):
                try:
                    resp = requests.post(
                        ANTHROPIC_URL,
                        headers={"x-api-key": anthropic_key,
                                 "anthropic-version": "2023-06-01",
                                 "content-type": "application/json"},
                        json={"model": "claude-sonnet-4-20250514", "max_tokens": 1500,
                              "messages": [{"role": "user", "content": prompt}]},
                        timeout=90,
                    )
                    resp.raise_for_status()
                    st.session_state["last_insight"] = resp.json()["content"][0]["text"]
                except requests.exceptions.HTTPError as e:
                    st.error(f"API error {e.response.status_code}: {e.response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

        if "last_insight" in st.session_state:
            st.markdown(st.session_state["last_insight"])
            st.download_button("⬇ Download insights", st.session_state["last_insight"],
                               f"insights_{sel_country}.txt", "text/plain")
