# app.py — patched version (defaults + percentile chips + attributes under cards + hover fix)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="UFC Fighter Comparison 🥊", layout="wide")

# ---- Adjust if your CSV is elsewhere ----
DATA_PATH = "ufc_clean.csv"   # or "data/ufc_clean.csv" if you moved it

COLS = {
    "name": "Name",
    "nickname": "nickname",
    "stance": "stance",
    "dob": "DOB",
    "height_cm": "Height(cm)",
    "reach_cm": "Reach(cm)",
    "weight_kg": "Weight(kg)",
    "wins": "Wins",
    "losses": "Losses",
    "draws": "Draws",
    "fights": "Fights",
    "win_pct": "Win%",
    "strikes_min": "Significant-Strikes-Landed-Per-Minute",
    "strike_acc": "Significant-Striking-Accuracy",
    "strikes_abs_min": "Significant-Strikes-Absorbed-Per-Minute",
    "strike_def": "Significant-Strike-Defence",
    "tds_15": "Average-Takedowns-Landed-Per-15-Minutes",
    "td_acc": "Takedown-Accuracy",
    "td_def": "takedown_defense",
    "subs_15": "average_submissions_attempted_per_15_minutes",
}

LABEL = {
    COLS["win_pct"]: "Win %",
    COLS["height_cm"]: "Height (cm)",
    COLS["reach_cm"]: "Reach (cm)",
    COLS["weight_kg"]: "Weight (kg)",
    COLS["strikes_min"]: "Strikes/min",
    COLS["strike_acc"]: "Striking Acc %",
    COLS["strikes_abs_min"]: "Strikes Abs/min",
    COLS["strike_def"]: "Strike Def %",
    COLS["tds_15"]: "TDs/15",
    COLS["td_acc"]: "TD Acc %",
    COLS["td_def"]: "TD Def %",
    COLS["subs_15"]: "Subs/15",
}

GROUPS = {
    "Physicals": [COLS["height_cm"], COLS["reach_cm"], COLS["weight_kg"]],
    "Record": [COLS["wins"], COLS["losses"], COLS["draws"], COLS["fights"], COLS["win_pct"]],
    "Striking": [COLS["strikes_min"], COLS["strike_acc"], COLS["strikes_abs_min"], COLS["strike_def"]],
    "Grappling": [COLS["tds_15"], COLS["td_acc"], COLS["td_def"], COLS["subs_15"]],
}

RADAR_METRICS = [
    COLS["strikes_min"],
    COLS["strike_acc"],
    COLS["tds_15"],
    COLS["td_acc"],
    COLS["td_def"],
    COLS["subs_15"],
]

# ---------- Helpers ----------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if COLS["dob"] in df.columns:
        df[COLS["dob"]] = pd.to_datetime(df[COLS["dob"]], errors="coerce")
        today = pd.Timestamp(datetime(2025, 8, 27))
        df["Age (years)"] = np.floor((today - df[COLS["dob"]]).dt.days / 365.25)
    else:
        df["Age (years)"] = np.nan
    if COLS["fights"] not in df.columns and {COLS["wins"], COLS["losses"], COLS["draws"]}.issubset(df.columns):
        df[COLS["fights"]] = df[COLS["wins"]] + df[COLS["losses"]] + df[COLS["draws"]]
    return df

def val(v):
    if pd.isna(v):
        return None
    return v

def fmt_num(v, decimals=1):
    if v is None:
        return "—"
    try:
        f = float(v)
        if f.is_integer():
            return f"{int(f)}"
        return f"{f:.{decimals}f}"
    except Exception:
        return str(v)

def get_row(df, name):
    return df.loc[df[COLS["name"]] == name].iloc[0]

def bar_compare(metric_col, a_row, b_row):
    a_val = val(a_row.get(metric_col))
    b_val = val(b_row.get(metric_col))
    label = LABEL.get(metric_col, metric_col)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[a_val], y=[f"🟥 {a_row[COLS['name']]}"], orientation="h",
        name=a_row[COLS["name"]],
        # ESCAPED braces so Plotly sees %{x} and f-string doesn't error
        hovertemplate=f"{label}: %{{x}}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=[b_val], y=[f"🟦 {b_row[COLS['name']]}"], orientation="h",
        name=b_row[COLS["name"]],
        hovertemplate=f"{label}: %{{x}}<extra></extra>"
    ))
    xs = [x for x in [a_val, b_val] if x is not None]
    x_max = max(xs) if xs else 1
    fig.update_layout(
        barmode="group",
        height=180,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        title=label,
        xaxis=dict(range=[0, x_max * 1.15 if x_max > 0 else 1]),
    )
    return fig

def normalize_minmax(series):
    s = series.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if s.empty:
        return series * 0
    mn, mx = s.min(), s.max()
    if mx - mn == 0:
        return series * 0
    return (series - mn) / (mx - mn)

def radar_compare(df, a_row, b_row, metrics):
    normed = {col: normalize_minmax(df[col]) for col in metrics if col in df.columns}
    a_vals, b_vals = [], []
    labels = [LABEL.get(m, m) for m in metrics]
    for col in metrics:
        s = normed.get(col)
        if s is None:
            a_vals.append(0.0); b_vals.append(0.0); continue
        a_vals.append(float(s.loc[a_row.name]) if pd.notna(s.loc[a_row.name]) else 0.0)
        b_vals.append(float(s.loc[b_row.name]) if pd.notna(s.loc[b_row.name]) else 0.0)
    labels += [labels[0]]
    a_vals += [a_vals[0]]
    b_vals += [b_vals[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=a_vals, theta=labels, fill='toself', name=a_row[COLS["name"]]))
    fig.add_trace(go.Scatterpolar(r=b_vals, theta=labels, fill='toself', name=b_row[COLS["name"]]))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=520, margin=dict(l=10, r=10, t=40, b=10),
        title="Multi-Metric Radar (normalized to dataset)"
    )
    return fig

# ---- Percentiles for chips ----
PCT_COLUMNS = [
    COLS["height_cm"], COLS["reach_cm"], COLS["weight_kg"],
    COLS["win_pct"],
    COLS["strikes_min"], COLS["strike_acc"], COLS["strikes_abs_min"], COLS["strike_def"],
    COLS["tds_15"], COLS["td_acc"], COLS["td_def"], COLS["subs_15"],
]

@st.cache_data
def compute_percentiles(df: pd.DataFrame, columns: list[str]) -> dict[str, pd.Series]:
    out = {}
    for col in columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        ranks = s.rank(method="average", na_option="keep")
        valid = ranks.notna()
        cnt = valid.sum()
        pct = pd.Series(np.nan, index=df.index)
        if cnt > 1:
            pct[valid] = (ranks[valid] - 1) / (cnt - 1) * 100.0
        elif cnt == 1:
            pct[valid] = 100.0
        out[col] = pct
    return out

def chip(value, pct=None, suffix=""):
    base = fmt_num(value)
    if pct is None or pd.isna(pct):
        return f"{base}{suffix}"
    return f"{base}{suffix} • p{int(round(pct))}"

def render_card(col, row, pct_map):
    with col:
        st.markdown(f"### {row[COLS['name']]}")
        if pd.notna(row.get(COLS["nickname"])) and str(row.get(COLS["nickname"])).strip():
            st.caption(f"“{str(row[COLS['nickname']]).strip()}”")

        top_bits = []
        if pd.notna(row.get("Age (years)")):
            top_bits.append(f"Age: {fmt_num(row.get('Age (years)'), 0)}")
        if pd.notna(row.get(COLS["stance"])):
            top_bits.append(f"Stance: {row[COLS['stance']]}")
        if top_bits:
            st.write(" • ".join(top_bits))

        # Physicals with percentile chips
        st.write(
            " | ".join([
                f"**📏 Height:** {chip(row.get(COLS['height_cm']), pct_map.get(COLS['height_cm'], pd.Series()).get(row.name), ' cm')}",
                f"**👐 Reach:** {chip(row.get(COLS['reach_cm']), pct_map.get(COLS['reach_cm'], pd.Series()).get(row.name), ' cm')}",
                f"**⚖️ Weight:** {chip(row.get(COLS['weight_kg']), pct_map.get(COLS['weight_kg'], pd.Series()).get(row.name), ' kg')}",
            ])
        )

        # Record with win% chip
        rec = f"{fmt_num(row.get(COLS['wins']),0)}–{fmt_num(row.get(COLS['losses']),0)}–{fmt_num(row.get(COLS['draws']),0)}"
        fights = fmt_num(row.get(COLS["fights"]), 0)
        winpct_chip = chip(row.get(COLS["win_pct"]), pct_map.get(COLS["win_pct"], pd.Series()).get(row.name))
        st.write(f"**Record:** {rec}  •  **Fights:** {fights}  •  **Win%:** {winpct_chip}")

        # Quick highlight strip (also with percentiles)
       

def render_attributes(col, row, pct_map):
    """Compact attributes block directly under the card (tiny text)."""
    with col:
        st.markdown("#### Attributes")
        phys = [
            f"{LABEL[COLS['height_cm']]}: {chip(row.get(COLS['height_cm']), pct_map.get(COLS['height_cm'], pd.Series()).get(row.name), ' cm')}",
            f"{LABEL[COLS['reach_cm']]}: {chip(row.get(COLS['reach_cm']), pct_map.get(COLS['reach_cm'], pd.Series()).get(row.name), ' cm')}",
            f"{LABEL[COLS['weight_kg']]}: {chip(row.get(COLS['weight_kg']), pct_map.get(COLS['weight_kg'], pd.Series()).get(row.name), ' kg')}",
        ]
        st.caption(" • ".join(phys))

        striking = [
            f"{LABEL[COLS['strikes_min']]}: {chip(row.get(COLS['strikes_min']), pct_map.get(COLS['strikes_min'], pd.Series()).get(row.name))}",
            f"{LABEL[COLS['strike_acc']]}: {chip(row.get(COLS['strike_acc']), pct_map.get(COLS['strike_acc'], pd.Series()).get(row.name), '%')}",
            f"{LABEL[COLS['strikes_abs_min']]}: {chip(row.get(COLS['strikes_abs_min']), pct_map.get(COLS['strikes_abs_min'], pd.Series()).get(row.name))}",
            f"{LABEL[COLS['strike_def']]}: {chip(row.get(COLS['strike_def']), pct_map.get(COLS['strike_def'], pd.Series()).get(row.name), '%')}",
        ]
        st.caption("**Striking:** " + " • ".join(striking))

        grappling = [
            f"{LABEL[COLS['tds_15']]}: {chip(row.get(COLS['tds_15']), pct_map.get(COLS['tds_15'], pd.Series()).get(row.name))}",
            f"{LABEL[COLS['td_acc']]}: {chip(row.get(COLS['td_acc']), pct_map.get(COLS['td_acc'], pd.Series()).get(row.name), '%')}",
            f"{LABEL[COLS['td_def']]}: {chip(row.get(COLS['td_def']), pct_map.get(COLS['td_def'], pd.Series()).get(row.name), '%')}",
            f"{LABEL[COLS['subs_15']]}: {chip(row.get(COLS['subs_15']), pct_map.get(COLS['subs_15'], pd.Series()).get(row.name))}",
        ]
        st.caption("**Grappling:** " + " • ".join(grappling))

        st.caption("*(Stats from dataset; small-sample fighters may have noisy rates.)*")

# ---------- UI ----------
st.title("UFC Fighter Comparison Tool 🥊")
st.write(
    "Compare **two UFC fighters** side-by-side across physical stats, records, and advanced performance metrics. "
    "Select two fighters below to get started."
)

df = load_data(DATA_PATH)
pct_map = compute_percentiles(df, PCT_COLUMNS)

# Smart defaults
famous_try = [
    "Conor McGregor", "Khabib Nurmagomedov", "Jon Jones", "Israel Adesanya",
    "Anderson Silva", "Georges St-Pierre", "Dustin Poirier", "Max Holloway"
]
names = sorted(df[COLS["name"]].unique().tolist())
present = [n for n in famous_try if n in names]
if len(present) >= 2:
    default_a, default_b = present[0], present[1]
else:
    if COLS["fights"] in df.columns:
        top2 = df.sort_values(COLS["fights"], ascending=False)[COLS["name"]].head(2).tolist()
        default_a, default_b = (top2 + names)[:2]
    else:
        default_a, default_b = names[0], names[1] if len(names) > 1 else names[0]

c1, c2 = st.columns(2)
with c1:
    fighter_a = st.selectbox("Choose Fighter A", names, index=names.index(default_a))
with c2:
    pre_b = default_b if default_b != fighter_a else (default_a if default_a != fighter_a else names[1])
    fighter_b = st.selectbox("Choose Fighter B", names, index=names.index(pre_b))

if fighter_a == fighter_b:
    st.warning("Please select two **different** fighters.")
    st.stop()

a_row = get_row(df, fighter_a)
b_row = get_row(df, fighter_b)

# Player cards + attributes underneath
st.subheader("Player Cards")
cc1, cc2 = st.columns(2)
render_card(cc1, a_row, pct_map)
render_card(cc2, b_row, pct_map)
render_attributes(cc1, a_row, pct_map)
render_attributes(cc2, b_row, pct_map)

st.markdown("---")
st.subheader("Attribute Comparison")

def plot_group(group_name, cols):
    st.markdown(f"### {group_name}")
    grid_cols = st.columns(2)
    for i, metric_col in enumerate(cols):
        with grid_cols[i % 2]:
            if metric_col not in df.columns:
                st.info(f"Missing: {metric_col}")
                continue
            fig = bar_compare(metric_col, a_row, b_row)
            st.plotly_chart(fig, use_container_width=True)
    if group_name == "Striking":
        st.caption("Strikes/min reflects output; accuracy and defense show efficiency and avoidance. Consider fights count—small samples can inflate numbers.")
    elif group_name == "Grappling":
        st.caption("TDs/15 indicates takedown pace; TD Acc % shows success rate; TD Def % shows defensive ability; Subs/15 reflects submission threat.")
    elif group_name == "Physicals":
        st.caption("Physical advantages (height, reach, weight) can influence style and matchups, but do not guarantee outcomes.")
    elif group_name == "Record":
        st.caption("Win% and record summarize outcomes; they don’t capture strength of schedule or style.")

for g, cols in GROUPS.items():
    plot_group(g, cols)

st.markdown("---")
st.subheader("Multi-Metric Radar")
st.caption("Metrics are min–max normalized on the full dataset for fair shape comparisons.")
radar_fig = radar_compare(df, a_row, b_row, RADAR_METRICS)
st.plotly_chart(radar_fig, use_container_width=True)

with st.expander("Data Notes & Caveats"):
    st.write(
        "- Percentile badges (pXX) compare the fighter to the entire dataset for that metric.\n"
        "- Some physical stats may have been imputed or set to 0 if missing based on your cleaning choices.\n"
        "- Win% and rate stats can be noisy for fighters with few bouts.\n"
        "- Radar chart uses min–max normalization across the dataset."
    )
