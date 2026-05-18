import sys
import datetime
import json
import random
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ITSM Ticket Assistant", page_icon="🛠️", layout="wide")

# Import all logic from app.py
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from app import init as _init, analyse_ticket, prioritize_ticket

# Wrap with Streamlit cache so they only run once per session
init = st.cache_resource(show_spinner="Loading AI models…")(_init)

st.title("ITSM Ticket Assistant")
st.caption("AI-powered ticket analysis and guided resolution by Samarth Inc")


def _format_ticket_id_value(value):
    """Normalize ticket ID numeric portions to realistic 5 digits (e.g., INC-12 -> INC-10012)."""
    if value is None:
        return value
    text = str(value).strip()
    if not text:
        return value

    match = re.match(r"^(.*?)(\d+)$", text)
    if match:
        prefix, number = match.groups()
        num = int(number)
        if num < 10000:
            num += 10000
        return f"{prefix}{num}"

    if text.isdigit():
        num = int(text)
        if num < 10000:
            num += 10000
        return str(num)
    return value


def render_ai_response(result):
    """Render AI output by extracting description and resolution steps from JSON."""
    parsed = None

    if isinstance(result, (dict, list)):
        parsed = result
    elif isinstance(result, str):
        text = result.strip()
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None

    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        parsed = parsed[0]

    def _pick_field(payload, candidates):
        if not isinstance(payload, dict):
            return None
        lowered = {str(k).strip().lower(): k for k in payload.keys()}
        for candidate in candidates:
            if candidate in lowered:
                return payload[lowered[candidate]]
        for low_key, original_key in lowered.items():
            if all(token in low_key for token in candidates[0].split()):
                return payload[original_key]
        return None

    def _render_steps(steps):
        if isinstance(steps, list):
            for step in steps:
                st.write(step)
            return
        st.write(steps)

    if isinstance(parsed, dict):
        st.markdown("### AI Response")
        desc = _pick_field(parsed, ["elaborated description", "description", "clear description", "detailed description"])
        steps = _pick_field(parsed, ["resolution steps", "steps", "resolution", "recommended steps"])

        if desc is not None:
            st.markdown("#### Clear Description")
            st.write(desc)

        if steps is not None:
            st.markdown("#### Resolution Steps")
            _render_steps(steps)

        if desc is not None or steps is not None:
            return

    st.markdown("### AI Response")
    st.write(result)


def _prepare_uploaded_df(uploaded_file):
    """Read uploaded Excel and normalize datetime fields for display/analysis."""
    uploaded_df = pd.read_excel(uploaded_file)
    for col in uploaded_df.select_dtypes(include=["datetime64[ns]", "datetimetz"]):
        uploaded_df[col] = uploaded_df[col].dt.strftime("%Y-%m-%d")

    # Normalize ticket-number display to 5 digits for common ticket ID columns.
    ticket_id_col = _find_column(uploaded_df, ["ticket id", "ticket", "incident id", "incident number", "case id"])
    if ticket_id_col:
        uploaded_df[ticket_id_col] = uploaded_df[ticket_id_col].apply(_format_ticket_id_value)

    # Uploaded priority is ignored; the app will always assign priority using AI.
    priority_columns = [c for c in uploaded_df.columns if str(c).strip().lower() == "priority"]
    uploaded_df = uploaded_df.drop(columns=priority_columns, errors="ignore")
    return uploaded_df


def _find_column(df, candidates):
    """Find first matching column by exact normalized name or token containment."""
    norm_to_col = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in norm_to_col:
            return norm_to_col[candidate]

    for col in df.columns:
        low = str(col).strip().lower()
        for candidate in candidates:
            parts = candidate.split()
            if all(part in low for part in parts):
                return col
    return None


def _to_text_series(df, columns):
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()


def _compute_ticket_kpis(df):
    """Compute requested KPI counts using schema-tolerant rules."""
    if df.empty:
        return {
            "duplicate": 0,
            "insufficient": 0,
            "sufficient_followup": 0,
            "unassigned": 0,
            "reopened": 0,
        }

    title_col = _find_column(df, ["title", "issue title", "subject", "summary"])
    desc_col = _find_column(df, ["description", "issue description", "details"])
    status_col = _find_column(df, ["status", "state", "ticket status"])
    assignee_col = _find_column(df, ["assignee", "assigned to", "owner", "agent"])
    reopen_col = _find_column(df, ["reopen count", "reopened count", "times reopened"])
    duplicate_col = _find_column(df, ["duplicate", "is duplicate", "redundant"])

    status_text = _to_text_series(df, [status_col] if status_col else [])
    followup_text = _to_text_series(df, [status_col, _find_column(df, ["remarks", "comments", "next action", "notes"])])

    duplicate_mask = pd.Series(False, index=df.index)
    if title_col or desc_col:
        keys = _to_text_series(df, [c for c in [title_col, desc_col] if c]).str.replace(r"\s+", " ", regex=True).str.strip()
        duplicate_mask = duplicate_mask | keys.duplicated(keep="first")
    duplicate_mask = duplicate_mask | status_text.str.contains(r"duplicate|duplicated|redundant", regex=True, na=False)
    if duplicate_col:
        dup_text = df[duplicate_col].fillna("").astype(str).str.strip().str.lower()
        duplicate_mask = duplicate_mask | dup_text.str.contains(r"^1$|^true$|^yes$|^y$|duplicate|redundant", regex=True, na=False)

    if desc_col:
        desc_text = df[desc_col].fillna("").astype(str).str.strip().str.lower()
        insufficient_mask = (
            (desc_text.str.len() < 25)
            | desc_text.str.contains(r"^na$|^n/a$|^none$|^unknown$|^not sure$|^help$", regex=True, na=False)
        )
    else:
        insufficient_mask = pd.Series(False, index=df.index)
    insufficient_mask = insufficient_mask | followup_text.str.contains(r"insufficient|missing info|need more info|incomplete", regex=True, na=False)

    needs_followup_mask = followup_text.str.contains(r"follow[ -]?up|pending|waiting|awaiting", regex=True, na=False)
    sufficient_followup_mask = (~insufficient_mask) & needs_followup_mask

    if assignee_col:
        assignee_text = df[assignee_col].fillna("").astype(str).str.strip().str.lower()
        unassigned_mask = assignee_text.eq("") | assignee_text.str.contains(r"unassigned|none|na|n/a|tbd", regex=True, na=False)
    else:
        unassigned_mask = pd.Series(False, index=df.index)

    reopened_mask = status_text.str.contains(r"re-?opened", regex=True, na=False)
    if reopen_col:
        reopen_num = pd.to_numeric(df[reopen_col], errors="coerce").fillna(0)
        reopened_mask = reopened_mask | (reopen_num > 0)

    return {
        "duplicate": int(duplicate_mask.sum()),
        "insufficient": int(insufficient_mask.sum()),
        "sufficient_followup": int(sufficient_followup_mask.sum()),
        "unassigned": int(unassigned_mask.sum()),
        "reopened": int(reopened_mask.sum()),
    }


@st.cache_data(show_spinner=False)
def _build_mock_dashboard_df(rows=120000, cache_version="v5_low_dup"):
    """Create mock tickets for demo dashboard visuals and KPIs."""
    categories = [
        "Network & Connectivity", "Hardware & Peripherals", "Software & Applications",
        "Email & Communication", "Security & Access", "IT Service Request", "Other"
    ]
    # Unequal weights so bar chart shows clear volume differences
    category_weights = [0.28, 0.08, 0.24, 0.10, 0.05, 0.18, 0.07]
    statuses = ["Open", "In Progress", "Resolved", "Reopened"]
    priorities = ["P1 - Critical", "P2 - High", "P3 - Medium", "P4 - Low"]

    today = datetime.date.today()

    # Build per-day weights: weekdays heavier, weekends lighter, random spike days
    day_offsets = list(range(30))
    day_weights = []
    for d in day_offsets:
        date = today - datetime.timedelta(days=d)
        # Mon=0 … Sun=6
        dow = date.weekday()
        base = 1.0 if dow < 5 else 0.35
        spike = random.uniform(1.0, 3.5) if random.random() < 0.15 else 1.0
        day_weights.append(base * spike)
    total_w = sum(day_weights)
    day_weights = [w / total_w for w in day_weights]

    data = []
    for i in range(rows):
        offset = random.choices(day_offsets, weights=day_weights, k=1)[0]
        opened_on = today - datetime.timedelta(days=offset)
        status = random.choices(statuses, weights=[0.35, 0.3, 0.25, 0.1], k=1)[0]
        is_redundant = random.random() < 0.03
        is_insufficient = random.random() < 0.2
        needs_followup = random.random() < 0.28
        is_unassigned = random.random() < 0.04
        is_reopened = status == "Reopened" or random.random() < 0.02

        data.append({
            "Ticket ID": f"MOCK-{100000 + i}",
            "Category": random.choices(categories, weights=category_weights, k=1)[0],
            "Status": status,
            "Priority": random.choice(priorities),
            "Raised On": opened_on.isoformat(),
            "is_redundant": is_redundant,
            "is_insufficient": is_insufficient,
            "needs_followup": needs_followup,
            "is_unassigned": is_unassigned,
            "is_reopened": is_reopened,
        })

    return pd.DataFrame(data)

# ── Load resources ────────────────────────────────────────────────────────────
rails, client, chunks, embeddings = init()
df = _build_mock_dashboard_df(rows=120000, cache_version="v5_low_dup")

# ── KPI Snapshot ──────────────────────────────────────────────────────────────
kpis = {
    "total": len(df),
    "open": int((df["Status"] == "Open").sum()),
    "duplicate": int(df["is_redundant"].sum()),
    "insufficient": int(df["is_insufficient"].sum()),
    "unassigned": int(df["is_unassigned"].sum()),
    "reopened": int(df["is_reopened"].sum()),
}

_KPI_CSS = """
<style>
.kpi-row { display:flex; gap:12px; margin-bottom:20px; flex-wrap:wrap; }
.kpi-box {
    flex:1; min-width:110px;
    background:#161622; border:1px solid #2a2a3e; border-radius:10px;
    padding:18px 14px 14px 14px; text-align:center;
}
.kpi-val { font-size:2rem; font-weight:700; line-height:1; }
.kpi-lbl { font-size:0.72rem; color:#888; margin-top:6px; text-transform:uppercase; letter-spacing:.05em; }
.kpi-red   { color:#E05252; }
.kpi-white { color:#ffffff; }
.kpi-amber { color:#F3A712; }
.chart-card {
    background:#161622; border:1px solid #2a2a3e; border-radius:10px;
    padding:18px 16px 12px 16px; margin-bottom:16px;
}
.chart-card h5 { color:#e0e0e0; margin:0 0 14px 0; font-size:0.95rem; }
</style>
"""
st.markdown(_KPI_CSS, unsafe_allow_html=True)

st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-box"><div class="kpi-val kpi-white">{kpis['total']}</div><div class="kpi-lbl">Total Tickets</div></div>
    <div class="kpi-box"><div class="kpi-val kpi-red">{kpis['open']}</div><div class="kpi-lbl">Open</div></div>
    <div class="kpi-box"><div class="kpi-val kpi-red">{kpis['duplicate']}</div><div class="kpi-lbl">Duplicate Tickets</div></div>
    <div class="kpi-box"><div class="kpi-val kpi-amber">{kpis['insufficient']}</div><div class="kpi-lbl">Insufficient Info</div></div>
    <div class="kpi-box"><div class="kpi-val kpi-red">{kpis['unassigned']}</div><div class="kpi-lbl">Unassigned</div></div>
    <div class="kpi-box"><div class="kpi-val kpi-red">{kpis['reopened']}</div><div class="kpi-lbl">Reopened</div></div>
</div>
""", unsafe_allow_html=True)

st.caption("Demo dashboard uses generated mock data for presentation only.")

# ── Row 1: Category bar + AI Decision donut ───────────────────────────────────
_chart_r1c1, _chart_r1c2 = st.columns(2)

with _chart_r1c1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    _cat_df = df["Category"].value_counts().rename_axis("Category").reset_index(name="Count")
    _cat_bar = (
        alt.Chart(_cat_df)
        .mark_bar(cornerRadiusEnd=4, color="#E05252")
        .encode(
            y=alt.Y("Category:N", sort="-x", title="Category", axis=alt.Axis(labelColor="#aaa", labelFontSize=11)),
            x=alt.X("Count:Q", title="Count", axis=alt.Axis(labelColor="#666", grid=False)),
            tooltip=["Category", "Count"],
        )
        .configure_view(strokeWidth=0)
        .configure_axis(domainColor="#2a2a3e", tickColor="#2a2a3e")
        .properties(title="Tickets with Highest Volume by Category",height=300, background="transparent")
    )
    st.altair_chart(_cat_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with _chart_r1c2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    _pri_df = df["Priority"].value_counts().rename_axis("Priority").reset_index(name="Count")
    _pri_colors = {
        "P1 - Critical": "#E05252",
        "P2 - High": "#F3A712",
        "P3 - Medium": "#4C78A8",
        "P4 - Low": "#2A9D8F",
    }
    _pri_df["Color"] = _pri_df["Priority"].map(_pri_colors).fillna("#888")
    _pri_pct = _pri_df.copy()
    _pri_pct["Percentage"] = (_pri_pct["Count"] / _pri_pct["Count"].sum() * 100).round(1)
    _hover = alt.selection_point(fields=["Priority"], on="pointerover", empty=False, clear="pointerout")

    _donut_base = alt.Chart(_pri_pct).mark_arc(innerRadius=65, outerRadius=110).encode(
        theta=alt.Theta("Count:Q"),
        color=alt.Color(
            "Priority:N",
            scale=alt.Scale(domain=list(_pri_colors.keys()), range=list(_pri_colors.values())),
            legend=alt.Legend(title=None, labelColor="#ccc", labelFontSize=11),
        ),
        tooltip=["Priority", "Count", alt.Tooltip("Percentage:Q", format=".1f", title="% Share")],
    )

    # Keep the same stack as base and show only hovered slice in the larger layer.
    _donut_hover = (
        alt.Chart(_pri_pct)
        .mark_arc(innerRadius=65, outerRadius=125, stroke="#ffffff", strokeWidth=1.0)
        .encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color(
                "Priority:N",
                scale=alt.Scale(domain=list(_pri_colors.keys()), range=list(_pri_colors.values())),
                legend=None,
            ),
            opacity=alt.condition(_hover, alt.value(1), alt.value(0)),
        )
    )

    _donut = (
        (_donut_base + _donut_hover)
        .add_params(_hover)
        .configure_view(strokeWidth=0)
        .properties(title="AI Priority Decision Breakdown",height=320, background="transparent")
    )
    st.altair_chart(_donut, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Row 2: Status distribution bar + Ticket quality bubble ────────────────────
_chart_r2c1, _chart_r2c2 = st.columns(2)

with _chart_r2c1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    _status_df = df["Status"].value_counts().rename_axis("Status").reset_index(name="Count")
    _status_colors = {"Open": "#E05252", "In Progress": "#F3A712", "Resolved": "#2A9D8F", "Reopened": "#D45087"}
    _status_df["Color"] = _status_df["Status"].map(_status_colors).fillna("#888")
    _status_bar = (
        alt.Chart(_status_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("Status:N", sort="-y", title="Status", axis=alt.Axis(labelColor="#aaa", labelFontSize=11)),
            y=alt.Y("Count:Q", title="Count", axis=alt.Axis(labelColor="#666", grid=True, gridColor="#2a2a3e")),
            color=alt.Color("Status:N", scale=alt.Scale(
                domain=list(_status_colors.keys()), range=list(_status_colors.values())
            ), legend=None),
            tooltip=["Status", "Count"],
        )
        .configure_view(strokeWidth=0)
        .configure_axis(domainColor="#2a2a3e", tickColor="#2a2a3e")
        .properties(title="Ticket Status Distribution", height=280, background="transparent")
    )
    st.altair_chart(_status_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with _chart_r2c2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    _qual_df = pd.DataFrame([
        {"Type": "Duplicate Tickets", "Count": kpis["duplicate"],    "Pct": round(kpis["duplicate"]    / kpis["total"] * 100, 1)},
        {"Type": "Insufficient Info", "Count": kpis["insufficient"], "Pct": round(kpis["insufficient"] / kpis["total"] * 100, 1)},
        {"Type": "Unassigned",        "Count": kpis["unassigned"],   "Pct": round(kpis["unassigned"]   / kpis["total"] * 100, 1)},
        {"Type": "Reopened",          "Count": kpis["reopened"],     "Pct": round(kpis["reopened"]     / kpis["total"] * 100, 1)},
        {"Type": "Normal",            "Count": kpis["open"],         "Pct": round(kpis["open"]         / kpis["total"] * 100, 1)},
    ])
    _qual_colors = {
        "Duplicate Tickets": "#E05252", "Insufficient Info": "#F3A712",
        "Unassigned": "#7A5195", "Reopened": "#D45087", "Normal": "#2A9D8F",
    }
    _bubble = (
        alt.Chart(_qual_df)
        .mark_point(filled=True, opacity=0.85)
        .encode(
            x=alt.X("Pct:Q", title="Pct", axis=alt.Axis(labelColor="#aaa", grid=False)),
            y=alt.Y("Count:Q", title="Count", axis=alt.Axis(labelColor="#aaa", gridColor="#2a2a3e")),
            size=alt.Size("Count:Q", scale=alt.Scale(range=[300, 2500]), legend=None),
            color=alt.Color("Type:N", scale=alt.Scale(
                domain=list(_qual_colors.keys()), range=list(_qual_colors.values())
            ), legend=alt.Legend(title=None, labelColor="#ccc", labelFontSize=10, orient="bottom", columns=3)),
            tooltip=["Type", "Count", alt.Tooltip("Pct:Q", format=".1f", title="% of Total")],
        )
        .configure_view(strokeWidth=0)
        .configure_axis(domainColor="#2a2a3e", tickColor="#2a2a3e")
        .properties(title="Ticket Quality Type Distribution", height=300, background="transparent")
    )
    st.altair_chart(_bubble, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Row 3: Daily ticket trend – simple line chart ─────────────────────────────
st.markdown('<div class="chart-card">', unsafe_allow_html=True)

_trend_df = (
    df.groupby("Raised On").size().rename("Tickets").reset_index().sort_values("Raised On")
)
_trend_df["Raised On"] = pd.to_datetime(_trend_df["Raised On"])

_trend_chart = (
    alt.Chart(_trend_df)
    .mark_line(color="#E05252", strokeWidth=2.5, point=alt.OverlayMarkDef(color="#E05252", size=40))
    .encode(
        x=alt.X("Raised On:T", title="Date", axis=alt.Axis(labelColor="#aaa", grid=False, format="%b %d")),
        y=alt.Y("Tickets:Q", title="Tickets", axis=alt.Axis(labelColor="#aaa", gridColor="#2a2a3e")),
        tooltip=[
            alt.Tooltip("Raised On:T", title="Date", format="%b %d"),
            alt.Tooltip("Tickets:Q", title="Tickets"),
        ],
    )
    .configure_view(strokeWidth=0)
    .configure_axis(domainColor="#2a2a3e", tickColor="#2a2a3e")
    .properties(title="Daily Ticket Trend (Last 30 Days)", height=250, background="transparent")
)

st.altair_chart(_trend_chart, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ── Source Systems ─────────────────────────────────────────────────────────────
st.subheader("Source Systems")
st.caption("Select a source system to connect and analyse ticket data.")

_CARD_CSS = """
<style>
.src-card {
    background: #1e1e2e;
    border: 1px solid #2e2e4e;
    border-radius: 12px;
    padding: 24px 20px 20px 20px;
    min-height: 160px;
    margin-bottom: 8px;
}
.src-card h4 { color: #ffffff; margin: 0 0 8px 0; font-size: 1.05rem; }
.src-card p  { color: #a0a0b8; font-size: 0.82rem; margin: 0; line-height: 1.5; }
</style>
"""
st.markdown(_CARD_CSS, unsafe_allow_html=True)

_SOURCES = [
    {
        "key": "mcp",
        "label": "MCP",
        "description": "Connect to the Model Context Protocol server to stream live ticket context and structured tool outputs.",
    },
    {
        "key": "api",
        "label": "API",
        "description": "Pull tickets directly from your ITSM REST API endpoint using bearer token authentication.",
    },
    {
        "key": "knowledge_base",
        "label": "Knowledge Base",
        "description": "Upload an Excel file of tickets and run AI-powered triage and resolution analysis row by row.",
    },
    {
        "key": "ivr",
        "label": "IVR",
        "description": "Ingest call transcripts from your Interactive Voice Response system for automated ticket creation.",
    },
    {
        "key": "iva",
        "label": "IVA",
        "description": "Analyse conversations from your Intelligent Virtual Assistant to surface recurring issues and gaps.",
    },
]

# Render cards in rows of 3
_cols_row1 = st.columns(3)
_cols_row2 = st.columns([1, 1, 2])   # IVR and IVA left-aligned, empty right

for _idx, _src in enumerate(_SOURCES):
    _col = _cols_row1[_idx] if _idx < 3 else _cols_row2[_idx - 3]
    with _col:
        st.markdown(
            f'<div class="src-card"><h4>{_src["label"]}</h4><p>{_src["description"]}</p></div>',
            unsafe_allow_html=True,
        )
        _btn_label = (
            "Open Knowledge Base"
            if _src["key"] == "knowledge_base"
            else f"Connect {_src['label']}"
        )
        if st.button(_btn_label, key=f"src_btn_{_src['key']}", use_container_width=True):
            st.session_state["selected_source"] = _src["key"]

_selected_source = st.session_state.get("selected_source")

if _selected_source and _selected_source != "knowledge_base":
    _src_label = next(s["label"] for s in _SOURCES if s["key"] == _selected_source)
    st.divider()
    st.success(f"**{_src_label}** source connected (demo mode). Live data ingestion will appear here.")
    st.info(
        f"In production, tickets ingested from **{_src_label}** will be displayed in an interactive "
        f"table and can be selected for AI triage and resolution analysis — identical to the Knowledge Base flow."
    )

st.divider()

# ── Knowledge Base – Upload Excel and Analyse ──────────────────────────────────
if _selected_source != "knowledge_base":
    st.markdown(
        "_Click **Open Knowledge Base** above to upload an Excel file and run AI analysis on your tickets._"
    )
    st.stop()

st.subheader("Knowledge Base – Upload Ticket Excel File")
st.caption("Upload an .xlsx/.xls file, then click a row in the preview table to run AI analysis.")

uploaded_file = st.file_uploader("Add Excel File", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        uploaded_df = _prepare_uploaded_df(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        st.stop()

    if uploaded_df.empty:
        st.warning("The uploaded file has no rows.")
        st.stop()

    file_token = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("uploaded_file_token") != file_token:
        st.session_state["uploaded_file_token"] = file_token
        st.session_state["last_uploaded_analysis_key"] = None
        st.session_state["last_uploaded_analysis_result"] = None
        st.session_state["last_uploaded_priority"] = None

    st.success("File uploaded successfully")
    table_event = st.dataframe(
        uploaded_df.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="uploaded_ticket_table",
    )

    selected_rows = table_event.selection.rows if table_event else []
    if not selected_rows:
        st.info("Select any row from the table, then click Start AI Analysis.")
        st.stop()

    selected_row_index = int(selected_rows[0])
    analysis_key = f"{file_token}:{selected_row_index}"
    st.caption(f"Selected row: {selected_row_index + 1}")

    run_analysis = st.button("Start AI Analysis", key="analyse_uploaded_selected")

    if run_analysis:
        selected_ticket = uploaded_df.iloc[selected_row_index].to_dict()

        with st.spinner("Assigning priority with AI…"):
            selected_ticket["Priority"] = prioritize_ticket(selected_ticket, rails)

        with st.spinner("Analysing uploaded ticket…"):
            result = analyse_ticket(selected_ticket, rails, client, chunks, embeddings)

        st.session_state["last_uploaded_analysis_key"] = analysis_key
        st.session_state["last_uploaded_analysis_result"] = result
        st.session_state["last_uploaded_priority"] = selected_ticket["Priority"]

    if st.session_state.get("last_uploaded_analysis_key") == analysis_key and st.session_state.get("last_uploaded_analysis_result") is not None:
        st.success(f"Analysis complete for selected row {selected_row_index + 1}")
        st.info(f"AI-assigned Priority: **{st.session_state.get('last_uploaded_priority', 'P3 - Medium')}**")
        render_ai_response(st.session_state.get("last_uploaded_analysis_result"))
    else:
        st.info("Click Start AI Analysis to generate results for the selected row.")
