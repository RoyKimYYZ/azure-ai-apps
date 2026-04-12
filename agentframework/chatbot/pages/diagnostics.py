"""Agent Diagnostics page — opens in a dedicated browser tab.

Shows rich context-window visualisation, token-usage charts, management
tips, and application tracing / logging for a selected agent.
"""

from __future__ import annotations

import html as _html
import json as _json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Ensure the chatbot package root is importable from the pages/ sub-dir.
_CHATBOT_DIR = str(Path(__file__).resolve().parent.parent)
if _CHATBOT_DIR not in sys.path:
    sys.path.insert(0, _CHATBOT_DIR)

# Ensure the project root is importable so we can reach the config package.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st  # noqa: E402
from diagnostics_store import (  # noqa: E402
    clear_agent_diagnostics,
    get_agent_diagnostics,
    get_all_agents,
    get_context_window_size,
)

from config import get_config  # noqa: E402

# ── Page config ──────────────────────────────────────────────────────
_cfg = get_config()
st.set_page_config(
    page_title=_cfg.ui.labels.diagnostics_title,
    page_icon=_cfg.ui.labels.diagnostics_icon,
    layout="wide",
)

# Hide the default Streamlit multi-page sidebar nav
st.markdown(
    '<style>[data-testid="stSidebarNav"]{display:none!important}</style>',
    unsafe_allow_html=True,
)

# ── Sidebar: agent selector & controls ───────────────────────────────
all_agents = get_all_agents()
params = st.query_params
query_agent = params.get("agent", "")

with st.sidebar:
    st.header("📊 Diagnostics")
    view_mode = st.radio("View", ["Overview", "Performance Diagnostics"], index=0)
    if all_agents:
        default_idx = all_agents.index(query_agent) if query_agent in all_agents else 0
        agent_name: str = st.selectbox("Agent", all_agents, index=default_idx)
    else:
        agent_name = ""
    st.divider()
    auto_refresh = st.toggle("Auto-refresh (5 s)", value=False)
    if st.button("🔄 Refresh now"):
        st.rerun()
    st.divider()
    if agent_name and st.button("🗑️ Clear diagnostics data", type="secondary"):
        clear_agent_diagnostics(agent_name)
        st.rerun()
    st.divider()
    st.caption("Open from the main chatbot via the sidebar diagnostics link.")

# ── Auto-refresh via JS ─────────────────────────────────────────────
if auto_refresh:
    st.markdown(
        f"<script>setTimeout(function(){{window.location.reload()}},{_cfg.ui.performance.auto_refresh_interval_ms});</script>",
        unsafe_allow_html=True,
    )

# ── Guard: need agent data ───────────────────────────────────────────
if not agent_name:
    st.info("No diagnostics data yet.  Start chatting with an agent and data will appear here.")
    st.stop()

data = get_agent_diagnostics(agent_name)
turns: list[dict] = data.get("turns", [])
app_logs: list[str] = data.get("app_logs", [])
errors: list[str] = data.get("errors", [])
performance_events: list[dict] = data.get("performance_events", [])

st.markdown(f"## 📊 {agent_name} — Diagnostics")

if not turns:
    st.info("No completion turns recorded yet for this agent.  Send a message and refresh.")
    st.stop()

latest = turns[-1]


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        normalized = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed
    except Exception:
        return None


def _display_timestamp(value: str | None) -> str:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return str(value or "")
    return parsed.astimezone(UTC).strftime("%m/%d/%Y")


def _group_conversations(turn_items: list[dict]) -> list[list[dict]]:
    grouped: list[list[dict]] = []
    current: list[dict] = []
    previous: dict | None = None
    previous_ts: datetime | None = None
    for index, turn in enumerate(turn_items, start=1):
        turn_copy = dict(turn)
        turn_copy["_turn_number"] = index
        current_ts = _parse_timestamp(turn.get("timestamp"))
        starts_new = False
        if previous is None:
            starts_new = True
        else:
            prev_messages = int(previous.get("messages_count") or 0)
            curr_messages = int(turn.get("messages_count") or 0)
            if curr_messages <= prev_messages or previous_ts and current_ts and (current_ts - previous_ts) > timedelta(minutes=30):
                starts_new = True
        if starts_new and current:
            grouped.append(current)
            current = []
        current.append(turn_copy)
        previous = turn_copy
        previous_ts = current_ts
    if current:
        grouped.append(current)
    return grouped


if view_mode == "Performance Diagnostics":
    st.markdown("### 🚀 Performance Diagnostics")
    import pandas as pd

    conversations = _group_conversations(turns)
    perf_df = pd.DataFrame(performance_events)
    if not perf_df.empty:
        perf_df["duration_ms"] = pd.to_numeric(perf_df["duration_ms"], errors="coerce").fillna(0.0)
    db_df = perf_df[perf_df["category"] == "db"].copy() if not perf_df.empty else pd.DataFrame()
    llm_df = perf_df[perf_df["category"] == "llm"].copy() if not perf_df.empty else pd.DataFrame()

    total_db_calls = int(len(db_df.index)) if not db_df.empty else 0
    avg_db_ms = float(db_df["duration_ms"].mean()) if not db_df.empty else 0.0
    slowest_event = perf_df.sort_values("duration_ms", ascending=False).head(1) if not perf_df.empty else pd.DataFrame()
    slowest_label = "n/a"
    slowest_value = 0.0
    if not slowest_event.empty:
        slowest_label = f"{slowest_event.iloc[0]['category']}:{slowest_event.iloc[0]['name']}"
        slowest_value = float(slowest_event.iloc[0]["duration_ms"])

    total_turn_latency = sum(float(turn.get("latency_s", 0.0) or 0.0) for turn in turns)
    avg_turns_per_conversation = (len(turns) / len(conversations)) if conversations else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Conversations", str(len(conversations)))
    m2.metric("Turns", str(len(turns)))
    m3.metric("DB Calls", str(total_db_calls))
    m4.metric("Avg DB Latency", f"{avg_db_ms:.1f} ms")
    m5.metric("Slowest Span", f"{slowest_value:.1f} ms")
    st.caption(f"Slowest measured operation: {slowest_label} | Average turns per conversation: {avg_turns_per_conversation:.1f} | Total turn latency: {total_turn_latency:.2f}s")

    summary_tab, conversations_tab, database_tab, spans_tab, export_tab = st.tabs([
        "Summary",
        "Conversations",
        "Database Calls",
        "Measured Spans",
        "LLM Export",
    ])

    with summary_tab:
        st.markdown("#### Turn Latency Over Time")
        turns_df = pd.DataFrame(
            {
                "Turn": list(range(1, len(turns) + 1)),
                "Latency (s)": [float(turn.get("latency_s", 0.0) or 0.0) for turn in turns],
                "Input Tokens": [int(turn.get("input_tokens", 0) or 0) for turn in turns],
                "Output Tokens": [int(turn.get("output_tokens", 0) or 0) for turn in turns],
            }
        ).set_index("Turn")
        st.line_chart(turns_df, use_container_width=True)

        if conversations:
            conv_rows = []
            for idx, conversation in enumerate(conversations, start=1):
                conv_rows.append(
                    {
                        "Conversation": idx,
                        "Turns": len(conversation),
                        "Latency (s)": round(sum(float(turn.get("latency_s", 0.0) or 0.0) for turn in conversation), 3),
                        "Input Tokens": sum(int(turn.get("input_tokens", 0) or 0) for turn in conversation),
                        "Output Tokens": sum(int(turn.get("output_tokens", 0) or 0) for turn in conversation),
                    }
                )
            conv_df = pd.DataFrame(conv_rows).set_index("Conversation")
            st.markdown("#### Conversation Totals")
            st.bar_chart(conv_df[["Latency (s)", "Turns"]], use_container_width=True)

        if not perf_df.empty:
            st.markdown("#### Average Span Latency By Operation")
            op_df = (
                perf_df.groupby(["category", "name"], dropna=False)["duration_ms"]
                .agg(["mean", "count", "max"])
                .reset_index()
                .sort_values("mean", ascending=False)
            )
            st.dataframe(op_df.rename(columns={"mean": "avg_ms", "count": "calls", "max": "max_ms"}), use_container_width=True, hide_index=True)

    with conversations_tab:
        if not conversations:
            st.caption("No conversation history available yet.")
        for idx, conversation in enumerate(conversations, start=1):
            start_ts = _display_timestamp(conversation[0].get("timestamp"))
            end_ts = _display_timestamp(conversation[-1].get("timestamp"))
            total_latency = sum(float(turn.get("latency_s", 0.0) or 0.0) for turn in conversation)
            header = f"Conversation {idx} · turns={len(conversation)} · latency={total_latency:.2f}s · {start_ts} to {end_ts}"
            with st.expander(header, expanded=(idx == len(conversations))):
                turn_rows = []
                for turn in conversation:
                    turn_rows.append(
                        {
                            "Turn": turn.get("_turn_number"),
                            "Date": _display_timestamp(turn.get("timestamp")),
                            "Model": turn.get("model", ""),
                            "Status": turn.get("status", ""),
                            "Latency (s)": round(float(turn.get("latency_s", 0.0) or 0.0), 3),
                            "Input": int(turn.get("input_tokens", 0) or 0),
                            "Output": int(turn.get("output_tokens", 0) or 0),
                            "Messages": int(turn.get("messages_count", 0) or 0),
                        }
                    )
                st.dataframe(pd.DataFrame(turn_rows), use_container_width=True, hide_index=True)
                req_ids = [str(turn.get("request_id") or "") for turn in conversation if turn.get("request_id")]
                if req_ids and not perf_df.empty:
                    conv_perf_df = perf_df[perf_df["request_id"].isin(req_ids)].copy()
                    if not conv_perf_df.empty:
                        st.markdown("**Measured operations for this conversation**")
                        st.dataframe(
                            conv_perf_df[["timestamp", "category", "name", "duration_ms", "status"]].rename(
                                columns={"timestamp": "Date", "duration_ms": "Duration (ms)"}
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

    with database_tab:
        st.markdown("#### Database and Repository Calls")
        if db_df.empty:
            st.caption("No structured database-call metrics recorded yet for this agent.")
        else:
            db_summary = (
                db_df.groupby("name", dropna=False)["duration_ms"]
                .agg(["count", "mean", "max"])
                .reset_index()
                .sort_values("mean", ascending=False)
            )
            db_chart = db_summary.set_index("name")[["mean", "max"]].rename(columns={"mean": "avg_ms", "max": "max_ms"})
            st.bar_chart(db_chart, use_container_width=True)
            st.dataframe(
                db_summary.rename(columns={"count": "calls", "mean": "avg_ms", "max": "max_ms", "name": "operation"}),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("#### Recent DB Calls")
            recent_db = db_df.sort_values("timestamp", ascending=False).head(100).copy()
            recent_db["timestamp"] = recent_db["timestamp"].map(_display_timestamp)
            st.dataframe(
                recent_db[["timestamp", "request_id", "name", "duration_ms", "status", "details"]].rename(
                    columns={"timestamp": "Date", "name": "operation", "duration_ms": "Duration (ms)"}
                ),
                use_container_width=True,
                hide_index=True,
            )

    with spans_tab:
        if perf_df.empty:
            st.caption("No measured spans captured yet.")
        else:
            category_counts = perf_df.groupby("category", dropna=False).size().rename("count")
            st.markdown("#### Span Count By Category")
            st.bar_chart(category_counts, use_container_width=True)

            st.markdown("#### Recent Measured Spans")
            recent_perf = perf_df.sort_values("timestamp", ascending=False).head(150).copy()
            recent_perf["timestamp"] = recent_perf["timestamp"].map(_display_timestamp)
            st.dataframe(
                recent_perf[["timestamp", "request_id", "category", "name", "duration_ms", "status", "details"]].rename(
                    columns={"timestamp": "Date", "duration_ms": "Duration (ms)"}
                ),
                use_container_width=True,
                hide_index=True,
            )

    with export_tab:
        conv_summaries = []
        for idx, conversation in enumerate(conversations, start=1):
            conv_summaries.append(
                {
                    "conversation": idx,
                    "start_date": _display_timestamp(conversation[0].get("timestamp")),
                    "end_date": _display_timestamp(conversation[-1].get("timestamp")),
                    "turns": len(conversation),
                    "total_latency_s": round(sum(float(turn.get("latency_s", 0.0) or 0.0) for turn in conversation), 3),
                    "input_tokens": sum(int(turn.get("input_tokens", 0) or 0) for turn in conversation),
                    "output_tokens": sum(int(turn.get("output_tokens", 0) or 0) for turn in conversation),
                }
            )

        top_spans = []
        if not perf_df.empty:
            for _, row in perf_df.sort_values("duration_ms", ascending=False).head(20).iterrows():
                top_spans.append(
                    {
                        "date": _display_timestamp(row.get("timestamp")),
                        "request_id": row.get("request_id", ""),
                        "category": row.get("category", ""),
                        "name": row.get("name", ""),
                        "duration_ms": round(float(row.get("duration_ms", 0.0) or 0.0), 2),
                        "status": row.get("status", ""),
                        "details": row.get("details", {}),
                    }
                )

        db_summary_rows = []
        if not db_df.empty:
            grouped = db_df.groupby("name", dropna=False)["duration_ms"].agg(["count", "mean", "max"]).reset_index()
            for _, row in grouped.sort_values("mean", ascending=False).iterrows():
                db_summary_rows.append(
                    {
                        "operation": row.get("name", ""),
                        "calls": int(row.get("count", 0) or 0),
                        "avg_ms": round(float(row.get("mean", 0.0) or 0.0), 2),
                        "max_ms": round(float(row.get("max", 0.0) or 0.0), 2),
                    }
                )

        export_payload = {
            "agent": agent_name,
            "generated_at": datetime.now(UTC).isoformat(),
            "summary": {
                "conversation_count": len(conversations),
                "turn_count": len(turns),
                "avg_turns_per_conversation": round(avg_turns_per_conversation, 2),
                "total_turn_latency_s": round(total_turn_latency, 3),
                "performance_event_count": int(len(perf_df.index)) if not perf_df.empty else 0,
                "db_call_count": total_db_calls,
                "avg_db_latency_ms": round(avg_db_ms, 2),
                "slowest_event": {"label": slowest_label, "duration_ms": round(slowest_value, 2)},
            },
            "conversation_summaries": conv_summaries,
            "db_summary": db_summary_rows,
            "top_spans": top_spans,
            "recent_turns": [
                {
                    "date": _display_timestamp(turn.get("timestamp")),
                    "request_id": turn.get("request_id", ""),
                    "model": turn.get("model", ""),
                    "status": turn.get("status", ""),
                    "latency_s": round(float(turn.get("latency_s", 0.0) or 0.0), 3),
                    "input_tokens": int(turn.get("input_tokens", 0) or 0),
                    "output_tokens": int(turn.get("output_tokens", 0) or 0),
                    "messages_count": int(turn.get("messages_count", 0) or 0),
                }
                for turn in turns[-25:]
            ],
            "prompt_for_llm": "Review these diagnostics, identify the biggest performance bottlenecks, and propose prioritized code changes with expected impact and validation steps.",
        }

        st.markdown("#### Copyable Diagnostics For LLM Analysis")
        st.text_area(
            "performance_diagnostics_export",
            value=_json.dumps(export_payload, indent=2, ensure_ascii=False, default=str),
            height=420,
            key="performance_diagnostics_export_text",
        )
        st.caption("Copy the JSON above and paste it into another LLM for optimization recommendations.")

    st.stop()

# ═════════════════════════════════════════════════════════════════════
# Section 1 – Key Metrics
# ═════════════════════════════════════════════════════════════════════
m1, m2, m3, m4, m5 = st.columns(5)
ctx_max = latest.get("context_window_max") or get_context_window_size(latest.get("model", ""))
used_tokens = latest.get("input_tokens") or 0
out_reserved = latest.get("output_reserved_tokens") or 0
utilization = ((used_tokens + out_reserved) / ctx_max * 100) if ctx_max else 0

m1.metric("Context Utilisation", f"{utilization:.1f}%")
m2.metric("Last Input Tokens", f"{used_tokens:,}")
m3.metric("Last Output Tokens", f"{latest.get('output_tokens', 0):,}")
m4.metric("Context Window", f"{ctx_max:,}")
m5.metric("Total Turns", str(len(turns)))

# ═════════════════════════════════════════════════════════════════════
# Section 2 – Context Window Visualisation
# ═════════════════════════════════════════════════════════════════════
st.markdown("### 🧠 Context Window Composition")

sys_tok = latest.get("system_prompt_est_tokens", 0)
ctx_tok = latest.get("context_provider_est_tokens", 0)
hist_tok = latest.get("chat_history_est_tokens", 0)
out_tok = out_reserved
available = max(0, ctx_max - sys_tok - ctx_tok - hist_tok - out_tok)

def _pct(val: int) -> float:
    return (val / ctx_max * 100) if ctx_max else 0

segments = [
    ("System Prompt", sys_tok, "#4a9eff"),
    ("Long-term Memory (Context Provider)", ctx_tok, "#2ecc71"),
    ("Chat History", hist_tok, "#f39c12"),
    ("Output Reserved", out_tok, "#9b59b6"),
    ("Available", available, "#2a2a2a"),
]

# Build stacked bar
bar_parts = ""
for label, tokens, colour in segments:
    pct = _pct(tokens)
    if pct < 0.3:
        continue
    bar_parts += (
        f'<div title="{label}: {tokens:,} tokens ({pct:.1f}%)" '
        f'style="width:{pct:.2f}%;background:{colour};display:flex;align-items:center;'
        f'justify-content:center;font-size:0.7rem;color:#fff;white-space:nowrap;overflow:hidden;">'
        f'{"" if pct < 4 else f"{pct:.0f}%"}</div>'
    )

st.markdown(
    f"""
    <div style="display:flex;height:44px;border-radius:10px;overflow:hidden;
                border:1px solid #444;margin:0.5rem 0 0.25rem 0;">
      {bar_parts}
    </div>
    """,
    unsafe_allow_html=True,
)

# Legend
legend_html = '<div style="display:flex;flex-wrap:wrap;gap:1.2rem;margin:0.3rem 0 1rem 0;">'
for label, tokens, colour in segments:
    legend_html += (
        f'<span style="display:inline-flex;align-items:center;gap:0.3rem;font-size:0.82rem;">'
        f'<span style="width:12px;height:12px;border-radius:3px;background:{colour};display:inline-block;"></span>'
        f'{label}: <strong>{tokens:,}</strong></span>'
    )
legend_html += "</div>"
st.markdown(legend_html, unsafe_allow_html=True)

# Breakdown table
st.markdown("#### Token Breakdown (Latest Turn)")
breakdown_cols = st.columns(5)
for i, (label, tokens, colour) in enumerate(segments):
    with breakdown_cols[i]:
        st.markdown(
            f'<div style="text-align:center;padding:0.6rem;border-radius:8px;'
            f'border:2px solid {colour};margin-bottom:0.3rem;">'
            f'<div style="font-size:1.4rem;font-weight:700;color:{colour};">{tokens:,}</div>'
            f'<div style="font-size:0.78rem;color:#bbb;">{label}</div></div>',
            unsafe_allow_html=True,
        )

# ═════════════════════════════════════════════════════════════════════
# Section 3 – Token Usage Over Time
# ═════════════════════════════════════════════════════════════════════
st.markdown("### 📈 Token Usage Over Time")

if len(turns) >= 2:
    import pandas as pd  # available via streamlit

    chart_data = pd.DataFrame(
        {
            "Turn": list(range(1, len(turns) + 1)),
            "Input Tokens": [t.get("input_tokens", 0) for t in turns],
            "Output Tokens": [t.get("output_tokens", 0) for t in turns],
            "Total Tokens": [t.get("total_tokens", 0) for t in turns],
        }
    )
    chart_data = chart_data.set_index("Turn")
    st.line_chart(chart_data, use_container_width=True)

    # Utilisation over time
    st.markdown("#### Context Utilisation % Over Time")
    util_data = pd.DataFrame(
        {
            "Turn": list(range(1, len(turns) + 1)),
            "Utilisation %": [
                round(
                    (t.get("input_tokens", 0) + t.get("output_reserved_tokens", 0))
                    / max(1, t.get("context_window_max", 0))
                    * 100,
                    1,
                )
                for t in turns
            ],
        }
    )
    util_data = util_data.set_index("Turn")
    st.area_chart(util_data, use_container_width=True)
else:
    st.caption("Need at least 2 turns to show trend charts.")

# ═════════════════════════════════════════════════════════════════════
# Section 4 – Context Window Management Tips
# ═════════════════════════════════════════════════════════════════════
st.markdown("### 💡 Context Window Management Tips")

with st.expander("What is the context window?", expanded=False):
    st.markdown(
        """
The **context window** is the maximum number of tokens a model can process in a single request.
It includes everything sent to the model:

| Component | Description |
|-----------|-------------|
| **System Prompt** | The agent's instructions and personality definition |
| **Context Provider** | Long-term memory injected automatically (user profile, body metrics, meal history) |
| **Chat History** | All prior user and assistant messages in the current conversation |
| **Output Reserved** | Tokens reserved for the model's response (`max_tokens` setting) |
| **Available** | Remaining capacity before hitting the limit |

When the context window fills up, the model may truncate older messages or the request may fail entirely.
"""
    )

with st.expander("Tips for managing the context window", expanded=True):
    st.markdown(
        """
**1. Start a New Chat when context grows large**
Click "New chat" in the sidebar to clear conversation history. Your long-term memory (profile, metrics, meals)
is preserved in the database — only the short-term chat history is cleared.

**2. Choose the right model for the task**
- **phi-4** has a 16K context window — best for quick, focused queries.  The app automatically
  uses a fresh thread per turn to avoid overflow.
- **gpt-5.2-chat** and **phi-4-mini-instruct** have 128K windows — suitable for longer conversations.

**3. Watch the utilisation percentage**
- 🟩 **< 50%** — Plenty of room.  Chat freely.
- 🟨 **50–80%** — Consider starting a new chat soon.
- 🟥 **> 80%** — Risk of truncation or errors.  Start a new chat.

**4. Reduce max_tokens when you don't need long responses**
Lowering "Max tokens" in the sidebar frees up more room for input context.

**5. Be concise in prompts**
Shorter, focused prompts consume fewer tokens and leave more room for context and output.

**6. Monitor the breakdown chart above**
If "Long-term Memory" is consuming a large share, the fitness memory database may have many
records.  The app limits recent meals and metrics automatically for smaller models.
"""
    )

# ═════════════════════════════════════════════════════════════════════
# Section 5 – Application Tracing, Logging & Errors
# ═════════════════════════════════════════════════════════════════════
st.markdown("### 📋 Application Tracing & Logs")

trace_tab, error_tab, history_tab = st.tabs(["🔍 Turn Trace Log", "❌ Error Log", "📊 Turn History"])

with trace_tab:
    # Show debug logs from all turns, most recent first
    all_debug: list[str] = []
    for idx, t in enumerate(reversed(turns), 1):
        turn_num = len(turns) - idx + 1
        logs = t.get("debug_logs", [])
        if logs:
            all_debug.append(f"── Turn {turn_num} ({t.get('timestamp', '?')[:19]}) ──")
            all_debug.extend(logs)
            all_debug.append("")

    if all_debug:
        log_text = "\n".join(all_debug[-300:])
        st.markdown(
            f'<div style="background:#0d1117;padding:1rem;border-radius:8px;'
            f'max-height:450px;overflow-y:auto;border:1px solid #333;">'
            f'<pre style="white-space:pre-wrap;color:#c9d1d9;margin:0;font-size:0.8rem;">'
            f'{_html.escape(log_text)}</pre></div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("No trace logs captured. Enable **Debug mode** in the chatbot sidebar.")

    # Also show app-level logs
    if app_logs:
        st.markdown("#### Application Logs")
        app_text = "\n".join(app_logs[-200:])
        st.markdown(
            f'<div style="background:#0d1117;padding:1rem;border-radius:8px;'
            f'max-height:300px;overflow-y:auto;border:1px solid #333;">'
            f'<pre style="white-space:pre-wrap;color:#8b949e;margin:0;font-size:0.78rem;">'
            f'{_html.escape(app_text)}</pre></div>',
            unsafe_allow_html=True,
        )

with error_tab:
    # Collect errors from turns and app_logs
    all_errors: list[str] = []
    for t in reversed(turns):
        errs = t.get("errors", [])
        if errs:
            all_errors.append(f"── Turn ({t.get('timestamp', '?')[:19]}) ──")
            all_errors.extend(errs)
    all_errors.extend(errors)

    if all_errors:
        err_text = "\n".join(all_errors[-200:])
        st.markdown(
            f'<div style="background:#1a0000;padding:1rem;border-radius:8px;'
            f'max-height:400px;overflow-y:auto;border:1px solid #660000;">'
            f'<pre style="white-space:pre-wrap;color:#ff6b6b;margin:0;font-size:0.8rem;">'
            f'{_html.escape(err_text)}</pre></div>',
            unsafe_allow_html=True,
        )
    else:
        st.success("No errors recorded. ✅")

with history_tab:
    import pandas as pd

    rows = []
    for idx, t in enumerate(turns, 1):
        status_icon = "✅" if t.get("status") == "ok" else "❌"
        rows.append(
            {
                "#": idx,
                "Time": t.get("timestamp", "")[:19].replace("T", " "),
                "Model": t.get("model", ""),
                "Provider": t.get("provider", ""),
                "Input": t.get("input_tokens", 0),
                "Output": t.get("output_tokens", 0),
                "Total": t.get("total_tokens", 0),
                "Latency (s)": round(t.get("latency_s", 0), 2),
                "Status": status_icon,
                "Messages": t.get("messages_count", 0),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════
# Section 6 – Per-Turn Detail Expanders (last 10)
# ═════════════════════════════════════════════════════════════════════
st.markdown("### 🔬 Recent Turn Details")
for t in reversed(turns[-10:]):
    ts_short = t.get("timestamp", "")[:19].replace("T", " ")
    model_name = t.get("model", "unknown")
    status_icon = "✅" if t.get("status") == "ok" else "❌"
    header = f"{status_icon} {ts_short} — {model_name} — in:{t.get('input_tokens', 0):,} out:{t.get('output_tokens', 0):,}"
    with st.expander(header, expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Input Tokens", f"{t.get('input_tokens', 0):,}")
        c2.metric("Output Tokens", f"{t.get('output_tokens', 0):,}")
        c3.metric("Latency", f"{t.get('latency_s', 0):.2f}s")

        # Mini context bar
        t_max = t.get("context_window_max", 0) or 1
        t_sys = t.get("system_prompt_est_tokens", 0)
        t_ctx = t.get("context_provider_est_tokens", 0)
        t_hist = t.get("chat_history_est_tokens", 0)
        t_out = t.get("output_reserved_tokens", 0)
        t_avail = max(0, t_max - t_sys - t_ctx - t_hist - t_out)
        mini_segments = [
            (t_sys, "#4a9eff"), (t_ctx, "#2ecc71"), (t_hist, "#f39c12"),
            (t_out, "#9b59b6"), (t_avail, "#2a2a2a"),
        ]
        mini_bar = ""
        for val, col in mini_segments:
            pct = val / t_max * 100
            if pct >= 0.3:
                mini_bar += f'<div style="width:{pct:.2f}%;background:{col};height:100%;"></div>'
        st.markdown(
            f'<div style="display:flex;height:16px;border-radius:6px;overflow:hidden;'
            f'border:1px solid #444;margin:0.3rem 0;">{mini_bar}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"System: {t_sys:,} · Context: {t_ctx:,} · History: {t_hist:,} · "
            f"Output reserved: {t_out:,} · Available: {t_avail:,} · Max: {t_max:,}"
        )

        # Debug logs
        dlogs = t.get("debug_logs", [])
        if dlogs:
            st.markdown("**Debug logs:**")
            st.code("\n".join(dlogs), language="text")
