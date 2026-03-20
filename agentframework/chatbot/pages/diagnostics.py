"""Agent Diagnostics page — opens in a dedicated browser tab.

Shows rich context-window visualisation, token-usage charts, management
tips, and application tracing / logging for a selected agent.
"""

from __future__ import annotations

import html as _html
import sys
from pathlib import Path

# Ensure the chatbot package root is importable from the pages/ sub-dir.
_CHATBOT_DIR = str(Path(__file__).resolve().parent.parent)
if _CHATBOT_DIR not in sys.path:
    sys.path.insert(0, _CHATBOT_DIR)

import streamlit as st  # noqa: E402
from diagnostics_store import (  # noqa: E402
    get_agent_diagnostics,
    get_all_agents,
    get_context_window_size,
    clear_agent_diagnostics,
    estimate_tokens,
)

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="Agent Diagnostics", page_icon="📊", layout="wide")

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
        "<script>setTimeout(function(){window.location.reload()},5000);</script>",
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

st.markdown(f"## 📊 {agent_name} — Diagnostics")

if not turns:
    st.info("No completion turns recorded yet for this agent.  Send a message and refresh.")
    st.stop()

latest = turns[-1]

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
