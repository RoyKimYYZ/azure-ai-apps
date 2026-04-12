"""⚙️ Admin — runtime configuration management.

This Streamlit page reads from and writes to ``appconfig.yaml`` via the
``config`` package.  Changes take effect when the user clicks
**💾 Save & Reload Config**.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

import streamlit as st  # noqa: E402
import yaml  # noqa: E402

from config import get_config, reload_config, save_config  # noqa: E402
from config.models import AppConfig  # noqa: E402
from config.secrets import redact_secret, resolve_env  # noqa: E402

# ── Page config ─────────────────────────────────

cfg = get_config()
st.set_page_config(page_title="Admin", page_icon="⚙️", layout="wide")
st.title("⚙️ Admin — Configuration")

if not cfg.admin.enabled:
    st.warning("Admin page is disabled in appconfig.yaml (`admin.enabled: false`).")
    st.stop()


# ── Helper to build a mutable copy ─────────────

def _mutable_cfg() -> dict:
    """Return the current config as a mutable dict in session state."""
    if "admin_cfg_dict" not in st.session_state:
        reload_config()  # always start from latest YAML on disk
        st.session_state["admin_cfg_dict"] = get_config().model_dump(by_alias=True)
    return st.session_state["admin_cfg_dict"]


def _save_and_reload() -> None:
    """Persist the edited config to disk and reload the singleton."""
    data = _mutable_cfg()
    new_cfg = AppConfig(**data)
    save_config(new_cfg)
    reload_config()
    st.session_state.pop("admin_cfg_dict", None)
    st.success("✅ Configuration saved and reloaded.")


def _reset_to_defaults() -> None:
    """Revert to Pydantic defaults, save, and reload."""
    default_cfg = AppConfig()
    save_config(default_cfg)
    reload_config()
    st.session_state.pop("admin_cfg_dict", None)
    st.success("✅ Configuration reset to defaults and reloaded.")


# ── Tabs ────────────────────────────────────────

tab_app, tab_providers, tab_agents, tab_db, tab_ui, tab_mcp, tab_diag = st.tabs(
    ["⚙️ App Settings", "🤖 AI Providers", "🧠 Agents", "🗄️ Database", "🎨 UI Settings", "🔧 MCP", "🩺 Diagnostics"]
)

d = _mutable_cfg()

# ── Tab 1: App Settings ────────────────────────

with tab_app:
    st.subheader("Application & Runtime")

    col1, col2 = st.columns(2)
    with col1:
        d["app"]["environment"] = st.selectbox(
            "Environment", ["dev", "staging", "prod"],
            index=["dev", "staging", "prod"].index(d["app"].get("environment", "dev")),
        )
        d["logging"]["level"] = st.selectbox(
            "Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"],
            index=["DEBUG", "INFO", "WARNING", "ERROR"].index(d["logging"].get("level", "INFO")),
        )
        d["logging"]["to_console"] = st.toggle("Log to console", value=d["logging"].get("to_console", True))
        d["logging"]["to_file"] = st.toggle("Log to file", value=d["logging"].get("to_file", False))
        d["logging"]["color"] = st.toggle("Color logging", value=d["logging"].get("color", True))

    with col2:
        d["runtime"]["port"] = st.number_input("Port", value=d["runtime"].get("port", 8000), min_value=1, max_value=65535)
        d["runtime"]["workers"] = st.number_input("Workers", value=d["runtime"].get("workers", 1), min_value=1, max_value=32)
        d["runtime"]["timeout"] = st.number_input("Timeout (s)", value=d["runtime"].get("timeout", 60), min_value=1)
        d["runtime"]["stream_tokens"] = st.toggle("Stream tokens", value=d["runtime"].get("stream_tokens", True))

    st.markdown("**Retry Settings**")
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        d["runtime"]["retry"]["rate_limit_base_delay"] = st.number_input(
            "Rate-limit base delay (s)", value=float(d["runtime"]["retry"].get("rate_limit_base_delay", 60)), min_value=1.0,
        )
    with r_col2:
        d["runtime"]["retry"]["rate_limit_max_delay"] = st.number_input(
            "Rate-limit max delay (s)", value=float(d["runtime"]["retry"].get("rate_limit_max_delay", 300)), min_value=1.0,
        )

# ── Tab 2: AI Providers ────────────────────────

with tab_providers:
    st.subheader("AI Providers")
    st.caption("Endpoint values are resolved from environment variables. API keys are redacted.")

    for i, prov in enumerate(d.get("ai", {}).get("providers", [])):
        with st.expander(f"**{prov.get('name', f'Provider {i}')}**", expanded=False):
            st.text_input("Name", value=prov.get("name", ""), key=f"prov_name_{i}", disabled=True)

            endpoint_env = prov.get("endpoint_env", "")
            resolved_ep = resolve_env(endpoint_env, prov.get("default_endpoint", "")) if endpoint_env else prov.get("default_endpoint", "")
            st.text_input("Endpoint (resolved)", value=resolved_ep, key=f"prov_ep_{i}", disabled=True)

            api_key_env = prov.get("api_key_env", "")
            if api_key_env:
                st.text_input("API Key", value=redact_secret(resolve_env(api_key_env)), key=f"prov_key_{i}", disabled=True)

            models_str = ", ".join(prov.get("models", []))
            new_models = st.text_input("Models (comma-separated)", value=models_str, key=f"prov_models_{i}")
            d["ai"]["providers"][i]["models"] = [m.strip() for m in new_models.split(",") if m.strip()]

            d["ai"]["providers"][i]["default_model"] = st.text_input(
                "Default model", value=prov.get("default_model", ""), key=f"prov_defmodel_{i}",
            )

# ── Tab 3: Agents ──────────────────────────────

with tab_agents:
    st.subheader("Agents")

    provider_names = [p.get("name", "") for p in d.get("ai", {}).get("providers", [])]

    for i, agent in enumerate(d.get("ai", {}).get("agents", [])):
        col_toggle, col_name = st.columns([0.15, 0.85])
        with col_toggle:
            is_enabled = d["ai"]["agents"][i].get("enabled", True)
            d["ai"]["agents"][i]["enabled"] = st.checkbox(
                "Enabled", value=is_enabled, key=f"agent_enabled_{i}",
                label_visibility="collapsed",
            )
        with col_name:
            status = "✅" if d["ai"]["agents"][i]["enabled"] else "⛔"
            with st.expander(f"{status} **{agent.get('name', f'Agent {i}')}**", expanded=False):
                st.text_input("Name", value=agent.get("name", ""), key=f"agent_name_{i}", disabled=True)
                d["ai"]["agents"][i]["description"] = st.text_input(
                    "Description", value=agent.get("description", ""), key=f"agent_desc_{i}",
                )
                current_prov = agent.get("provider", "")
                if current_prov in provider_names:
                    prov_idx = provider_names.index(current_prov)
                else:
                    prov_idx = 0
                d["ai"]["agents"][i]["provider"] = st.selectbox(
                    "Default provider", provider_names, index=prov_idx, key=f"agent_prov_{i}",
                )

    st.divider()
    st.subheader("Global AI Defaults")
    defs = d.get("ai", {}).get("defaults", {})
    dc1, dc2 = st.columns(2)
    with dc1:
        defs["temperature"] = st.slider("Default temperature", 0.0, 1.0, float(defs.get("temperature", 0.2)), 0.05, key="admin_def_temp")
        defs["max_tokens"] = st.number_input("Default max tokens", value=int(defs.get("max_tokens", 512)), min_value=1, max_value=16384, key="admin_def_maxt")
        defs["top_p"] = st.slider("Default top P", 0.0, 1.0, float(defs.get("top_p", 1.0)), 0.05, key="admin_def_topp")
    with dc2:
        defs["verify_tls"] = st.toggle("Verify TLS", value=defs.get("verify_tls", True), key="admin_def_tls")
        defs["debug_mode"] = st.toggle("Debug mode", value=defs.get("debug_mode", True), key="admin_def_debug")
        defs["debug_log_max_lines"] = st.number_input("Debug log max lines", value=int(defs.get("debug_log_max_lines", 200)), min_value=10, key="admin_def_loglines")

# ── Tab 4: Database ────────────────────────────

with tab_db:
    st.subheader("Database")

    backend_options = ["sqlite", "azuresql"]
    current_backend = d.get("database", {}).get("default_backend", "sqlite")
    d["database"]["default_backend"] = st.radio(
        "Backend", backend_options,
        index=backend_options.index(current_backend) if current_backend in backend_options else 0,
    )

    if d["database"]["default_backend"] == "sqlite":
        d["database"]["sqlite"]["path"] = st.text_input(
            "SQLite path", value=d["database"].get("sqlite", {}).get("path", "agentframework.db"),
        )
    else:
        st.markdown("**Azure SQL** (secrets resolved from environment)")
        sql_cfg = d.get("database", {}).get("azure_sql", {})
        st.text_input("Server (env)", value=resolve_env(sql_cfg.get("server_env", "")), disabled=True)
        st.text_input("Database (env)", value=resolve_env(sql_cfg.get("database_env", "")), disabled=True)
        sql_cfg["schema"] = st.text_input("Schema", value=sql_cfg.get("schema", "dbo"))
        sql_cfg["connection_timeout"] = st.number_input(
            "Connection timeout (s)", value=int(sql_cfg.get("connection_timeout", 30)), min_value=1,
        )

        if st.button("🔗 Test Connection"):
            try:
                import pyodbc
                server = resolve_env(sql_cfg.get("server_env", ""))
                database = resolve_env(sql_cfg.get("database_env", ""))
                driver = sql_cfg.get("driver", "ODBC Driver 18 for SQL Server")
                encrypt = sql_cfg.get("encrypt", True)
                trust_cert = sql_cfg.get("trust_server_certificate", False)
                auth_mode = sql_cfg.get("auth_mode", "defaultazurecredential").strip().lower()
                conn_str = (
                    f"Driver={{{driver}}};"
                    f"Server=tcp:{server},1433;"
                    f"Database={database};"
                    f"Encrypt={'yes' if encrypt else 'no'};"
                    f"TrustServerCertificate={'yes' if trust_cert else 'no'};"
                    f"Connection Timeout=10;"
                )

                if auth_mode in {"adminpassword", "sqlpassword", "sql-password"}:
                    admin_user = resolve_env(sql_cfg.get("admin_user_env", ""))
                    admin_password = resolve_env(sql_cfg.get("admin_password_env", ""))
                    if not admin_user or not admin_password:
                        st.error("❌ SQL auth mode requires AZURE_SQL_ADMIN_USER and AZURE_SQL_ADMIN_PASSWORD env vars.")
                    else:
                        with pyodbc.connect(f"{conn_str}UID={admin_user};PWD={admin_password};", timeout=10) as conn:
                            conn.execute("SELECT 1")
                        st.success("✅ Connection successful (SQL auth)!")
                else:
                    # DefaultAzureCredential / Entra ID token auth
                    from azure.identity import DefaultAzureCredential
                    client_id = resolve_env(get_config().azure.identity.client_id_env)
                    kwargs = {"managed_identity_client_id": client_id} if client_id else {}
                    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False, **kwargs)
                    token = credential.get_token("https://database.windows.net/.default")
                    token_bytes = token.token.encode("utf-16-le")
                    token_struct = bytes([len(token_bytes) & 0xFF, (len(token_bytes) >> 8) & 0xFF]) + token_bytes
                    SQL_COPT_SS_ACCESS_TOKEN = 1256
                    with pyodbc.connect(conn_str, attrs_before={SQL_COPT_SS_ACCESS_TOKEN: token_struct}, timeout=10) as conn:
                        conn.execute("SELECT 1")
                    st.success("✅ Connection successful (Entra ID)!")
            except Exception as exc:
                st.error(f"❌ Connection failed: {exc}")

# ── Tab 5: UI Settings ─────────────────────────

with tab_ui:
    st.subheader("UI Settings")

    st.markdown("**Theme**")
    ui_theme = d.get("ui", {}).get("theme", {})
    tc1, tc2 = st.columns(2)
    with tc1:
        ui_theme["page_title"] = st.text_input("Page title", value=ui_theme.get("page_title", "AI Foundry Chatbot"))
        ui_theme["page_icon"] = st.text_input("Page icon", value=ui_theme.get("page_icon", "🤖"))
    with tc2:
        ui_theme["layout"] = st.selectbox("Layout", ["wide", "centered"], index=0 if ui_theme.get("layout") == "wide" else 1)

    st.divider()
    st.markdown("**Labels**")
    ui_labels = d.get("ui", {}).get("labels", {})
    label_keys = list(ui_labels.keys())
    lc1, lc2 = st.columns(2)
    for idx, key in enumerate(label_keys):
        col = lc1 if idx % 2 == 0 else lc2
        with col:
            ui_labels[key] = st.text_input(f"Label: {key}", value=ui_labels.get(key, ""), key=f"label_{key}")

    st.divider()
    st.markdown("**Chat**")
    ui_chat = d.get("ui", {}).get("chat", {})
    cc1, cc2 = st.columns(2)
    with cc1:
        ui_chat["welcome_message"] = st.text_input("Welcome message", value=ui_chat.get("welcome_message", ""))
        ui_chat["user_avatar"] = st.text_input("User avatar", value=ui_chat.get("user_avatar", "🧑"))
        ui_chat["assistant_avatar"] = st.text_input("Assistant avatar", value=ui_chat.get("assistant_avatar", "🤖"))
    with cc2:
        ui_chat["max_display_messages"] = st.number_input("Max display messages", value=int(ui_chat.get("max_display_messages", 100)), min_value=1)
        ui_chat["message_truncate_length"] = st.number_input("Message truncate length", value=int(ui_chat.get("message_truncate_length", 120)), min_value=10)

    st.divider()
    st.markdown("**Fitness**")
    ui_fitness = d.get("ui", {}).get("fitness", {})
    fc1, fc2 = st.columns(2)
    with fc1:
        ui_fitness["recent_meals_count"] = st.number_input("Recent meals count", value=int(ui_fitness.get("recent_meals_count", 6)), min_value=1, max_value=50)
        ui_fitness["recent_messages_count"] = st.number_input("Recent messages count", value=int(ui_fitness.get("recent_messages_count", 6)), min_value=1, max_value=50)
    with fc2:
        ui_fitness["message_truncate_length"] = st.number_input("Fitness msg truncate", value=int(ui_fitness.get("message_truncate_length", 120)), min_value=10)
        ui_fitness["default_food_prompt"] = st.text_input("Default food prompt", value=ui_fitness.get("default_food_prompt", ""))

# ── Tab 6: MCP ──────────────────────────────────

with tab_mcp:
    st.subheader("MCP — GitHub API Settings")
    gh = d.get("mcp", {}).get("github", {})
    mc1, mc2 = st.columns(2)
    with mc1:
        gh["max_list_results"] = st.number_input("Max list results", value=int(gh.get("max_list_results", 30)), min_value=1, key="mcp_list")
        gh["max_search_results"] = st.number_input("Max search results", value=int(gh.get("max_search_results", 10)), min_value=1, key="mcp_search")
    with mc2:
        gh["max_file_bytes"] = st.number_input("Max file bytes", value=int(gh.get("max_file_bytes", 102400)), min_value=1024, key="mcp_bytes")
        gh["max_return_lines"] = st.number_input("Max return lines", value=int(gh.get("max_return_lines", 500)), min_value=10, key="mcp_lines")

# ── Tab 7: Diagnostics ─────────────────────────

with tab_diag:
    st.subheader("Configuration Diagnostics")

    config_path = Path(__file__).resolve().parents[2] / "appconfig.yaml"
    st.markdown(f"**Config file:** `{config_path}`")
    if config_path.exists():
        import datetime as _dt
        mtime = config_path.stat().st_mtime
        st.markdown(f"**Last modified:** {_dt.datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M:%S}")

    st.divider()
    st.markdown("**Current effective config** (secrets redacted)")
    display_cfg = get_config().model_dump(by_alias=True)
    # Redact *_env resolved values — show only env var names, not their values
    st.json(display_cfg)

    st.divider()
    col_dl, col_reset = st.columns(2)
    with col_dl:
        yaml_export = yaml.dump(display_cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)
        st.download_button("📥 Export config as YAML", data=yaml_export, file_name="appconfig-export.yaml", mime="text/yaml")
    with col_reset:
        st.markdown(
            '<a href="/diagnostics" target="_blank" style="display:inline-flex;align-items:center;gap:0.35rem;'
            'font-size:0.85rem;color:#4a9eff;text-decoration:none;">📊 Open Performance Diagnostics</a>',
            unsafe_allow_html=True,
        )

# ── Footer: Save & Reload / Reset ──────────────

st.divider()
footer_left, footer_right = st.columns([1, 1])
with footer_left:
    if st.button("💾 Save & Reload Config", type="primary", use_container_width=True):
        _save_and_reload()
with footer_right:
    if st.button("↩️ Reset to Defaults", use_container_width=True):
        _reset_to_defaults()
