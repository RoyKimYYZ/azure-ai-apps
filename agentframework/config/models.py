"""Pydantic v2 models for every section of appconfig.yaml."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ── App metadata ────────────────────────────────


class AppMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = "AgentFramework Chatbot"
    version: str = "0.1.0"
    environment: str = "dev"


# ── Runtime / server ────────────────────────────


class RetryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    rate_limit_base_delay: float = 60.0
    rate_limit_max_delay: float = 300.0


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    timeout: int = 60
    retry: RetryConfig = Field(default_factory=RetryConfig)
    stream_tokens: bool = True


# ── Logging ─────────────────────────────────────


class LoggingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    level: str = "INFO"
    to_console: bool = True
    to_file: bool = False
    file_path: str = "logs/agentframework.log"
    color: bool = True


# ── Observability ───────────────────────────────


class ObservabilityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    service_name: str = "agentframework"
    appinsights_connection_string_env: str = "APPINSIGHTS_CONNECTION_STRING"
    otlp_endpoint_env: str = "OTEL_EXPORTER_OTLP_ENDPOINT"


# ── Database ────────────────────────────────────


class SqliteConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str = "agentframework.db"


class AzureSqlConfig(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    server_env: str = "AZURE_SQL_SERVER"
    database_env: str = "AZURE_SQL_DATABASE"
    schema_name: str = Field(default="dbo", validation_alias="schema")
    driver: str = "ODBC Driver 18 for SQL Server"
    auth_mode: str = "defaultazurecredential"
    admin_user_env: str = "AZURE_SQL_ADMIN_USER"
    admin_password_env: str = "AZURE_SQL_ADMIN_PASSWORD"
    encrypt: bool = True
    trust_server_certificate: bool = False
    connection_timeout: int = 30


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_backend: str = "sqlite"
    sqlite: SqliteConfig = Field(default_factory=SqliteConfig)
    azure_sql: AzureSqlConfig = Field(default_factory=lambda: AzureSqlConfig())


# ── Azure services ──────────────────────────────


class AzureIdentityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    client_id_env: str = "AZURE_CLIENT_ID"
    tenant_id_env: str = "AZURE_TENANT_ID"
    client_secret_env: str = "AZURE_CLIENT_SECRET"


class AzureOpenAIConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    endpoint_env: str = "AZURE_OPENAI_ENDPOINT"
    api_key_env: str = "AZURE_OPENAI_API_KEY"
    chat_deployment: str = "gpt-5.2-chat"
    embedding_deployment: str = "text-embedding-ada-002"
    responses_deployment: str = "gpt-5.2-responses"
    api_version_env: str = "AZURE_OPENAI_API_VERSION"


class AzureConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    identity: AzureIdentityConfig = Field(default_factory=AzureIdentityConfig)
    openai: AzureOpenAIConfig = Field(default_factory=AzureOpenAIConfig)


# ── AI: providers, agents, defaults ─────────────


class ProviderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    endpoint_env: str = ""
    api_key_env: str = ""
    models: list[str] = Field(default_factory=list)
    model_env: str = ""
    default_model: str = ""
    default_endpoint: str = ""
    index_name_env: str = ""
    default_index_name: str = ""
    embedding_model: str = ""
    request_model: str = ""
    chat_completions_models: list[str] = Field(default_factory=list)


class AgentAvailableProvider(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    models: list[str] = Field(default_factory=list)


class AgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    enabled: bool = True
    provider: str = ""
    model: str = ""
    description: str = ""
    prompt_file: str = ""
    available_providers: list[AgentAvailableProvider] = Field(default_factory=list)
    extra_models: list[str] = Field(default_factory=list)
    model_backends: dict[str, str] = Field(default_factory=dict)


class AIDefaultsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    temperature: float = 0.2
    temperature_min: float = 0.0
    temperature_max: float = 1.0
    temperature_step: float = 0.05
    top_p: float = 1.0
    top_p_min: float = 0.0
    top_p_max: float = 1.0
    top_p_step: float = 0.05
    max_tokens: int = 512
    max_tokens_min: int = 1
    max_tokens_max: int = 4096
    verify_tls: bool = True
    debug_mode: bool = True
    debug_log_max_lines: int = 200
    extraction_temperature: float = 1.0


class AIConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    providers: list[ProviderConfig] = Field(default_factory=list)
    agents: list[AgentConfig] = Field(default_factory=list)
    defaults: AIDefaultsConfig = Field(default_factory=AIDefaultsConfig)


# ── UI settings ─────────────────────────────────


class UIThemeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    page_title: str = "AI Foundry Chatbot"
    page_icon: str = "🤖"
    layout: str = "wide"
    sidebar_state: str = "expanded"


class UILabelsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_selector: str = "Agent"
    provider_selector: str = "AI Provider"
    model_selector: str = "Model"
    temperature: str = "Temperature"
    max_tokens: str = "Max tokens"
    top_p: str = "Top P"
    verify_tls: str = "Verify TLS"
    debug_mode: str = "Debug mode"
    new_chat: str = "🗑️ New chat"
    user_name: str = "User name"
    refresh_memory: str = "Refresh memory snapshot"
    food_upload: str = "Upload food image (optional)"
    user_profile: str = "User Profile"
    prompt_library: str = "Prompt library"
    system_prompt: str = "System prompt"
    template_editor: str = "Template editor"
    diagnostics_link: str = "📊 Open Diagnostics"
    diagnostics_title: str = "Agent Diagnostics"
    diagnostics_icon: str = "📊"


class UISidebarConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    show_completion_metrics: bool = True
    metrics_max_entries: int = 50
    show_diagnostics_link: bool = True
    agent_dropdown_visible_agents: list[str] = Field(
        default_factory=list,
        description="List of agent names to show in the dropdown. Empty list means show all enabled agents."
    )


class UIChatConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    welcome_message: str = "Hello! How can I help you today?"
    user_avatar: str = "🧑"
    assistant_avatar: str = "🤖"
    max_display_messages: int = 100
    message_truncate_length: int = 120
    error_truncate_length: int = 140


class UIFitnessConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    recent_meals_count: int = 6
    recent_messages_count: int = 6
    message_truncate_length: int = 120
    accepted_image_types: list[str] = Field(
        default_factory=lambda: ["png", "jpg", "jpeg", "webp", "bmp"]
    )
    default_food_prompt: str = "What are the macronutrients in this meal?"
    default_user_id: str = "default-user"


class UIPerformanceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_events: int = 120
    auto_refresh_interval_ms: int = 5000


class UIConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    theme: UIThemeConfig = Field(default_factory=UIThemeConfig)
    labels: UILabelsConfig = Field(default_factory=UILabelsConfig)
    sidebar: UISidebarConfig = Field(default_factory=UISidebarConfig)
    chat: UIChatConfig = Field(default_factory=UIChatConfig)
    fitness: UIFitnessConfig = Field(default_factory=UIFitnessConfig)
    performance: UIPerformanceConfig = Field(default_factory=UIPerformanceConfig)


# ── MCP ─────────────────────────────────────────


class MCPGithubConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    token_env: str = "GITHUB_TOKEN"
    alt_token_env: str = "GH_PAT"
    api_base: str = "https://api.github.com"
    max_list_results: int = 30
    max_search_results: int = 10
    max_file_bytes: int = 102400
    max_return_lines: int = 500


class MCPConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    github: MCPGithubConfig = Field(default_factory=MCPGithubConfig)


# ── External Identities / OAuth ──────────────────


class OAuthProviderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_name: str  # 'microsoft', 'google', 'twitter'
    client_id_env: str  # Environment variable name for client ID
    client_secret_env: str  # Environment variable name for client secret
    authority_url: str = ""  # For Microsoft Entra ID
    scope: str = "openid profile email"  # Default OIDC scopes
    enabled: bool = True


class ExternalIdentitiesConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False  # Feature toggle for OAuth/OIDC
    require_email_verified: bool = False  # Whether to enforce email_verified=true
    providers: list[OAuthProviderConfig] = Field(default_factory=list)


class DemoModeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False  # Feature toggle for anonymous demo
    demo_user_id: str = "demo-user"  # User ID to bind demo sessions to
    demo_user_name: str = "Demo User"  # Display name for demo user
    writable: bool = True  # Whether demo mode allows writes


# ── Admin ───────────────────────────────────────


class AdminConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True


class TemporaryAccessConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    allow_anonymous_admin_page: bool = False
    allow_anonymous_diagnostics_page: bool = False


class UserPermissionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    email: str
    admin: bool = False
    diagnostics: bool = False
    allowed_agents: list[str] = Field(default_factory=list)


class RbacConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    super_admin_emails: list[str] = Field(default_factory=list)
    default_authenticated_agents: list[str] = Field(default_factory=lambda: ["Fitness Nutrition"])
    user_permissions: list[UserPermissionConfig] = Field(default_factory=list)


# ── Root config model ───────────────────────────


class AppConfig(BaseModel):
    """Root configuration model.  Composed from all section models."""

    app: AppMeta = Field(default_factory=AppMeta)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    azure: AzureConfig = Field(default_factory=AzureConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    external_identities: ExternalIdentitiesConfig = Field(default_factory=ExternalIdentitiesConfig)
    demo_mode: DemoModeConfig = Field(default_factory=DemoModeConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)
    temporary_access: TemporaryAccessConfig = Field(default_factory=TemporaryAccessConfig)
    rbac: RbacConfig = Field(default_factory=RbacConfig)
