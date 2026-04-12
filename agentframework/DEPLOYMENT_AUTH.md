# Deployment Guide: External Identities & Demo Mode

This guide covers deploying the agentframework chatbot with OAuth/OIDC authentication and demo mode support to Azure App Service and AKS.

## Feature Summary

- **OAuth/OIDC Login**: Support for Microsoft Entra, Google, and Twitter/X providers
- **Demo Mode**: Anonymous access for marketing/onboarding with data isolation
- **Data Isolation**: Strict self-only access control for authenticated users
- **Config-Driven**: All auth behavior controlled via `appconfig.yaml` and environment variables

## Prerequisites

### Database Schema Migrations

Before deploying, ensure the database schema includes the external identity columns.

**For Azure SQL:**

```bash
# Apply the migration
sqlcmd -S <server>.database.windows.net -d <database> -i sql/003_external_identities_azuresql.sql

# Verify
sqlcmd -S <server>.database.windows.net -d <database> -Q "SELECT TOP 3 COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'users' AND COLUMN_NAME IN ('auth_provider', 'provider_subject_id', 'email', 'email_verified', 'last_login_at');"
```

**For SQLite (local dev):**

```bash
# Apply the migration
sqlite3 agentframework.db < sql/003_external_identities_sqlite.sql

# Verify
sqlite3 agentframework.db "PRAGMA table_info(users);" | grep -E 'auth_provider|provider_subject_id|email|email_verified|last_login_at'
```

### OAuth Provider Applications

Register your application with each provider you want to support. You'll need client credentials and callback URLs.

#### Microsoft Entra ID

1. Go to [Azure Portal](https://portal.azure.com) → Microsoft Entra ID → App registrations
2. Create a new app registration
3. Choose `Accounts in any organizational directory and personal Microsoft accounts` if you want friends or external testers to sign in with Microsoft accounts
3. Set `Redirect URI` to:
   - **Local dev**: `http://localhost:8501/auth/callback`
   - **Azure App Service**: `https://<your-app>.azurewebsites.net/auth/callback`
   - **AKS**: Your ingress URL + `/auth/callback`
4. Grant `openid`, `profile`, `email` scopes
5. Copy `Client ID` and generate `Client Secret`
6. Use `https://login.microsoftonline.com/common/v2.0` as the authority URL for multitenant + personal-account testing
7. Set `OAUTH_MICROSOFT_TENANT_ID` only if you later switch back to a tenant-specific authority

#### Google OAuth 2.0

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new OAuth 2.0 web application
3. Add authorized redirect URIs (same as above for different environments)
4. Grant `openid`, `profile`, `email` scopes
5. Copy `Client ID` and `Client Secret`

#### Twitter / X OAuth 2.0

1. Apply for [Twitter Developer Program](https://developer.twitter.com/) access
2. Create a new app or edit existing app
3. Set up OAuth 2.0 with PKCE
4. Add `Redirect URIs` (same pattern as above)
5. Request `users.read` and `tweet.read` scopes
6. Copy `Client ID` and `Client Secret`
7. **Note**: Twitter may not provide email; the implementation gracefully handles this

## Configuration

### Enabling OAuth/OIDC

Edit `agentframework/appconfig.yaml`:

```yaml
external_identities:
  enabled: true  # Enable OAuth provider login
  require_email_verified: false  # Set to true to enforce verified email
  providers:
    - provider_name: microsoft
      client_id_env: OAUTH_MICROSOFT_CLIENT_ID
      client_secret_env: OAUTH_MICROSOFT_CLIENT_SECRET
      authority_url: https://login.microsoftonline.com/common/v2.0
      scope: "openid profile email"
      enabled: true
    - provider_name: google
      client_id_env: OAUTH_GOOGLE_CLIENT_ID
      client_secret_env: OAUTH_GOOGLE_CLIENT_SECRET
      scope: "openid profile email"
      enabled: true
    - provider_name: twitter
      client_id_env: OAUTH_TWITTER_CLIENT_ID
      client_secret_env: OAUTH_TWITTER_CLIENT_SECRET
      scope: "tweet.read users.read"
      enabled: true
```

### Enabling Demo Mode

Edit `agentframework/appconfig.yaml`:

```yaml
demo_mode:
  enabled: true  # Allow unauthenticated access
  demo_user_id: bob  # User ID for demo sessions
  demo_user_name: "Demo User"  # Display name
  writable: true  # Allow demo users to write data
```

**Important**: Demo mode is beneficial for marketing/onboarding but should be disabled in production by default unless you specifically want anonymous access.

### Setting Environment Variables

Copy and customize `chatbot/.env-sample` to `chatbot/.env`:

```bash
cp chatbot/.env-sample chatbot/.env
```

Fill in OAuth provider credentials:

```bash
# Microsoft Entra ID
OAUTH_MICROSOFT_CLIENT_ID="<your-client-id>"
OAUTH_MICROSOFT_CLIENT_SECRET="<your-client-secret>"
OAUTH_MICROSOFT_TENANT_ID=""  # Optional when using /common

# Google
OAUTH_GOOGLE_CLIENT_ID="<your-client-id>"
OAUTH_GOOGLE_CLIENT_SECRET="<your-client-secret>"

# Twitter
OAUTH_TWITTER_CLIENT_ID="<your-client-id>"
OAUTH_TWITTER_CLIENT_SECRET="<your-client-secret>"

# Callback URL (update for your environment)
OAUTH_CALLBACK_HOST="http://localhost:8501"
```

## Deployment Scenarios

### Local Development (SQLite + Demo Mode)

```bash
cd agentframework

# Copy .env template
cp chatbot/.env-sample chatbot/.env

# Edit .env to enable demo mode (or just use defaults)
# Leave OAuth credentials empty; demo mode will work

# Create/migrate SQLite database
sqlite3 agentframework.db < sql/001_fitness_memory_sqlite.sql
sqlite3 agentframework.db < sql/003_external_identities_sqlite.sql

# Run the app
uv run streamlit run chatbot/chatbot.py
```

Navigate to `http://localhost:8501` and click "🎭 Continue as Demo" to test.

### Azure App Service (Azure SQL + OAuth)

**Step 1: Build and push Docker image to ACR**

```bash
# From repo root
az acr build \
  --registry rkimacr \
  --image agentframework-chatbot:v1 \
  --file agentframework/chatbot-Dockerfile \
  agentframework/
```

**Step 2: Configure managed identity and ACR pull**

```bash
bash agentframework/configure_webapp_acr.sh
```

**Step 3: Deploy with environment variables**

```bash
# Update agentframework/chatbot/.env with production settings:
# - FITNESS_DB_BACKEND: azuresql
# - Azure SQL credentials
# - OAuth provider credentials
# - OAUTH_CALLBACK_HOST: https://<your-app>.azurewebsites.net
# - external_identities and demo_mode settings in appconfig.yaml

# Deploy to App Service
bash agentframework/deploy_webapp.sh agentframework-chatbot v1
```

**Step 4: Verify**

```bash
# Check container logs
az webapp log tail \
  --name aiagentrk \
  --resource-group app-service-linux \
  --subscription 2dac1c43-b88c-412f-8b6c-89295fe465de
```

### AKS Deployment (Azure SQL + OAuth)

**Step 1: Build and push Docker image to ACR**

```bash
# Same as App Service step 1
az acr build \
  --registry rkimacr \
  --image agentframework-chatbot:v1 \
  --file agentframework/chatbot-Dockerfile \
  agentframework/
```

**Step 2: Configure ConfigMap and Secret**

Update `agentframework/aks/configmap.yaml`:

```yaml
data:
  EXTERNAL_IDENTITIES_ENABLED: "true"
  DEMO_MODE_ENABLED: "false"  # Or "true" if desired
  OAUTH_CALLBACK_HOST: "https://<your-ingress-hostname>"
  # ... other config variables
```

Update `agentframework/aks/secret.yaml-sample` → `secret.yaml`:

```bash
cp agentframework/aks/secret.yaml-sample agentframework/aks/secret.yaml
```

Fill in OAuth credentials in `secret.yaml`:

```yaml
stringData:
  OAUTH_MICROSOFT_CLIENT_ID: "<client-id>"
  OAUTH_MICROSOFT_CLIENT_SECRET: "<client-secret>"
  # ... other OAuth secrets
```

**Step 3: Apply manifests**

```bash
# Create namespace if needed
kubectl create namespace agentframework --dry-run=client -o yaml | kubectl apply -f -

# Apply ConfigMap
kubectl apply -f agentframework/aks/configmap.yaml -n agentframework

# Apply Secret (DO NOT commit secret.yaml to git)
kubectl apply -f agentframework/aks/secret.yaml -n agentframework

# Apply/update Deployment
kubectl apply -f agentframework/aks/deployment.yaml -n agentframework

# Verify rollout
kubectl rollout status deployment/agentframework-chatbot -n agentframework
```

**Step 4: Set up ingress and configure callback URL**

If using an ingress controller:

```bash
# Get ingress hostname
kubectl get ingress agentframework-chatbot-ingress -n agentframework -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```

Update OAuth provider configurations with the correct callback URL:
- Microsoft Entra: Add redirect URI to app registration
- Google: Add authorized redirect URI
- Twitter: Add redirect URI to app settings

Update `OAUTH_CALLBACK_HOST` in configmap to match your ingress URL.

## Verification Checklist

- [ ] Database schema includes `auth_provider`, `provider_subject_id`, `email`, `email_verified`, `last_login_at` columns
- [ ] OAuth provider apps are registered and callback URLs are configured
- [ ] Environment variables (credentials, callback URL) are set in deployment
- [ ] App starts successfully: check logs for no auth initialization errors
- [ ] Demo mode: anonymous button appears and demo session binds to `bob` user
- [ ] OAuth login: clicking provider button initiates login flow (or shows error if not fully implemented)
- [ ] Data access: authenticated users can only see their own data; cross-user access is denied
- [ ] Database: new login records appear in `dbo.users` with provider identity columns populated

## Troubleshooting

### "Auth provider is not properly configured"

- Check that `external_identities.enabled=true` in `appconfig.yaml`
- Verify OAuth provider credentials are set in environment variables
- Ensure provider name matches configured provider in config

### "Redirect URI mismatch" from OAuth provider

- OAuth callback URL must match exactly (including https:// vs http://)
- Update OAUTH_CALLBACK_HOST in deployment config
- Re-register or update provider app with correct callback URL

### "No authenticated session" error on data operations

- If not using OAuth, enable demo mode in appconfig.yaml
- Ensure user clicks OAuth login or demo button before accessing features
- Check Streamlit session_state for auth session object

### Database migration errors

- Ensure ODBC Driver 18 for SQL Server is installed (for Azure SQL)
- Check connection credentials and database access permissions
- Verify SQL syntax compatibility (migration files support both T-SQL and SQLite)

## OAuth Flow Implementation Status

**Current Status**: The identity infrastructure and config/secrets management are in place. The actual OAuth token exchange workflow is not yet implemented in the Streamlit app.

For Microsoft login aimed at friends or external testers, prefer the `common` authority rather than a tenant-specific endpoint so both Entra users from other tenants and personal Microsoft accounts can sign in.

**Next Steps**:
1. Implement OIDC Discovery and issuer metadata retrieval
2. Use `msal` (Microsoft), `authlib` (Google), or `requests-oauthlib` (Twitter) to handle token exchange
3. Extract claims from ID token and use `IdentityClaimNormalizer` to normalize
4. Call `create_user_with_provider_identity()` or `update_provider_login_metadata()` via repository
5. Set Streamlit session with `set_auth_session()`

See [auth_identity.py](../chatbot/auth_identity.py) and [auth_gate.py](../chatbot/auth_gate.py) for integration points.

## Support

For issues or questions, see the main [README.md](../README.md).
