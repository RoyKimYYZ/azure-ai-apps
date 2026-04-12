# Database Guide (agentframework)

This guide covers both the local SQLite database used today and the Azure SQL artifacts used for schema creation and migration.

## Files

- `001_fitness_memory_sqlite.sql` — SQLite migration script (initial schema)
- `001_fitness_memory_azuresql.sql` — Azure SQL-compatible version of the same schema
- `002_structured_outputs_azuresql.sql` — Azure SQL schema for `structured_outputs`
- `003_external_identities_sqlite.sql` — SQLite migration for OAuth/OIDC identity columns
- `003_external_identities_azuresql.sql` — Azure SQL migration for OAuth/OIDC identity columns
- `create_azure_sql_db.sh` — Azure CLI script to create an Azure SQL database
- `migrate_sqlite_to_azure_sql.py` — SQLite to Azure SQL migration utility
- `../agentframework.db` — local app database
- `../sample-db/agentframework-template.db` — sample pre-created template database

## Prerequisites

- SQLite CLI installed (`sqlite3`)
- Run commands from either:
  - repo root: `/home/rkadmin/azure-ai-apps`
  - app folder: `/home/rkadmin/azure-ai-apps/agentframework`

Check installation:

```bash
sqlite3 --version
```

## Choose the active database backend

Set the durable memory backend in [.env-sample](.env-sample) / `.env` with `FITNESS_DB_BACKEND`.

### SQLite mode

```bash
FITNESS_DB_BACKEND="sqlite"
FITNESS_DB_PATH="agentframework.db"
```

Use this for local development with the project-local SQLite file.

### Azure SQL mode

```bash
FITNESS_DB_BACKEND="azuresql"
AZURE_SQL_SERVER="<logical-server-name>.database.windows.net"
AZURE_SQL_DATABASE="<database-name>"
AZURE_SQL_SCHEMA="dbo"
AZURE_SQL_DRIVER="ODBC Driver 18 for SQL Server"
AZURE_SQL_AUTH_MODE="defaultazurecredential"
AZURE_SQL_ENCRYPT="true"
AZURE_SQL_TRUST_SERVER_CERTIFICATE="false"
AZURE_SQL_CONNECTION_TIMEOUT="30"
```

Use this when the fitness durable memory should connect to Azure SQL instead of SQLite.

## Prepare a Local Database

### Option A: Use existing `agentframework.db`

From repo root:

```bash
sqlite3 agentframework/agentframework.db < agentframework/sql/001_fitness_memory_sqlite.sql
```

From `agentframework/`:

```bash
sqlite3 agentframework.db < sql/001_fitness_memory_sqlite.sql
```

### Option B: Create a new database file

From `agentframework/`:

```bash
sqlite3 my-local.db < sql/001_fitness_memory_sqlite.sql
```

## Manage the Database

Open interactive shell:

```bash
sqlite3 agentframework.db
```

Useful shell commands:

```sql
.tables
.schema users
PRAGMA foreign_keys;
```

Enable foreign keys for a session:

```sql
PRAGMA foreign_keys = ON;
```

## Verify Schema and Indexes

From `agentframework/`:

```bash
sqlite3 agentframework.db ".tables"
sqlite3 agentframework.db ".schema meal_events"
sqlite3 agentframework.db "PRAGMA index_list('meal_events');"
sqlite3 agentframework.db "PRAGMA index_list('body_metric_events');"
```

Expected core tables:

- `users`
- `body_metric_events`
- `meal_events`
- `agent_session_memory`
- `ingestion_runs`

## Quick Data Validation Checks

From `agentframework/`:

```bash
sqlite3 agentframework.db "SELECT name, sql FROM sqlite_master WHERE type='table' AND name='meal_events';"
sqlite3 agentframework.db "SELECT COUNT(*) FROM meal_events;"
sqlite3 agentframework.db "SELECT COUNT(*) FROM users;"
```

JSON validity check example:

```bash
sqlite3 agentframework.db "SELECT meal_event_id FROM meal_events WHERE llm_structured_output_json IS NOT NULL AND json_valid(llm_structured_output_json)=0;"
```

## Backup, Reset, and Recreate

Backup:

```bash
cp agentframework.db agentframework.db.bak
```

Recreate from scratch:

```bash
rm -f agentframework.db
sqlite3 agentframework.db < sql/001_fitness_memory_sqlite.sql
```

Create/regenerate template DB:

```bash
mkdir -p sample-db
rm -f sample-db/agentframework-template.db
sqlite3 sample-db/agentframework-template.db < sql/001_fitness_memory_sqlite.sql
sqlite3 sample-db/agentframework-template.db ".tables"
```

## Azure SQL Prerequisites

- Azure CLI installed and logged in
- Microsoft Entra ID access to the target Azure SQL server/database
- Microsoft ODBC Driver 18 for SQL Server installed on the machine running the migration script
- Python dependencies installed from `pyproject.toml`

Optional environment variables from [../.env-sample](../.env-sample):

- `AZURE_SQL_ADMIN_USER`
- `AZURE_SQL_ADMIN_PASSWORD`
- `AZURE_SQL_SERVER`
- `AZURE_SQL_DATABASE`
- `AZURE_SQL_SCHEMA`
- `AZURE_SQL_DRIVER`
- `AZURE_SQL_RESOURCE_GROUP`
- `AZURE_SQL_LOCATION`
- `AZURE_SQL_ENTRA_ADMIN_NAME`
- `AZURE_SQL_ENTRA_ADMIN_OBJECT_ID`

## Create an Azure SQL Database

From `agentframework/`:

```bash
bash sql/create_azure_sql_db.sh \
  --resource-group <resource-group> \
  --server <logical-server-name> \
  --database <database-name>
```

If the logical server does not exist yet, the same script can create it:

```bash
bash sql/create_azure_sql_db.sh \
  --resource-group <resource-group> \
  --location <azure-region> \
  --server <logical-server-name> \
  --database <database-name> \
  --create-server \
  --sql-admin-user <temporary-sql-admin-user> \
  --sql-admin-password '<temporary-sql-admin-password>' \
  --entra-admin-name '<entra-admin-display-name>' \
  --entra-admin-object-id '<entra-admin-object-id>'
```

## Apply the Azure SQL Schema

Apply schema files in order against the target Azure SQL database:

1. `sql/001_fitness_memory_azuresql.sql` — Initial schema creation
2. `sql/002_structured_outputs_azuresql.sql` — Structured outputs table
3. `sql/003_external_identities_azuresql.sql` — OAuth/OIDC identity columns (required for consumer login)

Using `sqlcmd` CLI (recommended):

```bash
# Apply initial schema
sqlcmd -S <server>.database.windows.net -d <database> -i sql/001_fitness_memory_azuresql.sql

# Apply structured outputs schema
sqlcmd -S <server>.database.windows.net -d <database> -i sql/002_structured_outputs_azuresql.sql

# Apply external identities schema (for OAuth/OIDC)
sqlcmd -S <server>.database.windows.net -d <database> -i sql/003_external_identities_azuresql.sql
```

Or using Azure Data Studio / SSMS GUI:
- Open the SQL file in your SQL editor
- Connect to your server and database
- Execute the entire script (F5)

The schema files create the following tables:

- `users` — User profiles with optional OAuth identity columns
- `body_metric_events` — Weight, waist, blood pressure measurements
- `meal_events` — Meal logs with nutrition analysis
- `agent_session_memory` — Agent conversation state
- `ingestion_runs` — Data ingestion audit trail
- `structured_outputs` — Cached LLM structured outputs

**External Identity Columns** (added by migration 003):

The `dbo.users` table includes these additional columns:
- `auth_provider` — OAuth provider name (e.g., 'microsoft', 'google', 'twitter')
- `provider_subject_id` — Provider's unique user identifier (e.g., OID from Microsoft Entra)
- `email` — User email address
- `email_verified` — BIT flag indicating whether email was verified by provider
- `last_login_at` — Timestamp of most recent login

These columns are optional and support provider-based authentication while maintaining backward compatibility with existing username-based records.

## Apply SQLite Migrations

Apply schema files in order to an existing SQLite database:

1. `sql/001_fitness_memory_sqlite.sql` — Initial schema creation (already applied to most existing databases)
2. `sql/003_external_identities_sqlite.sql` — OAuth/OIDC identity columns (required for consumer login)

From `agentframework/` folder:

```bash
# Apply external identities migration to existing database
sqlite3 agentframework.db < sql/003_external_identities_sqlite.sql
```

Or using interactive shell:

```bash
sqlite3 agentframework.db
sqlite> pragma foreign_keys = on;
sqlite> .read sql/003_external_identities_sqlite.sql
sqlite> .quit
```

Verify the migration applied successfully:

```bash
sqlite3 agentframework.db "PRAGMA table_info(users);" | grep -E 'auth_provider|provider_subject_id|email|email_verified|last_login_at'
sqlite3 agentframework.db "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='users' AND name LIKE 'uq_users_auth_provider%';"
```

Expected output:
- 5 new columns appear in table_info output
- 3 new indexes appear in the index listing (uq_users_auth_provider_subject, ix_users_email, ix_users_last_login_at)

To apply migrations to a fresh database:

```bash
# Create new local database with all migrations
sqlite3 my-fitness.db < sql/001_fitness_memory_sqlite.sql
sqlite3 my-fitness.db < sql/003_external_identities_sqlite.sql
```

## Migrate Data from SQLite to Azure SQL

From `agentframework/`:

```bash
uv run python sql/migrate_sqlite_to_azure_sql.py \
  --sqlite-db agentframework.db \
  --sql-server <logical-server-name> \
  --sql-database <database-name>
```

To perform a full refresh of the target tables before inserting data:

```bash
uv run python sql/migrate_sqlite_to_azure_sql.py \
  --sqlite-db agentframework.db \
  --sql-server <logical-server-name> \
  --sql-database <database-name> \
  --truncate-target
```

Migration order is dependency-safe and includes the shared `structured_outputs` table.

## Troubleshooting

### 1) `sqlite3: command not found`

Install SQLite CLI on your OS, then rerun `sqlite3 --version`.

### 2) `no such table: ...`

Migration was not applied to the DB you are querying.

- Confirm current path with `pwd`
- Re-run migration against the correct file
- Verify with `.tables`

### 3) `database is locked`

Another process has an open write transaction.

- Close other tools/shells using the DB
- Retry after a few seconds
- If needed, restart the process holding the connection

### 4) `CHECK constraint failed`

Input data violates schema constraints.

Common cases:
- `metric_type='blood_pressure'` requires `value_secondary`
- unit mismatch (for example, `weight` must be `lbs` or `kg`)
- invalid `status` in `ingestion_runs`
- invalid JSON in `llm_structured_output_json` or `session_json`

### 5) Foreign key rows not enforced

SQLite enforces FKs per connection.

- Ensure your app/CLI session executes `PRAGMA foreign_keys = ON;`

### 6) Azure SQL migration fails with ODBC driver errors

Install Microsoft ODBC Driver 18 for SQL Server, then re-run the migration utility.

Typical Linux package name:

- `msodbcsql18`

### 7) Azure SQL migration fails with authentication errors

- Confirm `az login` completed with the correct tenant/account
- Confirm the signed-in identity has access to the target database
- Confirm the target SQL server has a Microsoft Entra administrator configured

## Minimal Smoke Test

From `agentframework/`:

```bash
sqlite3 agentframework.db "INSERT INTO users (user_id, name) VALUES ('u_demo','Demo User');"
sqlite3 agentframework.db "INSERT INTO meal_events (meal_event_id,user_id,occurred_at,meal_type,calories_kcal,llm_structured_output_json) VALUES ('m_demo','u_demo',datetime('now'),'lunch',550,'{\"calories\":550}');"
sqlite3 agentframework.db "SELECT user_id,name FROM users WHERE user_id='u_demo';"
sqlite3 agentframework.db "SELECT meal_event_id,calories_kcal FROM meal_events WHERE meal_event_id='m_demo';"
```

Cleanup test rows:

```bash
sqlite3 agentframework.db "DELETE FROM meal_events WHERE meal_event_id='m_demo';"
sqlite3 agentframework.db "DELETE FROM users WHERE user_id='u_demo';"
```
