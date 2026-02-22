# SQLite Database Guide (agentframework)

This guide covers how to prepare, manage, verify, and troubleshoot the SQLite database used by the fitness memory schema.

## Files

- `001_fitness_memory_sqlite.sql` — SQLite migration script
- `001_fitness_memory_azuresql.sql` — Azure SQL-compatible version of the same schema
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
