PRAGMA foreign_keys = ON;

BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    external_user_key TEXT,
    name TEXT NOT NULL,
    birthday_mmddyyyy TEXT,
    height_value REAL,
    height_unit TEXT,
    city TEXT,
    country TEXT,
    sex TEXT,
    timezone TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    CHECK (birthday_mmddyyyy IS NULL OR birthday_mmddyyyy GLOB '[0-1][0-9]/[0-3][0-9]/[1-2][0-9][0-9][0-9]'),
    CHECK (height_value IS NULL OR height_value > 0)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_users_external_user_key
    ON users(external_user_key)
    WHERE external_user_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS body_metric_events (
    event_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    metric_type TEXT NOT NULL,
    value_primary REAL NOT NULL,
    value_secondary REAL,
    unit TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    source TEXT,
    confidence REAL,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    CHECK (metric_type IN ('weight', 'waist', 'blood_pressure')),
    CHECK (value_primary > 0),
    CHECK (value_secondary IS NULL OR value_secondary > 0),
    CHECK (
        (metric_type = 'blood_pressure' AND value_secondary IS NOT NULL) OR
        (metric_type <> 'blood_pressure' AND value_secondary IS NULL)
    ),
    CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    CHECK (
        (metric_type = 'weight' AND unit IN ('lbs', 'kg')) OR
        (metric_type = 'waist' AND unit IN ('in', 'cm')) OR
        (metric_type = 'blood_pressure' AND unit = 'mmHg')
    )
);

CREATE INDEX IF NOT EXISTS ix_body_metric_events_user_observed_at
    ON body_metric_events(user_id, observed_at DESC);

CREATE INDEX IF NOT EXISTS ix_body_metric_events_metric_type
    ON body_metric_events(metric_type);

CREATE TABLE IF NOT EXISTS meal_events (
    meal_event_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    meal_type TEXT,
    source_image_uri TEXT,
    source_hash TEXT,
    calories_kcal REAL,
    protein_g REAL,
    carbs_g REAL,
    fat_g REAL,
    fiber_g REAL,
    sugar_g REAL,
    sodium_mg REAL,
    unit_system TEXT,
    confidence REAL,
    model_name TEXT,
    model_version TEXT,
    prompt_version TEXT,
    llm_structured_output_json TEXT,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    CHECK (meal_type IS NULL OR meal_type IN ('breakfast', 'lunch', 'dinner', 'snack', 'other')),
    CHECK (unit_system IS NULL OR unit_system IN ('metric', 'imperial')),
    CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    CHECK (calories_kcal IS NULL OR calories_kcal >= 0),
    CHECK (protein_g IS NULL OR protein_g >= 0),
    CHECK (carbs_g IS NULL OR carbs_g >= 0),
    CHECK (fat_g IS NULL OR fat_g >= 0),
    CHECK (fiber_g IS NULL OR fiber_g >= 0),
    CHECK (sugar_g IS NULL OR sugar_g >= 0),
    CHECK (sodium_mg IS NULL OR sodium_mg >= 0),
    CHECK (llm_structured_output_json IS NULL OR json_valid(llm_structured_output_json))
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_meal_events_user_source_hash
    ON meal_events(user_id, source_hash)
    WHERE source_hash IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_meal_events_user_occurred_at
    ON meal_events(user_id, occurred_at DESC);

CREATE INDEX IF NOT EXISTS ix_meal_events_meal_type
    ON meal_events(meal_type);

CREATE TABLE IF NOT EXISTS agent_session_memory (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_key TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    session_json TEXT NOT NULL,
    summary_text TEXT,
    last_event_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    CHECK (json_valid(session_json))
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_session_memory_user_session_agent
    ON agent_session_memory(user_id, session_key, agent_name);

CREATE INDEX IF NOT EXISTS ix_agent_session_memory_user_last_event
    ON agent_session_memory(user_id, last_event_at DESC);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    idempotency_key TEXT,
    request_json TEXT,
    response_json TEXT,
    structured_output_json TEXT,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    CHECK (status IN ('started', 'completed', 'failed')),
    CHECK (request_json IS NULL OR json_valid(request_json)),
    CHECK (response_json IS NULL OR json_valid(response_json)),
    CHECK (structured_output_json IS NULL OR json_valid(structured_output_json))
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_ingestion_runs_idempotency
    ON ingestion_runs(user_id, source_type, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_ingestion_runs_user_created
    ON ingestion_runs(user_id, created_at DESC);

COMMIT;
