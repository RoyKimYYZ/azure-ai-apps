BEGIN TRANSACTION;

IF OBJECT_ID('dbo.users', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.users (
        user_id NVARCHAR(64) NOT NULL PRIMARY KEY,
        external_user_key NVARCHAR(128) NULL,
        name NVARCHAR(200) NOT NULL,
        birthday_mmddyyyy CHAR(10) NULL,
        height_value DECIMAL(10, 3) NULL,
        height_unit NVARCHAR(16) NULL,
        city NVARCHAR(120) NULL,
        country NVARCHAR(120) NULL,
        sex NVARCHAR(32) NULL,
        timezone NVARCHAR(100) NULL,
        created_at DATETIME2(3) NOT NULL CONSTRAINT DF_users_created_at DEFAULT SYSUTCDATETIME(),
        updated_at DATETIME2(3) NOT NULL CONSTRAINT DF_users_updated_at DEFAULT SYSUTCDATETIME(),
        is_active BIT NOT NULL CONSTRAINT DF_users_is_active DEFAULT (1),
        CONSTRAINT CK_users_birthday_format CHECK (
            birthday_mmddyyyy IS NULL OR birthday_mmddyyyy LIKE '[0-1][0-9]/[0-3][0-9]/[1-2][0-9][0-9][0-9]'
        ),
        CONSTRAINT CK_users_height_positive CHECK (height_value IS NULL OR height_value > 0)
    );
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_users_external_user_key' AND object_id = OBJECT_ID('dbo.users'))
BEGIN
    CREATE UNIQUE INDEX UQ_users_external_user_key ON dbo.users(external_user_key)
    WHERE external_user_key IS NOT NULL;
END;

IF OBJECT_ID('dbo.body_metric_events', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.body_metric_events (
        event_id NVARCHAR(64) NOT NULL PRIMARY KEY,
        user_id NVARCHAR(64) NOT NULL,
        metric_type NVARCHAR(32) NOT NULL,
        value_primary DECIMAL(12, 4) NOT NULL,
        value_secondary DECIMAL(12, 4) NULL,
        unit NVARCHAR(16) NOT NULL,
        observed_at DATETIME2(3) NOT NULL,
        source NVARCHAR(64) NULL,
        confidence DECIMAL(5, 4) NULL,
        notes NVARCHAR(1000) NULL,
        created_at DATETIME2(3) NOT NULL CONSTRAINT DF_body_metric_events_created_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT FK_body_metric_events_user FOREIGN KEY (user_id) REFERENCES dbo.users(user_id) ON DELETE CASCADE,
        CONSTRAINT CK_body_metric_events_metric_type CHECK (metric_type IN ('weight', 'waist', 'blood_pressure')),
        CONSTRAINT CK_body_metric_events_values CHECK (value_primary > 0 AND (value_secondary IS NULL OR value_secondary > 0)),
        CONSTRAINT CK_body_metric_events_bp_secondary CHECK (
            (metric_type = 'blood_pressure' AND value_secondary IS NOT NULL) OR
            (metric_type <> 'blood_pressure' AND value_secondary IS NULL)
        ),
        CONSTRAINT CK_body_metric_events_conf CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
        CONSTRAINT CK_body_metric_events_units CHECK (
            (metric_type = 'weight' AND unit IN ('lbs', 'kg')) OR
            (metric_type = 'waist' AND unit IN ('in', 'cm')) OR
            (metric_type = 'blood_pressure' AND unit = 'mmHg')
        )
    );
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_body_metric_events_user_observed_at' AND object_id = OBJECT_ID('dbo.body_metric_events'))
BEGIN
    CREATE INDEX IX_body_metric_events_user_observed_at ON dbo.body_metric_events(user_id, observed_at DESC);
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_body_metric_events_metric_type' AND object_id = OBJECT_ID('dbo.body_metric_events'))
BEGIN
    CREATE INDEX IX_body_metric_events_metric_type ON dbo.body_metric_events(metric_type);
END;

IF OBJECT_ID('dbo.meal_events', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.meal_events (
        meal_event_id NVARCHAR(64) NOT NULL PRIMARY KEY,
        user_id NVARCHAR(64) NOT NULL,
        occurred_at DATETIME2(3) NOT NULL,
        meal_type NVARCHAR(16) NULL,
        source_image_uri NVARCHAR(2048) NULL,
        source_hash NVARCHAR(128) NULL,
        calories_kcal DECIMAL(12, 3) NULL,
        protein_g DECIMAL(12, 3) NULL,
        carbs_g DECIMAL(12, 3) NULL,
        fat_g DECIMAL(12, 3) NULL,
        fiber_g DECIMAL(12, 3) NULL,
        sugar_g DECIMAL(12, 3) NULL,
        sodium_mg DECIMAL(12, 3) NULL,
        unit_system NVARCHAR(16) NULL,
        confidence DECIMAL(5, 4) NULL,
        model_name NVARCHAR(128) NULL,
        model_version NVARCHAR(128) NULL,
        prompt_version NVARCHAR(128) NULL,
        llm_structured_output_json NVARCHAR(MAX) NULL,
        notes NVARCHAR(2000) NULL,
        created_at DATETIME2(3) NOT NULL CONSTRAINT DF_meal_events_created_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT FK_meal_events_user FOREIGN KEY (user_id) REFERENCES dbo.users(user_id) ON DELETE CASCADE,
        CONSTRAINT CK_meal_events_meal_type CHECK (meal_type IS NULL OR meal_type IN ('breakfast', 'lunch', 'dinner', 'snack', 'other')),
        CONSTRAINT CK_meal_events_unit_system CHECK (unit_system IS NULL OR unit_system IN ('metric', 'imperial')),
        CONSTRAINT CK_meal_events_conf CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
        CONSTRAINT CK_meal_events_nonnegative CHECK (
            (calories_kcal IS NULL OR calories_kcal >= 0) AND
            (protein_g IS NULL OR protein_g >= 0) AND
            (carbs_g IS NULL OR carbs_g >= 0) AND
            (fat_g IS NULL OR fat_g >= 0) AND
            (fiber_g IS NULL OR fiber_g >= 0) AND
            (sugar_g IS NULL OR sugar_g >= 0) AND
            (sodium_mg IS NULL OR sodium_mg >= 0)
        ),
        CONSTRAINT CK_meal_events_structured_json CHECK (
            llm_structured_output_json IS NULL OR ISJSON(llm_structured_output_json) = 1
        )
    );
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_meal_events_user_source_hash' AND object_id = OBJECT_ID('dbo.meal_events'))
BEGIN
    CREATE UNIQUE INDEX UQ_meal_events_user_source_hash ON dbo.meal_events(user_id, source_hash)
    WHERE source_hash IS NOT NULL;
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_meal_events_user_occurred_at' AND object_id = OBJECT_ID('dbo.meal_events'))
BEGIN
    CREATE INDEX IX_meal_events_user_occurred_at ON dbo.meal_events(user_id, occurred_at DESC);
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_meal_events_meal_type' AND object_id = OBJECT_ID('dbo.meal_events'))
BEGIN
    CREATE INDEX IX_meal_events_meal_type ON dbo.meal_events(meal_type);
END;

IF OBJECT_ID('dbo.agent_session_memory', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.agent_session_memory (
        memory_id NVARCHAR(64) NOT NULL PRIMARY KEY,
        user_id NVARCHAR(64) NOT NULL,
        session_key NVARCHAR(128) NOT NULL,
        agent_name NVARCHAR(128) NOT NULL,
        session_json NVARCHAR(MAX) NOT NULL,
        summary_text NVARCHAR(2000) NULL,
        last_event_at DATETIME2(3) NULL,
        created_at DATETIME2(3) NOT NULL CONSTRAINT DF_agent_session_memory_created_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT FK_agent_session_memory_user FOREIGN KEY (user_id) REFERENCES dbo.users(user_id) ON DELETE CASCADE,
        CONSTRAINT CK_agent_session_memory_json CHECK (ISJSON(session_json) = 1)
    );
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_agent_session_memory_user_session_agent' AND object_id = OBJECT_ID('dbo.agent_session_memory'))
BEGIN
    CREATE UNIQUE INDEX UQ_agent_session_memory_user_session_agent
        ON dbo.agent_session_memory(user_id, session_key, agent_name);
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_agent_session_memory_user_last_event' AND object_id = OBJECT_ID('dbo.agent_session_memory'))
BEGIN
    CREATE INDEX IX_agent_session_memory_user_last_event
        ON dbo.agent_session_memory(user_id, last_event_at DESC);
END;

IF OBJECT_ID('dbo.ingestion_runs', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ingestion_runs (
        run_id NVARCHAR(64) NOT NULL PRIMARY KEY,
        user_id NVARCHAR(64) NOT NULL,
        source_type NVARCHAR(64) NOT NULL,
        idempotency_key NVARCHAR(128) NULL,
        request_json NVARCHAR(MAX) NULL,
        response_json NVARCHAR(MAX) NULL,
        structured_output_json NVARCHAR(MAX) NULL,
        status NVARCHAR(16) NOT NULL,
        error NVARCHAR(2000) NULL,
        created_at DATETIME2(3) NOT NULL CONSTRAINT DF_ingestion_runs_created_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT FK_ingestion_runs_user FOREIGN KEY (user_id) REFERENCES dbo.users(user_id) ON DELETE CASCADE,
        CONSTRAINT CK_ingestion_runs_status CHECK (status IN ('started', 'completed', 'failed')),
        CONSTRAINT CK_ingestion_runs_request_json CHECK (request_json IS NULL OR ISJSON(request_json) = 1),
        CONSTRAINT CK_ingestion_runs_response_json CHECK (response_json IS NULL OR ISJSON(response_json) = 1),
        CONSTRAINT CK_ingestion_runs_structured_json CHECK (structured_output_json IS NULL OR ISJSON(structured_output_json) = 1)
    );
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_ingestion_runs_idempotency' AND object_id = OBJECT_ID('dbo.ingestion_runs'))
BEGIN
    CREATE UNIQUE INDEX UQ_ingestion_runs_idempotency
        ON dbo.ingestion_runs(user_id, source_type, idempotency_key)
        WHERE idempotency_key IS NOT NULL;
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ingestion_runs_user_created' AND object_id = OBJECT_ID('dbo.ingestion_runs'))
BEGIN
    CREATE INDEX IX_ingestion_runs_user_created ON dbo.ingestion_runs(user_id, created_at DESC);
END;

COMMIT TRANSACTION;
