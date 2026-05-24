-- Prompt Library (Azure SQL)
-- Stores user-managed prompt templates that can be assigned to one or more agents.
-- Schema is intentionally extensible: new ad-hoc fields land in extra_json (JSON object).

BEGIN TRANSACTION;

IF OBJECT_ID('dbo.prompt_library', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.prompt_library (
        prompt_id         NVARCHAR(64)   NOT NULL PRIMARY KEY,
        user_id           NVARCHAR(64)   NOT NULL,
        title             NVARCHAR(200)  NOT NULL,
        body              NVARCHAR(MAX)  NOT NULL,
        description       NVARCHAR(1000) NULL,
        -- JSON array of agent display names this prompt is assigned to.
        -- Empty array (N'[]') means "available for any agent".
        agent_names_json  NVARCHAR(MAX)  NOT NULL CONSTRAINT DF_prompt_library_agent_names DEFAULT N'[]',
        tags_json         NVARCHAR(MAX)  NOT NULL CONSTRAINT DF_prompt_library_tags        DEFAULT N'[]',
        visibility        NVARCHAR(16)   NOT NULL CONSTRAINT DF_prompt_library_visibility  DEFAULT N'private',
        is_active         BIT            NOT NULL CONSTRAINT DF_prompt_library_is_active   DEFAULT (1),
        sort_order        INT            NOT NULL CONSTRAINT DF_prompt_library_sort_order  DEFAULT (0),
        usage_count       INT            NOT NULL CONSTRAINT DF_prompt_library_usage_count DEFAULT (0),
        last_used_at      DATETIME2(3)   NULL,
        created_at        DATETIME2(3)   NOT NULL CONSTRAINT DF_prompt_library_created_at  DEFAULT SYSUTCDATETIME(),
        updated_at        DATETIME2(3)   NOT NULL CONSTRAINT DF_prompt_library_updated_at  DEFAULT SYSUTCDATETIME(),
        extra_json        NVARCHAR(MAX)  NULL,
        CONSTRAINT FK_prompt_library_users      FOREIGN KEY (user_id) REFERENCES dbo.users(user_id),
        CONSTRAINT CK_prompt_library_visibility CHECK (visibility IN (N'private', N'shared', N'global'))
    );
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_prompt_library_user' AND object_id = OBJECT_ID('dbo.prompt_library'))
BEGIN
    CREATE INDEX IX_prompt_library_user ON dbo.prompt_library(user_id);
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_prompt_library_visibility' AND object_id = OBJECT_ID('dbo.prompt_library'))
BEGIN
    CREATE INDEX IX_prompt_library_visibility ON dbo.prompt_library(visibility);
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_prompt_library_is_active' AND object_id = OBJECT_ID('dbo.prompt_library'))
BEGIN
    CREATE INDEX IX_prompt_library_is_active ON dbo.prompt_library(is_active);
END;

COMMIT TRANSACTION;
