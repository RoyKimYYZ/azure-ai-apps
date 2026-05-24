-- Prompt Library (SQLite)
-- Stores user-managed prompt templates that can be assigned to one or more agents.
-- Schema is intentionally extensible: new ad-hoc fields land in extra_json (JSON object).

CREATE TABLE IF NOT EXISTS prompt_library (
    prompt_id        TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    title            TEXT NOT NULL,
    body             TEXT NOT NULL,
    description      TEXT,
    -- JSON array of agent display names this prompt is assigned to.
    -- Empty array ('[]') means "available for any agent".
    agent_names_json TEXT NOT NULL DEFAULT '[]',
    tags_json        TEXT NOT NULL DEFAULT '[]',
    -- private  -> visible only to owner
    -- shared   -> reserved for future team sharing (treated as private for v1)
    -- global   -> visible to every authenticated user (admin-only to set)
    visibility       TEXT NOT NULL DEFAULT 'private'
                     CHECK (visibility IN ('private', 'shared', 'global')),
    is_active        INTEGER NOT NULL DEFAULT 1,
    sort_order       INTEGER NOT NULL DEFAULT 0,
    usage_count      INTEGER NOT NULL DEFAULT 0,
    last_used_at     TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    extra_json       TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE INDEX IF NOT EXISTS idx_prompt_library_user        ON prompt_library(user_id);
CREATE INDEX IF NOT EXISTS idx_prompt_library_visibility  ON prompt_library(visibility);
CREATE INDEX IF NOT EXISTS idx_prompt_library_is_active   ON prompt_library(is_active);
