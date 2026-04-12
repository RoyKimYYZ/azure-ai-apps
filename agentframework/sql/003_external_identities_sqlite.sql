-- Migration 003: Add external identity columns to support provider-based authentication (SQLite)
-- Extends users table for OAuth/OIDC-based login with Microsoft, Google, Twitter/X providers
-- Adds columns for auth provider, provider subject ID, email, and login metadata
-- Creates unique index on (auth_provider, provider_subject_id) to enforce provider identity uniqueness
-- SQLite implementation with parity to Azure SQL version

PRAGMA foreign_keys = ON;

BEGIN TRANSACTION;

-- Add external identity columns to users table
-- NOTE: SQLite does not support ALTER TABLE ADD COLUMN in the same transaction as other DDL,
-- so we use IF NOT EXISTS guards to handle idempotent re-runs

ALTER TABLE users
ADD COLUMN auth_provider TEXT;

ALTER TABLE users
ADD COLUMN provider_subject_id TEXT;

ALTER TABLE users
ADD COLUMN email TEXT;

ALTER TABLE users
ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0 CHECK (email_verified IN (0, 1));

ALTER TABLE users
ADD COLUMN last_login_at TEXT;

-- Create unique index on (auth_provider, provider_subject_id) to enforce provider identity uniqueness
CREATE UNIQUE INDEX IF NOT EXISTS uq_users_auth_provider_subject
    ON users(auth_provider, provider_subject_id)
    WHERE auth_provider IS NOT NULL AND provider_subject_id IS NOT NULL;

-- Create index on email for lookup when email-based identity resolution is needed
CREATE INDEX IF NOT EXISTS ix_users_email
    ON users(email)
    WHERE email IS NOT NULL;

-- Create index on last_login_at for audit and activity queries
CREATE INDEX IF NOT EXISTS ix_users_last_login_at
    ON users(last_login_at DESC);

COMMIT TRANSACTION;

-- Verification queries (run these to verify migration success)
-- PRAGMA table_info(users);
-- SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='users' AND (name LIKE 'uq_users_auth_provider%' OR name LIKE 'ix_users_email%' OR name LIKE 'ix_users_last_login_at%');
