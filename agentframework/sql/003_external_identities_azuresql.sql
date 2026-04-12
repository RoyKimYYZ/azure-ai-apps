-- Migration 003: Add external identity columns to support provider-based authentication
-- Extends dbo.users table for OAuth/OIDC-based login with Microsoft, Google, Twitter/X providers
-- Adds columns for auth provider, provider subject ID, email, and login metadata
-- Creates unique index on (auth_provider, provider_subject_id) to enforce provider identity uniqueness

BEGIN TRANSACTION;

-- Add external identity columns to dbo.users (individually for idempotence/safety)
IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.users') AND name = 'auth_provider')
BEGIN
    ALTER TABLE dbo.users ADD auth_provider NVARCHAR(32) NULL;
    PRINT 'Added column dbo.users.auth_provider';
END;

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.users') AND name = 'provider_subject_id')
BEGIN
    ALTER TABLE dbo.users ADD provider_subject_id NVARCHAR(256) NULL;
    PRINT 'Added column dbo.users.provider_subject_id';
END;

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.users') AND name = 'email')
BEGIN
    ALTER TABLE dbo.users ADD email NVARCHAR(254) NULL;
    PRINT 'Added column dbo.users.email';
END;

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.users') AND name = 'email_verified')
BEGIN
    ALTER TABLE dbo.users ADD email_verified BIT NOT NULL CONSTRAINT DF_users_email_verified DEFAULT (0);
    PRINT 'Added column dbo.users.email_verified';
END;

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.users') AND name = 'last_login_at')
BEGIN
    ALTER TABLE dbo.users ADD last_login_at DATETIME2(3) NULL;
    PRINT 'Added column dbo.users.last_login_at';
END;

-- Create unique index on (auth_provider, provider_subject_id) to enforce provider identity uniqueness
IF EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.users') AND name = 'auth_provider')
   AND EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.users') AND name = 'provider_subject_id')
   AND NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_users_auth_provider_subject' AND object_id = OBJECT_ID('dbo.users'))
BEGIN
    CREATE UNIQUE INDEX UQ_users_auth_provider_subject ON dbo.users(auth_provider, provider_subject_id)
    WHERE auth_provider IS NOT NULL AND provider_subject_id IS NOT NULL;
    
    PRINT 'Created unique index UQ_users_auth_provider_subject on (auth_provider, provider_subject_id)';
END;

-- Create index on email for lookup when email-based identity resolution is needed
IF EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.users') AND name = 'email')
    AND NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_users_email' AND object_id = OBJECT_ID('dbo.users'))
BEGIN
    CREATE INDEX IX_users_email ON dbo.users(email)
    WHERE email IS NOT NULL;
    
    PRINT 'Created index IX_users_email for email lookups';
END;

-- Create index on last_login_at for audit and activity queries
IF EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.users') AND name = 'last_login_at')
    AND NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_users_last_login_at' AND object_id = OBJECT_ID('dbo.users'))
BEGIN
    CREATE INDEX IX_users_last_login_at ON dbo.users(last_login_at DESC);
    
    PRINT 'Created index IX_users_last_login_at for login activity queries';
END;

COMMIT TRANSACTION;

-- Verification queries (run these to verify migration success)
-- SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'users' AND COLUMN_NAME IN ('auth_provider', 'provider_subject_id', 'email', 'email_verified', 'last_login_at');
-- SELECT name FROM sys.indexes WHERE object_id = OBJECT_ID('dbo.users') AND name LIKE 'UQ_users_auth_provider%' OR name LIKE 'IX_users_email%' OR name LIKE 'IX_users_last_login_at%';
