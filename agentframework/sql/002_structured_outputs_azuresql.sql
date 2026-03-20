BEGIN TRANSACTION;

IF OBJECT_ID('dbo.structured_outputs', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.structured_outputs (
        id INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
        steps_json NVARCHAR(MAX) NOT NULL,
        rationale NVARCHAR(MAX) NOT NULL,
        type NVARCHAR(128) NOT NULL,
        created_at DATETIME2(3) NOT NULL,
        CONSTRAINT CK_structured_outputs_steps_json CHECK (ISJSON(steps_json) = 1)
    );
END;

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_structured_outputs_created_at' AND object_id = OBJECT_ID('dbo.structured_outputs'))
BEGIN
    CREATE INDEX IX_structured_outputs_created_at
        ON dbo.structured_outputs(created_at DESC);
END;

COMMIT TRANSACTION;
