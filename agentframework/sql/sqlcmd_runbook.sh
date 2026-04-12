#!/usr/bin/env bash
# =============================================================================
# sqlcmd_runbook.sh — Azure SQL queries for the agentframework fitness database.
#
# VS Code usage:  Select lines → right-click → "Run Selected Text in Active Terminal"
#   Step 1: Select and run the CONNECTION SETUP block below (once per terminal session)
#   Step 2: Select and run any individual query block you need
#
# Full script:  SQL_PASSWORD='...' bash sql/sqlcmd_runbook.sh
#
# Tables: dbo.users, dbo.body_metric_events, dbo.meal_events,
#         dbo.agent_session_memory, dbo.ingestion_runs, dbo.structured_outputs
# =============================================================================

# =============================================================================
# >>> CONNECTION SETUP — select this 1 line and run first <<<
# =============================================================================
read -s -p "Enter username for Azure SQL (e.g. rkadmin): " SQL_USER
echo
read -s -p "Enter SQL password for $SQL_USER: " SQL_PASSWORD
echo
export SQLCMD="sqlcmd -S rkaks-sqlserver.database.windows.net -d rkaksDB -U $SQL_USER -P $SQL_PASSWORD"

# =============================================================================
# INSTALL sqlcmd (run once)
# =============================================================================
# curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | sudo gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg
# echo "deb [arch=amd64,arm64,armhf signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/ubuntu/22.04/prod jammy main" | sudo tee /etc/apt/sources.list.d/microsoft-prod.list
# sudo apt-get update && sudo apt-get install -y sqlcmd

# =============================================================================
# List all tables
# =============================================================================
$SQLCMD -Q "SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_SCHEMA, TABLE_NAME"

# =============================================================================
# Users
# =============================================================================
$SQLCMD -Q "SELECT user_id, name, birthday_mmddyyyy, height_value, height_unit, city, sex, timezone, FORMAT(created_at, 'yyyy-MM-dd HH:mm') AS created, FORMAT(updated_at, 'yyyy-MM-dd HH:mm') AS updated FROM dbo.users"

# =============================================================================
# Recent body metrics (last 10)
# =============================================================================
$SQLCMD -Q "SELECT TOP 10 event_id, user_id, metric_type, value_primary, value_secondary, unit, FORMAT(observed_at, 'yyyy-MM-dd HH:mm') AS observed, source FROM dbo.body_metric_events ORDER BY observed_at DESC"

# =============================================================================
# Recent meals (last 10)
# =============================================================================
$SQLCMD -Q "SELECT TOP 10 meal_event_id, user_id, meal_type, calories_kcal, protein_g, carbs_g, fat_g, FORMAT(occurred_at, 'yyyy-MM-dd HH:mm') AS occurred, source_type FROM dbo.meal_events ORDER BY occurred_at DESC"

# =============================================================================
# Detected foods for latest meal
# =============================================================================
$SQLCMD -Q "SELECT TOP 1 meal_event_id, meal_type, FORMAT(occurred_at, 'yyyy-MM-dd HH:mm') AS occurred, detected_food_labels_json FROM dbo.meal_events ORDER BY occurred_at DESC"

# =============================================================================
# Recent ingestion runs (last 5)
# =============================================================================
$SQLCMD -Q "SELECT TOP 5 run_id, user_id, source_type, status, FORMAT(created_at, 'yyyy-MM-dd HH:mm') AS created, error FROM dbo.ingestion_runs ORDER BY created_at DESC"

# =============================================================================
# Agent session memory (last 10)
# =============================================================================
$SQLCMD -Q "SELECT session_key, role, LEFT(content, 80) AS content_preview, FORMAT(created_at, 'yyyy-MM-dd HH:mm') AS created FROM dbo.agent_session_memory ORDER BY created_at DESC OFFSET 0 ROWS FETCH NEXT 10 ROWS ONLY"

# =============================================================================
# Row counts per table
# =============================================================================
$SQLCMD -Q "SELECT 'users' AS tbl, COUNT(*) AS cnt FROM dbo.users UNION ALL SELECT 'body_metric_events', COUNT(*) FROM dbo.body_metric_events UNION ALL SELECT 'meal_events', COUNT(*) FROM dbo.meal_events UNION ALL SELECT 'agent_session_memory', COUNT(*) FROM dbo.agent_session_memory UNION ALL SELECT 'ingestion_runs', COUNT(*) FROM dbo.ingestion_runs"

# =============================================================================
# INSERT examples (edit values, then select and run)
# =============================================================================

# --- Insert a new user ---
# $SQLCMD -Q "INSERT INTO dbo.users (user_id, name, birthday_mmddyyyy, height_value, height_unit, city, sex, timezone) VALUES ('u_testuser', 'Test User', '01/15/1990', 175.0, 'cm', 'Toronto', 'male', 'America/Toronto')"

# --- Insert a body metric event ---
# $SQLCMD -Q "INSERT INTO dbo.body_metric_events (event_id, user_id, metric_type, value_primary, unit, observed_at, source) VALUES (NEWID(), 'u_testuser', 'weight', 80.5, 'kg', SYSUTCDATETIME(), 'manual')"

# --- Insert a meal event ---
# $SQLCMD -Q "INSERT INTO dbo.meal_events (meal_event_id, user_id, occurred_at, meal_type, source_type, calories_kcal, protein_g, carbs_g, fat_g) VALUES (NEWID(), 'u_testuser', SYSUTCDATETIME(), 'lunch', 'manual', 650, 40, 60, 25)"

# =============================================================================
# UPDATE examples (edit values, then select and run)
# =============================================================================

# --- Update a user profile ---
# $SQLCMD -Q "UPDATE dbo.users SET city = 'Vancouver', updated_at = SYSUTCDATETIME() WHERE user_id = 'u_testuser'"

# --- Update a body metric ---
# $SQLCMD -Q "UPDATE dbo.body_metric_events SET value_primary = 79.0, notes = 'corrected reading' WHERE event_id = '<paste-event-id-here>'"

# --- Update meal macros ---
# $SQLCMD -Q "UPDATE dbo.meal_events SET calories_kcal = 700, protein_g = 45, carbs_g = 65, fat_g = 28 WHERE meal_event_id = '<paste-meal-event-id-here>'"

# =============================================================================
# DELETE examples (careful!)
# =============================================================================

# --- Delete a test user (cascades to body_metric_events and meal_events) ---
# $SQLCMD -Q "DELETE FROM dbo.users WHERE user_id = 'u_testuser'"
