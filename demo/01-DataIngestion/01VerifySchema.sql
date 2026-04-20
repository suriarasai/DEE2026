
-- Verify the schema exists under the workspace catalog

SHOW SCHEMAS IN workspace;

-- Verify the volume exists under the rebu schema

SHOW VOLUMES IN workspace.rebu;



-- ============================================================
-- ALTERNATIVE: Create schema and volume via SQL
-- (Use this if you prefer SQL over the UI, or for automation)
-- ============================================================

-- Create schema if not exists
--CREATE SCHEMA IF NOT EXISTS workspace.rebu;

-- Create managed volume
--CREATE VOLUME IF NOT EXISTS workspace.rebu.raw;

-- Confirm creation
--SELECT 'Schema and volume created successfully' AS status;
