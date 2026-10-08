#!/bin/bash
# Runs once on first start of the postgres volume: extra databases + a read-only role for Grafana.
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE mlflow;
    CREATE DATABASE prefect;
    CREATE SCHEMA IF NOT EXISTS app;
    CREATE SCHEMA IF NOT EXISTS raw;
    CREATE ROLE grafana_ro LOGIN PASSWORD '${GRAFANA_RO_PASSWORD}';
    GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO grafana_ro;
    GRANT USAGE ON SCHEMA app TO grafana_ro;
    ALTER DEFAULT PRIVILEGES FOR ROLE ${POSTGRES_USER} IN SCHEMA app GRANT SELECT ON TABLES TO grafana_ro;
EOSQL
