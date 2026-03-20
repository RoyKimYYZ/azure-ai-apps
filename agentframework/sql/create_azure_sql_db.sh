#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Create an Azure SQL Database for agentframework.

This script is Entra-friendly for data-plane access, but Azure SQL logical server creation
still requires a temporary SQL admin login/password at control-plane creation time.

Examples:
  # Create a database on an existing logical server
  ./sql/create_azure_sql_db.sh \
    --resource-group my-rg \
    --server my-sql-server \
    --database agentframeworkdb

  # Create the logical server if missing, then create the database
  ./sql/create_azure_sql_db.sh \
    --resource-group my-rg \
    --location eastus2 \
    --server my-sql-server \
    --database agentframeworkdb \
    --create-server \
    --sql-admin-user sqladminagent \
    --sql-admin-password 'ChangeMe123!' \
    --entra-admin-name 'Jane Doe' \
    --entra-admin-object-id '00000000-0000-0000-0000-000000000000'
EOF
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: Required command not found: $1" >&2
    exit 1
  fi
}

RESOURCE_GROUP="${AZURE_SQL_RESOURCE_GROUP:-}"
LOCATION="${AZURE_SQL_LOCATION:-}"
SERVER_NAME="${AZURE_SQL_SERVER_NAME:-${AZURE_SQL_SERVER:-}}"
DATABASE_NAME="${AZURE_SQL_DATABASE:-}"
SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:-}"
SERVICE_OBJECTIVE="${AZURE_SQL_SERVICE_OBJECTIVE:-Basic}"
MAX_SIZE="${AZURE_SQL_MAX_SIZE:-5GB}"
CREATE_SERVER="false"
ALLOW_AZURE_SERVICES="false"
ALLOW_CLIENT_IP="false"
CLIENT_IP="${AZURE_SQL_CLIENT_IP:-}"
SQL_ADMIN_USER="${AZURE_SQL_ADMIN_USER:-}"
SQL_ADMIN_PASSWORD="${AZURE_SQL_ADMIN_PASSWORD:-}"
ENTRA_ADMIN_NAME="${AZURE_SQL_ENTRA_ADMIN_NAME:-}"
ENTRA_ADMIN_OBJECT_ID="${AZURE_SQL_ENTRA_ADMIN_OBJECT_ID:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resource-group)
      RESOURCE_GROUP="$2"
      shift 2
      ;;
    --location)
      LOCATION="$2"
      shift 2
      ;;
    --server)
      SERVER_NAME="$2"
      shift 2
      ;;
    --database)
      DATABASE_NAME="$2"
      shift 2
      ;;
    --subscription)
      SUBSCRIPTION_ID="$2"
      shift 2
      ;;
    --service-objective)
      SERVICE_OBJECTIVE="$2"
      shift 2
      ;;
    --max-size)
      MAX_SIZE="$2"
      shift 2
      ;;
    --create-server)
      CREATE_SERVER="true"
      shift
      ;;
    --allow-azure-services)
      ALLOW_AZURE_SERVICES="true"
      shift
      ;;
    --allow-client-ip)
      ALLOW_CLIENT_IP="true"
      shift
      ;;
    --client-ip)
      CLIENT_IP="$2"
      shift 2
      ;;
    --sql-admin-user)
      SQL_ADMIN_USER="$2"
      shift 2
      ;;
    --sql-admin-password)
      SQL_ADMIN_PASSWORD="$2"
      shift 2
      ;;
    --entra-admin-name)
      ENTRA_ADMIN_NAME="$2"
      shift 2
      ;;
    --entra-admin-object-id)
      ENTRA_ADMIN_OBJECT_ID="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_command az
require_command curl

if [[ -z "$RESOURCE_GROUP" || -z "$SERVER_NAME" || -z "$DATABASE_NAME" ]]; then
  echo "ERROR: --resource-group, --server, and --database are required." >&2
  usage >&2
  exit 1
fi

if [[ -n "$SUBSCRIPTION_ID" ]]; then
  az account set --subscription "$SUBSCRIPTION_ID"
fi

server_exists="false"
if az sql server show --resource-group "$RESOURCE_GROUP" --name "$SERVER_NAME" >/dev/null 2>&1; then
  server_exists="true"
fi

if [[ "$server_exists" != "true" ]]; then
  if [[ "$CREATE_SERVER" != "true" ]]; then
    echo "ERROR: SQL server '$SERVER_NAME' does not exist in resource group '$RESOURCE_GROUP'." >&2
    echo "Re-run with --create-server and provide --location, --sql-admin-user, and --sql-admin-password." >&2
    exit 1
  fi

  if [[ -z "$LOCATION" || -z "$SQL_ADMIN_USER" || -z "$SQL_ADMIN_PASSWORD" ]]; then
    echo "ERROR: --location, --sql-admin-user, and --sql-admin-password are required when --create-server is used." >&2
    exit 1
  fi

  echo "Creating logical SQL server '$SERVER_NAME' in '$RESOURCE_GROUP'..."
  az sql server create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$SERVER_NAME" \
    --location "$LOCATION" \
    --admin-user "$SQL_ADMIN_USER" \
    --admin-password "$SQL_ADMIN_PASSWORD" \
    --output table
fi

if [[ -n "$ENTRA_ADMIN_NAME" && -n "$ENTRA_ADMIN_OBJECT_ID" ]]; then
  echo "Configuring Microsoft Entra administrator '$ENTRA_ADMIN_NAME'..."
  az sql server ad-admin create \
    --resource-group "$RESOURCE_GROUP" \
    --server "$SERVER_NAME" \
    --display-name "$ENTRA_ADMIN_NAME" \
    --object-id "$ENTRA_ADMIN_OBJECT_ID" \
    --output table
else
  echo "Skipping Entra admin configuration. Provide --entra-admin-name and --entra-admin-object-id to configure it."
fi

if [[ "$ALLOW_AZURE_SERVICES" == "true" ]]; then
  echo "Allowing Azure services through the firewall..."
  az sql server firewall-rule create \
    --resource-group "$RESOURCE_GROUP" \
    --server "$SERVER_NAME" \
    --name AllowAzureServices \
    --start-ip-address 0.0.0.0 \
    --end-ip-address 0.0.0.0 \
    --output table
fi

if [[ "$ALLOW_CLIENT_IP" == "true" ]]; then
  if [[ -z "$CLIENT_IP" ]]; then
    CLIENT_IP="$(curl -fsSL https://api.ipify.org)"
  fi
  echo "Allowing current client IP '$CLIENT_IP' through the firewall..."
  az sql server firewall-rule create \
    --resource-group "$RESOURCE_GROUP" \
    --server "$SERVER_NAME" \
    --name AllowCurrentClient \
    --start-ip-address "$CLIENT_IP" \
    --end-ip-address "$CLIENT_IP" \
    --output table
fi

echo "Creating database '$DATABASE_NAME' on server '$SERVER_NAME'..."
az sql db create \
  --resource-group "$RESOURCE_GROUP" \
  --server "$SERVER_NAME" \
  --name "$DATABASE_NAME" \
  --service-objective "$SERVICE_OBJECTIVE" \
  --max-size "$MAX_SIZE" \
  --output table

echo
echo "Done."
echo "Server FQDN: ${SERVER_NAME}.database.windows.net"
echo "Database: ${DATABASE_NAME}"
echo "Next steps:"
echo "  1. Apply Azure SQL schema files under sql/."
echo "  2. Run sqlite-to-Azure SQL migration script with Entra ID auth."
