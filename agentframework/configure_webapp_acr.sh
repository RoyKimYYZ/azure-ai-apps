#!/usr/bin/env bash
set -euo pipefail

# Configure Azure Web App for Containers to pull images from ACR
# using managed identity (cross-subscription, passwordless).

# --- Configuration ---
WEBAPP_NAME="aiagentrk"
WEBAPP_RG="app-service-linux"
WEBAPP_SUB=""

ACR_NAME="rkimacr"
ACR_RG="enterprise"
ACR_SUB=""
ACR_LOGIN_SERVER="rkimacr.azurecr.io"

# Set this to your actual image name and tag
IMAGE="${ACR_LOGIN_SERVER}/${1:-agentframework-chatbot:latest}"

# --- 1. Enable system-assigned managed identity on the web app ---
echo "Enabling managed identity on ${WEBAPP_NAME}..."
az webapp identity assign \
  --name "$WEBAPP_NAME" \
  --resource-group "$WEBAPP_RG" \
  --subscription "$WEBAPP_SUB"

# --- 2. Get the principal ID ---
PRINCIPAL_ID=$(az webapp identity show \
  --name "$WEBAPP_NAME" \
  --resource-group "$WEBAPP_RG" \
  --subscription "$WEBAPP_SUB" \
  --query principalId -o tsv)
echo "Principal ID: ${PRINCIPAL_ID}"

# --- 3. Get the ACR resource ID ---
ACR_ID=$(az acr show \
  --name "$ACR_NAME" \
  --resource-group "$ACR_RG" \
  --subscription "$ACR_SUB" \
  --query id -o tsv)
echo "ACR ID: ${ACR_ID}"

# --- 4. Grant AcrPull role to the web app's managed identity ---
echo "Assigning AcrPull role..."
az role assignment create \
  --assignee "$PRINCIPAL_ID" \
  --role AcrPull \
  --scope "$ACR_ID"

# --- 5. Configure the web app to use managed identity for ACR pulls ---
echo "Configuring managed identity ACR pull..."
az webapp config set \
  --name "$WEBAPP_NAME" \
  --resource-group "$WEBAPP_RG" \
  --subscription "$WEBAPP_SUB" \
  --generic-configurations '{"acrUseManagedIdentityCreds": true}'

# --- 6. Set the container image ---
echo "Setting container image to ${IMAGE}..."
az webapp config container set \
  --name "$WEBAPP_NAME" \
  --resource-group "$WEBAPP_RG" \
  --subscription "$WEBAPP_SUB" \
  --registry-server "$ACR_LOGIN_SERVER" \
  --image "$IMAGE"

echo "Done. Web app ${WEBAPP_NAME} configured to pull from ${ACR_LOGIN_SERVER}."
