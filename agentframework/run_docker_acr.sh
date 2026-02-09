#!/usr/bin/env bash
set -euo pipefail

# Runbook: Build and push Docker image to Azure Container Registry
# Target registry: rkimacr.azurecr.io
# Subscription: de8b8186-01b8-4948-978f-f0dfd6a3655d

SUBSCRIPTION_ID="de8b8186-01b8-4948-978f-f0dfd6a3655d"
ACR_NAME="rkimacr"
ACR_LOGIN_SERVER="${ACR_NAME}.azurecr.io"

# Image settings (allow override via env)
IMAGE_NAME="agentframework-chatbot"
IMAGE_TAG="latest"
# Optional: set a fixed tag by exporting IMAGE_TAG before running
export IMAGE_TAG="v1"

# Authenticate (interactive)
# az login
az account set --subscription "${SUBSCRIPTION_ID}"

# Login to ACR (token-based to avoid credential helper issues on Linux)
ACR_TOKEN="$(az acr login --name "${ACR_NAME}" --expose-token --output tsv --query accessToken)"
DOCKER_CONFIG_DIR="$(mktemp -d)"
trap 'rm -rf "${DOCKER_CONFIG_DIR}"' EXIT
export DOCKER_CONFIG="${DOCKER_CONFIG_DIR}"
cat > "${DOCKER_CONFIG_DIR}/config.json" <<'JSON'
{
	"auths": {}
}
JSON
echo "${ACR_TOKEN}" | docker login "${ACR_LOGIN_SERVER}" --username 00000000-0000-0000-0000-000000000000 --password-stdin

# Build the image (from this folder)
# Force legacy builder to avoid buildx plugin issues
export DOCKER_BUILDKIT=0
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f chatbot-Dockerfile .

# Tag for ACR
FULL_IMAGE_NAME="${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"
echo "Tagging image as: ${FULL_IMAGE_NAME}"
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${FULL_IMAGE_NAME}"

# Push to ACR
docker push "${FULL_IMAGE_NAME}"
echo "Pushed: ${FULL_IMAGE_NAME}"

kubectl rollout restart -n default deployment/agentframework-chatbot
kubectl rollout status -n default deployment/agentframework-chatbot

docker run -p 8501:8501 --env-file chatbot/.env rkimacr.azurecr.io/agentframework-chatbot:v1

# exec into container with image agentframework-chatbot
CONTAINER_ID="$(docker ps --filter "ancestor=${FULL_IMAGE_NAME}" --format '{{.ID}}' | head -n1)"
docker exec -it "${CONTAINER_ID}" /bin/bash

docker container ls
docker logs -f 6d7f1cfa5afe
docker exec -it f0202ec1b535 /bin/bash 

# Manage docker images
docker images
docker run 
docker image prune -f # Remove dangling images

docker rmi ea46bf3500c6 938293b497fd 938293b497fd 


docker container ls -a


# ---------------------------
# Optional: Podman build/push
# ---------------------------
# ACR token login for Podman (daemonless)
podman login "${ACR_LOGIN_SERVER}" --username 00000000-0000-0000-0000-000000000000 --password "${ACR_TOKEN}"

# Build with Podman
podman build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f chatbot-Dockerfile .

# Tag for ACR
FULL_IMAGE_NAME_PODMAN="${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"
podman tag "${IMAGE_NAME}:${IMAGE_TAG}" "${FULL_IMAGE_NAME_PODMAN}"

# Push with Podman
podman push "${FULL_IMAGE_NAME_PODMAN}"

echo "Pushed (podman): ${FULL_IMAGE_NAME_PODMAN}"



## Docker Management

#Remove dangling images: 
docker image prune
#Remove unused images: 
docker image prune -a
#Remove specific images: 
docker rmi c96583b1aa23