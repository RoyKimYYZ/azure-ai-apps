#!/usr/bin/env bash
set -euo pipefail

# Build and push the agentframework Streamlit app image to Azure Container Registry.
# This builds chatbot-Dockerfile, which runs chatbot/chatbot.py.

SUBSCRIPTION_ID="${SUBSCRIPTION_ID:-de8b8186-01b8-4948-978f-f0dfd6a3655d}"
ACR_NAME="${ACR_NAME:-rkimacr}"
ACR_LOGIN_SERVER="${ACR_LOGIN_SERVER:-${ACR_NAME}.azurecr.io}"

# Image settings (allow override via env)
IMAGE_NAME="${IMAGE_NAME:-agentframework-chatbot}"
IMAGE_TAG="${IMAGE_TAG:-v1}"
DOCKERFILE="${DOCKERFILE:-chatbot-Dockerfile}"
IMAGE_PULL_POLICY_NOTE="Build context is the agentframework folder and the image entrypoint is chatbot/chatbot.py"

# Optional rollout settings
RESTART_DEPLOYMENT="${RESTART_DEPLOYMENT:-true}"
K8S_NAMESPACE="${K8S_NAMESPACE:-default}"
K8S_DEPLOYMENT_NAME="${K8S_DEPLOYMENT_NAME:-agentframework-chatbot}"


echo "Building ${IMAGE_NAME}:${IMAGE_TAG} from ${DOCKERFILE}"
echo "${IMAGE_PULL_POLICY_NOTE}"

# Authenticate (interactive)
# az login
az account set --subscription "${SUBSCRIPTION_ID}"

# Login to ACR (token-based to avoid credential helper issues on Linux)
ACR_TOKEN="$(az acr login --name "${ACR_NAME}" --expose-token --output tsv --query accessToken)"
DOCKER_CONFIG_DIR="$(mktemp -d)" # Create a temporary directory for Docker config to avoid conflicts with existing credentials
trap 'rm -rf "${DOCKER_CONFIG_DIR}"' EXIT # Clean up temp directory on exit
export DOCKER_CONFIG="${DOCKER_CONFIG_DIR}" # Point Docker to use the temporary config directory
cat > "${DOCKER_CONFIG_DIR}/config.json" <<'JSON'
{
	"auths": {}
}
JSON
echo "${ACR_TOKEN}" | docker login "${ACR_LOGIN_SERVER}" --username 00000000-0000-0000-0000-000000000000 --password-stdin

# Build the image from the current folder
docker build --no-cache -t "${IMAGE_NAME}:${IMAGE_TAG}" -f "${DOCKERFILE}" .

# Tag for ACR
FULL_IMAGE_NAME="${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"
echo "Tagging image as: ${FULL_IMAGE_NAME}"
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${FULL_IMAGE_NAME}"

# Push to ACR
docker push "${FULL_IMAGE_NAME}"
echo "Pushed: ${FULL_IMAGE_NAME}"

# Run a test container locally (optional)
# echo "Running test container locally..."
# docker run --rm -p 8502:8501 --env-file .env "${FULL_IMAGE_NAME}"


if [[ "${RESTART_DEPLOYMENT}" == "true" ]]; then
	echo "Restarting Kubernetes deployment ${K8S_DEPLOYMENT_NAME} in namespace ${K8S_NAMESPACE}"
	kubectl rollout restart -n "${K8S_NAMESPACE}" deployment/"${K8S_DEPLOYMENT_NAME}"
	kubectl rollout status -n "${K8S_NAMESPACE}" deployment/"${K8S_DEPLOYMENT_NAME}"
fi

echo "Done."