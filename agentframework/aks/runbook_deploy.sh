#!/usr/bin/env bash
set -euo pipefail # Strict mode so that the script fails on errors

# Runbook: Deploy agentframework chatbot to Kubernetes
# Usage:
#   NAMESPACE=chatbot \
#   KUBE_CONTEXT=my-context \
#   IMAGE=rkimacr.azurecr.io/agentframework:v1 \
#   INGRESS_HOST=agentframework.example.com \
#   ./runbook_deploy.sh

NAMESPACE="default"
# Sets KUBE_CONTEXT to the existing environment value if provided; otherwise defaults to an empty string.
# Syntax: VAR="${VAR:-default}" uses parameter expansion to supply a fallback when VAR is unset or null.
KUBE_CONTEXT="${KUBE_CONTEXT:-}"
IMAGE="${IMAGE:-rkimacr.azurecr.io/agentframework-chatbot:v1}"
INGRESS_HOST="${INGRESS_HOST:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script directory: $SCRIPT_DIR"



# Export INGRESS_IP for further use
INGRESS_IP="${INGRESS_IP:-$(kubectl get svc ingress-nginx-controller -n ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].ip}')}"
if [ -z "$INGRESS_IP" ]; then
  echo "ERROR: INGRESS_IP is empty. Is ingress-nginx installed and has an external IP?" >&2
  kubectl get svc -n ingress-nginx -o wide >&2 || true
  exit 1
fi
export INGRESS_IP
echo "INGRESS_IP=$INGRESS_IP"
INGRESS_HOST=$INGRESS_IP

echo "Current kubectl context:"
kubectl config current-context

echo "Ensure namespace exists: $NAMESPACE"
kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || kubectl create namespace "$NAMESPACE"

if [[ -n "$INGRESS_HOST" ]]; then
  echo "Update ingress host patch to $INGRESS_HOST"
  cat > "$SCRIPT_DIR/ingress-host-patch.yaml" <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: agentframework
spec:
  rules:
    - host: ${INGRESS_HOST}
EOF
fi

if [[ -n "$IMAGE" ]]; then
  echo "Update image to $IMAGE"
  kustomize edit set image "rkimacr.azurecr.io/agentframework-chatbot:v1"
fi

echo "Apply kustomization"
kubectl apply -n "$NAMESPACE" -k "$SCRIPT_DIR"

echo "Wait for rollout"
kubectl rollout status -n "$NAMESPACE" deployment/agentframework-chatbot
kubectl rollout restart -n "$NAMESPACE" deployment/agentframework-chatbot
kubectl get pod -n default -l app=agentframework-chatbot

echo "Show service and ingress"
kubectl get svc -n "$NAMESPACE" agentframework-chatbot
kubectl get ingress -n "$NAMESPACE" agentframework-chatbot

echo "\nDone."

# Configure AKS to pull from ACR
az account set --subscription "ed4bb153-37db-4f9e-99b0-dc0f00cd8be3"
az aks update --name rkaksdev --resource-group aks-solution --attach-acr /subscriptions/de8b8186-01b8-4948-978f-f0dfd6a3655d/resourceGroups/enterprise/providers/Microsoft.ContainerRegistry/registries/rkimacr 

KUBELET_ID=$(az aks show -n rkaksdev -g aks-solution --query identityProfile.kubeletidentity.objectId -o tsv)
ACR_ID=$(az acr show -n rkimacr --query id -o tsv --subscription de8b8186-01b8-4948-978f-f0dfd6a3655d )
az role assignment list --assignee $KUBELET_ID --scope $ACR_ID -o table