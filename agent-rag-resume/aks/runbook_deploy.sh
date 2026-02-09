#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# Runbook: Build, push, and deploy agent-rag-resume to AKS
#
# Usage:
#   ./runbook_deploy.sh              # deploy only (image already in ACR)
#   ./runbook_deploy.sh build        # build + push + deploy
#
# Environment overrides:
#   NAMESPACE=default
#   IMAGE=rkimacr.azurecr.io/agent-rag-resume:v1
#   ACR_NAME=rkimacr
#   AKS_CLUSTER=rkaksdev
#   AKS_RG=aks-solution
# ──────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

NAMESPACE="${NAMESPACE:-default}"
IMAGE="${IMAGE:-rkimacr.azurecr.io/agent-rag-resume:v1}"
ACR_NAME="${ACR_NAME:-rkimacr}"
AKS_CLUSTER="${AKS_CLUSTER:-rkaksdev}"
AKS_RG="${AKS_RG:-aks-solution}"
MODE="${1:-deploy}"

echo "═══════════════════════════════════════════════════"
echo "  agent-rag-resume — AKS Deployment Runbook"
echo "═══════════════════════════════════════════════════"
echo "  Namespace : $NAMESPACE"
echo "  Image     : $IMAGE"
echo "  AKS       : $AKS_CLUSTER ($AKS_RG)"
echo "  ACR       : $ACR_NAME"
echo "  Mode      : $MODE"
echo "═══════════════════════════════════════════════════"

# ── Step 1: Build and push (optional) ─────────────────────────
if [[ "$MODE" == "build" ]]; then
  echo ""
  echo "▶ Logging in to ACR: $ACR_NAME"
  az acr login --name "$ACR_NAME"

  echo "▶ Building Docker image: $IMAGE"
  docker build -t "$IMAGE" -f "$PROJECT_DIR/Dockerfile" "$PROJECT_DIR"

  echo "▶ Pushing image to ACR"
  docker push "$IMAGE"

  echo "✓ Image pushed: $IMAGE"
fi

# ── Step 2: Ensure ACR is attached to AKS ─────────────────────
echo ""
echo "▶ Ensuring ACR is attached to AKS cluster"
az aks update \
  --name "$AKS_CLUSTER" \
  --resource-group "$AKS_RG" \
  --attach-acr "$ACR_NAME" \
  2>/dev/null || echo "  (ACR already attached or cross-sub — skipping)"

# ── Step 3: Detect Ingress external IP ─────────────────────────
echo ""
echo "▶ Detecting Ingress-NGINX external IP"
INGRESS_IP="${INGRESS_IP:-$(kubectl get svc ingress-nginx-controller \
  -n ingress-nginx \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)}"

if [[ -z "$INGRESS_IP" ]]; then
  echo "  ⚠ Could not detect Ingress IP — ingress-nginx may not be installed"
  echo "  Continuing without host-based routing..."
else
  echo "  Ingress IP: $INGRESS_IP"
fi

# ── Step 4: Verify secret.yaml exists ──────────────────────────
if [[ ! -f "$SCRIPT_DIR/secret.yaml" ]]; then
  echo ""
  echo "ERROR: secret.yaml not found in $SCRIPT_DIR"
  echo "  cp secret.yaml-sample secret.yaml"
  echo "  # then fill in your API keys"
  exit 1
fi

# ── Step 5: Ensure namespace exists ────────────────────────────
echo ""
echo "▶ Ensuring namespace: $NAMESPACE"
kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 \
  || kubectl create namespace "$NAMESPACE"

# ── Step 6: Apply Kustomize manifests ──────────────────────────
echo ""
echo "▶ Applying Kustomize manifests"
kubectl apply -n "$NAMESPACE" -k "$SCRIPT_DIR"

# ── Step 7: Wait for rollout ───────────────────────────────────
echo ""
echo "▶ Waiting for rollout to complete"
kubectl rollout status -n "$NAMESPACE" deployment/agent-rag-resume --timeout=120s

# ── Step 8: Show status ───────────────────────────────────────
echo ""
echo "▶ Deployment status"
kubectl get pods -n "$NAMESPACE" -l app=agent-rag-resume
echo ""
kubectl get svc -n "$NAMESPACE" agent-rag-resume
echo ""
kubectl get ingress -n "$NAMESPACE" agent-rag-resume

echo ""
echo "═══════════════════════════════════════════════════"
if [[ -n "${INGRESS_IP:-}" ]]; then
  echo "  ✓ App URL: http://${INGRESS_IP}/agent-rag-resume"
else
  echo "  ✓ Deployed. Use kubectl port-forward to access locally."
fi
echo "═══════════════════════════════════════════════════"
