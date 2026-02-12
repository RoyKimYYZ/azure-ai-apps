#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Runbook: Build, push, and deploy rag-chatapp-retail to AKS
#
# Run each section below one at a time (copy-paste into terminal).
# ──────────────────────────────────────────────────────────────

# ── Variables — edit these to match your environment ───────────
export NAMESPACE="default"
export IMAGE="rkimacr.azurecr.io/rag-chatapp-retail:v1"
export ACR_NAME="rkimacr"
export AKS_CLUSTER="rkaksdev"
export AKS_RG="aks-solution"

# ── Step 1: Login to Azure ─────────────────────────────────────
az login
az acr login --name $ACR_NAME

# ── Step 2: Build Docker image ─────────────────────────────────
cd rag-chatapp-retail
docker build -t $IMAGE -f Dockerfile .

# ── Step 3: Push image to ACR ──────────────────────────────────
docker push $IMAGE

# ── Step 4: Get AKS credentials ───────────────────────────────
az aks get-credentials \
  --name $AKS_CLUSTER \
  --resource-group $AKS_RG \
  --overwrite-existing

# ── Step 5: Attach ACR to AKS ─────────────────────────────────
az aks update \
  --name $AKS_CLUSTER \
  --resource-group $AKS_RG \
  --attach-acr $ACR_NAME

# ── Step 6: Create secret.yaml from sample ─────────────────────
# cp aks/secret.yaml-sample aks/secret.yaml
# Then edit aks/secret.yaml and fill in:
#   AIPROJECT_CONNECTION_STRING  — from Azure AI Foundry project
#   AZURE_OPENAI_KEY            — from Azure OpenAI resource

# ── Step 7: Create namespace (if needed) ───────────────────────
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# ── Step 8: Apply Kustomize manifests ──────────────────────────
kubectl apply -n $NAMESPACE -k aks/

# ── Step 9: Wait for rollout ──────────────────────────────────
kubectl rollout status -n $NAMESPACE deployment/rag-chatapp-retail --timeout=120s

# ── Step 10: Check status ─────────────────────────────────────
kubectl get pods -n $NAMESPACE -l app=rag-chatapp-retail
kubectl get svc -n $NAMESPACE rag-chatapp-retail
kubectl get ingress -n $NAMESPACE rag-chatapp-retail

# ── Step 11: Get Ingress IP ───────────────────────────────────
kubectl get svc ingress-nginx-controller -n ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

# ── Step 12: Test the app ─────────────────────────────────────
# curl http://<INGRESS_IP>/rag-chatapp-retail/health
# curl -X POST http://<INGRESS_IP>/rag-chatapp-retail/api/chat \
#   -H "Content-Type: application/json" \
#   -d '{"messages":[{"role":"user","content":"I need a tent for 4 people"}]}'

# ── Port-forward alternative (no ingress needed) ──────────────
# kubectl port-forward -n $NAMESPACE svc/rag-chatapp-retail 8000:80
# curl http://localhost:8000/health

