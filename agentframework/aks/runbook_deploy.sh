#!/usr/bin/env bash
set -euo pipefail # Strict mode so that the script fails on errors

# Runbook: Deploy agentframework chatbot to Kubernetes
# Usage:
#   NAMESPACE=chatbot \
#   KUBE_CONTEXT=my-context \
#   IMAGE=rkimacr.azurecr.io/agentframework:v1 \
#   INGRESS_HOST=agentframework.example.com \
#   ./runbook_deploy.sh

NAMESPACE="${NAMESPACE:-default}"
# Sets KUBE_CONTEXT to the existing environment value if provided; otherwise defaults to an empty string.
# Syntax: VAR="${VAR:-default}" uses parameter expansion to supply a fallback when VAR is unset or null.
KUBE_CONTEXT="${KUBE_CONTEXT:-}"
IMAGE="${IMAGE:-rkimacr.azurecr.io/agentframework-chatbot:v1}"
INGRESS_HOST="${INGRESS_HOST:-ai.roykim.ca}"

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
echo "INGRESS_HOST=$INGRESS_HOST"

if ! kubectl get crd clusterissuers.cert-manager.io >/dev/null 2>&1; then
  echo "WARNING: cert-manager CRDs are not installed. Install cert-manager before applying TLS resources." >&2
fi

echo "Current kubectl context:"
kubectl config current-context

echo "Ensure namespace exists: $NAMESPACE"
kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || kubectl create namespace "$NAMESPACE"

cd "$SCRIPT_DIR"

if [[ -n "$INGRESS_HOST" ]]; then
  echo "Update ingress host patch to $INGRESS_HOST"
  cat > "$SCRIPT_DIR/ingress-host-patch.yaml" <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: agentframework-chatbot
spec:
  tls:
    - hosts:
        - ${INGRESS_HOST}
      secretName: agentframework-chatbot-tls
  rules:
    - host: ${INGRESS_HOST}
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: agentframework-chatbot
                port:
                  number: 80
EOF
fi

if [[ -n "$IMAGE" ]]; then
  echo "Update image to $IMAGE"
  kustomize edit set image "rkimacr.azurecr.io/agentframework-chatbot:v1"
fi

kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.17.1/cert-manager.yaml

if kubectl get crd clusterissuers.cert-manager.io >/dev/null 2>&1; then
  echo "Apply kustomization with cert-manager resources"
  kubectl apply -n "$NAMESPACE" -k "$SCRIPT_DIR"
else
  echo "Apply app resources without cert-manager ClusterIssuer"
  kubectl apply -n "$NAMESPACE" -f "$SCRIPT_DIR/configmap.yaml"
  kubectl apply -n "$NAMESPACE" -f "$SCRIPT_DIR/secret.yaml"
  kubectl apply -n "$NAMESPACE" -f "$SCRIPT_DIR/service.yaml"
  kubectl apply -n "$NAMESPACE" -f "$SCRIPT_DIR/deployment.yaml"
  kubectl apply -n "$NAMESPACE" -f "$SCRIPT_DIR/hpa.yaml"
  kubectl apply -n "$NAMESPACE" -f "$SCRIPT_DIR/ingress.yaml"
  kubectl apply -n "$NAMESPACE" -f "$SCRIPT_DIR/ingress-host-patch.yaml"
  echo "Install cert-manager, then rerun this script to create [ClusterIssuer] and the TLS secret." >&2
fi

echo "Wait for rollout"
kubectl rollout status -n "$NAMESPACE" deployment/agentframework-chatbot
kubectl rollout restart -n "$NAMESPACE" deployment/agentframework-chatbot
kubectl get pod -n "$NAMESPACE" -l app=agentframework-chatbot

echo "Show service and ingress"
kubectl get svc -n "$NAMESPACE" agentframework-chatbot
kubectl get ingress -n "$NAMESPACE" agentframework-chatbot

echo "\nDone."

# Configure AKS to pull from ACR
az account set --subscription "ed4bb153-37db-4f9e-99b0-dc0f00cd8be3"
az aks update --name rkaksdev --resource-group aks-solution --attach-acr /subscriptions/de8b8186-01b8-4948-978f-f0dfd6a3655d/resourceGroups/enterprise/providers/Microsoft.ContainerRegistry/registries/rkimacr 

KUBELET_ID=$(az aks show -n rkaksdev -g aks-solution --query identityProfile.kubeletidentity.objectId -o tsv)
echo "KUBELET_ID: $KUBELET_ID"
ACR_ID=$(az acr show -n rkimacr --query id -o tsv --subscription de8b8186-01b8-4948-978f-f0dfd6a3655d )
echo "ACR_ID: $ACR_ID"
az role assignment list --assignee $KUBELET_ID --scope $ACR_ID -o table