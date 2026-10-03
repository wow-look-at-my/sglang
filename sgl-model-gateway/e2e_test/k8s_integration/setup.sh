#!/usr/bin/env bash
# Setup script for K8s integration tests.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLUSTER_NAME="smg-test"
NAMESPACE="smg-test"
CONTEXT="kind-${CLUSTER_NAME}"
MANIFESTS_DIR="${SCRIPT_DIR}/manifests"

log() { echo "==> $*"; }

teardown() {
    log "Tearing down..."
    if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
        kind delete cluster --name "$CLUSTER_NAME"
    else
        log "Cluster '${CLUSTER_NAME}' not found, nothing to tear down."
    fi
    log "Done."
}

if [[ "${1:-}" == "teardown" ]]; then
    teardown
    exit 0
fi

if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
    log "Kind cluster '${CLUSTER_NAME}' already exists"
else
    log "Creating kind cluster '${CLUSTER_NAME}'..."
    kind create cluster --name "$CLUSTER_NAME"
fi

kubectl config use-context "$CONTEXT"

cd "$REPO_ROOT"
if [[ "${SKIP_DOCKER_BUILD:-}" == "1" ]]; then
    log "SKIP_DOCKER_BUILD=1 — skipping docker build, expecting smg-gateway:test to exist"
    if ! docker image inspect smg-gateway:test >/dev/null 2>&1; then
        log "ERROR: smg-gateway:test not found locally; cannot continue"
        exit 1
    fi
else
    log "Building gateway Docker image (this may take 5-10 minutes on first run)..."
    docker build -f e2e_test/k8s_integration/Dockerfile.gateway -t smg-gateway:test .
fi

log "Loading smg-gateway:test image into kind..."
kind load docker-image smg-gateway:test --name "$CLUSTER_NAME"

log "Ensuring python:3.12-slim is available in kind..."
if ! docker image inspect python:3.12-slim >/dev/null 2>&1; then
    log "Pulling python:3.12-slim..."
    docker pull python:3.12-slim
fi
if ! kind load docker-image python:3.12-slim --name "$CLUSTER_NAME" 2>/dev/null; then
    log "kind load failed (multi-arch image), pulling inside kind node..."
    docker exec "${CLUSTER_NAME}-control-plane" crictl pull docker.io/library/python:3.12-slim
fi

log "Applying namespace and RBAC..."
kubectl --context "$CONTEXT" apply -f "${MANIFESTS_DIR}/namespace.yaml"
kubectl --context "$CONTEXT" apply -f "${MANIFESTS_DIR}/rbac.yaml"

log "Creating fake-worker ConfigMap..."
kubectl --context "$CONTEXT" -n "$NAMESPACE" create configmap fake-worker-script \
    --from-file="fake_worker.py=${SCRIPT_DIR}/fake_worker.py" \
    --dry-run=client -o yaml | kubectl --context "$CONTEXT" apply -f -

log "Deploying SMG gateway..."
kubectl --context "$CONTEXT" apply -f "${MANIFESTS_DIR}/gateway.yaml"

log "Waiting for gateway to be ready..."
kubectl --context "$CONTEXT" -n "$NAMESPACE" rollout status deployment/smg-gateway --timeout=180s

log ""
log "Setup complete! Run the integration tests with:"
log "  pytest e2e_test/k8s_integration/ -v -s"
log ""
log "To tear down:"
log "  ./e2e_test/k8s_integration/setup.sh teardown"
