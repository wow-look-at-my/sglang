#!/usr/bin/env bash
# Builds libgoipc on a CI runner and exports where it and the downloaded policy binary are.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
GO_IPC_REF="${GO_IPC_REF:-typed-service}"
WORK="${RUNNER_TEMP:-/tmp}/go-ipc"
BIN="$REPO_ROOT/tools/sched-policy/build/sglang-sched-policy"

chmod +x "$BIN"
rm -rf "$WORK"
git clone --depth 1 --branch "$GO_IPC_REF" --recurse-submodules --shallow-submodules \
	https://github.com/wow-look-at-my/go-ipc.git "$WORK"
make -C "$WORK/c" build/libgoipc.so

echo "SGLANG_SCHED_POLICY_BIN=$BIN" >> "$GITHUB_ENV"
echo "GOIPC_LIBRARY=$WORK/c/build/libgoipc.so" >> "$GITHUB_ENV"
