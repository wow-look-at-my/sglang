#!/usr/bin/env bash
# Builds the policy process with go-toolchain and libgoipc on a CI runner, and exports where both are.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
GO_IPC_REF="${GO_IPC_REF:-typed-service}"
WORK="${RUNNER_TEMP:-/tmp}/go-ipc"
TOOLCHAIN="${RUNNER_TEMP:-/tmp}/go-toolchain"
BIN="$REPO_ROOT/tools/sched-policy/build/sglang-sched-policy"

curl -fsSL --retry 3 -o "$TOOLCHAIN" "https://dl.pazer.build/go-toolchain?os=linux&arch=amd64"
(cd "$REPO_ROOT/tools/sched-policy" && sh "$TOOLCHAIN")
sh "$BIN" --assimilate
"$BIN" --help

rm -rf "$WORK"
git clone --depth 1 --branch "$GO_IPC_REF" --recurse-submodules --shallow-submodules \
	https://github.com/wow-look-at-my/go-ipc.git "$WORK"
make -C "$WORK/c" build/libgoipc.so

echo "SGLANG_SCHED_POLICY_BIN=$BIN" >> "$GITHUB_ENV"
echo "GOIPC_LIBRARY=$WORK/c/build/libgoipc.so" >> "$GITHUB_ENV"
