#!/usr/bin/env bash
# Builds the policy process with go-toolchain and libgoipc on a CI runner, and exports where both are.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
GO_IPC_REF="${GO_IPC_REF:-typed-service}"
WORK="${RUNNER_TEMP:-/tmp}/go-ipc"
TOOLCHAIN_DIR="${RUNNER_TEMP:-/tmp}/go-toolchain-bin"
BIN="$REPO_ROOT/tools/sched-policy/build/sglang-sched-policy"

mkdir -p "$TOOLCHAIN_DIR"
curl -fsSL --retry 3 -o "$TOOLCHAIN_DIR/register-ape-binfmt.sh" \
	https://raw.githubusercontent.com/wow-look-at-my/go-toolchain/master/.github/scripts/register-ape-binfmt.sh
bash "$TOOLCHAIN_DIR/register-ape-binfmt.sh"
curl -fsSL --retry 3 -o "$TOOLCHAIN_DIR/go-toolchain" "https://dl.pazer.build/go-toolchain?os=linux&arch=amd64"
chmod +x "$TOOLCHAIN_DIR/go-toolchain"
export PATH="$TOOLCHAIN_DIR:$PATH"
(cd "$REPO_ROOT/tools/sched-policy" && go-toolchain)
"$BIN" --help

rm -rf "$WORK"
git clone --depth 1 --branch "$GO_IPC_REF" --recurse-submodules --shallow-submodules \
	https://github.com/wow-look-at-my/go-ipc.git "$WORK"
make -C "$WORK/c" build/libgoipc.so

echo "SGLANG_SCHED_POLICY_BIN=$BIN" >> "$GITHUB_ENV"
echo "GOIPC_LIBRARY=$WORK/c/build/libgoipc.so" >> "$GITHUB_ENV"
