#!/usr/bin/env bash
# The scheduler's Go policy runtime on a CI runner, in subcommands.
set -euo pipefail

USAGE="usage: install_sched_policy_runtime.sh build-goipc OUT_DIR | install POLICY_DIR GOIPC_DIR"
CMD="${1:?$USAGE}"

build_goipc() {
	local out="$1"
	local ref="${GO_IPC_REF:-master}"
	local work="${RUNNER_TEMP:-/tmp}/go-ipc"
	rm -rf "$work"
	git clone --depth 1 --branch "$ref" --recurse-submodules --shallow-submodules \
		https://github.com/wow-look-at-my/go-ipc.git "$work"
	make -C "$work/c" "$work/c/build/libgoipc.so"
	mkdir -p "$out"
	cp "$work/c/build/libgoipc.so" "$out/libgoipc.so"
	ls -l "$out"
}

if [ "$CMD" = build-goipc ]; then
	build_goipc "${2:?$USAGE}"
	exit 0
fi
if [ "$CMD" != install ]; then
	echo "$USAGE" >&2
	exit 2
fi

POLICY_DIR="${2:?$USAGE}"
GOIPC_DIR="${3:?$USAGE}"
BIN="$POLICY_DIR/sglang-sched-policy"
LIB="$GOIPC_DIR/libgoipc.so"

if [ ! -f "$BIN" ]; then
	echo "::error::$BIN is missing from the sched-policy hand-off" >&2
	ls -l "$POLICY_DIR" >&2
	exit 1
fi
if [ ! -f "$LIB" ]; then
	echo "::error::$LIB is missing from the libgoipc hand-off" >&2
	ls -l "$GOIPC_DIR" >&2
	exit 1
fi
chmod +x "$BIN"

# The policy process is an APE; the scheduler execs it directly, which needs the binfmt entry.
SCRIPT="${RUNNER_TEMP:-/tmp}/register-ape-binfmt.sh"
curl -fsSL --retry 3 -o "$SCRIPT" \
	https://raw.githubusercontent.com/wow-look-at-my/go-toolchain/master/.github/scripts/register-ape-binfmt.sh
bash "$SCRIPT"
"$BIN" --help

echo "SGLANG_SCHED_POLICY_BIN=$BIN" >> "$GITHUB_ENV"
echo "GOIPC_LIBRARY=$LIB" >> "$GITHUB_ENV"
