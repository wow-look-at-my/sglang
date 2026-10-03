#!/bin/bash
# Install protoc and a Rust toolchain (rustup/cargo).
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$(id -u)" = "0" ]; then
    SUDO=""
elif command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
else
    SUDO=""
fi

${SUDO} bash "${SCRIPT_DIR}/install_protoc.sh"
bash "${SCRIPT_DIR}/install_rustup.sh"
