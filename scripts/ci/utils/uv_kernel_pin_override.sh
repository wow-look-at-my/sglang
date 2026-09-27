#!/usr/bin/env bash
# Prints uv pip arguments that let an install proceed.
set -euo pipefail

KERNEL_PIN=$(grep -Po -m1 '(?<="sglang-kernel==)[^"]+' python/pyproject.toml)
if curl -fsS --max-time 15 --retry 3 -o /dev/null "https://pypi.org/pypi/sglang-kernel/${KERNEL_PIN}/json"; then
  exit 0
fi
echo "::warning::sglang-kernel==${KERNEL_PIN} is not on PyPI; installing the latest published sglang-kernel" >&2
echo "sglang-kernel" > "${RUNNER_TEMP}/uv-overrides.txt"
echo "--override ${RUNNER_TEMP}/uv-overrides.txt"
