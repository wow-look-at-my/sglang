#!/bin/sh
# Teach python/sglang/kernels/aot/setup_rocm.py to build for gfx1151 (Strix Halo).
set -e

FILE="${1:?usage: sgl-kernel-gfx1151.sh <path to setup_rocm.py>}"
UTILS="$(dirname "${FILE}")/include/utils.h"

GATE_OLD='if amdgpu_target not in ["gfx942", "gfx950", "gfx1250"]:'
GATE_NEW='if amdgpu_target not in ["gfx942", "gfx950", "gfx1250", "gfx1151"]:'

FLAGS_OLD='    f"-DSGL_TOPK_DYNAMIC_SMEM_BYTES={topk_dynamic_smem_bytes}",'
FLAGS_NEW='    f"-DSGL_TOPK_DYNAMIC_SMEM_BYTES={topk_dynamic_smem_bytes}",
    # gfx1151 is wave32; pin both compiler passes to it (see utils.h below).
    *(["-DSGL_ROCM_WARP_SIZE=32"] if amdgpu_target == "gfx1151" else []),'

WARP_OLD='#if defined(__GFX9__) || !defined(__HIP_DEVICE_COMPILE__)
#define WARP_SIZE 64'
WARP_NEW='#if defined(SGL_ROCM_WARP_SIZE)
#define WARP_SIZE SGL_ROCM_WARP_SIZE
#elif defined(__GFX9__) || !defined(__HIP_DEVICE_COMPILE__)
#define WARP_SIZE 64'

for pattern in "${GATE_OLD}" "${FLAGS_OLD}"; do
    if ! grep -qF "${pattern}" "${FILE}"; then
        echo "ERROR: expected line not found in ${FILE}:" >&2
        echo "  ${pattern}" >&2
        echo "setup_rocm.py changed upstream; re-check this patch before building." >&2
        exit 1
    fi
done

if ! grep -qF "${WARP_OLD}" "${UTILS}"; then
    echo "ERROR: expected WARP_SIZE block not found in ${UTILS}." >&2
    echo "utils.h changed upstream; re-check the wave32 fix before building." >&2
    exit 1
fi

python3 - "${UTILS}" "${WARP_OLD}" "${WARP_NEW}" <<'PY'
import sys

path, old, new = sys.argv[1:4]
with open(path) as f:
    src = f.read()
with open(path, "w") as f:
    f.write(src.replace(old, new, 1))
PY

python3 - "${FILE}" "${GATE_OLD}" "${GATE_NEW}" "${FLAGS_OLD}" "${FLAGS_NEW}" <<'PY'
import sys

path, gate_old, gate_new, flags_old, flags_new = sys.argv[1:6]
with open(path) as f:
    src = f.read()
src = src.replace(gate_old, gate_new)
src = src.replace(flags_old, flags_new, 1)
with open(path, "w") as f:
    f.write(src)
PY

echo "Patched ${FILE} for gfx1151:"
grep -nF -e "${GATE_NEW}" -e "SGL_ROCM_WARP_SIZE" "${FILE}"
echo "Patched ${UTILS} for wave32:"
grep -nF "SGL_ROCM_WARP_SIZE" "${UTILS}"
