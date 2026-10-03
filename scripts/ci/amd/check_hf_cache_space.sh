#!/bin/bash
# Report HuggingFace cache headroom before a large checkpoint is used, and clear stale download artifacts.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_REPO_ID="${1:?model repo id, e.g. amd/Qwen3.8-2.4T-A95B-Quark-MXFP4}"
REQUIRED_GIB="${2:-0}"

HF_CACHE="${HF_HOME:-/sgl-data/hf-cache}/hub"
# HuggingFace stores `org/name` as `models--org--name`.
MODEL_DIR="$HF_CACHE/models--${MODEL_REPO_ID//\//--}"

avail_gib() {
    df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'
}

report() {
    echo "=== HF cache space ($1) ==="
    df -h "$HF_CACHE" 2>/dev/null || df -h /sgl-data 2>/dev/null || true
    echo "==========================="
}

check_hf_cache_space() {
    if [[ ! -d "$HF_CACHE" ]]; then
        echo "HF cache $HF_CACHE does not exist yet; nothing to report."
        return 0
    fi

    report "before"

    if [[ -d "$MODEL_DIR" ]]; then
        echo "✓ ${MODEL_REPO_ID} is already cached at ${MODEL_DIR};" \
             "no download needed regardless of free space."
    else
        echo "${MODEL_REPO_ID} is NOT cached; it must be downloaded."
    fi

    # Abandoned partial downloads are pure waste and safe to drop.
    python3 "${SCRIPT_DIR}/../utils/cleanup_hf_cache.py" || true

    report "after"

    local avail
    avail=$(avail_gib "$HF_CACHE")
    if [[ -z "$avail" ]]; then
        echo "WARNING: could not read free space from df."
        return 0
    fi
    echo "Free space: ${avail} GiB."

    if [[ -d "$MODEL_DIR" ]] || (( REQUIRED_GIB == 0 )) || (( avail >= REQUIRED_GIB )); then
        return 0
    fi

    echo "=============================================================="
    echo "WARNING: ${MODEL_REPO_ID} is not cached and only ${avail} GiB is"
    echo "         free, against roughly ${REQUIRED_GIB} GiB of weights. The"
    echo "         download will likely fail with ENOSPC partway through."
    echo ""
    echo "         /sgl-data is shared by the whole AMD fleet, so this is a"
    echo "         capacity problem rather than something this job can clear:"
    echo "         deleting other checkpoints to make room just moves the"
    echo "         failure onto whichever job needed them next. Raising it"
    echo "         needs the runner owners."
    echo "=============================================================="
    return 0
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    check_hf_cache_space "$@"
fi
