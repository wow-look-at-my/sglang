#!/bin/bash
# Remove the per-job uv venv created by ci_install_dependency.sh.

# Best-effort cleanup: never fail the job.
set +e
set -u

# Bound the persistent uv cache (~/.cache/uv, bind-mounted and shared across
# all runner containers).
if command -v uv >/dev/null 2>&1; then
    cache_dir="$(uv cache dir 2>/dev/null || echo "${HOME:-/root}/.cache/uv")"
    [ -d "$cache_dir" ] || cache_dir=/
    used="$(df --output=pcent "$cache_dir" 2>/dev/null | tr -dc '0-9')"
    if [ "${used:-0}" -ge 85 ]; then
        echo "uv cache filesystem at ${used}% — pruning"
        uv cache prune --ci >/dev/null 2>&1 || true
    fi
fi

# Skip entirely when venv mode is disabled — no /tmp/sglang-ci-* dir exists and there's nothing to sweep.
USE_VENV_RAW="${USE_VENV:-true}"
case "$(printf '%s' "$USE_VENV_RAW" | tr '[:upper:]' '[:lower:]')" in
    1 | true | yes) ;;
    *)
        echo "USE_VENV=${USE_VENV_RAW}: skipping venv cleanup"
        exit 0
        ;;
esac

# Prefer the path propagated via GITHUB_ENV. Fallback: glob for any venv from
# this run+job (covers the case where install crashed before exporting the path).
if [ -n "${SGLANG_CI_VENV_PATH:-}" ] && [ -d "$SGLANG_CI_VENV_PATH" ]; then
    if rm -rf "$SGLANG_CI_VENV_PATH"; then
        echo "Cleaned up venv: $SGLANG_CI_VENV_PATH"
    else
        echo "::warning::Failed to remove $SGLANG_CI_VENV_PATH — runner cron should sweep /tmp/sglang-ci-*"
    fi
else
    matched=0
    for venv in /tmp/sglang-ci-${GITHUB_RUN_ID:-unknownrun}-${GITHUB_JOB:-unknownjob}-*; do
        [ -d "$venv" ] || continue
        matched=1
        if rm -rf "$venv"; then
            echo "Cleaned up venv (via glob): $venv"
        else
            echo "::warning::Failed to remove $venv — runner cron should sweep /tmp/sglang-ci-*"
        fi
    done
    [ "$matched" -eq 0 ] && echo "No venv to clean for run=${GITHUB_RUN_ID:-?} job=${GITHUB_JOB:-?}"
fi

# Sweep stale venvs from cancelled/crashed jobs that never reached cleanup.
stale_count=0
for venv in /tmp/sglang-ci-*; do
    [ -d "$venv" ] || continue
    if find "$venv" -maxdepth 0 -mmin +240 -print -quit | grep -q .; then
        rm -rf "$venv" && stale_count=$((stale_count + 1))
    fi
done
[ "$stale_count" -gt 0 ] && echo "Swept $stale_count stale venv(s) older than 4h"

exit 0
