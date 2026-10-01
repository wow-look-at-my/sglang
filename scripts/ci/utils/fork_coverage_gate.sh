#!/usr/bin/env bash
# Statement coverage gate for the fork's scheduler consensus components:
# the cross-rank collectives a TP scheduler deadlocks on when ranks diverge.
set -euo pipefail

cd "$(dirname "$0")/../../.."

uv pip install pytest pytest-cov

tests=test/registered/unit/managers
python3 -m pytest -q \
	"$tests/scheduler_components/test_rank0_consensus.py" \
	"$tests/scheduler_components/test_eviction_throttle.py" \
	"$tests/scheduler_components/test_prefill_decode_balancer.py" \
	"$tests/test_scheduler_eviction_throttle_init.py" \
	--cov=sglang.srt.managers.scheduler_components.rank0_consensus \
	--cov=sglang.srt.managers.scheduler_components.eviction_throttle \
	--cov=sglang.srt.managers.scheduler_components.prefill_decode_balancer \
	--cov-config=scripts/ci/utils/fork_coverage_gate.coveragerc \
	--cov-report=term-missing \
	--cov-fail-under=95
