#!/usr/bin/env bash
# Runs the simulator's tests with coverage and fails if any file of the log
# replay work (the trace layer, the log-derived workload, the report and the
# command) is under 95% statement coverage. The scenario engine and KV pool
# are not held to the bar here; they have their own contract tests.
set -euo pipefail
cd "$(dirname "$0")/.."
go test ./... -count=1 -coverprofile=cover.out -coverpkg=./... "$@"
awk -v min=95 -v FILES="cmd/schedsim/ internal/trace/parse.go internal/trace/boot.go internal/trace/calib_boots.go internal/sched/sched.go internal/sim/logreplay.go internal/report/report.go" \
  -f scripts/filecov.awk cover.out | sort -k3
