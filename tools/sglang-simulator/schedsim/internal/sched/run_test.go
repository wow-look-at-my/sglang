package sched

import (
	"strings"
	"testing"

	"schedsim/internal/trace"
	"schedsim/internal/trace/tracetest"
)

func TestWorkloadFromRunPricesTheRunsOwnChunks(t *testing.T) {
	boots, err := trace.ParseBoots(tracetest.Stalled("w", 10).String())
	if err != nil {
		t.Fatal(err)
	}
	b := boots[0]
	m := b.Summarize(4096)
	runs := b.PrefillStretches(4096, 8)
	if len(runs) != 1 {
		t.Fatalf("stretches = %+v", runs)
	}
	w := WorkloadFromRun(b.Steps, runs[0], m)
	if len(w.Chunks) != 10 || w.TotalTokens != 10*4096 || w.ChunkSize != 4096 || w.RunningReqs != 4 {
		t.Errorf("workload = %+v", w)
	}
	if w.DecodeStepSeconds != m.DecodeStepSecs || w.DecodeTokensPerStep != m.DecodeStepToks {
		t.Error("decode figures are not the boot's")
	}
	if !strings.Contains(w.Source, "10 cold 4096-token chunks") {
		t.Errorf("source = %s", w.Source)
	}
	if w.Chunks[0].Seconds != trace.StepSeconds(b.Steps[runs[0].Start]) {
		t.Error("chunk cost is not the step's measured seconds")
	}
	// The single-window builder and the run builder agree on the same run.
	if full := WorkloadFromLog(b.Steps, m, 4096); full.TotalTokens != w.TotalTokens || len(full.Chunks) != len(w.Chunks) {
		t.Errorf("WorkloadFromLog %d/%d vs WorkloadFromRun %d/%d", full.TotalTokens, len(full.Chunks), w.TotalTokens, len(w.Chunks))
	}
	// Nothing running is modelled as one request, as for WorkloadFromLog.
	idle := runs[0]
	idle.RunningAtStart = 0
	if got := WorkloadFromRun(b.Steps, idle, m).RunningReqs; got != 1 {
		t.Errorf("idle run modelled with %d running", got)
	}
}
