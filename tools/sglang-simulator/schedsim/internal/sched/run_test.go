package sched

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"schedsim/internal/trace"
	"schedsim/internal/trace/tracetest"
)

func TestWorkloadFromRunPricesTheRunsOwnChunks(t *testing.T) {
	boots, err := trace.ParseBoots(tracetest.Stalled("w", 10).String())
	require.Nil(t, err)

	b := boots[0]
	m := b.Summarize(4096)
	runs := b.PrefillStretches(4096, 8)
	require.Equal(t, 1, len(runs))

	w := WorkloadFromRun(b.Steps, runs[0], m)
	assert.False(t, len(w.Chunks) != 10 || w.TotalTokens != 10*4096 || w.ChunkSize != 4096 || w.RunningReqs != 4)

	assert.False(t, w.DecodeStepSeconds != m.DecodeStepSecs || w.DecodeTokensPerStep != m.DecodeStepToks)

	assert.Contains(t, w.Source, "10 cold 4096-token chunks")

	assert.Equal(t, trace.StepSeconds(b.Steps[runs[0].Start]), w.Chunks[0].Seconds)

	// The single-window builder and the run builder agree on the same run.
	full := WorkloadFromLog(b.Steps, m, 4096)
	assert.False(t, full.TotalTokens != w.TotalTokens || len(full.Chunks) != len(w.Chunks))

	// Nothing running is modelled as one request, as for WorkloadFromLog.
	idle := runs[0]
	idle.RunningAtStart = 0
	got := WorkloadFromRun(b.Steps, idle, m).RunningReqs
	assert.Equal(t, 1, got)

}
