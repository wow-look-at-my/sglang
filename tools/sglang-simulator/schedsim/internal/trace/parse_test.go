package trace

import (
	"math"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"schedsim/internal/trace/tracetest"
)

// incident is the bare-format incident log every test in this file reads.
var incident = tracetest.DefaultIncident

func mustParse(t *testing.T) []Step {
	t.Helper()
	steps, err := Parse(tracetest.Incident(incident).String())
	require.Nil(t, err)

	return steps
}

func TestParseReadsEverySchedulerStep(t *testing.T) {
	steps := mustParse(t)
	require.NotEqual(t, 0, len(steps))

	for _, s := range steps {
		require.Greater(t, s.Throughput, 0)

		require.False(t, s.Kind == Prefill && s.NewTokens <= 0)

		require.False(t, s.Kind == Decode && s.AcceptLen <= 0)

	}
}

func TestParseKeepsPrefillAndDecodeApart(t *testing.T) {
	steps := mustParse(t)
	var prefills, decodes int
	for _, s := range steps {
		if s.Kind == Prefill {
			prefills++
			continue
		}
		decodes++
		// A decode line never carries prefill-only fields.
		require.False(t, s.NewTokens != 0 || s.Pending != 0)

	}
	require.False(t, prefills == 0 || decodes == 0)

}

// The cold window must be the incident's C1: what its first chunk left
// pending, every one of its full chunks, and the isolation the log shows:
// full chunks, no cache reuse, nothing else on the GPU.
func TestColdPrefillIsTheReportedOne(t *testing.T) {
	steps := mustParse(t)
	m := Summarize(steps, 4096)

	require.GreaterOrEqual(t, m.ColdStart, 0)

	want := incident.C1Len() - incident.Chunk
	require.Equal(t, want, m.PendingAtCold)

	require.Equal(t, incident.C1Chunks, m.ColdChunks)

	require.Equal(t, 0, m.RunningAtCold)

	require.Equal(t, 0, m.DecodeInCold)

	for i := m.ColdStart; i < m.ColdEnd; i++ {
		s := steps[i]
		require.Equal(t, Prefill, s.Kind)

		require.False(t, s.NewTokens != 4096 || s.HitTokens != 0)

	}
}

// The prefill sag is measured, not invented: the steady band is the log's
// second chunk and the bottom its deepest one, each the rate the line printed.
func TestMeasuredSagMatchesTheReportedNumbers(t *testing.T) {
	steps := mustParse(t)
	m := Summarize(steps, 4096)

	want := incident.C1ChunkTPS(1)
	require.LessOrEqual(t, math.Abs(m.ColdSteadyTP-want), 0.01)

	want := incident.C1ChunkTPS(incident.C1Chunks - 1)
	require.LessOrEqual(t, math.Abs(m.ColdMinTP-want), 0.01)

	require.Less(t, m.ColdMinTP, m.ColdSteadyTP)

}

// Queue growth inside the window is the operator's "#queue-req climbs".
func TestQueueGrowsDuringTheColdPrefill(t *testing.T) {
	steps := mustParse(t)
	m := Summarize(steps, 4096)

	require.Greater(t, m.QueuePeakInCold, m.QueueAtCold)

}

// Decode stat lines are emitted every DecodeLogInterval iterations, so the
// absence of decode lines in the window bounds decode there rather than proving
// zero. Pin the arithmetic the report relies on.
func TestDecodeLineIntervalBoundsTheWindow(t *testing.T) {
	steps := mustParse(t)
	m := Summarize(steps, 4096)

	require.False(t, m.DecodeStepSecs <= 0 || m.DecodeStepToks <= 0)

	normal := m.NormalDecodeStepsInColdWindow()
	bound := float64(MaxDecodeStepsInColdWindow())
	require.Greater(t, normal, bound)

	lo := m.GenRateForSteps(MaxDecodeStepsInColdWindow())
	require.Less(t, lo, 8)

	got := m.StepsForGenRate(8)
	require.Greater(t, got, bound)

}

// The collapsed rate reads back as a share of steps: the band the operator
// reported is partial starvation, well under the total starvation the log shows.
func TestCollapseBandReadsBackAsADutyCycle(t *testing.T) {
	steps := mustParse(t)
	m := Summarize(steps, 4096)

	lo := m.DecodeDutyCycleForRate(8)
	hi := m.DecodeDutyCycleForRate(20)
	require.False(t, lo <= 0 || hi <= lo)

	require.Less(t, hi, 1)

	// A conversation getting the pre-collapse share of steps would run at the
	// baseline rate; the band must be far below it.
	require.GreaterOrEqual(t, m.DecodeDutyCycleForRate(m.BaselineGenTP), hi)

}

func TestParseRejectsLogWithoutSteps(t *testing.T) {
	_, err := Parse("nothing here\n")
	require.NotNil(t, err)

}

func TestParseReportsTheOffendingLine(t *testing.T) {
	_, err := Parse("TP0] Prefill batch, #new-seq: 1, #new-token: 4096, #cached-token: 0, full token usage: 0.01, mamba usage: 0.04, #running-req: 0, #queue-req: 0, #pending-token: 8, cuda graph: False, input throughput (token/s): 0.00\n")
	require.NotNil(t, err)

	require.Contains(t, err.Error(), "line 1")

}
