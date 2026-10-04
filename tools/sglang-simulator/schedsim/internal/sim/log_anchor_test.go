package sim

import (
	"bytes"
	"math"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"schedsim/internal/trace"
)

// The engine is calibrated on the embedded production log, not on the scenarios it is
// run on: the cost coefficients come from `internal/trace/calib.go`. These tests pin
// the behaviours the log states as fact and check the fitted model against the log
// lines it was solved on.

// The composition depends on the page rounding, the token budget, the cede's
// half-chunk rule and which prefix the pool reports for the follow-up, so it fails
// if any of those drift.
func TestReplayReproducesTheLoggedBatch(t *testing.T) {
	sc := ScenarioA(testEpisode())
	cost := NewCost(calib(t))
	res := Run(sc, DefaultConfig(ModeOld, cost), sc.Seeds[0])

	var found *PrefillLogLine
	for i, pl := range res.PrefillLog {
		if pl.NewSeq == 3 {
			found = &res.PrefillLog[i]
			break
		}
	}
	require.NotNil(t, found)

	assert.Equal(t, 4000, found.Tokens)

	assert.Equal(t, 248320, found.Hit)

	var got []int
	for _, it := range found.Items {
		got = append(got, it.Extend)
	}
	sort.Ints(got)
	want := []int{263, 921, 2816}
	require.Equal(t, len(want), len(got))

	for i := range want {
		require.Equal(t, want[i], got[i])

	}
}

// TestOldRunsNoDecodeDuringTheColdPrefills is the log's own shape. Upstream
// prefill-priority launches a prefill batch whenever one can be formed, so the replay
// of OLD has to show the same silence.
func TestOldRunsNoDecodeDuringTheColdPrefills(t *testing.T) {
	steps := mustSteps(t)
	start := -1
	for i, s := range steps {
		if s.Kind == trace.Prefill && s.Pending+s.NewTokens > 400_000 {
			start = i
			break
		}
	}
	require.GreaterOrEqual(t, start, 0)

	chunks := 0
	for _, s := range steps[start:] {
		require.NotEqual(t, trace.Decode, s.Kind)

		if s.NewTokens != BaselineChunkSize {
			require.Equal(t, 105, chunks)

			require.False(t, s.NewSeq != 3 || s.NewTokens != 4000 || s.HitTokens != 248320)

			break
		}
		chunks++
	}

	sc := ScenarioA(testEpisode())
	cost := NewCost(calib(t))
	res := Run(sc, DefaultConfig(ModeOld, cost), sc.Seeds[0])
	c1 := FindTag(res, "C1")
	require.NotNil(t, c1)

	n := 0
	for _, b := range res.Batches {
		if !b.IsPrefill && b.Start < c1.FirstTok {
			n++
		}
	}
	assert.Equal(t, 0, n)

}

// TestDecodeRateMatchesTheLog checks the decode model against the lines FitDecode
// solved: a decode line whose predecessor is also a decode line, on one request. Its
// reported gen throughput covers a whole interval of decode steps, so accepted tokens
// over throughput is a step time the model has to reproduce. Lines that follow a
// prefill are excluded because their throughput window contains the prefill; they are
// reported, not fitted, and the gap between both populations is the reason.
func TestDecodeRateMatchesTheLog(t *testing.T) {
	steps := mustSteps(t)
	d := calib(t).Decode
	var steadyRel, otherRel []float64
	for i, s := range steps {
		if s.Kind != trace.Decode || s.Throughput <= 0 || s.RunningReq <= 0 {
			continue
		}
		implied := math.Max(s.AcceptLen, 1) / s.Throughput
		model := d.StepSeconds(s.RunningReq, s.FullTokens) / float64(s.RunningReq)
		rel := math.Abs(model - implied)
		rel /= implied
		if i > 0 && steps[i-1].Kind == trace.Decode && s.RunningReq == 1 {
			steadyRel = append(steadyRel, rel)
			continue
		}
		otherRel = append(otherRel, rel)
	}
	require.Equal(t, d.Samples, len(steadyRel))

	sort.Float64s(steadyRel)
	sort.Float64s(otherRel)
	mean := 0.0
	for _, v := range steadyRel {
		mean += v
	}
	mean /= float64(len(steadyRel))
	t.Logf("steady single-request decode lines: n=%d median %.2f%% mean %.2f%% max %.2f%%",
		len(steadyRel), 100*steadyRel[len(steadyRel)/2], 100*mean, 100*steadyRel[len(steadyRel)-1])
	t.Logf("lines whose throughput window holds a prefill: n=%d median %.2f%% max %.2f%% (excluded by FitDecode)",
		len(otherRel), 100*otherRel[len(otherRel)/2], 100*otherRel[len(otherRel)-1])
	assert.LessOrEqual(t, steadyRel[len(steadyRel)-1], 0.05)

	assert.LessOrEqual(t, mean, 0.01)

}

// TestRunIsDeterministic pins the entry point's promise: the same scenario, policy and
// seed produce byte-identical tables, and a parallel suite reports what a serial one
// does. Both orderings dispatch the same runs through different goroutines, so this
// also covers the suite's own aggregation.
func TestRunIsDeterministic(t *testing.T) {
	cost := NewCost(calib(t))
	scs := []Scenario{ScenarioA(testEpisode()), ScenarioB(5), ScenarioThrash(1.5, 600)}
	var first []byte
	for _, workers := range []int{1, 4} {
		var buf bytes.Buffer
		WriteTables(&buf, RunSuite(scs, cost, DefaultConfig, workers))
		if first == nil {
			first = buf.Bytes()
			continue
		}
		assert.True(t, bytes.Equal(first, buf.Bytes()))

	}
}

func calib(t *testing.T) trace.Calibration {
	t.Helper()
	return trace.Calibrate(mustSteps(t), BaselineChunkSize, 64)
}

func mustSteps(t *testing.T) []trace.Step {
	t.Helper()
	return incidentSteps()
}
