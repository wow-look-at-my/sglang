package sched

import (
	"testing"

	"github.com/stretchr/testify/require"
	"schedsim/internal/trace"
)

func workload(t *testing.T) (Workload, trace.Metrics) {
	t.Helper()
	steps, err := trace.Parse(trace.EmbeddedLog)
	require.Nil(t, err)

	m := trace.Summarize(steps, 4096)
	require.GreaterOrEqual(t, m.ColdStart, 0)

	return WorkloadFromLog(steps, m, 4096), m
}

// The property the collapse is about: under prefill priority no decode step runs
// in the cold-prefill window, and under the balancer many do. This is the
// before/after the operator asked for, and it fails if either policy's rule is
// changed so the effect disappears.
func TestOldPolicyStarvesDecodeAndNewPolicyDoesNot(t *testing.T) {
	w, _ := workload(t)

	oldRes := Simulate(w, w.Params(PolicyPrefillPriority), 0)
	require.Equal(t, 0, oldRes.DecodeSteps)

	require.Equal(t, float64(0), oldRes.GeneratedTokens)

	newRes := Simulate(w, w.Params(PolicyTimeBalance), 0)
	require.Greater(t, newRes.DecodeSteps, 0)

	require.Greater(t, newRes.EffectiveGenTPS, oldRes.EffectiveGenTPS)

}

// The fix must not livelock prefill: both policies finish every chunk.
func TestBothPoliciesCompleteTheColdPrefill(t *testing.T) {
	w, _ := workload(t)

	for _, p := range []Policy{PolicyPrefillPriority, PolicyTimeBalance} {
		r := Simulate(w, w.Params(p), 0)
		require.True(t, r.PrefillCompleted)

		require.Equal(t, len(w.Chunks), r.PrefillChunksRun)

	}
}

// Old-policy starvation is total: one gap as long as the whole window.
func TestOldPolicyGapIsTheWholeWindow(t *testing.T) {
	w, _ := workload(t)
	r := Simulate(w, w.Params(PolicyPrefillPriority), 0)
	diff := r.LongestDecodeGap - r.WindowSeconds
	require.False(t, diff > 1e-9 || diff < -1e-9)

	n := Simulate(w, w.Params(PolicyTimeBalance), 0)
	require.Less(t, n.LongestDecodeGap, r.LongestDecodeGap)

}

// A scheduling parameter must move the result: the model is not a pair of
// hard-coded numbers. More decode rounds per prefill batch buys more generation
// and delays the prefill, monotonically.
func TestDecodeIntervalMovesBothOutcomes(t *testing.T) {
	w, _ := workload(t)

	var prevGen, prevFinish float64
	for _, n := range []int{1, 2, 4, 8} {
		p := w.Params(PolicyFixedInterval)
		p.Interval = n
		r := Simulate(w, p, 0)

		require.False(t, n > 1 && r.EffectiveGenTPS <= prevGen)

		require.False(t, n > 1 && r.WindowSeconds <= prevFinish)

		require.True(t, r.PrefillCompleted)

		prevGen, prevFinish = r.EffectiveGenTPS, r.WindowSeconds
	}
}

// The prefill share is the balancer's split knob, so changing it must change how
// the window is divided: a larger prefill share means a faster prefill and less
// generation, monotonically.
func TestPrefillShareTradesPrefillTimeAgainstGeneration(t *testing.T) {
	w, _ := workload(t)

	var prevGen, prevFinish float64
	first := true
	for _, share := range []float64{0.25, 0.5, 0.75, 0.9} {
		p := w.Params(PolicyTimeBalance)
		p.PrefillShare = share
		r := Simulate(w, p, 0)

		if !first {
			require.Less(t, r.EffectiveGenTPS, prevGen)

			require.Less(t, r.WindowSeconds, prevFinish)

		}
		first = false
		prevGen, prevFinish = r.EffectiveGenTPS, r.WindowSeconds
	}
}

func TestBalancedShareSplitsGPUTimeEvenly(t *testing.T) {
	w, _ := workload(t)

	p := w.Params(PolicyTimeBalance)
	r := Simulate(w, p, 0)

	var prefillSecs, decodeSecs float64
	for _, e := range r.Trace {
		if e.IsPrefill {
			prefillSecs += e.Seconds
		} else {
			decodeSecs += e.Seconds
		}
	}
	require.False(t, prefillSecs <= 0 || decodeSecs <= 0)

	ratio := prefillSecs / (prefillSecs + decodeSecs)
	require.False(t, ratio < 0.4 || ratio > 0.6)

}

// Costs must come from the log: the model's prefill seconds must equal the sum
// of the measured chunk durations, not a constant.
func TestPrefillCostIsTheMeasuredLogCost(t *testing.T) {
	w, m := workload(t)

	r := Simulate(w, w.Params(PolicyPrefillPriority), 0)
	diff := r.PrefillGPUSeconds - m.ColdSeconds
	require.False(t, diff > 1e-9 || diff < -1e-9)

	require.Equal(t, m.ColdSeconds, r.WindowSeconds)

}

// Decode cost scaling is a knob because the log only ever decodes one request.
// With more than one running request the knob must bite: a higher per-request
// cost buys fewer decode steps and a longer window.
func TestDecodeCostKnobResponds(t *testing.T) {
	w, _ := workload(t)

	flat := w.Params(PolicyTimeBalance)
	flat.RunningReqs = 3
	flat.DecodePerReqFraction = 0
	rFlat := Simulate(w, flat, 0)

	scaled := w.Params(PolicyTimeBalance)
	scaled.RunningReqs = 3
	scaled.DecodePerReqFraction = 1.0
	rScaled := Simulate(w, scaled, 0)

	// Both generate for conversations, so compare step counts and window, which
	// the per-request cost does govern.
	require.Less(t, rScaled.DecodeSteps, rFlat.DecodeSteps)

	require.Greater(t, rScaled.WindowSeconds, rFlat.WindowSeconds)

}

// A decode step advances every running request. Since batch-size scaling of
// decode cost is not measurable from this log, the default model charges a flat
// cost per step, and the consequence must be exactly what "flat" means: the
// batch generates proportionally more while each conversation's rate is
// unchanged.
func TestFlatDecodeCostScalesGenerationWithTheBatch(t *testing.T) {
	w, _ := workload(t)

	base := w.Params(PolicyTimeBalance)
	base.RunningReqs = 1
	rBase := Simulate(w, base, 0)
	perReq := rBase.GeneratedTokens

	for _, reqs := range []int{2, 4} {
		p := w.Params(PolicyTimeBalance)
		p.RunningReqs = reqs
		r := Simulate(w, p, 0)

		require.Equal(t, rBase.DecodeSteps, r.DecodeSteps)

		want := perReq * float64(reqs)
		diff := r.GeneratedTokens - want
		require.False(t, diff > 1e-6 || diff < -1e-6)

	}
}

// Once per-request cost is charged, sharing the window must cost each
// conversation: per-request generation falls as the batch grows.
func TestPerRequestDecodeCostDividesGeneration(t *testing.T) {
	w, _ := workload(t)

	var prevPerReq float64
	first := true
	for _, reqs := range []int{2, 4, 8} {
		p := w.Params(PolicyTimeBalance)
		p.RunningReqs = reqs
		p.DecodePerReqFraction = 1.0
		r := Simulate(w, p, 0)

		require.Greater(t, r.GeneratedTokens, 0)

		perReq := r.GeneratedTokens / float64(reqs)
		require.False(t, !first && perReq >= prevPerReq)

		first = false
		prevPerReq = perReq
	}
}

func TestQueueBalanceIgnoresRequestsThatCannotRun(t *testing.T) {
	w, _ := workload(t)

	even := Simulate(w, w.Params(PolicyTimeBalance), 0)
	p := w.Params(PolicyQueueBalance)
	alone := Simulate(w, p, 0)
	queued := Simulate(w, p, 3)

	require.False(t, queued.DecodeSteps != alone.DecodeSteps || queued.WindowSeconds != alone.WindowSeconds)

	d := even.DecodeSteps - alone.DecodeSteps
	require.False(t, d < 0 || d > len(w.Chunks))

}

// Mixed-chunk tokens ride on top of decode's half and never replace it: a
// chunk that gives each running request one token is still followed by pure
// decode worth its own time, so no stream collapses to a token per chunk.
func TestMixedChunkRidesOnTopOfDecodeHalf(t *testing.T) {
	w, _ := workload(t)

	p := w.Params(PolicyQueueBalance)
	p.RunningReqs = 5
	plain := Simulate(w, p, 0)
	p.MixedChunk = true
	r := Simulate(w, p, 0)

	var chunkSecs, decodeSecs, longest float64
	for _, c := range w.Chunks {
		chunkSecs += c.Seconds
	}
	for _, e := range r.Trace {
		longest = max(longest, e.Seconds)
		if !e.IsPrefill {
			decodeSecs += e.Seconds
		}
	}
	// The window ends with the last chunk, which decode has not yet repaid.
	owed := chunkSecs - w.Chunks[len(w.Chunks)-1].Seconds
	require.GreaterOrEqual(t, decodeSecs, owed-p.DecodeStepSeconds)

	require.Greater(t, r.GeneratedTokens, plain.GeneratedTokens)

	require.LessOrEqual(t, r.LongestDecodeGap, longest+1e-9)

}

// Every chunk the workload carries must be a step the log actually contains.
func TestWorkloadChunksComeFromTheLog(t *testing.T) {
	w, m := workload(t)

	require.Equal(t, m.ColdChunks, len(w.Chunks))

	for _, c := range w.Chunks {
		require.Equal(t, 4096, c.Tokens)

		require.Greater(t, c.Seconds, 0)

		// The log reports throughput to decimals, so the identity holds to
		// that rounding rather than exactly.
		back := float64(c.Tokens) / c.Seconds
		rel := (back - c.InputTPS) / c.InputTPS
		require.False(t, rel > 1e-4 || rel < -1e-4)

	}
}
