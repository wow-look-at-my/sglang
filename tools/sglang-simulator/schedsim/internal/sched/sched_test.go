package sched

import (
	"testing"

	"schedsim/internal/trace"
	"schedsim/internal/trace/tracetest"
)

func workload(t *testing.T) (Workload, trace.Metrics) {
	t.Helper()
	steps, err := trace.Parse(tracetest.Incident(tracetest.DefaultIncident).String())
	if err != nil {
		t.Fatalf("parse incident log: %v", err)
	}
	m := trace.Summarize(steps, 4096)
	if m.ColdStart < 0 {
		t.Fatal("no cold prefill stretch in the incident log")
	}
	return WorkloadFromLog(steps, m, 4096), m
}

// The property the collapse is about: under prefill priority no decode step runs
// in the cold-prefill window, and under the balancer many do. This is the
// before/after the operator asked for, and it fails if either policy's rule is
// changed so the effect disappears.
func TestOldPolicyStarvesDecodeAndNewPolicyDoesNot(t *testing.T) {
	w, _ := workload(t)

	oldRes := Simulate(w, w.Params(PolicyPrefillPriority), 0)
	if oldRes.DecodeSteps != 0 {
		t.Fatalf("old policy ran %d decode steps in the window, want 0", oldRes.DecodeSteps)
	}
	if oldRes.GeneratedTokens != 0 {
		t.Fatalf("old policy generated %v tokens, want 0", oldRes.GeneratedTokens)
	}

	newRes := Simulate(w, w.Params(PolicyTimeBalance), 0)
	if newRes.DecodeSteps <= 0 {
		t.Fatalf("new policy ran %d decode steps in the window, want > 0", newRes.DecodeSteps)
	}
	if newRes.EffectiveGenTPS <= oldRes.EffectiveGenTPS {
		t.Fatalf("new policy gen %.2f tok/s is not above old %.2f", newRes.EffectiveGenTPS, oldRes.EffectiveGenTPS)
	}
}

// The fix must not livelock prefill: both policies finish every chunk.
func TestBothPoliciesCompleteTheColdPrefill(t *testing.T) {
	w, _ := workload(t)

	for _, p := range []Policy{PolicyPrefillPriority, PolicyTimeBalance} {
		r := Simulate(w, w.Params(p), 0)
		if !r.PrefillCompleted {
			t.Fatalf("%s ran %d of %d chunks", p, r.PrefillChunksRun, len(w.Chunks))
		}
		if r.PrefillChunksRun != len(w.Chunks) {
			t.Fatalf("%s ran %d chunks, want %d", p, r.PrefillChunksRun, len(w.Chunks))
		}
	}
}

// Old-policy starvation is total: one gap as long as the whole window.
func TestOldPolicyGapIsTheWholeWindow(t *testing.T) {
	w, _ := workload(t)
	r := Simulate(w, w.Params(PolicyPrefillPriority), 0)
	if diff := r.LongestDecodeGap - r.WindowSeconds; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("old policy longest decode gap %.6f s, window %.6f s; want them equal",
			r.LongestDecodeGap, r.WindowSeconds)
	}
	n := Simulate(w, w.Params(PolicyTimeBalance), 0)
	if n.LongestDecodeGap >= r.LongestDecodeGap {
		t.Fatalf("new policy gap %.6f s is not shorter than old %.6f s", n.LongestDecodeGap, r.LongestDecodeGap)
	}
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

		if n > 1 && r.EffectiveGenTPS <= prevGen {
			t.Fatalf("interval %d gave %.2f tok/s, not above the previous %.2f", n, r.EffectiveGenTPS, prevGen)
		}
		if n > 1 && r.WindowSeconds <= prevFinish {
			t.Fatalf("interval %d finished in %.2f s, not after the previous %.2f", n, r.WindowSeconds, prevFinish)
		}
		if !r.PrefillCompleted {
			t.Fatalf("interval %d left the prefill incomplete", n)
		}
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
			if r.EffectiveGenTPS >= prevGen {
				t.Fatalf("share %.2f gave %.2f tok/s, not below the previous %.2f", share, r.EffectiveGenTPS, prevGen)
			}
			if r.WindowSeconds >= prevFinish {
				t.Fatalf("share %.2f finished in %.2f s, not before the previous %.2f", share, r.WindowSeconds, prevFinish)
			}
		}
		first = false
		prevGen, prevFinish = r.EffectiveGenTPS, r.WindowSeconds
	}
}

// The balancer's own rule is symmetric at 50/50, so the split it produces must
// be near even. This is what makes PrefillShare=0.5 a reproduction of the
// shipped rule rather than a tunable invented here.
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
	if prefillSecs <= 0 || decodeSecs <= 0 {
		t.Fatalf("one class got no GPU time: prefill %.3f s, decode %.3f s", prefillSecs, decodeSecs)
	}
	ratio := prefillSecs / (prefillSecs + decodeSecs)
	if ratio < 0.4 || ratio > 0.6 {
		t.Fatalf("prefill took %.1f%% of contended GPU time at share 0.5, want ~50%%", ratio*100)
	}
}

// Costs must come from the log: the model's prefill seconds must equal the sum
// of the measured chunk durations, not a constant.
func TestPrefillCostIsTheMeasuredLogCost(t *testing.T) {
	w, m := workload(t)

	r := Simulate(w, w.Params(PolicyPrefillPriority), 0)
	if diff := r.PrefillGPUSeconds - m.ColdSeconds; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("model prefill %.6f s, log measured %.6f s", r.PrefillGPUSeconds, m.ColdSeconds)
	}
	if r.WindowSeconds != m.ColdSeconds {
		t.Fatalf("old policy window %.6f s should be exactly the measured prefill %.6f s",
			r.WindowSeconds, m.ColdSeconds)
	}
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

	// Both generate for three conversations, so compare step counts and window,
	// which the per-request cost does govern.
	if rScaled.DecodeSteps >= rFlat.DecodeSteps {
		t.Fatalf("decode cost knob did nothing: %d steps at flat cost, %d at 3x",
			rFlat.DecodeSteps, rScaled.DecodeSteps)
	}
	if rScaled.WindowSeconds <= rFlat.WindowSeconds {
		t.Fatalf("window stayed at %.2f s despite higher decode cost", rFlat.WindowSeconds)
	}
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

		if r.DecodeSteps != rBase.DecodeSteps {
			t.Fatalf("flat cost changed the step count at %d requests: %d vs %d",
				reqs, r.DecodeSteps, rBase.DecodeSteps)
		}
		want := perReq * float64(reqs)
		if diff := r.GeneratedTokens - want; diff > 1e-6 || diff < -1e-6 {
			t.Fatalf("%d requests generated %.1f tokens, want %.1f with flat per-request cost",
				reqs, r.GeneratedTokens, want)
		}
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

		if r.GeneratedTokens <= 0 {
			t.Fatalf("%d running requests generated nothing", reqs)
		}
		perReq := r.GeneratedTokens / float64(reqs)
		if !first && perReq >= prevPerReq {
			t.Fatalf("%d requests got %.1f tokens each, not below the previous %.1f",
				reqs, perReq, prevPerReq)
		}
		first = false
		prevPerReq = perReq
	}
}

// Requests queued behind the chunk cannot join it, so they must not weigh the
// split: the revised balancer gives decode the shipped 50/50 share whatever
// the queue, carrying at most one decode step of overshoot per chunk.
func TestQueueBalanceIgnoresRequestsThatCannotRun(t *testing.T) {
	w, _ := workload(t)

	even := Simulate(w, w.Params(PolicyTimeBalance), 0)
	p := w.Params(PolicyQueueBalance)
	alone := Simulate(w, p, 0)
	queued := Simulate(w, p, 3)

	if queued.DecodeSteps != alone.DecodeSteps || queued.WindowSeconds != alone.WindowSeconds {
		t.Fatalf("queue changed the run: %d steps over %.3f s, alone %d over %.3f s",
			queued.DecodeSteps, queued.WindowSeconds, alone.DecodeSteps, alone.WindowSeconds)
	}
	if d := even.DecodeSteps - alone.DecodeSteps; d < 0 || d > len(w.Chunks) {
		t.Fatalf("revised %d decode steps, shipped %d: more than one step of carry per chunk",
			alone.DecodeSteps, even.DecodeSteps)
	}
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
	if decodeSecs < owed-p.DecodeStepSeconds {
		t.Fatalf("pure decode %.2f s against %.2f s of chunks: rows replaced decode's half", decodeSecs, owed)
	}
	if r.GeneratedTokens <= plain.GeneratedTokens {
		t.Fatalf("mixed generated %.0f tokens, plain %.0f: rows added nothing", r.GeneratedTokens, plain.GeneratedTokens)
	}
	if r.LongestDecodeGap > longest+1e-9 {
		t.Fatalf("gap %.3f s exceeds the longest batch %.3f s", r.LongestDecodeGap, longest)
	}
}

// Every chunk the workload carries must be a step the log actually contains.
func TestWorkloadChunksComeFromTheLog(t *testing.T) {
	w, m := workload(t)

	if len(w.Chunks) != m.ColdChunks {
		t.Fatalf("workload has %d chunks, log's cold window has %d", len(w.Chunks), m.ColdChunks)
	}
	for i, c := range w.Chunks {
		if c.Tokens != 4096 {
			t.Fatalf("chunk %d has %d tokens, want 4096", i, c.Tokens)
		}
		if c.Seconds <= 0 {
			t.Fatalf("chunk %d has %v seconds", i, c.Seconds)
		}
		// The log reports throughput to two decimals, so the identity holds to
		// that rounding rather than exactly.
		back := float64(c.Tokens) / c.Seconds
		if rel := (back - c.InputTPS) / c.InputTPS; rel > 1e-4 || rel < -1e-4 {
			t.Fatalf("chunk %d: %.2f tok/s and %v s disagree with %d tokens by %.4f",
				i, c.InputTPS, c.Seconds, c.Tokens, rel)
		}
	}
}
