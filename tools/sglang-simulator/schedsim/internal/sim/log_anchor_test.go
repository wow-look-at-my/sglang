package sim

import (
	"bytes"
	"math"
	"sort"
	"testing"

	"schedsim/internal/trace"
)

// The engine is calibrated on the embedded production log, not on the scenarios it is
// run on: the cost coefficients come from `internal/trace/calib.go`. These tests pin
// the behaviours the log states as fact and check the fitted model against the log
// lines it was solved on.

// TestReplayReproducesTheLoggedBatch pins the batch the log reports at its
// "#new-seq: 3" line: 4000 new tokens and 248320 cached, made of three items of
// 263, 921 and 2816 tokens. The composition depends on the page rounding, the token
// budget, the cede's half-chunk rule and which prefix the pool reports for the
// follow-up, so it fails if any of those drift.
func TestReplayReproducesTheLoggedBatch(t *testing.T) {
	sc := ScenarioA()
	cost := NewCost(calib(t))
	res := Run(sc, DefaultConfig(ModeOld, cost), sc.Seeds[0])

	var found *PrefillLogLine
	for i, pl := range res.PrefillLog {
		if pl.NewSeq == 3 {
			found = &res.PrefillLog[i]
			break
		}
	}
	if found == nil {
		t.Fatalf("no prefill batch with #new-seq 3; the run logged %d prefill lines", len(res.PrefillLog))
	}
	if found.Tokens != 4000 {
		t.Errorf("#new-token = %d, want the log's 4000", found.Tokens)
	}
	if found.Hit != 248320 {
		t.Errorf("#cached-token = %d, want the log's 248320", found.Hit)
	}
	var got []int
	for _, it := range found.Items {
		got = append(got, it.Extend)
	}
	sort.Ints(got)
	want := []int{263, 921, 2816}
	if len(got) != len(want) {
		t.Fatalf("the batch carries %d items %v, want %v", len(got), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("item %d = %d, want %d (batch %v)", i, got[i], want[i], got)
		}
	}
}

// TestOldRunsNoDecodeDuringTheColdPrefills is the log's own shape. C1's first chunk is
// the prefill line whose pending work plus itself exceeds 400K; from there the log runs
// 105 chunks of 4096 tokens with no decode line between them, and only the batch that
// finishes C1 (#new-seq 3, 4000 new tokens) is followed by decode. Upstream
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
	if start < 0 {
		t.Fatal("the log has no cold run over 400K pending tokens")
	}
	chunks := 0
	for _, s := range steps[start:] {
		if s.Kind == trace.Decode {
			t.Fatalf("the log shows a decode line (%d) after %d of C1's chunks", s.Line, chunks)
		}
		if s.NewTokens != BaselineChunkSize {
			if chunks != 105 {
				t.Fatalf("C1's cold run is %d full chunks, want 105 (stopped at #new-token %d)",
					chunks, s.NewTokens)
			}
			if s.NewSeq != 3 || s.NewTokens != 4000 || s.HitTokens != 248320 {
				t.Fatalf("the batch after C1's chunks is #new-seq %d, #new-token %d, #cached-token %d,"+
					" want 3, 4000, 248320", s.NewSeq, s.NewTokens, s.HitTokens)
			}
			break
		}
		chunks++
	}

	sc := ScenarioA()
	cost := NewCost(calib(t))
	res := Run(sc, DefaultConfig(ModeOld, cost), sc.Seeds[0])
	c1 := FindTag(res, "C1")
	if c1 == nil {
		t.Fatal("scenario A produced no C1")
	}
	n := 0
	for _, b := range res.Batches {
		if !b.IsPrefill && b.Start < c1.FirstTok {
			n++
		}
	}
	if n != 0 {
		t.Errorf("OLD launched %d decode batches before C1's first token, want 0", n)
	}
}

// TestDecodeRateMatchesTheLog checks the decode model against the lines FitDecode
// solved: a decode line whose predecessor is also a decode line, on one request. Its
// reported gen throughput covers a whole interval of decode steps, so accepted tokens
// over throughput is a step time the model has to reproduce. Lines that follow a
// prefill are excluded because their throughput window contains the prefill; they are
// reported, not fitted, and the gap between the two populations is the reason.
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
	if len(steadyRel) != d.Samples {
		t.Fatalf("checked %d steady decode lines, the fit used %d", len(steadyRel), d.Samples)
	}
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
	if steadyRel[len(steadyRel)-1] > 0.05 {
		t.Errorf("the decode model misses a steady line by %.1f%%", 100*steadyRel[len(steadyRel)-1])
	}
	if mean > 0.01 {
		t.Errorf("the decode model misses the steady lines by %.1f%% on average", 100*mean)
	}
}

// TestRunIsDeterministic pins the entry point's promise: the same scenario, policy and
// seed produce byte-identical tables, and a parallel suite reports what a serial one
// does. The two orderings dispatch the same runs through different goroutines, so this
// also covers the suite's own aggregation.
func TestRunIsDeterministic(t *testing.T) {
	cost := NewCost(calib(t))
	scs := []Scenario{ScenarioA(), ScenarioB(5), ScenarioThrash(1.5, 600)}
	var first []byte
	for _, workers := range []int{1, 4} {
		var buf bytes.Buffer
		WriteTables(&buf, RunSuite(scs, cost, DefaultConfig, workers))
		if first == nil {
			first = buf.Bytes()
			continue
		}
		if !bytes.Equal(first, buf.Bytes()) {
			t.Errorf("the second suite run differs from the first (1 worker vs %d)", workers)
		}
	}
}

func calib(t *testing.T) trace.Calibration {
	t.Helper()
	return trace.Calibrate(mustSteps(t), BaselineChunkSize, 64)
}

func mustSteps(t *testing.T) []trace.Step {
	t.Helper()
	steps, err := trace.Parse(trace.EmbeddedLog)
	if err != nil {
		t.Fatalf("parsing the embedded log: %v", err)
	}
	return steps
}
