package sim

import (
	"testing"

	"schedsim/internal/trace"
)

func log2Boot(t *testing.T, index int) *trace.Boot {
	t.Helper()
	boots, err := trace.ParseBoots(trace.Log2())
	if err != nil {
		t.Fatal(err)
	}
	return &boots[index]
}

// Boot 4 is the 38-minute lifetime that held the day's worst pile-up: 14
// conversations decoding behind a chain of cold prompts at 00:08-00:18Z.
func TestLogReplayRebuildsBootFour(t *testing.T) {
	b := log2Boot(t, 4)
	l, err := BuildLogReplay(b, 4096, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if l.Shared < 10000 || l.Shared > 14000 {
		t.Errorf("shared prefix %d, want the ~12k-token system prompt", l.Shared)
	}
	if len(l.Turns) < 800 || len(l.Turns) > 1000 {
		t.Errorf("turns = %d; the boot completed 849 requests", len(l.Turns))
	}
	if l.Convs < 40 || l.Convs > 120 {
		t.Errorf("conversations = %d", l.Convs)
	}
	if l.Returning < 10 {
		t.Errorf("returning conversations = %d; the boot re-prefilled evicted prompts repeatedly", l.Returning)
	}
	if m := l.MeasuredOuts(); m < len(l.Turns)*3/4 {
		t.Errorf("reply length measured for %d of %d turns", m, len(l.Turns))
	}
	if c := l.ColdTurns(); c < 20 {
		t.Errorf("cold turns = %d", c)
	}
	// Arrivals are monotone and prompt sizes plausible for a 524288 context.
	for i, tn := range l.Turns {
		if i > 0 && tn.At < l.Turns[i-1].At {
			t.Fatalf("turn %d arrives before turn %d", i, i-1)
		}
		if tn.Input <= 0 || tn.Input > 524288 || tn.Out <= 0 || tn.Out > maxReply {
			t.Errorf("turn %d: input %d out %d", i, tn.Input, tn.Out)
		}
		if tn.Hit > tn.Input {
			t.Errorf("turn %d: hit %d over input %d", i, tn.Hit, tn.Input)
		}
	}
	// Every conversation's turns grow: the next prompt carries the last.
	for conv, idx := range l.byConv {
		for k := 1; k < len(idx); k++ {
			prev, cur := l.Turns[idx[k-1]], l.Turns[idx[k]]
			if cur.Kind == KindAgent && cur.Hit < prev.Input {
				t.Errorf("conv %d turn %d hit %d below the previous prompt %d", conv, k, cur.Hit, prev.Input)
			}
		}
	}
	// Initial releases one turn per conversation, and OnFinish walks the rest
	// in order, never earlier than logged.
	init := l.Initial()
	if len(init) != l.Convs {
		t.Errorf("Initial released %d, want %d", len(init), l.Convs)
	}
	r := init[0]
	next := l.OnFinish(r, r.Arrival+1000)
	if len(l.byConv[r.Conv]) > 1 {
		if len(next) != 1 || next[0].Arrival < r.Arrival+1000 {
			t.Errorf("OnFinish released %v", next)
		}
	} else if len(next) != 0 {
		t.Errorf("a one-turn conversation released %v", next)
	}
}

func TestLogReplayRunsUnderEveryPolicy(t *testing.T) {
	b := log2Boot(t, 4)
	l, err := BuildLogReplay(b, 4096, 0, 300)
	if err != nil {
		t.Fatal(err)
	}
	cal, _ := trace.CalibrateBoots([]trace.Boot{*b}, 4096, 64)
	cost := NewCost(cal)
	sc := Scenario{
		Name: "boot 4 first 300 s", Key: "boot4",
		Build:    func(int64, Cost) Workload { return l.Reset() },
		HardStop: l.Window + 120, Window: l.Window,
		MaxRunning: b.Args.MaxRunningRequests, HostMul: 0, Seeds: []int64{1},
	}
	rows := RunSuite([]Scenario{sc}, cost, DefaultConfig, 3)
	for i, mode := range Modes {
		m := rows[0].Metrics[i]
		if m.CompletedTurns == 0 {
			t.Errorf("%s completed no turns", mode)
		}
		if m.LongestStall <= 0 {
			t.Errorf("%s reports no stall at all", mode)
		}
	}
	old, new := rows[0].Metrics[0], rows[0].Metrics[2]
	if new.LongestStall >= old.LongestStall {
		t.Errorf("longest stall OLD %.1f s, NEW %.1f s", old.LongestStall, new.LongestStall)
	}
}

func TestChainTokensStopsAtThePartialChunk(t *testing.T) {
	full := func(pending int) trace.Step {
		return trace.Step{Kind: trace.Prefill, NewSeq: 1, NewTokens: 4096, Pending: pending}
	}
	// A 10,000-token prompt behind a 300,000-token queue: first chunk, one
	// full chunk, a partial one, then the queued prompt's own first chunk.
	prefill := []trace.Step{full(305904), full(301808), {Kind: trace.Prefill, NewSeq: 1, NewTokens: 1808, Pending: 300000}, full(295904)}
	if got := chainTokens(prefill, 0, 4096); got != 4096+1808 {
		t.Errorf("chain = %d, want 5904 (pending counted the queue)", got)
	}
}

func TestSplitArrivalsSharesALineAmongItsRequests(t *testing.T) {
	one := trace.Step{Kind: trace.Prefill, NewSeq: 1, NewTokens: 4096, HitTokens: 12288, Pending: 400000}
	if got := splitArrivals(one, 4096, 400000); len(got) != 1 || got[0].hit+got[0].nw+got[0].pending != 416384 {
		t.Errorf("one request: %+v", got)
	}
	two := trace.Step{Kind: trace.Prefill, NewSeq: 2, NewTokens: 4037, HitTokens: 12288, Pending: 402160}
	got := splitArrivals(two, 4096, 402160)
	if len(got) != 2 || got[0].pending != 402160 || got[0].nw != 4037 || got[1].hit != 12288 {
		t.Errorf("chunked prompt plus follow-up: %+v", got)
	}
	three := trace.Step{Kind: trace.Prefill, NewSeq: 3, NewTokens: 4000, HitTokens: 248320}
	got = splitArrivals(three, 4096, 0)
	if len(got) != 3 || got[1].hit != 248320/3 {
		t.Errorf("three follow-ups: %+v", got)
	}
}
