package sim

import (
	"testing"

	"schedsim/internal/trace"
	"schedsim/internal/trace/tracetest"
)

// corpusBoot is boot index of the generated multi-boot log, with what the generator wrote into it.
func corpusBoot(t *testing.T, index int) (*trace.Boot, *tracetest.Traffic) {
	t.Helper()
	gen := tracetest.Corpus()
	boots, err := trace.ParseBoots(tracetest.Text(gen))
	if err != nil {
		t.Fatal(err)
	}
	return &boots[index], gen[index]
}

func TestLogReplayRebuildsABoot(t *testing.T) {
	b, g := corpusBoot(t, 2)
	l, err := BuildLogReplay(b, 4096, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if l.Shared != g.Spec.Shared {
		t.Errorf("shared prefix %d, want the %d-token system prompt", l.Shared, g.Spec.Shared)
	}
	if len(l.Turns) != g.Turns {
		t.Errorf("turns = %d; the boot completed %d requests", len(l.Turns), g.Turns)
	}
	if l.Convs != g.Convs+g.FreshColds {
		t.Errorf("conversations = %d, want %d plus %d fresh cold prompts", l.Convs, g.Convs, g.FreshColds)
	}
	if l.Returning != g.Returns {
		t.Errorf("returning conversations = %d; the boot re-prefilled %d evicted prompts", l.Returning, g.Returns)
	}
	if m := l.MeasuredOuts(); m != g.Measured {
		t.Errorf("reply length measured for %d of %d turns, a later turn followed %d", m, len(l.Turns), g.Measured)
	}
	if c := l.ColdTurns(); c != g.ColdTurns {
		t.Errorf("cold turns = %d, want %d", c, g.ColdTurns)
	}
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
	// Initial releases one turn per conversation, and OnFinish walks the rest in order, never earlier than logged.
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
	b, _ := corpusBoot(t, 2)
	l, err := BuildLogReplay(b, 4096, 0, 300)
	if err != nil {
		t.Fatal(err)
	}
	cal, _ := trace.CalibrateBoots([]trace.Boot{*b}, 4096, 64)
	cost := NewCost(cal)
	sc := Scenario{
		Name: "boot 2 first 300 s", Key: "boot2",
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
