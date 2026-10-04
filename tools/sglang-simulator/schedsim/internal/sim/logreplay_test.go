package sim

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"schedsim/internal/trace"
	"schedsim/internal/trace/tracetest"
)

// corpusBoot is boot index of the generated multi-boot log, with what the generator wrote into it.
func corpusBoot(t *testing.T, index int) (*trace.Boot, *tracetest.Traffic) {
	t.Helper()
	gen := tracetest.Corpus()
	boots, err := trace.ParseBoots(tracetest.Text(gen))
	require.Nil(t, err)

	return &boots[index], gen[index]
}

func TestLogReplayRebuildsABoot(t *testing.T) {
	b, g := corpusBoot(t, 2)
	l, err := BuildLogReplay(b, 4096, 0, 0)
	require.Nil(t, err)

	assert.Equal(t, g.Spec.Shared, l.Shared)

	assert.Equal(t, g.Turns, len(l.Turns))

	assert.Equal(t, g.Convs+g.FreshColds, l.Convs)

	assert.Equal(t, g.Returns, l.Returning)

	m := l.MeasuredOuts()
	assert.Equal(t, g.Measured, m)

	c := l.ColdTurns()
	assert.Equal(t, g.ColdTurns, c)

	for i, tn := range l.Turns {
		require.False(t, i > 0 && tn.At < l.Turns[i-1].At)

		assert.False(t, tn.Input <= 0 || tn.Input > 524288 || tn.Out <= 0 || tn.Out > maxReply)

		assert.LessOrEqual(t, tn.Hit, tn.Input)

	}
	// Every conversation's turns grow: the next prompt carries the last.
	for _, idx := range l.byConv {
		for k := 1; k < len(idx); k++ {
			prev, cur := l.Turns[idx[k-1]], l.Turns[idx[k]]
			assert.False(t, cur.Kind == KindAgent && cur.Hit < prev.Input)

		}
	}
	// Initial releases one turn per conversation, and OnFinish walks the rest in order, never earlier than logged.
	init := l.Initial()
	assert.Equal(t, l.Convs, len(init))

	r := init[0]
	next := l.OnFinish(r, r.Arrival+1000)
	if len(l.byConv[r.Conv]) > 1 {
		assert.False(t, len(next) != 1 || next[0].Arrival < r.Arrival+1000)

	} else if len(next) != 0 {
		t.Errorf("a one-turn conversation released %v", next)
	}
}

func TestLogReplayRunsUnderEveryPolicy(t *testing.T) {
	b, _ := corpusBoot(t, 2)
	l, err := BuildLogReplay(b, 4096, 0, 300)
	require.Nil(t, err)

	cal, _ := trace.CalibrateBoots([]trace.Boot{*b}, 4096, 64)
	cost := NewCost(cal)
	sc := Scenario{
		Name: "boot 2 first 300 s", Key: "boot2",
		Build:    func(int64, Cost) Workload { return l.Reset() },
		HardStop: l.Window + 120, Window: l.Window,
		MaxRunning: b.Args.MaxRunningRequests, HostMul: 0, Seeds: []int64{1},
	}
	rows := RunSuite([]Scenario{sc}, cost, DefaultConfig, 3)
	for i, _ := range Modes {
		m := rows[0].Metrics[i]
		assert.NotEqual(t, 0, m.CompletedTurns)

		assert.Greater(t, m.LongestStall, 0)

	}
	old, new := rows[0].Metrics[0], rows[0].Metrics[2]
	assert.Less(t, new.LongestStall, old.LongestStall)

}

func TestChainTokensStopsAtThePartialChunk(t *testing.T) {
	full := func(pending int) trace.Step {
		return trace.Step{Kind: trace.Prefill, NewSeq: 1, NewTokens: 4096, Pending: pending}
	}
	prefill := []trace.Step{full(305904), full(301808), {Kind: trace.Prefill, NewSeq: 1, NewTokens: 1808, Pending: 300000}, full(295904)}
	got := chainTokens(prefill, 0, 4096)
	assert.Equal(t, 4096+1808, got)

}

func TestSplitArrivalsSharesALineAmongItsRequests(t *testing.T) {
	one := trace.Step{Kind: trace.Prefill, NewSeq: 1, NewTokens: 4096, HitTokens: 12288, Pending: 400000}
	got := splitArrivals(one, 4096, 400000)
	assert.False(t, len(got) != 1 || got[0].hit+got[0].nw+got[0].pending != 416384)

	two := trace.Step{Kind: trace.Prefill, NewSeq: 2, NewTokens: 4037, HitTokens: 12288, Pending: 402160}
	got := splitArrivals(two, 4096, 402160)
	assert.False(t, len(got) != 2 || got[0].pending != 402160 || got[0].nw != 4037 || got[1].hit != 12288)

	three := trace.Step{Kind: trace.Prefill, NewSeq: 3, NewTokens: 4000, HitTokens: 248320}
	got = splitArrivals(three, 4096, 0)
	assert.False(t, len(got) != 3 || got[1].hit != 248320/3)

}
