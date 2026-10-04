package sim

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"schedsim/internal/trace"
	"schedsim/internal/trace/tracetest"
)

func bootOf(t *testing.T, l *tracetest.Log) *trace.Boot {
	t.Helper()
	boots, err := trace.ParseBoots(l.String())
	require.Nil(t, err)

	return &boots[0]
}

func TestBuildLogReplayErrors(t *testing.T) {
	bare, _ := trace.ParseBoots(tracetest.Incident(tracetest.DefaultIncident).String())
	_, err := BuildLogReplay(&bare[0], 4096, 0, 0)
	assert.False(t, err == nil || !strings.Contains(err.Error(), "no timestamps"))

	decodeOnly := bootOf(t, tracetest.New("w", tracetest.Start).Steady(2, 1, 100))
	_, err := BuildLogReplay(decodeOnly, 4096, 0, 0)
	assert.False(t, err == nil || !strings.Contains(err.Error(), "no prefill step"))

	one := bootOf(t, tracetest.New("w", tracetest.Start).Prefill(tracetest.Prefill{NewTokens: 100}).Steady(1, 1, 100))
	_, err := BuildLogReplay(one, 4096, 3600, 0)
	assert.False(t, err == nil || !strings.Contains(err.Error(), "no requests in the window"))

	// A request with no tokens at all is dropped, leaving nothing.
	empty := bootOf(t, tracetest.New("w", tracetest.Start).Prefill(tracetest.Prefill{NewTokens: 0, TPS: 1}))
	_, err := BuildLogReplay(empty, 4096, 0, 0)
	assert.NotNil(t, err)

}

func TestBuildLogReplayReconstructsConversations(t *testing.T) {
	const chunk = 4096
	l := tracetest.New("w", tracetest.Start).ServerArgs(tracetest.DefaultArgs)
	// Conversation multiple opens on the shared 12288-token prompt and, after a 400-token reply, sends its second turn.
	l.Prefill(tracetest.Prefill{NewTokens: 1000, Hit: 12288})
	l.Steady(2, 1, 50000)
	l.Prefill(tracetest.Prefill{NewTokens: 700, Hit: 13288 + 400, Running: 1})
	l.Prefill(tracetest.Prefill{NewTokens: 3000, Hit: 12288, Running: 1})
	l.Advance(1)
	l.Prefill(tracetest.Prefill{NewTokens: 6000, Hit: 12288, Running: 2})
	l.Steady(2, 3, 50000)
	l.Prefill(tracetest.Prefill{NewSeq: 2, NewTokens: 1200, Hit: 15288 + 300 + 18288 + 200, Running: 3})
	l.Steady(2, 3, 50000)
	// A cold prompt of multiple chunks and a partial one, chunked.
	const coldLen = 40*chunk + 100
	l.Prefill(tracetest.Prefill{NewTokens: chunk, Pending: coldLen + 3000, Running: 3, Queue: 1})
	for left := coldLen - chunk; left > 0; left -= chunk {
		n := chunk
		if left < n {
			n = left
		}
		l.Prefill(tracetest.Prefill{NewTokens: n, Pending: left + 3000, Running: 3, Queue: 1})
	}
	l.Prefill(tracetest.Prefill{NewTokens: 3000, Hit: 12288, Running: 4})
	l.Steady(3, 5, 260000)
	// The cold prompt comes back after eviction: same size plus its reply.
	l.Advance(5)
	l.Prefill(tracetest.Prefill{NewTokens: chunk, Pending: coldLen + 500, Running: 2})
	for left := coldLen + 500 - chunk; left > 0; left -= chunk {
		n := chunk
		if left < n {
			n = left
		}
		l.Prefill(tracetest.Prefill{NewTokens: n, Pending: left, Running: 2})
	}
	l.Steady(2, 3, 260000)
	b := bootOf(t, l)
	r, err := BuildLogReplay(b, chunk, 0, 0)
	require.Nil(t, err)

	assert.Equal(t, 12288, r.Shared)

	assert.False(t, r.Convs != 5 || r.Returning != 1 || r.ColdTurns() != 2 || len(r.Turns) != 9)

	assert.False(t, r.Agents() != r.Convs || r.SharedPrefix() != r.Shared)

	first := r.Turns[0]
	assert.False(t, !first.Measured || first.Out != 400 || first.Input != 13288)

	second := r.Turns[1]
	assert.False(t, second.Conv != 0 || second.Kind != KindAgent || second.Hit != 13688)

	f2, f1 := r.Turns[4], r.Turns[5]
	assert.False(t, f2.Conv != 2 || f1.Conv != 1 || f2.Hit != 18288 || f1.Hit != 15288 || f2.Input != 18288+600)

	// The cold prompt's size is its own chunks, not the queue behind it.
	cold := r.Turns[6]
	assert.False(t, cold.Kind != KindCold || cold.Input != coldLen || cold.Conv != 3)

	// The returning prompt is the cold conversation again, with its reply
	// measured from the size difference.
	queued := r.Turns[7]
	assert.False(t, queued.Conv != 4 || queued.Kind != KindAgent || queued.Input != 15288)

	back := r.Turns[8]
	assert.False(t, !back.Returning || back.Conv != cold.Conv || !cold.Measured || cold.Out != 500)

	m := r.MeasuredOuts()
	assert.Equal(t, 2, m)

	init := r.Initial()
	assert.Equal(t, 5, len(init))

	nxt := r.OnFinish(init[0], 0)
	assert.False(t, len(nxt) != 1 || nxt[0].Conv != 0 || nxt[0].Arrival != r.Turns[1].At)

	again := r.OnFinish(nxt[0], 0)
	assert.Equal(t, 0, len(again))

	late := r.OnFinish(init[1], 1e6)
	assert.False(t, len(late) != 1 || late[0].Arrival != 1e6)

	fresh := r.Reset()
	assert.False(t, len(fresh.Initial()) != 5 || len(r.next) == 0)

	// A window that starts after the first turns sees them as new, and a context cap clamps the prompt.
	b.Args.ContextLength = 10000
	late, err := BuildLogReplay(b, chunk, 3, 20)
	require.Nil(t, err)

	for _, tn := range late.Turns {
		assert.False(t, tn.Input > 10000 || tn.At > 20)

	}
	assert.LessOrEqual(t, late.Window, 20)

}

func TestSharedPrefixNeedsThreeHits(t *testing.T) {
	steps := []trace.Step{{Kind: trace.Prefill, HitTokens: 500}, {Kind: trace.Prefill, HitTokens: 500}}
	got := sharedPrefix(steps)
	assert.Equal(t, 0, got)

	steps = append(steps, trace.Step{Kind: trace.Prefill, HitTokens: 500}, trace.Step{Kind: trace.Prefill, HitTokens: 40000})
	got := sharedPrefix(steps)
	assert.Equal(t, 500, got)

	assert.False(t, medianInt(nil, 7) != 7 || medianInt([]int{3, 1, 2}, 0) != 2)

	got := chainTokens([]trace.Step{{NewTokens: 100}}, 0, 4096)
	assert.Equal(t, 0, got)

}
