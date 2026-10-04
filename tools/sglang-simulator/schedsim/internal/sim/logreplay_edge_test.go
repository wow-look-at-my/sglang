package sim

import (
	"strings"
	"testing"

	"schedsim/internal/trace"
	"schedsim/internal/trace/tracetest"
)

func bootOf(t *testing.T, l *tracetest.Log) *trace.Boot {
	t.Helper()
	boots, err := trace.ParseBoots(l.String())
	if err != nil {
		t.Fatal(err)
	}
	return &boots[0]
}

func TestBuildLogReplayErrors(t *testing.T) {
	bare, _ := trace.ParseBoots(trace.EmbeddedLog)
	if _, err := BuildLogReplay(&bare[0], 4096, 0, 0); err == nil || !strings.Contains(err.Error(), "no timestamps") {
		t.Errorf("bare boot: %v", err)
	}
	decodeOnly := bootOf(t, tracetest.New("w", tracetest.Start).Steady(2, 1, 100))
	if _, err := BuildLogReplay(decodeOnly, 4096, 0, 0); err == nil || !strings.Contains(err.Error(), "no prefill step") {
		t.Errorf("decode-only boot: %v", err)
	}
	one := bootOf(t, tracetest.New("w", tracetest.Start).Prefill(tracetest.Prefill{NewTokens: 100}).Steady(1, 1, 100))
	if _, err := BuildLogReplay(one, 4096, 3600, 0); err == nil || !strings.Contains(err.Error(), "no requests in the window") {
		t.Errorf("window past the log: %v", err)
	}
	// A request with no tokens at all is dropped, leaving nothing.
	empty := bootOf(t, tracetest.New("w", tracetest.Start).Prefill(tracetest.Prefill{NewTokens: 0, TPS: 1}))
	if _, err := BuildLogReplay(empty, 4096, 0, 0); err == nil {
		t.Error("an empty request was replayed")
	}
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
	if err != nil {
		t.Fatal(err)
	}
	if r.Shared != 12288 {
		t.Errorf("shared prefix %d", r.Shared)
	}
	if r.Convs != 5 || r.Returning != 1 || r.ColdTurns() != 2 || len(r.Turns) != 9 {
		t.Errorf("convs %d returning %d cold %d turns %d", r.Convs, r.Returning, r.ColdTurns(), len(r.Turns))
	}
	if r.Agents() != r.Convs || r.SharedPrefix() != r.Shared {
		t.Error("accessors")
	}
	if first := r.Turns[0]; !first.Measured || first.Out != 400 || first.Input != 13288 {
		t.Errorf("first turn = %+v", first)
	}
	if second := r.Turns[1]; second.Conv != 0 || second.Kind != KindAgent || second.Hit != 13688 {
		t.Errorf("follow-up = %+v", second)
	}
	f2, f1 := r.Turns[4], r.Turns[5]
	if f2.Conv != 2 || f1.Conv != 1 || f2.Hit != 18288 || f1.Hit != 15288 || f2.Input != 18288+600 {
		t.Errorf("follow-ups = %+v, %+v", f2, f1)
	}
	// The cold prompt's size is its own chunks, not the queue behind it.
	cold := r.Turns[6]
	if cold.Kind != KindCold || cold.Input != coldLen || cold.Conv != 3 {
		t.Errorf("cold turn = %+v", cold)
	}
	// The returning prompt is the cold conversation again, with its reply
	// measured from the size difference.
	if queued := r.Turns[7]; queued.Conv != 4 || queued.Kind != KindAgent || queued.Input != 15288 {
		t.Errorf("queued request = %+v", queued)
	}
	back := r.Turns[8]
	if !back.Returning || back.Conv != cold.Conv || !cold.Measured || cold.Out != 500 {
		t.Errorf("returning turn = %+v, cold %+v", back, cold)
	}
	if m := r.MeasuredOuts(); m != 2 {
		t.Errorf("measured replies = %d", m)
	}
	init := r.Initial()
	if len(init) != 5 {
		t.Errorf("Initial released %d", len(init))
	}
	nxt := r.OnFinish(init[0], 0)
	if len(nxt) != 1 || nxt[0].Conv != 0 || nxt[0].Arrival != r.Turns[1].At {
		t.Errorf("OnFinish released %+v", nxt)
	}
	if again := r.OnFinish(nxt[0], 0); len(again) != 0 {
		t.Errorf("a finished conversation released %+v", again)
	}
	if late := r.OnFinish(init[1], 1e6); len(late) != 1 || late[0].Arrival != 1e6 {
		t.Errorf("a turn after a late reply released %+v", late)
	}
	fresh := r.Reset()
	if len(fresh.Initial()) != 5 || len(r.next) == 0 {
		t.Error("Reset did not start the cursors over")
	}
	// A window that starts after the first turns sees them as new, and a context cap clamps the prompt.
	b.Args.ContextLength = 10000
	late, err := BuildLogReplay(b, chunk, 3, 20)
	if err != nil {
		t.Fatal(err)
	}
	for _, tn := range late.Turns {
		if tn.Input > 10000 || tn.At > 20 {
			t.Errorf("turn %+v outside cap or window", tn)
		}
	}
	if late.Window > 20 {
		t.Errorf("window %v", late.Window)
	}
}

func TestSharedPrefixNeedsThreeHits(t *testing.T) {
	steps := []trace.Step{{Kind: trace.Prefill, HitTokens: 500}, {Kind: trace.Prefill, HitTokens: 500}}
	if got := sharedPrefix(steps); got != 0 {
		t.Errorf("two hits gave %d", got)
	}
	steps = append(steps, trace.Step{Kind: trace.Prefill, HitTokens: 500}, trace.Step{Kind: trace.Prefill, HitTokens: 40000})
	if got := sharedPrefix(steps); got != 500 {
		t.Errorf("three hits gave %d", got)
	}
	if medianInt(nil, 7) != 7 || medianInt([]int{3, 1, 2}, 0) != 2 {
		t.Error("medianInt")
	}
	if got := chainTokens([]trace.Step{{NewTokens: 100}}, 0, 4096); got != 0 {
		t.Errorf("chain of a whole request = %d", got)
	}
}
