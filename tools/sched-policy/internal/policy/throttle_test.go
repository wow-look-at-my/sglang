package policy

import "testing"

const (
	deviceTokens    = 1000
	secondsPerToken = 1e-3
)

func throttle(hostTokens int) *Throttle {
	return NewThrottle(deviceTokens, hostTokens, func() float64 { return secondsPerToken })
}

func seq(from, n int) []int32 {
	out := make([]int32, n)
	for i := range out {
		out[i] = int32(from + i)
	}
	return out
}

// serveTurn has a conversation turn arrive, run, and finish with ctx tokens.
func serveTurn(th *Throttle, c *clock, rid string, ctx []int32) {
	th.OnRequestQueued(rid, ctx[:len(ctx)-10], c.now)
	c.now += 1
	th.OnRequestFinished(rid, len(ctx), ctx[len(ctx)-TailTokens:], c.now)
}

type holdOpts struct {
	deviceHit  int
	wouldEvict bool
	queuedAt   float64
}

func hold(th *Throttle, c *clock, rid string, inputLen int, o holdOpts) bool {
	return th.ShouldHold(rid, inputLen, o.deviceHit, inputLen+50, o.wouldEvict, o.queuedAt, c.now)
}

func twoLive(th *Throttle, c *clock) (a, b []int32) {
	a, b = seq(0, 400), seq(10000, 400)
	serveTurn(th, c, "a1", a)
	serveTurn(th, c, "b1", b)
	th.OnRequestQueued("a2", append(append([]int32{}, a...), seq(7, 20)...), c.now)
	return a, b
}

func TestNewConversationIsHeldWhenItWouldEvictALiveOne(t *testing.T) {
	c := &clock{now: 100}
	th := throttle(0)
	twoLive(th, c)
	th.OnRequestQueued("c1", seq(50000, 300), c.now)
	th.BeginPass([]string{"a2", "c1"})
	if !hold(th, c, "c1", 300, holdOpts{wouldEvict: true, queuedAt: c.now}) {
		t.Fatal("a thrashing admission was not held")
	}
}

func TestAdmittedOnceItFitsWhatHostMemoryCanRestore(t *testing.T) {
	c := &clock{now: 100}
	th := throttle(4 * deviceTokens)
	twoLive(th, c)
	th.OnRequestQueued("c1", seq(50000, 300), c.now)
	th.BeginPass([]string{"a2", "c1"})
	if hold(th, c, "c1", 300, holdOpts{wouldEvict: true, queuedAt: c.now}) {
		t.Fatal("held although the host tier keeps every prefix")
	}
}

func TestNeverHoldsWithoutEvictionOrForAResidentConversation(t *testing.T) {
	c := &clock{now: 100}
	th := throttle(0)
	twoLive(th, c)
	th.BeginPass([]string{"a2"})
	if hold(th, c, "a2", 420, holdOpts{queuedAt: c.now}) {
		t.Fatal("held without eviction")
	}
	if hold(th, c, "a2", 420, holdOpts{deviceHit: 410, wouldEvict: true, queuedAt: c.now}) {
		t.Fatal("held a resident conversation")
	}
}

func TestIdleConversationPastTheLongestReturnGapIsNotLive(t *testing.T) {
	c := &clock{now: 100}
	th := throttle(0)
	twoLive(th, c)
	c.now += 5
	th.OnRequestQueued("c1", seq(50000, 300), c.now)
	th.BeginPass([]string{"a2", "c1"})
	if hold(th, c, "c1", 300, holdOpts{wouldEvict: true, queuedAt: c.now}) {
		t.Fatal("an idle conversation counted as live")
	}
}

func TestOnlyTheOldestHeldRequestAgesInToEvict(t *testing.T) {
	c := &clock{now: 100}
	th := throttle(0)
	twoLive(th, c)
	th.OnRequestQueued("c1", seq(50000, 300), c.now)
	th.OnRequestQueued("d1", seq(60000, 300), c.now)
	queuedAt := c.now
	rebuild := deviceTokens * secondsPerToken
	evict := func(rid string) bool {
		return hold(th, c, rid, 300, holdOpts{wouldEvict: true, queuedAt: queuedAt})
	}

	c.now = queuedAt + 0.5*rebuild
	th.BeginPass([]string{"a2", "c1", "d1"})
	if !evict("c1") || !evict("d1") {
		t.Fatal("evicted before the pool rebuild time")
	}
	c.now = queuedAt + rebuild
	th.BeginPass([]string{"a2", "c1", "d1"})
	if evict("c1") || !evict("d1") {
		t.Fatal("FIFO aging broke")
	}
	th.OnAdmitted(true, c.now)
	th.BeginPass([]string{"a2", "d1"})
	if !evict("d1") {
		t.Fatal("a second eviction did not wait a full rebuild")
	}
}

func TestAbortedRequestStopsKeepingItsConversationLive(t *testing.T) {
	c := &clock{now: 100}
	th := throttle(0)
	twoLive(th, c)
	c.now += 5
	th.OnRequestQueued("c1", seq(50000, 600), c.now)
	th.BeginPass([]string{"a2", "c1"})
	if !hold(th, c, "c1", 600, holdOpts{wouldEvict: true, queuedAt: c.now}) {
		t.Fatal("not held with a2 queued")
	}
	th.BeginPass([]string{"c1"})
	if hold(th, c, "c1", 600, holdOpts{wouldEvict: true, queuedAt: c.now}) {
		t.Fatal("an aborted request kept its conversation live")
	}
}

func TestLedgerRecognizesTheLongestReturningContextAndPrunes(t *testing.T) {
	c := &clock{now: 100}
	th := throttle(0)
	short, long := seq(0, 200), seq(0, 400)
	serveTurn(th, c, "s", short)
	serveTurn(th, c, "l", long)
	th.OnRequestQueued("r", seq(0, 450), c.now)
	th.OnRequestQueued("r", seq(0, 450), c.now)
	conv := th.ledger.ConversationOf("r")
	if conv == nil || conv.length != 450 {
		t.Fatalf("returning conversation %+v", conv)
	}
	th.OnRequestFinished("unknown", 10, seq(0, 10), c.now)
	// More 400-token conversations exceed what the pool retains.
	for i, base := range []int{20000, 30000, 40000} {
		serveTurn(th, c, string(rune('x'+i)), seq(base, 400))
	}
	if n := len(th.ledger.conversations); n > 3 {
		t.Fatalf("%d conversations retained for a %d-token pool", n, deviceTokens)
	}
}
