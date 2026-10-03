package sim

import "sort"

// Throttle is EvictionThrottle (scheduler_components/eviction_throttle.py):
// it holds a request back when admitting it would evict a cached prefix.
type Throttle struct {
	// Capacity is the larger cache tier: write-through mirrors every cached prefix on host.
	Capacity int
	// DeviceTokens bounds the aging wait.
	DeviceTokens int
	// PerToken is the measured prefill rate that turns tokens into seconds.
	PerToken func() float64

	convs     map[int]*ledConv
	order     []int
	maxGap    float64
	lastEvict float64
	headTaken bool
	retained  int
}

type ledConv struct {
	id           int
	tokens       int
	lastDone     float64
	finishedOnce bool
	active       int
}

// NewThrottle returns a throttle for a cache with the given tiers. perToken is
// the balancer's measured prefill cost, the same rate the stall bound prices.
func NewThrottle(deviceTokens, hostTokens int, perToken func() float64) *Throttle {
	capacity := maxI(deviceTokens, hostTokens)
	return &Throttle{
		Capacity:     capacity,
		DeviceTokens: deviceTokens,
		PerToken:     perToken,
		convs:        map[int]*ledConv{},
		lastEvict:    -1e18,
		retained:     capacity,
	}
}

// OnQueue registers a request of a conversation as live from its arrival. The
// gap between one turn finishing and the next arriving defines "live" for every
// conversation, so a longer turn model widens the held-back set.
func (t *Throttle) OnQueue(id, tokens int, now float64) {
	c := t.convs[id]
	if c == nil {
		c = &ledConv{id: id}
		t.convs[id] = c
		t.order = append(t.order, id)
	} else if c.active == 0 && c.finishedOnce {
		t.maxGap = maxF(t.maxGap, now-c.lastDone)
	}
	c.active++
	c.tokens = maxI(c.tokens, tokens)
}

// OnFinish records the conversation's new full context length.
func (t *Throttle) OnFinish(id, tokens int, now float64) {
	c := t.convs[id]
	if c == nil {
		c = &ledConv{id: id}
		t.convs[id] = c
		t.order = append(t.order, id)
	}
	c.active = maxI(c.active-1, 0)
	c.tokens = tokens
	c.lastDone = now
	c.finishedOnce = true
	t.prune()
}

// isLive is ConversationLedger.is_live.
func (t *Throttle) isLive(c *ledConv, now float64) bool {
	return c.active > 0 || (c.finishedOnce && now-c.lastDone <= t.maxGap)
}

// liveTokens is ConversationLedger.live_tokens.
func (t *Throttle) liveTokens(exclude int, now float64) int {
	sum := 0
	for _, id := range t.order {
		c := t.convs[id]
		if id != exclude && t.isLive(c, now) {
			sum += c.tokens
		}
	}
	return sum
}

// prune is ConversationLedger._prune: the ledger keeps the most recent
// conversations up to what the cache can hold, since anything beyond that would
// miss anyway.
func (t *Throttle) prune() {
	ids := append([]int(nil), t.order...)
	sort.SliceStable(ids, func(i, j int) bool {
		a, b := t.convs[ids[i]], t.convs[ids[j]]
		if a.lastDone != b.lastDone {
			return a.lastDone > b.lastDone
		}
		return ids[i] < ids[j]
	})
	kept := make([]int, 0, len(ids))
	total := 0
	for _, id := range ids {
		c := t.convs[id]
		total += c.tokens
		if c.active > 0 || total <= t.retained {
			kept = append(kept, id)
			continue
		}
		delete(t.convs, id)
	}
	t.order = kept
}

// BeginPass starts one scan of the waiting queue.
func (t *Throttle) BeginPass() { t.headTaken = false }

// ShouldHold is should_hold: a request whose own conversation is already
// resident on device is never held, only the first eviction-triggering
// candidate of a pass is adjudicated, and a held head goes through once it has
// waited longer than rebuilding the device pool would take.
func (t *Throttle) ShouldHold(id, inputLen, deviceHit, totalTokens int, wouldEvict bool, queuedAt, now float64) bool {
	// Half the input separates a conversation's own resident context from a hit
	// on nothing but a shared system prompt.
	if !wouldEvict || 2*deviceHit >= inputLen {
		return false
	}
	if t.headTaken {
		return true
	}
	// A plain admit leaves the head unclaimed, so the scan keeps adjudicating.
	if t.liveTokens(id, now)+totalTokens <= t.Capacity {
		return false
	}
	t.headTaken = true
	waited := now - maxF(t.lastEvict, queuedAt)
	return waited < float64(t.DeviceTokens)*t.PerToken()
}

// OnAdmitted remembers an admission that displaced cached prefixes; the aging
// clock for the next head restarts there.
func (t *Throttle) OnAdmitted(evicted bool, now float64) {
	if evicted {
		t.lastEvict = now
	}
}
