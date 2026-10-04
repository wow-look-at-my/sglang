// Package kv models the device KV cache and the HiCache host tier the
// scheduler admits against: one cached prefix per conversation, whole-prefix
// eviction to host, and the reload or recompute an evicted prefix costs.
package kv

import "sort"

// ReloadSecondsPerToken is what a host-resident prefix costs to bring back per token.
const ReloadSecondsPerToken = 0.5e-6

// Entry is one conversation's cached prefix, the radix leaf a returning turn
// matches against.
type Entry struct {
	Conv    int
	DevLen  int
	HostLen int
	LastUse float64
}

// Pool is the cache the scheduler admits against.
type Pool struct {
	DevCap  int
	HostCap int

	entries   map[int]*Entry
	inflight  int // tokens reserved by admitted, unfinished requests
	devCached int // device tokens held by cached prefixes no request owns
	hostUsed  int
	seen      map[int]bool

	// Recomputes counts, per conversation, the turns that found less than half of their input cached after the conversation had been served once.
	Recomputes map[int]int
	// ReloadTokens and RecomputeTokens are the prefix tokens brought back from host and recomputed instead.
	ReloadTokens    int
	RecomputeTokens int
	EvictedTokens   int
}

func New(devCap, hostCap int) *Pool {
	return &Pool{
		DevCap:     devCap,
		HostCap:    hostCap,
		entries:    map[int]*Entry{},
		seen:       map[int]bool{},
		Recomputes: map[int]int{},
	}
}

// Free is the device tokens admission may still take.
func (p *Pool) Free() int { return p.DevCap - p.inflight - p.devCached }

// Capacity is what the cache can keep across a turn boundary: write-through
// mirrors every cached prefix on host.
func (p *Pool) Capacity() int {
	if p.HostCap > p.DevCap {
		return p.HostCap
	}
	return p.DevCap
}

// Warm seeds a conversation's prefix as an earlier turn left it.
func (p *Pool) Warm(conv, tokens int, now float64) {
	e := &Entry{Conv: conv, LastUse: now}
	switch {
	case tokens <= p.Free():
		e.DevLen = tokens
		p.devCached += tokens
	case p.HostCap-p.hostUsed >= tokens:
		e.HostLen = tokens
		p.hostUsed += tokens
	default:
		return
	}
	p.entries[conv] = e
	p.seen[conv] = true
}

// Lookup is match_prefix: a device hit if the prefix is resident, otherwise a
// host hit to load back.
func (p *Pool) Lookup(conv, inputLen int) (devHit, hostHit int) {
	e := p.entries[conv]
	if e == nil {
		return 0, 0
	}
	if e.DevLen > 0 {
		return min(e.DevLen, inputLen-1), 0
	}
	return 0, min(e.HostLen, inputLen-1)
}

// Victims lists, least recently used first, the cached device prefixes that
// must go to free short tokens, excluding the requesting conversation's own.
func (p *Pool) Victims(conv, short int) []*Entry {
	var cand []*Entry
	for _, e := range p.entries {
		if e.DevLen > 0 && e.Conv != conv {
			cand = append(cand, e)
		}
	}
	sort.Slice(cand, func(i, j int) bool {
		if cand[i].LastUse != cand[j].LastUse {
			return cand[i].LastUse < cand[j].LastUse
		}
		return cand[i].Conv < cand[j].Conv
	})
	var out []*Entry
	got := 0
	for _, e := range cand {
		if got >= short {
			break
		}
		out = append(out, e)
		got += e.DevLen
	}
	return out
}

// WouldEvict reports whether admitting a reservation of tokens needs a cached
// prefix of somebody else's to go.
func (p *Pool) WouldEvict(conv, tokens int) bool {
	if tokens <= p.Free() {
		return false
	}
	return len(p.Victims(conv, tokens-p.Free())) > 0
}

// hostDrops lists host copies (LRU first, excluding conv) to discard so tokens
// more can be written to host.
func (p *Pool) hostDrops(conv, tokens int) []*Entry {
	short := tokens - (p.HostCap - p.hostUsed)
	if short <= 0 {
		return nil
	}
	var cand []*Entry
	for _, e := range p.entries {
		if e.HostLen > 0 && e.DevLen == 0 && e.Conv != conv {
			cand = append(cand, e)
		}
	}
	sort.Slice(cand, func(i, j int) bool {
		if cand[i].LastUse != cand[j].LastUse {
			return cand[i].LastUse < cand[j].LastUse
		}
		return cand[i].Conv < cand[j].Conv
	})
	var out []*Entry
	got := 0
	for _, e := range cand {
		if got >= short {
			break
		}
		out = append(out, e)
		got += e.HostLen
	}
	return out
}

// Eviction is one prefix's passage out of the device cache, snapshotted
// before it happens: `ToHost` says whether a copy survived there.
type Eviction struct {
	Conv   int
	Tokens int
	ToHost bool
}

// Evict moves one device prefix to host, dropping colder host copies to make
// room. A prefix larger than the whole host tier is lost outright, and so is
// each host copy dropped for it; both come back as evictions the caller prices.
func (p *Pool) Evict(e *Entry) []Eviction {
	var out []Eviction
	p.devCached -= e.DevLen
	p.EvictedTokens += e.DevLen
	toHost := p.HostCap > 0 && e.DevLen <= p.HostCap
	if toHost {
		add := e.DevLen - e.HostLen
		for _, d := range p.hostDrops(e.Conv, add) {
			p.hostUsed -= d.HostLen
			out = append(out, Eviction{Conv: d.Conv, Tokens: d.HostLen})
			d.HostLen = 0
			if d.DevLen == 0 {
				delete(p.entries, d.Conv)
			}
		}
		p.hostUsed += add
		e.HostLen = e.DevLen
	} else {
		p.hostUsed -= e.HostLen
		e.HostLen = 0
	}
	out = append(out, Eviction{Conv: e.Conv, Tokens: e.DevLen, ToHost: toHost})
	e.DevLen = 0
	if e.HostLen == 0 {
		delete(p.entries, e.Conv)
	}
	return out
}

// Admit takes ownership of a request's cached prefix and reserves its
// footprint, evicting LRU prefixes the way the PrefillAdder's allocation would.
// It returns the prefixes it evicted, so the caller can price what that costs
// to rebuild.
func (p *Pool) Admit(conv, inputLen, outTarget, devHit, hostHit int, now float64) []Eviction {
	if e := p.entries[conv]; e != nil && e.DevLen > 0 {
		p.devCached -= e.DevLen
		e.DevLen = 0
		if e.HostLen == 0 {
			delete(p.entries, conv)
		}
	}
	reserved := inputLen + outTarget
	var evicted []Eviction
	if short := reserved - p.Free(); short > 0 {
		for _, v := range p.Victims(conv, short) {
			evicted = append(evicted, p.Evict(v)...)
		}
	}
	p.inflight += reserved
	if e := p.entries[conv]; e != nil {
		e.LastUse = now
	}
	p.seen[conv] = true
	if hostHit > 0 {
		p.ReloadTokens += hostHit
	}
	return evicted
}

// CountRecompute records what a request found on arrival: a conversation that
// had been served before and now matches less than half of its input is
// paying for its whole prefix again.
func (p *Pool) CountRecompute(conv, inputLen, devHit, hostHit int) {
	if !p.seen[conv] {
		return
	}
	if 2*(devHit+hostHit) < inputLen {
		p.Recomputes[conv]++
		p.RecomputeTokens += inputLen - devHit - hostHit
	}
}

// Release returns a request's reservation and caches its context as its
// conversation's prefix.
func (p *Pool) Release(conv, ctx int, now float64, reserved int) {
	p.inflight -= reserved
	e := p.entries[conv]
	if e == nil {
		e = &Entry{Conv: conv}
		p.entries[conv] = e
	}
	e.DevLen = min(ctx, max(p.Free(), 0))
	p.devCached += e.DevLen
	e.LastUse = now
	if e.DevLen == 0 && e.HostLen == 0 {
		delete(p.entries, conv)
	}
	p.seen[conv] = true
}

// Owned returns a request its own reservation back at launch time, when it
// stops being a cached prefix and becomes the request's working set.
func (p *Pool) Owned(conv, tokens int) {
	if e := p.entries[conv]; e != nil && e.DevLen > 0 {
		p.devCached -= e.DevLen
		e.DevLen = 0
		if e.HostLen == 0 {
			delete(p.entries, conv)
		}
	}
	p.inflight += tokens
}

// RebuildSeconds is the cost of bringing evicted prefixes back: a reload where
// a copy landed on host, a recompute where it did not. recomputeSecPerTok is
// the measured prefill rate at the conversation's context.
func (p *Pool) RebuildSeconds(evicted []Eviction, recomputeSecPerTok float64) float64 {
	t := 0.0
	for _, e := range evicted {
		if e.ToHost {
			t += float64(e.Tokens) * ReloadSecondsPerToken
		} else {
			t += float64(e.Tokens) * recomputeSecPerTok
		}
	}
	return t
}

// CachedConversations reports how many prefixes are resident on device.
func (p *Pool) CachedConversations() int {
	n := 0
	for _, e := range p.entries {
		if e.DevLen > 0 {
			n++
		}
	}
	return n
}
