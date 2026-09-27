package sim

import (
	"math"
	"sort"
)

// The eviction throttle mirrors python/sglang/srt/managers/
// scheduler_components/eviction_throttle.py on master.

type ledgerConv struct {
	length       int
	lastFinish   float64
	finishedOnce bool
	active       map[int]bool
}

type ledger struct {
	retained int
	byRid    map[int]*ledgerConv
	index    map[*Conv]*ledgerConv
	convs    []*ledgerConv
	owner    map[*ledgerConv]*Conv
	maxGap   float64
}

func newLedger(retained int) *ledger {
	return &ledger{retained: retained, byRid: map[int]*ledgerConv{},
		index: map[*Conv]*ledgerConv{}, owner: map[*ledgerConv]*Conv{}}
}

func (l *ledger) onQueued(rid int, conv *Conv, tokens int, now float64) {
	if _, ok := l.byRid[rid]; ok {
		return
	}
	lc := l.index[conv]
	if lc != nil && lc.length > tokens {
		lc = nil
	}
	if lc == nil {
		lc = &ledgerConv{length: tokens, active: map[int]bool{}}
		l.convs = append(l.convs, lc)
	} else if len(lc.active) == 0 {
		l.maxGap = math.Max(l.maxGap, now-lc.lastFinish)
	}
	lc.active[rid] = true
	lc.length = max(lc.length, tokens)
	l.byRid[rid] = lc
	l.owner[lc] = conv
}

func (l *ledger) onFinished(rid int, length int, now float64) {
	lc, ok := l.byRid[rid]
	if !ok {
		return
	}
	delete(l.byRid, rid)
	delete(lc.active, rid)
	l.unindex(lc)
	lc.length = length
	lc.lastFinish = now
	lc.finishedOnce = true
	l.index[l.owner[lc]] = lc
	l.prune()
}

func (l *ledger) dropAbsent(present map[int]bool) {
	for rid, lc := range l.byRid {
		if !present[rid] {
			delete(l.byRid, rid)
			delete(lc.active, rid)
		}
	}
}

func (l *ledger) isLive(lc *ledgerConv, now float64) bool {
	return len(lc.active) > 0 || now-lc.lastFinish <= l.maxGap
}

func (l *ledger) liveTokens(now float64, exclude *ledgerConv) int {
	n := 0
	for _, lc := range l.convs {
		if lc != exclude && l.isLive(lc, now) {
			n += lc.length
		}
	}
	return n
}

func (l *ledger) unindex(lc *ledgerConv) {
	if conv := l.owner[lc]; conv != nil && l.index[conv] == lc {
		delete(l.index, conv)
	}
}

func (l *ledger) prune() {
	sorted := append([]*ledgerConv(nil), l.convs...)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].lastFinish > sorted[j].lastFinish })
	kept, total := sorted[:0:0], 0
	for _, lc := range sorted {
		total += lc.length
		if len(lc.active) > 0 || total <= l.retained {
			kept = append(kept, lc)
		} else {
			l.unindex(lc)
		}
	}
	l.convs = kept
}

const (
	verdictAdmit = iota
	verdictAdmitAged
	verdictHold
)

type throttle struct {
	deviceTokens      int
	capacity          int
	secondsPerToken   func() float64
	ledger            *ledger
	lastEvictingAdmit float64
	headTaken         bool
}

func newThrottle(deviceTokens, hostTokens int, spt func() float64) *throttle {
	capacity := max(deviceTokens, hostTokens)
	return &throttle{deviceTokens: deviceTokens, capacity: capacity, secondsPerToken: spt,
		ledger: newLedger(capacity), lastEvictingAdmit: math.Inf(-1)}
}

func (t *throttle) beginPass(present map[int]bool) {
	t.headTaken = false
	t.ledger.dropAbsent(present)
}

func (t *throttle) shouldHold(rid, inputLen, deviceHit, totalTokens int, wouldEvict bool, queuedAt, now float64) bool {
	if !wouldEvict || 2*deviceHit >= inputLen {
		return false
	}
	if t.headTaken {
		return true
	}
	verdict := t.localVerdict(rid, totalTokens, queuedAt, now)
	t.headTaken = verdict != verdictAdmit
	return verdict == verdictHold
}

func (t *throttle) onAdmitted(evicted bool, now float64) {
	if evicted {
		t.lastEvictingAdmit = now
	}
}

func (t *throttle) localVerdict(rid, totalTokens int, queuedAt, now float64) int {
	lc := t.ledger.byRid[rid]
	if t.ledger.liveTokens(now, lc)+totalTokens <= t.capacity {
		return verdictAdmit
	}
	waited := now - math.Max(queuedAt, t.lastEvictingAdmit)
	if waited < float64(t.deviceTokens)*t.secondsPerToken() {
		return verdictHold
	}
	return verdictAdmitAged
}
