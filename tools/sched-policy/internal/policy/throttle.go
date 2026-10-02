package policy

import (
	"github.com/wow-look-at-my/go-containers/set"
	"math"
	"slices"
	"sort"
)

// Throttle holds a conversation back instead of letting it thrash the prefix
// cache.
//
// Admission's capacity predicate is the allocator's free tokens plus the
// tree's evictable tokens. What admission can take is the cached prefix of a
// conversation between turns. When the conversations in use need more than
// the cache can keep, admitting one evicts another, which on its next turn
// evicts another, and every turn reloads or recomputes its prefix.
//
// The throttle keeps a ledger of the conversations served, recognizes a
// returning one by the tail of its previous context, and holds a request back
// when admitting it would evict while the live conversations plus it do not
// fit what the cache can keep:
//
//   - A conversation is live while a request of it is queued or running, or
//     while its idle time is within the longest gap between one of its turns
//     finishing and the next arriving observed so far.
//   - What the cache can keep is the device pool, or the HiCache host tier
//     when larger.
//   - A request whose own context is mostly resident on device is never held.
//   - Held requests keep FIFO order. The oldest may evict once it has waited,
//     since its arrival or the previous evicting admission, as long as
//     recomputing the whole device pool takes at the measured prefill rate.
type Throttle struct {
	deviceTokens           int
	capacity               int
	prefillSecondsPerToken func() float64
	ledger                 *Ledger
	lastEvictingAdmit      float64
	headTaken              bool
}

// TailTokens is how many trailing tokens recognize a returning conversation.
// Arbitrary; long enough that unrelated contexts do not collide.
const TailTokens = 64

// Verdict is the throttle's decision for one request.
type Verdict int

const (
	Admit Verdict = iota
	AdmitAged
	Hold
)

func NewThrottle(deviceTokens, hostTokens int, prefillSecondsPerToken func() float64) *Throttle {
	capacity := max(deviceTokens, hostTokens)
	return &Throttle{
		deviceTokens:           deviceTokens,
		capacity:               capacity,
		prefillSecondsPerToken: prefillSecondsPerToken,
		ledger:                 NewLedger(capacity),
		lastEvictingAdmit:      math.Inf(-1),
	}
}

func (t *Throttle) OnRequestQueued(rid string, tokens []int32, now float64) {
	t.ledger.OnQueued(rid, tokens, now)
}

// OnRequestFinished takes the request's full context length and the last
// TailTokens of it.
func (t *Throttle) OnRequestFinished(rid string, length int, tail []int32, now float64) {
	t.ledger.OnFinished(rid, length, tail, now)
}

// BeginPass starts one scan of the waiting queue. present lists the requests
// queued, running or in flight; any other request the ledger holds active
// left without finishing (an abort) and is let go.
func (t *Throttle) BeginPass(present []string) {
	t.headTaken = false
	t.ledger.DropAbsent(present)
}

// ShouldHold decides one waiting request. totalTokens is what admission
// reserves for it; queuedAt its wait-queue entry time.
func (t *Throttle) ShouldHold(rid string, inputLen, deviceHit, totalTokens int, wouldEvict bool, queuedAt, now float64) bool {
	if t.headTaken {
		return true
	}
	verdict := Admit
	// Half the input separates a conversation's own resident context from a
	// hit on nothing but a shared system prompt.
	if wouldEvict && 2*deviceHit < inputLen {
		verdict = t.localVerdict(rid, totalTokens, queuedAt, now)
	}
	t.headTaken = verdict != Admit
	return verdict == Hold
}

func (t *Throttle) OnAdmitted(evicted bool, now float64) {
	if evicted {
		t.lastEvictingAdmit = now
	}
}

func (t *Throttle) localVerdict(rid string, totalTokens int, queuedAt, now float64) Verdict {
	conv := t.ledger.ConversationOf(rid)
	if t.ledger.LiveTokens(now, conv)+totalTokens <= t.capacity {
		return Admit
	}
	waited := now - max(queuedAt, t.lastEvictingAdmit)
	if waited < float64(t.deviceTokens)*t.prefillSecondsPerToken() {
		return Hold
	}
	return AdmitAged
}

type conversation struct {
	length       int
	tail         string
	lastFinish   float64
	finishedOnce bool
	activeRids   map[string]bool
}

// Ledger is the conversations the scheduler has served, independent of what
// is cached.
type Ledger struct {
	retainedTokens int
	byRid          map[string]*conversation
	// length -> tail -> conversation, for the returning-conversation lookup.
	index         map[int]map[string]*conversation
	conversations []*conversation
	MaxGap        float64
}

func NewLedger(retainedTokens int) *Ledger {
	return &Ledger{
		retainedTokens: retainedTokens,
		byRid:          map[string]*conversation{},
		index:          map[int]map[string]*conversation{},
	}
}

func (l *Ledger) ConversationOf(rid string) *conversation { return l.byRid[rid] }

func (l *Ledger) OnQueued(rid string, tokens []int32, now float64) {
	if _, requeued := l.byRid[rid]; requeued {
		return
	}
	conv := l.findReturning(tokens)
	if conv == nil {
		conv = &conversation{length: len(tokens), tail: tailKey(tokens, len(tokens)), activeRids: map[string]bool{}}
		l.conversations = append(l.conversations, conv)
	} else if len(conv.activeRids) == 0 {
		l.MaxGap = max(l.MaxGap, now-conv.lastFinish)
	}
	conv.activeRids[rid] = true
	conv.length = max(conv.length, len(tokens))
	l.byRid[rid] = conv
}

func (l *Ledger) OnFinished(rid string, length int, tail []int32, now float64) {
	conv, ok := l.byRid[rid]
	if !ok {
		return
	}
	delete(l.byRid, rid)
	delete(conv.activeRids, rid)
	l.unindex(conv)
	conv.length = length
	conv.tail = tailKey(tail, len(tail))
	conv.lastFinish = now
	conv.finishedOnce = true
	byTail, ok := l.index[conv.length]
	if !ok {
		byTail = map[string]*conversation{}
		l.index[conv.length] = byTail
	}
	byTail[conv.tail] = conv
	l.prune()
}

func (l *Ledger) DropAbsent(present []string) {
	keep := set.New[string]()
	for _, rid := range present {
		keep.Add(rid)
	}
	for rid, conv := range l.byRid {
		if !keep.Contains(rid) {
			delete(l.byRid, rid)
			delete(conv.activeRids, rid)
		}
	}
}

func (l *Ledger) isLive(conv *conversation, now float64) bool {
	return len(conv.activeRids) > 0 || now-conv.lastFinish <= l.MaxGap
}

// LiveTokens sums the live conversations other than exclude.
func (l *Ledger) LiveTokens(now float64, exclude *conversation) int {
	total := 0
	for _, conv := range l.conversations {
		if conv != exclude && l.isLive(conv, now) {
			total += conv.length
		}
	}
	return total
}

func (l *Ledger) findReturning(tokens []int32) *conversation {
	var best *conversation
	for length, byTail := range l.index {
		if length > len(tokens) || (best != nil && length <= best.length) {
			continue
		}
		if conv, ok := byTail[tailKey(tokens, length)]; ok {
			best = conv
		}
	}
	return best
}

func (l *Ledger) unindex(conv *conversation) {
	byTail, ok := l.index[conv.length]
	if !ok || byTail[conv.tail] != conv {
		return
	}
	delete(byTail, conv.tail)
	if len(byTail) == 0 {
		delete(l.index, conv.length)
	}
}

func (l *Ledger) prune() {
	sorted := slices.Clone(l.conversations)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].lastFinish > sorted[j].lastFinish })
	kept, total := sorted[:0:0], 0
	for _, conv := range sorted {
		total += conv.length
		if len(conv.activeRids) > 0 || total <= l.retainedTokens {
			kept = append(kept, conv)
		} else {
			l.unindex(conv)
		}
	}
	l.conversations = kept
}

func tailKey(tokens []int32, length int) string {
	start := max(0, length-TailTokens)
	b := make([]byte, 0, 4*(length-start))
	for _, t := range tokens[start:length] {
		b = append(b, byte(t), byte(t>>8), byte(t>>16), byte(t>>24))
	}
	return string(b)
}
