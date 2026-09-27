package sim

import (
	"container/heap"
	"math"
	"sort"

	"schedsim/internal/cost"
)

// Policy selects the scheduler under test.
type Policy int

const (
	// Old is upstream before this work: prefill always wins the step.
	Old Policy = iota
	// Prev is the fixed half-and-half time balancer with the half-chunk cede.
	Prev
	// New is the scheduler on master.
	New
)

func (p Policy) String() string {
	return [...]string{"OLD", "PREV", "NEW"}[p]
}

// Config is one deployment.
type Config struct {
	Policy Policy
	Cost   cost.Model

	ChunkSize  int
	PageSize   int
	MaxRunning int
	HostTokens int
	// Overlap is the overlap scheduler, on by default.
	Overlap bool
	// NoMixedChunk opts NEW out of mixed chunked prefill, which it resolves on.
	NoMixedChunk bool
	// ShortestFirst selects shortest-prefill-first instead of FCFS.
	ShortestFirst bool
	NewTokenRatio float64
	ClipMaxNew    int
}

// Workload feeds requests into a run.
type Workload interface {
	Start(e *Engine)
	OnFinish(e *Engine, r *Request)
}

const (
	idlePoll   = 0.01
	idleGiveUp = 3600.0
)

// Engine runs one policy over one workload.
type Engine struct {
	cfg   Config
	Cache *Cache
	now   float64

	arrivals arrivalHeap
	waiting  []*Request
	running  []*Request
	chunked  *Request
	nextID   int
	lastEnd  float64

	bal balancer
	nb  *newBalancer
	thr *throttle

	Rec *Recorder
}

// NewEngine builds an engine; the workload seeds it in Run.
func NewEngine(cfg Config) *Engine {
	e := &Engine{cfg: cfg, Cache: NewCache(cfg.Cost.DevicePoolTokens, cfg.HostTokens), Rec: newRecorder()}
	switch cfg.Policy {
	case Prev:
		e.bal = &prevBalancer{}
	case New:
		nb := &newBalancer{}
		e.bal, e.nb = nb, nb
		e.thr = newThrottle(cfg.Cost.DevicePoolTokens, cfg.HostTokens, nb.secondsPerToken)
	}
	return e
}

// Now is the simulated clock.
func (e *Engine) Now() float64 { return e.now }

func (e *Engine) mixed() bool { return e.cfg.Policy == New && !e.cfg.NoMixedChunk }

// Submit queues a request to arrive at r.Arrival.
func (e *Engine) Submit(r *Request) {
	r.ID = e.nextID
	e.nextID++
	heap.Push(&e.arrivals, r)
}

// Run simulates until horizon, or until the workload drains when horizon is
// infinite.
func (e *Engine) Run(w Workload, horizon float64) {
	w.Start(e)
	var prev *batch
	idleSince := math.Inf(1)
	for e.now < horizon {
		e.admitArrivals()
		b := e.decide()
		if b != nil {
			e.launch(b)
		}
		hadPrev := prev != nil
		if e.cfg.Overlap {
			if prev != nil {
				e.now = math.Max(e.now, prev.end)
				e.process(prev, w)
			}
			prev = b
		} else if b != nil {
			e.now = b.end
			e.process(b, w)
		}
		if b != nil || hadPrev {
			idleSince = math.Inf(1)
			continue
		}
		if len(e.waiting) == 0 && len(e.running) == 0 && e.chunked == nil {
			if len(e.arrivals) == 0 {
				break
			}
			e.now = math.Max(e.now, e.arrivals[0].Arrival)
			continue
		}
		// Queued work that cannot start yet: the loop keeps polling.
		idleSince = math.Min(idleSince, e.now)
		if e.now-idleSince > idleGiveUp {
			e.Rec.Stuck = true
			break
		}
		e.now += idlePoll
	}
	if prev != nil && prev.end <= horizon {
		e.now = prev.end
		e.process(prev, w)
	}
	e.Rec.End = math.Min(e.now, horizon)
	if math.IsInf(horizon, 1) {
		e.Rec.End = e.now
		e.Rec.Drained = !e.Rec.Stuck
	}
}

func (e *Engine) admitArrivals() {
	for len(e.arrivals) > 0 && e.arrivals[0].Arrival <= e.now {
		r := heap.Pop(&e.arrivals).(*Request)
		r.chain = append([]*Segment(nil), r.Conv.Chain...)
		r.prefixLen = r.Conv.Len
		r.target = r.InputLen()
		r.queuedAt = r.Arrival
		e.waiting = append(e.waiting, r)
		e.Rec.arrived(r)
		if e.thr != nil {
			e.thr.ledger.onQueued(r.ID, r.Conv, r.InputLen(), e.now)
		}
	}
}

// decodable are the running requests the next decode step advances.
func (e *Engine) decodable() []*Request {
	var out []*Request
	for _, r := range e.running {
		if r.remainingOutput() > 1e-9 {
			out = append(out, r)
		}
	}
	return out
}

func (e *Engine) decide() *batch {
	runnable := e.decodable()
	if e.nb != nil {
		e.nb.target = 0
		if r := e.chunked; r != nil {
			c := e.cfg.ChunkSize
			last := r.target - c
			e.nb.target = 2 * (e.cfg.Cost.BatchOverhead + e.cfg.Cost.ExtendSeconds(last, c))
			if !e.cfg.Overlap {
				e.nb.target /= 2
			}
		}
	}
	if e.bal != nil {
		pending := len(e.waiting) > 0 || e.chunked != nil
		if e.bal.shouldDefer(pending, len(runnable) > 0) {
			return e.decodeBatch(runnable)
		}
	}
	if b := e.prefillBatch(runnable); b != nil {
		return b
	}
	return e.decodeBatch(runnable)
}

// arrivalHeap orders requests by arrival time, then submission order.
type arrivalHeap []*Request

func (h arrivalHeap) Len() int { return len(h) }
func (h arrivalHeap) Less(i, j int) bool {
	if h[i].Arrival != h[j].Arrival {
		return h[i].Arrival < h[j].Arrival
	}
	return h[i].ID < h[j].ID
}
func (h arrivalHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *arrivalHeap) Push(x any)   { *h = append(*h, x.(*Request)) }
func (h *arrivalHeap) Pop() any {
	old := *h
	r := old[len(old)-1]
	*h = old[:len(old)-1]
	return r
}

func sortFCFS(rs []*Request) {
	sort.SliceStable(rs, func(i, j int) bool { return rs[i].queuedAt < rs[j].queuedAt })
}
