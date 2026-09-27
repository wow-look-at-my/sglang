package sim

import (
	"math"
	"math/rand"
	"sort"

	"schedsim/internal/kv"
	"schedsim/internal/trace"
)

// PolicyName selects the waiting-queue order shared by every mode.
type PolicyName string

const (
	// PolicyFCFS is the default --schedule-policy.
	PolicyFCFS PolicyName = "fcfs"
	// PolicySPF is shortest-prefill-first.
	PolicySPF PolicyName = "spf"
)

// Config is one run's knobs: the policy mode plus the deployment it runs on.
type Config struct {
	Mode   Mode
	Cost   Cost
	Policy PolicyName

	ChunkSize  int
	Page       int
	MaxRunning int
	// HostMul is the HiCache host tier as a multiple of the device pool.
	HostMul float64
	// Overlap is the overlap scheduler: batch N+1 is chosen and launched before
	// batch N's result is processed, so the balancer is charged at completion.
	Overlap bool
	// MixedChunk folds one decode row per running request into each prefill batch.
	MixedChunk bool

	Seed        int64
	HardStop    float64
	LogInterval int
}

// DefaultConfig returns the deployment the log describes, with the mode's own
// batch-composition rule.
func DefaultConfig(mode Mode, cost Cost) Config {
	return Config{
		Mode:        mode,
		Cost:        cost,
		Policy:      PolicyFCFS,
		ChunkSize:   cost.Cal.ChunkSize,
		Page:        cost.Cal.Page,
		MaxRunning:  6,
		HostMul:     4,
		Overlap:     true,
		MixedChunk:  FeaturesOf(mode).MixedChunk,
		LogInterval: trace.DecodeLogInterval,
	}
}

// Batch is one forward pass the scheduler launched.
type Batch struct {
	Seq       int
	IsPrefill bool
	// Items are the extend rows: one per request the batch prefills.
	Items []Item
	// Rows are the running requests a mixed batch decodes alongside its prefill.
	Rows []*Request
	// Decode is a pure decode batch's running requests.
	Decode []*Request
	// DecodeToks is the tokens each Decode request advanced, index-aligned.
	DecodeToks []int
	// ExtendTokens counts every extend token in the batch, decode rows included.
	ExtendTokens int
	// ReloadSeconds is the host-to-device copy the batch pays before its kernels.
	ReloadSeconds float64
	// NewSeq is how many requests the batch prefilled, as #new-seq reports it.
	NewSeq int
	// Continuation is the request whose chunk this batch continues, if any: its
	// own already-computed tokens are not a cache hit in the log's counting.
	Continuation *Request
	Start        float64
	End          float64
}

// Item is one request's contribution to an extend batch.
type Item struct {
	Req          *Request
	Extend       int
	PrefixBefore int
	Completes    bool
}

// DecodeLogLine is one decode stat line: the interval's tokens over its seconds.
type DecodeLogLine struct {
	T       float64
	Gen     float64
	Running int
}

// PrefillLogLine is one prefill stat line as the log would print it.
type PrefillLogLine struct {
	T      float64
	Tokens int
	Hit    int
	NewSeq int
	Queue  int
	Items  []Item
}

// ColdWindow bounds one cold prompt's prefill: arrival to first token, the
// stretch the decode-rate metrics are measured inside.
type ColdWindow struct {
	Tag      string
	Arrival  float64
	FirstTok float64
}

// Done reports whether the prompt ever reached its first token.
func (w ColdWindow) Done() bool { return w.FirstTok >= 0 }

// Result is one run: the trace every metric is computed from.
type Result struct {
	Cfg        Config
	Requests   []*Request
	Batches    []*Batch
	DecodeLog  []DecodeLogLine
	PrefillLog []PrefillLogLine
	// Windows lists every cold prompt's prefill stretch in arrival order; a
	// Request's Win field is its index here.
	Windows     []ColdWindow
	Pool        *kv.Pool
	Agents      int
	BusyPrefill float64
	BusyDecode  float64
	End         float64
	// Held counts admissions the eviction throttle turned back.
	Held int
}

// engine is the scheduler's state plus the event loop.
type engine struct {
	cfg    Config
	cost   Cost
	wl     Workload
	rng    *rand.Rand
	res    *Result
	pool   *kv.Pool
	bal    *Balancer
	th     *Throttle
	feat   Features
	shared int

	waiting []*Request
	chunked *Request
	running []*Request

	now       float64
	gpuFree   float64
	inflight  []*Batch
	future    []*Request
	decodeCt  int
	genSince  int
	lastLogT  float64
	cedeReq   *Request
	cedeBase  int
	cededToks int
}

// Run simulates one scenario's workload under one policy.
func Run(sc Scenario, cfg Config, seed int64) *Result {
	cfg.Seed = seed
	cfg.HardStop = sc.HardStop
	if sc.MaxRunning > 0 {
		cfg.MaxRunning = sc.MaxRunning
	}
	if sc.HostMul > 0 {
		cfg.HostMul = sc.HostMul
	}
	e := newEngine(cfg, sc.Build(seed, cfg.Cost))
	e.run()
	return e.res
}

// RunWorkload simulates an explicit workload, for a test that scripts arrivals.
func RunWorkload(cfg Config, wl Workload) *Result {
	e := newEngine(cfg, wl)
	e.run()
	return e.res
}

func newEngine(cfg Config, wl Workload) *engine {
	dev := cfg.Cost.DeviceTokens()
	shared := 0
	if m, ok := wl.(interface{ SharedPrefix() int }); ok {
		shared = m.SharedPrefix()
	}
	if shared > dev {
		shared = dev
	}
	host := int(float64(dev) * cfg.HostMul)
	pool := kv.New(maxI(dev-shared, 0), maxI(host-shared, 0))
	e := &engine{
		cfg: cfg, cost: cfg.Cost, wl: wl, shared: shared,
		rng:  rand.New(rand.NewSource(cfg.Seed*7919 + 13)),
		pool: pool,
		feat: FeaturesOf(cfg.Mode),
		res:  &Result{Cfg: cfg, Pool: pool, Agents: wl.Agents()},
	}
	e.bal = NewBalancer(e.feat, cfg.ChunkSize)
	if e.feat.Throttle {
		e.th = NewThrottle(dev, host, e.bal.PrefillSecondsPerToken)
	}
	// A conversation's context starts out cached wherever the workload says so:
	// an agent's context is the residue of the turns before this run began.
	if seeds := wlSeeds(wl); len(seeds) > 0 {
		ids := make([]int, 0, len(seeds))
		for id := range seeds {
			ids = append(ids, id)
		}
		sort.Ints(ids)
		for _, id := range ids {
			if n := seeds[id]; n > 0 {
				pool.Warm(id, n, 0)
			}
		}
	}
	e.add(wl.Initial())
	return e
}

// wlSeeds lists the conversations whose prefix is cached before the run starts.
func wlSeeds(wl Workload) map[int]int {
	if w, ok := wl.(interface{ WarmSeeds() map[int]int }); ok {
		return w.WarmSeeds()
	}
	return nil
}

func (e *engine) add(rs []*Request) {
	for _, r := range rs {
		if r.AcceptQ == 0 {
			e.drawAcceptance(r)
		}
		e.res.Requests = append(e.res.Requests, r)
		e.future = append(e.future, r)
	}
	sortRequests(e.future)
}

// drawAcceptance picks this request's per-draft acceptance rate from the lengths
// the log reported, so the speculative gain is the measured one.
func (e *engine) drawAcceptance(r *Request) {
	L := e.cost.AcceptLengthMean()
	if list := e.cost.AcceptList(); len(list) > 0 {
		L = list[e.rng.Intn(len(list))] * L / meanOf(list)
	}
	maxLen := float64(e.cost.NumDraft()) + 1
	L = clampF(L, 1.01, maxLen-0.01)
	r.AcceptQ = AcceptProb(L, e.cost.NumDraft())
}

// coldInFlight reports whether a cold prefill is queued or chunking, which is
// what marks a later arrival as having arrived during a cold prefill.
func (e *engine) coldInFlight() bool {
	if e.chunked != nil && e.chunked.Kind == KindCold {
		return true
	}
	for _, w := range e.waiting {
		if w.Kind == KindCold {
			return true
		}
	}
	return false
}

func (e *engine) ingest() {
	for len(e.future) > 0 && e.future[0].Arrival <= e.now {
		r := e.future[0]
		e.future = e.future[1:]
		r.ArrivedDuringCold = e.coldInFlight()
		if r.Kind == KindCold {
			r.Win = len(e.res.Windows)
			e.res.Windows = append(e.res.Windows, ColdWindow{Tag: r.Tag, Arrival: r.Arrival})
		}
		if e.th != nil {
			e.th.OnQueue(r.Conv, r.InputLen, e.now)
		}
		e.waiting = append(e.waiting, r)
	}
}

func (e *engine) run() {
	depth := 0
	if e.cfg.Overlap {
		depth = 1
	}
	for {
		if e.cfg.HardStop > 0 && e.now > e.cfg.HardStop {
			break
		}
		e.ingest()
		b := e.nextBatch()
		if b != nil {
			e.launch(b)
		}
		if len(e.inflight) > depth || (b == nil && len(e.inflight) > 0) {
			first := e.inflight[0]
			e.inflight = e.inflight[1:]
			e.finish(first)
			continue
		}
		if b != nil {
			continue
		}
		if len(e.future) == 0 {
			break
		}
		e.now = math.Max(e.now, e.future[0].Arrival)
	}
	e.res.End = e.now
}

// nextBatch is get_next_batch_to_run: the balancer's deferral decision, then
// prefill whenever one can be formed, decode only otherwise.
func (e *engine) nextBatch() *Batch {
	if e.feat.Balancer {
		prefillPending := len(e.waiting) > 0 || e.chunked != nil
		if !e.bal.ShouldDeferPrefill(prefillPending, len(e.running) > 0, e.chunked != nil) {
			if b := e.prefillBatch(true); b != nil {
				return b
			}
		}
	} else if b := e.prefillBatch(false); b != nil {
		return b
	}
	if len(e.running) > 0 {
		return &Batch{Decode: append([]*Request(nil), e.running...)}
	}
	return nil
}

// prefillBatch is get_new_batch_prefill: the chunked request's next chunk first,
// capped by the cede and the balancer's token budget, then the waiting queue in
// order until a request cannot be added.
func (e *engine) prefillBatch(policyOn bool) *Batch {
	if e.chunked == nil && len(e.waiting) == 0 {
		return nil
	}
	runningBs := len(e.running)
	if e.chunked == nil && runningBs >= e.cfg.MaxRunning {
		return nil
	}
	if e.cfg.Policy == PolicySPF {
		sort.SliceStable(e.waiting, func(i, j int) bool {
			return e.waiting[i].Work() < e.waiting[j].Work()
		})
	}
	// The cede compares uncached work, so every candidate's prefix match must be
	// current before it runs; match_prefix_for_req is the same refresh.
	chunkedWas := e.chunked
	for _, r := range e.waiting {
		devHit, hostHit := e.prefixHit(r)
		r.Prefix = e.sharedTokens(r) + devHit + hostHit
	}
	budget := e.cfg.ChunkSize
	if policyOn {
		if limit := e.bal.PrefillTokenBudget(); limit >= 0 && limit < budget {
			budget = limit
		}
	}
	var items []Item
	var reload float64
	if e.chunked != nil {
		limit := -1
		switch e.feat.Cede {
		case CedeHalf:
			limit = e.cedeHalf(e.chunked, budget, runningBs)
		case CedeFair:
			limit = e.cedeFair(e.chunked, budget, runningBs)
		}
		if limit < 0 && e.cfg.Policy == PolicySPF {
			limit = e.spfChunkLimit(e.chunked, budget)
		}
		rem := budget
		if limit >= 0 {
			rem = minI(rem, limit)
		}
		remaining := e.chunked.InputLen - e.chunked.Prefix
		n := minI(remaining, rem)
		if n > 0 {
			items = append(items, Item{
				Req: e.chunked, Extend: n, PrefixBefore: e.chunked.Prefix, Completes: n == remaining,
			})
			budget -= ceilPage(n, e.cfg.Page)
			if n == remaining {
				e.chunked = nil
			}
		}
	}
	hasChunked := e.chunked != nil
	kept := make([]bool, len(e.waiting))
	if e.th != nil {
		e.th.BeginPass()
	}
	for i, r := range e.waiting {
		if len(items) >= e.cfg.MaxRunning-runningBs || budget <= 0 {
			break
		}
		devHit, hostHit := e.prefixHit(r)
		own := r.InputLen - e.sharedTokens(r)
		r.Prefix = e.sharedTokens(r) + devHit + hostHit
		fits, wouldEvict := e.evictPlan(r, own, devHit)
		if !fits {
			break
		}
		deviceHit := minI(e.sharedTokens(r)+devHit, r.InputLen)
		// A held request is skipped rather than ending the scan: ShouldHold holds
		// back every later candidate that would also evict, and one that fits
		// without eviction takes its own place in the batch.
		if e.th != nil && e.th.ShouldHold(r.Conv, r.InputLen, deviceHit, own+r.OutTarget,
			wouldEvict, r.Arrival, e.now) {
			e.res.Held++
			continue
		}
		left := r.InputLen - r.Prefix
		work := left
		if work > budget {
			if hasChunked {
				break
			}
			end := (r.Prefix + budget/e.cfg.Page*e.cfg.Page) / e.cfg.Page * e.cfg.Page
			work = end - r.Prefix
			if work <= 0 {
				break
			}
		}
		items = append(items, Item{Req: r, Extend: work, PrefixBefore: r.Prefix, Completes: work == left})
		reload += float64(hostHit) * e.cost.ReloadSecondsPerToken()
		e.admit(r, own, devHit, hostHit, wouldEvict)
		if work < left {
			e.chunked = r
			hasChunked = true
		}
		budget -= ceilPage(work, e.cfg.Page)
		kept[i] = true
	}
	if len(items) == 0 {
		return nil
	}
	rest := make([]*Request, 0, len(e.waiting))
	for i, r := range e.waiting {
		if !kept[i] {
			rest = append(rest, r)
		}
	}
	e.waiting = rest
	b := &Batch{IsPrefill: true, Items: items, ReloadSeconds: reload}
	if chunkedWas != nil && items[0].Req == chunkedWas {
		b.Continuation = chunkedWas
	}
	for _, it := range items {
		b.NewSeq++
		b.ExtendTokens += it.Extend
	}
	if e.cfg.MixedChunk && runningBs > 0 {
		b.Rows = append([]*Request(nil), e.running...)
		b.ExtendTokens += len(b.Rows)
	}
	return b
}

// sharedTokens is the part of a request's input every conversation carries,
// which stays resident while conversation prefixes come and go.
func (e *engine) sharedTokens(r *Request) int { return minI(e.shared, r.InputLen) }

// prefixHit is match_prefix over the conversation's own prefix; the shared
// system prompt is resident regardless of what eviction did to the conversations.
func (e *engine) prefixHit(r *Request) (devHit, hostHit int) {
	own := r.InputLen - e.sharedTokens(r)
	if own <= 0 {
		return 0, 0
	}
	return e.pool.Lookup(r.Conv, own)
}

// evictPlan reports whether freeing the cached prefixes admission would have to
// reclaim is enough, and whether anything would go at all. The requesting
// conversation's own resident prefix is reused rather than re-reserved, so it
// counts as free.
func (e *engine) evictPlan(r *Request, own, devHit int) (fits, wouldEvict bool) {
	short := own + r.OutTarget - (e.pool.Free() + devHit)
	if short <= 0 {
		return true, false
	}
	got := 0
	for _, v := range e.pool.Victims(r.Conv, short) {
		got += v.DevLen
	}
	return got >= short, got > 0
}

// admit takes the request's reservation and records what its prefix match found.
// The pool re-derives the eviction set from its own state, which is the same set
// evictPlan listed for the throttle's decision.
func (e *engine) admit(r *Request, own, devHit, hostHit int, wouldEvict bool) {
	e.pool.CountRecompute(r.Conv, r.InputLen, minI(e.sharedTokens(r)+devHit, r.InputLen), hostHit)
	e.pool.Admit(r.Conv, own, r.OutTarget, devHit, hostHit, e.now)
	r.Reserved = own + r.OutTarget
	r.ReloadToks = hostHit
	if e.th != nil {
		e.th.OnAdmitted(wouldEvict, e.now)
	}
}

// launch runs a batch on the GPU: it starts when the GPU is free, and its
// requests advance by the end of the step.
func (e *engine) launch(b *Batch) {
	b.Seq = len(e.res.Batches)
	b.Start = math.Max(e.now, e.gpuFree)
	if b.IsPrefill {
		items := make([]trace.ExtendItem, 0, len(b.Items)+len(b.Rows))
		for _, it := range b.Items {
			items = append(items, trace.ExtendItem{
				Tokens: it.Extend, MidCtx: float64(it.PrefixBefore) + float64(it.Extend)/2,
			})
		}
		// A decode row is one more extend token in the same pass.
		for _, r := range b.Rows {
			items = append(items, trace.ExtendItem{Tokens: 1, MidCtx: float64(r.Context())})
		}
		b.End = b.Start + e.cost.ExtendSeconds(items) + b.ReloadSeconds
		for _, it := range b.Items {
			r := it.Req
			if r.PrefillStart < 0 {
				r.PrefillStart = b.Start
			}
			r.Prefix += it.Extend
			if it.Completes {
				// The prefill samples the first token; the request joins the
				// running batch at the next decision.
				r.OutDone = 1
				if r.OutDone > r.OutTarget {
					r.OutDone = r.OutTarget
				}
			}
		}
		e.advanceRows(b.Rows)
		e.res.BusyPrefill += b.End - b.Start
		e.bal.OnLaunch(true, b.ExtendTokens, len(b.Rows), e.now)
	} else {
		sumCtx := 0
		for _, r := range b.Decode {
			sumCtx += r.Context()
		}
		b.End = b.Start + e.cost.DecodeSeconds(len(b.Decode), sumCtx)
		b.DecodeToks = make([]int, len(b.Decode))
		keep := e.running[:0]
		for i, r := range b.Decode {
			n := e.stepTokens(r)
			b.DecodeToks[i] = n
			r.OutDone += n
			if r.OutDone >= r.OutTarget {
				r.OutDone = r.OutTarget
				continue
			}
			keep = append(keep, r)
		}
		e.running = keep
		e.res.BusyDecode += b.End - b.Start
		e.bal.OnLaunch(false, 0, 0, e.now)
	}
	e.gpuFree = b.End
	e.inflight = append(e.inflight, b)
	e.res.Batches = append(e.res.Batches, b)
}

// advanceRows steps the requests a mixed prefill batch decodes and drops the ones
// that ran out: speculative decoding degrades to a plain decode inside a mixed
// step, so every row is one token.
func (e *engine) advanceRows(rows []*Request) {
	if len(rows) == 0 {
		return
	}
	keep := e.running[:0]
	for _, r := range e.running {
		if r.OutDone < r.OutTarget {
			r.OutDone++
		}
		if r.OutDone < r.OutTarget {
			keep = append(keep, r)
		}
	}
	e.running = keep
}

// stepTokens draws one decode step's accepted tokens: the sampled token plus the
// drafts accepted in sequence.
func (e *engine) stepTokens(r *Request) int {
	n := 1
	for k := 0; k < e.cost.NumDraft(); k++ {
		if e.rng.Float64() >= r.AcceptQ {
			break
		}
		n++
	}
	if left := r.OutTarget - r.OutDone; n > left {
		n = left
	}
	if n < 1 {
		n = 1
	}
	return n
}

// finish processes a batch's result. This is where the overlap scheduler has
// reached, and where the balancer is charged for the batch it launched.
func (e *engine) finish(b *Batch) {
	e.now = math.Max(e.now, b.End)
	if b.IsPrefill {
		tok, hit := 0, 0
		for _, it := range b.Items {
			tok += it.Extend
			// #cached-token counts what a newly added request matched; the
			// continuation's own chunks are its computed tokens, not a cache hit.
			if it.Req != b.Continuation {
				hit += it.PrefixBefore
			}
			if !it.Completes {
				continue
			}
			r := it.Req
			e.firstToken(r, b.End)
			if r.Finish < 0 {
				e.running = append(e.running, r)
			}
		}
		for _, r := range b.Rows {
			e.deliver(r, b.End, 1)
			e.settle(r, b.End)
		}
		e.res.PrefillLog = append(e.res.PrefillLog, PrefillLogLine{
			T: b.End, Tokens: tok, Hit: hit, NewSeq: b.NewSeq, Queue: len(e.waiting), Items: b.Items,
		})
	} else {
		for i, r := range b.Decode {
			n := b.DecodeToks[i]
			if n <= 0 {
				continue
			}
			e.deliver(r, b.End, n)
			e.genSince += n
			e.settle(r, b.End)
		}
		e.decodeCt++
		interval := maxI(e.cfg.LogInterval, 1)
		if e.decodeCt%interval == 0 {
			e.res.DecodeLog = append(e.res.DecodeLog, DecodeLogLine{
				T: b.End, Gen: float64(e.genSince) / math.Max(b.End-e.lastLogT, 1e-9), Running: len(b.Decode),
			})
			e.genSince = 0
			e.lastLogT = b.End
		}
	}
	e.bal.OnFinish(e.now)
}

// firstToken records a prefilled request's first token, which the prefill batch
// itself sampled, and closes the cold window it opened.
func (e *engine) firstToken(r *Request, at float64) {
	if r.FirstTok >= 0 {
		return
	}
	r.FirstTok = at
	e.deliver(r, at, 1)
	if r.Win >= 0 && r.Win < len(e.res.Windows) {
		e.res.Windows[r.Win].FirstTok = at
	}
	e.settle(r, at)
}

func (e *engine) deliver(r *Request, at float64, n int) {
	if n <= 0 {
		return
	}
	r.Deliveries = append(r.Deliveries, Delivery{T: at, N: n})
}

// settle closes a request that has streamed its whole answer.
func (e *engine) settle(r *Request, at float64) {
	if r.Finish >= 0 || r.OutDone < r.OutTarget {
		return
	}
	r.Finish = at
	ctx := r.Context()
	own := ctx - minI(e.shared, ctx)
	e.pool.Release(r.Conv, own, at, r.Reserved)
	r.Reserved = 0
	if e.th != nil {
		e.th.OnFinish(r.Conv, ctx, at)
	}
	e.add(e.wl.OnFinish(r, at))
}

// cedeHalf is SchedulePolicy.cede_chunk_budget outside shortest-prefill-first:
// the chunked request keeps at least half of every chunk.
func (e *engine) cedeHalf(chunked *Request, budget, runningBs int) int {
	page := e.cfg.Page
	if budget < 2*page || len(e.waiting) == 0 {
		return -1
	}
	maxReserved := budget / 2 / page * page
	if e.cfg.Policy == PolicySPF {
		maxReserved = budget - page
	}
	return e.reserveCeded(chunked, budget, maxReserved, e.cfg.MaxRunning-runningBs-1)
}

// cedeFair is the revised cede: it amortizes the half over the chunked request's
// progress since it first ceded, so a chunk it ran alone banks room for a
// follow-up of up to a whole chunk in one step.
func (e *engine) cedeFair(chunked *Request, budget, runningBs int) int {
	page := e.cfg.Page
	if budget < 2*page || len(e.waiting) == 0 {
		return -1
	}
	if e.cedeReq != chunked {
		e.cedeReq = chunked
		e.cedeBase = chunked.Prefix
		e.cededToks = 0
	}
	maxReserved := budget - page
	if e.cfg.Policy != PolicySPF {
		progress := chunked.Prefix - e.cedeBase
		fair := (progress - e.cededToks + budget) / 2 / page * page
		maxReserved = minI(maxReserved, fair)
	}
	return e.reserveCeded(chunked, budget, maxReserved, e.cfg.MaxRunning-runningBs-1)
}

// reserveCeded moves every waiting request with less uncached work than the
// chunked request has left ahead of the queue and reserves budget for it; the
// result is what the chunked request may still take, -1 for the whole budget.
func (e *engine) reserveCeded(chunked *Request, budget, maxReserved, maxNewReqs int) int {
	page := e.cfg.Page
	remaining := chunked.InputLen - chunked.Prefix
	var shorter []*Request
	reserved := 0
	for _, r := range e.waiting {
		if maxNewReqs >= 0 && len(shorter) >= maxNewReqs {
			break
		}
		if reserved+page > maxReserved {
			break
		}
		work := r.Work()
		charge := ceilPage(work, page)
		if work < remaining && reserved+charge <= maxReserved && e.admissible(r) {
			shorter = append(shorter, r)
			reserved += charge
		}
	}
	if reserved == 0 {
		return -1
	}
	e.cededToks += reserved
	chosen := map[int]bool{}
	for _, r := range shorter {
		chosen[r.ID] = true
	}
	nq := append([]*Request(nil), shorter...)
	for _, r := range e.waiting {
		if !chosen[r.ID] {
			nq = append(nq, r)
		}
	}
	e.waiting = nq
	return (budget - reserved) / page * page
}

// admissible is the cede's memory predicate: a reservation the pool cannot
// honour is budget the chunk would waste.
func (e *engine) admissible(r *Request) bool {
	own := r.InputLen - e.sharedTokens(r)
	devHit, _ := e.prefixHit(r)
	fits, _ := e.evictPlan(r, own, devHit)
	return fits
}

// spfChunkLimit is the pre-balancer shortest-prefill-first chunk cap, read by
// the sweep's SPF variant.
func (e *engine) spfChunkLimit(chunked *Request, budget int) int {
	page := e.cfg.Page
	if budget < 2*page {
		return -1
	}
	remaining := chunked.InputLen - chunked.Prefix
	reserved := 0
	for _, r := range e.waiting {
		work := r.Work()
		charge := ceilPage(work, page)
		if work >= remaining || reserved+charge > budget-page {
			break
		}
		reserved += charge
	}
	if reserved == 0 {
		return -1
	}
	return (budget - reserved) / page * page
}

func ceilPage(n, page int) int {
	if page <= 1 {
		return n
	}
	return ceilDiv(n, page) * page
}

func ceilDiv(a, b int) int { return (a + b - 1) / b }

func meanOf(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}
