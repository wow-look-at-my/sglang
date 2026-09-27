package sim

import "math"

type extendItem struct {
	req       *Request
	start, n  int
	completes bool
}

type batch struct {
	isPrefill bool
	items     []extendItem
	rows      []*Request
	reload    int
	tokens    int
	start     float64
	end       float64
	gen       map[*Request]float64
}

func pageDown(n, page int) int { return n / page * page }

func ceilDiv(a, b int) int { return (a + b - 1) / b }

func (e *Engine) clipNew(r *Request) int {
	left := r.MaxNew - int(r.generated)
	return max(0, min(left, e.cfg.ClipMaxNew))
}

// runningReserve is what PrefillAdder holds back for the running requests'
// future decode tokens.
func (e *Engine) runningReserve() int {
	total := 0.0
	for _, r := range e.running {
		total += float64(e.clipNew(r)) * e.cfg.NewTokenRatio
	}
	return int(total)
}

// match refreshes a waiting request's prefix match, as init_next_round_input
// and match_prefix_for_req do.
func (e *Engine) match(r *Request) (dev, host []*Segment) {
	dev, host = e.Cache.Match(r.chain)
	r.prefixIdx = tokensOf(dev)
	return dev, host
}

func (e *Engine) admissionTokens(r *Request) int {
	return r.target - r.prefixIdx + e.clipNew(r) + e.cfg.PageSize
}

func (e *Engine) prefillBatch(runnable []*Request) *batch {
	if e.chunked == nil && (len(e.waiting) == 0 || len(e.running) >= e.cfg.MaxRunning) {
		return nil
	}
	chunk := e.cfg.ChunkSize
	var rows []*Request
	if e.mixed() && !e.chunkFollows(chunk) {
		rows = runnable
	}
	b := &batch{isPrefill: true, rows: rows}
	remChunk := chunk - len(rows)
	reserve := e.runningReserve()
	remTotal := e.Cache.Free() + e.Cache.Evictable() - reserve
	remNoEvict := e.Cache.Free() - reserve

	hasChunked := false
	if e.chunked != nil {
		r := e.chunked
		limit, limited := e.cedeBudget(remChunk)
		n := min(r.target-r.done, remChunk, remTotal)
		if limited {
			n = min(n, limit)
		}
		completes := n >= r.target-r.done
		if !completes {
			n = pageDown(n, e.cfg.PageSize)
		}
		if n > 0 && e.Cache.Alloc(n) {
			b.items = append(b.items, extendItem{req: r, start: r.done, n: n, completes: completes})
			r.done += n
			r.own += n
			remChunk -= n
			remTotal -= n
			remNoEvict -= n
			if completes {
				// add_chunked_req reserves the output of the chunk that finishes the prompt.
				remTotal -= e.clipNew(r)
				remNoEvict -= e.clipNew(r)
				e.chunked = nil
			} else {
				hasChunked = true
			}
		} else {
			hasChunked = true
		}
	}

	if e.thr != nil {
		e.thr.beginPass(e.presentRids())
	}
	admitted := 0
	var kept []*Request
	stop := false
	for i, r := range e.waiting {
		if stop {
			kept = append(kept, e.waiting[i:]...)
			break
		}
		if admitted >= e.cfg.MaxRunning-len(e.running) {
			kept = append(kept, e.waiting[i:]...)
			break
		}
		wouldEvict := false
		if e.thr != nil {
			e.match(r)
			total := e.admissionTokens(r)
			wouldEvict = total >= remNoEvict
			if e.thr.shouldHold(r.ID, r.InputLen(), r.prefixIdx, total, wouldEvict, r.queuedAt, e.now) {
				kept = append(kept, r)
				continue
			}
		}
		dev, host := e.match(r)
		res := e.addOne(b, r, dev, host, &remChunk, &remTotal, &remNoEvict, hasChunked)
		switch res {
		case addFull, addChunked:
			admitted++
			if res == addChunked {
				hasChunked = true
			}
			if e.thr != nil {
				e.thr.onAdmitted(wouldEvict, e.now)
			}
		default:
			kept = append(kept, r)
			stop = true
		}
	}
	e.waiting = kept
	if len(b.items) == 0 {
		return nil
	}
	return b
}

// chunkFollows mirrors PrefillDecodeBalancer.chunk_follows: the next chunk
// launches before this batch is charged, so it carries the decode rows instead.
func (e *Engine) chunkFollows(chunk int) bool {
	return e.nb != nil && e.nb.burstContinues(e.cfg.Overlap, e.chunked != nil && e.chunked.target-e.chunked.done > chunk)
}

const (
	addFull = iota
	addChunked
	addStop
)

func (e *Engine) addOne(b *batch, r *Request, dev, host []*Segment, remChunk, remTotal, remNoEvict *int, hasChunked bool) int {
	devLen, hostLen := tokensOf(dev), tokensOf(host)
	total := r.target - devLen + e.clipNew(r) + e.cfg.PageSize
	// add_one_req pins the prefix before the check, so it is not evictable room.
	pinned := 0
	for _, s := range dev {
		if s.lock == 0 {
			pinned += s.Tokens
		}
	}
	if total >= *remTotal-pinned {
		return addStop
	}
	extend := r.target - devLen - hostLen
	n, full := extend, true
	if extend > *remChunk {
		if *remChunk <= 0 || hasChunked {
			return addStop
		}
		n, full = pageDown(*remChunk, e.cfg.PageSize), false
		if n <= 0 {
			return addStop
		}
	}
	e.Cache.Lock(dev, e.now)
	if len(host) > 0 {
		if !e.Cache.LoadBack(host) {
			e.Cache.Unlock(dev, e.now)
			return addStop
		}
		e.Cache.Lock(host, e.now)
		b.reload += hostLen
	}
	if !e.Cache.Alloc(n) {
		e.Cache.Unlock(append(append([]*Segment(nil), dev...), host...), e.now)
		return addStop
	}
	r.locked = append(append([]*Segment(nil), dev...), host...)
	r.rest = r.chain[len(r.locked):]
	r.done = devLen + hostLen
	if !r.matched {
		r.matched = true
		e.Rec.matched(r, devLen+hostLen)
	}
	b.items = append(b.items, extendItem{req: r, start: r.done, n: n, completes: full})
	r.done += n
	r.own += n
	*remChunk -= n
	consumed := n
	if full {
		consumed += e.clipNew(r)
	}
	*remTotal -= consumed + pinned
	*remNoEvict -= consumed
	if !full {
		e.chunked = r
		return addChunked
	}
	return addFull
}

func (e *Engine) presentRids() map[int]bool {
	present := map[int]bool{}
	for _, r := range e.waiting {
		present[r.ID] = true
	}
	for _, r := range e.running {
		present[r.ID] = true
	}
	if e.chunked != nil {
		present[e.chunked.ID] = true
	}
	return present
}

func (e *Engine) decodeBatch(runnable []*Request) *batch {
	if len(runnable) == 0 {
		return nil
	}
	rows := runnable
	for {
		need := 0
		for _, r := range rows {
			need += stepTokens(r, e.cfg.Cost.AcceptLen)
		}
		if e.Cache.Alloc(need) {
			for _, r := range rows {
				r.own += stepTokens(r, e.cfg.Cost.AcceptLen)
			}
			return &batch{rows: rows}
		}
		if !e.retractOne() {
			return nil
		}
		if rows = e.decodable(); len(rows) == 0 {
			return nil
		}
	}
}

// stepTokens is how many whole tokens one step adds to r's KV.
func stepTokens(r *Request, perStep float64) int {
	g := math.Min(perStep, r.remainingOutput())
	before := r.generated + r.inFlight
	return int(math.Floor(before+g)) - int(math.Floor(before))
}

// retractOne sends the newest idle running request back to the queue, as
// update_running_batch does when decode runs out of memory.
func (e *Engine) retractOne() bool {
	for i := len(e.running) - 1; i >= 0; i-- {
		r := e.running[i]
		if r.inFlight > 0 {
			continue
		}
		e.running = append(e.running[:i], e.running[i+1:]...)
		e.Cache.Release(r.own)
		e.Cache.Unlock(r.locked, e.now)
		r.own, r.locked, r.rest = 0, nil, nil
		r.target = r.InputLen() + int(r.generated)
		r.done = 0
		r.retracted = true
		e.waiting = append(e.waiting, r)
		sortFCFS(e.waiting)
		e.Rec.Retractions++
		return true
	}
	return false
}

func (e *Engine) launch(b *batch) {
	m := e.cfg.Cost
	b.gen = map[*Request]float64{}
	if b.isPrefill {
		compute := 0.0
		for _, it := range b.items {
			compute += m.ExtendSeconds(it.start, it.n)
			b.tokens += it.n
		}
		for _, r := range b.rows {
			compute += m.ExtendSeconds(r.ctx(), 1)
			if e.Cache.Alloc(stepTokens(r, 1)) {
				r.own += stepTokens(r, 1)
			}
			b.gen[r] = math.Min(1, r.remainingOutput())
		}
		b.tokens += len(b.rows)
		b.start = math.Max(e.now, e.gpuFree())
		b.end = b.start + m.PrefillBatchSeconds(compute, b.reload)
		for _, it := range b.items {
			if it.completes {
				e.running = append(e.running, it.req)
				b.gen[it.req] = 1
			}
		}
	} else {
		ctx := 0
		for _, r := range b.rows {
			ctx += r.ctx()
			b.gen[r] = math.Min(m.AcceptLen, r.remainingOutput())
		}
		b.start = math.Max(e.now, e.gpuFree())
		b.end = b.start + m.DecodeSeconds(len(b.rows), ctx)
	}
	for r, g := range b.gen {
		r.inFlight += g
	}
	e.lastEnd = b.end
	if e.bal != nil {
		e.bal.onLaunched(e.now, b.isPrefill, b.tokens, len(b.rows))
	}
	e.Rec.launched(b)
}

func (e *Engine) gpuFree() float64 { return e.lastEnd }

func (e *Engine) process(b *batch, w Workload) {
	if e.bal != nil {
		e.bal.onFinished(b.end)
	}
	for _, r := range orderedGen(b) {
		g := b.gen[r]
		r.inFlight -= g
		r.generated += g
		e.Rec.delivered(r, b.end, g)
		if r.generated >= float64(r.OutputLen)-1e-9 && r.inFlight <= 1e-9 {
			e.finish(r, w)
		}
	}
}

// orderedGen returns the batch's requests in a fixed order so runs repeat.
func orderedGen(b *batch) []*Request {
	out := make([]*Request, 0, len(b.gen))
	for r := range b.gen {
		out = append(out, r)
	}
	sortByID(out)
	return out
}

func sortByID(rs []*Request) {
	for i := 1; i < len(rs); i++ {
		for j := i; j > 0 && rs[j].ID < rs[j-1].ID; j-- {
			rs[j], rs[j-1] = rs[j-1], rs[j]
		}
	}
}

func (e *Engine) finish(r *Request, w Workload) {
	for i, x := range e.running {
		if x == r {
			e.running = append(e.running[:i], e.running[i+1:]...)
			break
		}
	}
	r.finish = e.now
	conv := r.Conv
	seg := &Segment{Tokens: r.NewTokens + int(r.generated), Parent: lastSeg(r.chain)}
	segs := append(append([]*Segment(nil), r.rest...), seg)
	e.Cache.Insert(segs, r.own, e.now)
	e.Cache.Unlock(r.locked, e.now)
	r.own = 0
	conv.Chain = append(append([]*Segment(nil), r.chain...), seg)
	conv.Len = r.prefixLen + seg.Tokens
	conv.Finished++
	conv.LastFinish = e.now
	if e.thr != nil {
		e.thr.ledger.onFinished(r.ID, conv.Len, e.now)
	}
	e.Rec.finished(r)
	w.OnFinish(e, r)
}

func lastSeg(chain []*Segment) *Segment {
	if len(chain) == 0 {
		return nil
	}
	return chain[len(chain)-1]
}
