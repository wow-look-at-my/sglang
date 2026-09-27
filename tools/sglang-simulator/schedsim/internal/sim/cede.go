package sim

import "github.com/wow-look-at-my/go-containers/set"

// cedeScanLimit mirrors _CEDE_SCAN_LIMIT in schedule_policy.py.
const cedeScanLimit = 128

type cedeLedger struct {
	rid        int
	basePrefix int
	ceded      int
}

func (e *Engine) shortestWork(r *Request) int {
	dev, host := e.match(r)
	return max(1, r.target-tokensOf(dev)-tokensOf(host))
}

// cedeBudget caps the chunked request's next chunk so shorter waiting
// requests prefill beside it. It returns the cap and whether one applies.
func (e *Engine) cedeBudget(budget, remTotal int) (int, bool) {
	page := e.cfg.PageSize
	switch {
	case e.cfg.Policy == Old && !e.cfg.ShortestFirst:
		return 0, false
	case budget < 2*page || len(e.waiting) == 0:
		return 0, false
	}
	r := e.chunked
	remaining := r.target - r.done
	maxNewReqs := -1
	var maxReserved int
	switch {
	case e.cfg.ShortestFirst:
		maxReserved = budget - page
	case e.cfg.Policy == New:
		maxNewReqs = e.cfg.MaxRunning - len(e.running) - 1
		if maxNewReqs <= 0 {
			return 0, false
		}
		e.syncCedeLedger(r)
		progress := r.done - e.cede.basePrefix
		fair := (progress - e.cede.ceded + budget) / 2
		maxReserved = min(budget-page, pageDown(fair, page))
	default:
		maxReserved = pageDown(budget/2, page)
	}
	if e.cfg.Policy == New && e.cfg.ShortestFirst {
		maxNewReqs = e.cfg.MaxRunning - len(e.running) - 1
		if maxNewReqs <= 0 {
			return 0, false
		}
		e.syncCedeLedger(r)
	}
	var shorter []*Request
	reserved := 0
	for i, w := range e.waiting {
		if i >= cedeScanLimit || reserved+page > maxReserved || len(shorter) == maxNewReqs {
			break
		}
		work := e.shortestWork(w)
		charge := ceilDiv(work, page) * page
		admissible := e.cfg.Policy != New || e.admissionTokens(w) < remTotal
		if work < remaining && reserved+charge <= maxReserved && admissible {
			shorter = append(shorter, w)
			reserved += charge
		}
	}
	if reserved == 0 {
		return 0, false
	}
	if e.cfg.Policy == New {
		e.cede.ceded += reserved
	}
	chosen := set.New[*Request]()
	for _, w := range shorter {
		chosen.Add(w)
	}
	reordered := append([]*Request(nil), shorter...)
	for _, w := range e.waiting {
		if !chosen.Contains(w) {
			reordered = append(reordered, w)
		}
	}
	e.waiting = reordered
	return pageDown(budget-reserved, page), true
}

func (e *Engine) syncCedeLedger(r *Request) {
	if r.ID != e.cede.rid || r.done < e.cede.basePrefix {
		e.cede = cedeLedger{rid: r.ID, basePrefix: r.done}
	}
}
