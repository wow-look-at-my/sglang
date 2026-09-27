package sim

// cedeScanLimit mirrors _CEDE_SCAN_LIMIT in schedule_policy.py.
const cedeScanLimit = 128

func (e *Engine) shortestWork(r *Request) int {
	dev, host := e.match(r)
	return max(1, r.target-tokensOf(dev)-tokensOf(host))
}

// cedeBudget caps the chunked request's next chunk so shorter waiting
// requests prefill beside it. It returns the cap and whether one applies.
func (e *Engine) cedeBudget(budget int) (int, bool) {
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
	if e.cfg.Policy == New {
		maxNewReqs = e.cfg.MaxRunning - len(e.running) - 1
		if maxNewReqs <= 0 {
			return 0, false
		}
	}
	maxReserved := pageDown(budget/2, page)
	if e.cfg.ShortestFirst {
		maxReserved = budget - page
	}
	var shorter []*Request
	chosen := make([]bool, len(e.waiting))
	reserved := 0
	for i, w := range e.waiting {
		if i >= cedeScanLimit || reserved+page > maxReserved || len(shorter) == maxNewReqs {
			break
		}
		work := e.shortestWork(w)
		charge := ceilDiv(work, page) * page
		if work < remaining && reserved+charge <= maxReserved {
			shorter = append(shorter, w)
			chosen[i] = true
			reserved += charge
		}
	}
	if reserved == 0 {
		return 0, false
	}
	reordered := append([]*Request(nil), shorter...)
	for i, w := range e.waiting {
		if !chosen[i] {
			reordered = append(reordered, w)
		}
	}
	e.waiting = reordered
	return pageDown(budget-reserved, page), true
}
