package sim

// Balancer is PrefillDecodeBalancer (prefill_decode_balancer.py), shared by
// both balancing modes: Prev selects the balance floored at zero with no
// token bound and no piggyback credit, and the default selects HEAD's rules.
type Balancer struct {
	Prev        bool
	BurstTokens int
	// PiggybackCredit prices a mixed batch's decode rows out of its charge.
	PiggybackCredit bool

	Debt        float64
	UnsettledPf float64
	UnsettledDc float64
	LastDecode  float64
	UnsettledN  int
	BurstUsed   int
	// ExtendSecond and ExtendTokens are the running measurement behind prefill_seconds_per_token.
	ExtendSecond float64
	ExtendTokens int
	// LastPrefillSecond and LastPrefillTokens are the marginal rate of the last prefill batch.
	LastPrefillSecond float64
	LastPrefillTokens int

	inFlight []launched
	busy     float64
}

type launched struct {
	IsPrefill bool
	Tokens    int
	Rows      int
}

// NewBalancer builds the controller a mode selects. chunkSize is
// chunked_prefill_size, used as the stall bound where the mode has one.
func NewBalancer(f Features, chunkSize int) *Balancer {
	b := &Balancer{Prev: f.Cede == CedeHalf}
	if f.BurstTokens {
		b.BurstTokens = chunkSize
	}
	b.PiggybackCredit = f.PiggybackCredit
	return b
}

func (b *Balancer) PrefillSecondsPerToken() float64 {
	if b.ExtendTokens == 0 {
		return 0
	}
	return b.ExtendSecond / float64(b.ExtendTokens)
}

// marginalSecondsPerToken is the last prefill batch's own cost per token,
// which is the rate the next chunk of the same prompt faces.
func (b *Balancer) marginalSecondsPerToken() float64 {
	if b.LastPrefillTokens == 0 {
		return 0
	}
	return b.LastPrefillSecond / float64(b.LastPrefillTokens)
}

// ShouldDeferPrefill is should_defer_prefill. Without contention the balance is
// cleared and nothing is deferred; continuesChunk says the pending prefill work
// is the next chunk of a request already being chunked.
func (b *Balancer) ShouldDeferPrefill(prefillPending, decodeRunnable, continuesChunk bool) bool {
	if !prefillPending || !decodeRunnable {
		b.Debt = 0
		b.UnsettledPf, b.UnsettledDc = 0, 0
		b.UnsettledN = 0
		b.BurstUsed = 0
		return false
	}
	if b.UnsettledN > 0 {
		if b.Prev {
			// Decode never banks credit: the balance stops at zero.
			b.Debt = maxF(b.Debt+b.UnsettledPf-b.UnsettledDc, 0)
		} else {
			// At most the last decode batch overshoots into the next decision.
			b.Debt = maxF(b.Debt+b.UnsettledPf-b.UnsettledDc, -b.LastDecode)
		}
		b.UnsettledPf, b.UnsettledDc = 0, 0
		b.UnsettledN = 0
	}
	if b.Debt <= 0 && !b.prefillInFlight() {
		b.BurstUsed = 0
	}
	if b.BurstTokens == 0 {
		return b.Debt > 0
	}
	if continuesChunk {
		return b.BurstUsed > 0
	}
	return b.BurstUsed >= b.BurstTokens
}

func (b *Balancer) PrefillTokenBudget(continuesChunk bool) int {
	if b.BurstTokens == 0 {
		return -1
	}
	average := b.PrefillSecondsPerToken()
	marginal := b.marginalSecondsPerToken()
	if continuesChunk && average > 0 && marginal > 0 {
		return maxI(0, int(float64(b.BurstTokens)*average/marginal))
	}
	if b.BurstUsed == 0 {
		return -1
	}
	return b.BurstTokens - b.BurstUsed
}

// OnLaunch records a launched batch, starting the busy clock when the GPU was
// idle. tokens are the batch's extend tokens, rows the requests a mixed batch
// decodes alongside them.
func (b *Balancer) OnLaunch(isPrefill bool, tokens, rows int, now float64) {
	if len(b.inFlight) == 0 {
		b.busy = now
	}
	b.inFlight = append(b.inFlight, launched{IsPrefill: isPrefill, Tokens: tokens, Rows: rows})
	if isPrefill {
		b.BurstUsed += tokens
	}
}

// OnFinish charges the oldest launched batch the elapsed time, from the
// completion rather than from its own launch.
func (b *Balancer) OnFinish(now float64) {
	if len(b.inFlight) == 0 {
		return
	}
	el := now - b.busy
	b.busy = now
	f := b.inFlight[0]
	b.inFlight = b.inFlight[1:]
	piggy := 0.0
	if f.Rows > 0 && b.PiggybackCredit {
		piggy = minF(float64(f.Rows)*b.PrefillSecondsPerToken(), el)
	}
	if f.IsPrefill && f.Tokens > 0 {
		b.ExtendSecond += el
		b.ExtendTokens += f.Tokens
		b.LastPrefillSecond, b.LastPrefillTokens = el, f.Tokens
	}
	if f.IsPrefill {
		b.UnsettledPf += el - piggy
	} else {
		b.UnsettledDc += el
		b.LastDecode = el
	}
	b.UnsettledN++
}

func (b *Balancer) prefillInFlight() bool {
	for _, f := range b.inFlight {
		if f.IsPrefill {
			return true
		}
	}
	return false
}

func maxF(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func minF(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func maxI(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minI(a, b int) int {
	if a < b {
		return a
	}
	return b
}
