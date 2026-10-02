// Package policy holds the scheduling policies every TP rank must decide
// identically.
package policy

// Balancer shares GPU time between prefill and decode while both have work.
//
// It measures how long each batch occupied the GPU, from the previous
// completion (or its own launch, if the GPU was idle) to its own completion,
// and keeps a running balance, debt, accumulated only while both classes
// contend (a prefill is pending and running requests can decode):
//
//   - Share. Prefill batches and decode batches get equal GPU time; only
//     batches that actually ran are charged.
//   - Stall bound. Between points where decode has caught up, at most
//     BurstTokens prefill tokens are launched. A chunked prompt's next chunk
//     is capped in seconds rather than tokens, since its cost per token rises
//     with the context attention reads.
//   - Piggybacked decode rows in a mixed batch are charged as prefill minus
//     their marginal cost.
//   - Decode banks at most the last decode batch's time.
type Balancer struct {
	// burstTokens is nil when chunked prefill is off: defer on debt alone.
	burstTokens *int

	debt              float64
	unsettledPrefill  float64
	unsettledDecode   float64
	lastDecode        float64
	unsettledBatches  int
	inFlight          []launched
	busySince         float64
	burstUsed         int
	extendSeconds     float64
	extendTokens      int
	lastPrefillSecs   float64
	lastPrefillTokens int
}

// BatchClass is a batch's role in the balance.
type BatchClass int

const (
	ClassOther BatchClass = iota
	ClassPrefill
	ClassDecode
)

type launched struct {
	class      BatchClass
	tokens     int
	decodeRows int
}

func NewBalancer(burstTokens *int) *Balancer {
	return &Balancer{burstTokens: burstTokens}
}

func (b *Balancer) Debt() float64 { return b.debt }

func (b *Balancer) PrefillSecondsPerToken() float64 {
	if b.extendTokens == 0 {
		return 0
	}
	return b.extendSeconds / float64(b.extendTokens)
}

func (b *Balancer) MarginalPrefillSecondsPerToken() float64 {
	if b.lastPrefillTokens == 0 {
		return 0
	}
	return b.lastPrefillSecs / float64(b.lastPrefillTokens)
}

// PrefillTokenBudget caps the next prefill batch's new tokens; nil means
// uncapped.
func (b *Balancer) PrefillTokenBudget(continuesChunk bool) *int {
	if b.burstTokens == nil {
		return nil
	}
	average := b.PrefillSecondsPerToken()
	marginal := b.MarginalPrefillSecondsPerToken()
	if continuesChunk && average > 0 && marginal > 0 {
		budget := max(0, int(float64(*b.burstTokens)*average/marginal))
		return &budget
	}
	if b.burstUsed == 0 {
		return nil
	}
	budget := *b.burstTokens - b.burstUsed
	return &budget
}

func (b *Balancer) ShouldDeferPrefill(prefillPending, decodeRunnable, continuesChunk bool) bool {
	if !(prefillPending && decodeRunnable) {
		// No contention: whichever class has work runs at full speed.
		b.debt = 0
		b.unsettledPrefill, b.unsettledDecode = 0, 0
		b.unsettledBatches = 0
		b.burstUsed = 0
		return false
	}
	if b.unsettledBatches > 0 {
		b.debt = max(b.debt+b.unsettledPrefill-b.unsettledDecode, -b.lastDecode)
		b.unsettledPrefill, b.unsettledDecode = 0, 0
		b.unsettledBatches = 0
	}
	if b.debt <= 0 && !b.prefillInFlight() {
		b.burstUsed = 0
	}
	if b.burstTokens == nil {
		return b.debt > 0
	}
	if continuesChunk {
		return b.burstUsed > 0
	}
	return b.burstUsed >= *b.burstTokens
}

// OnBatchLaunched records a launch at now; tokens are the extend tokens and
// decodeRows the running requests a mixed batch decodes alongside its prefill.
func (b *Balancer) OnBatchLaunched(class BatchClass, tokens, decodeRows int, now float64) {
	if len(b.inFlight) == 0 {
		b.busySince = now
	}
	b.inFlight = append(b.inFlight, launched{class, tokens, decodeRows})
	if class == ClassPrefill {
		b.burstUsed += tokens
	}
}

// OnBatchFinished charges the oldest launched batch, whose result was just
// processed at now.
func (b *Balancer) OnBatchFinished(now float64) {
	if len(b.inFlight) == 0 {
		// Batch launched outside get_next_batch_to_run (disaggregation loops).
		return
	}
	elapsed := now - b.busySince
	b.busySince = now
	batch := b.inFlight[0]
	b.inFlight = b.inFlight[1:]
	// A decode row adds one token to the extend pass; its attention over its
	// own context is not counted, so the estimate errs toward decode time.
	piggyback := min(float64(batch.decodeRows)*b.PrefillSecondsPerToken(), elapsed)
	if batch.class == ClassPrefill && batch.tokens > 0 {
		b.extendSeconds += elapsed
		b.extendTokens += batch.tokens
		b.lastPrefillSecs, b.lastPrefillTokens = elapsed, batch.tokens
	}
	switch batch.class {
	case ClassOther:
		return
	case ClassPrefill:
		b.unsettledPrefill += elapsed - piggyback
	case ClassDecode:
		b.unsettledDecode += elapsed
		b.lastDecode = elapsed
	}
	b.unsettledBatches++
}

func (b *Balancer) prefillInFlight() bool {
	for _, l := range b.inFlight {
		if l.class == ClassPrefill {
			return true
		}
	}
	return false
}
