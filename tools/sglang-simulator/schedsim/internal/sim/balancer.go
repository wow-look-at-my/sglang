package sim

import "math"

// balancer decides whether a prefill batch may run while decode has work.
type balancer interface {
	shouldDefer(prefillPending, decodeRunnable, continuesChunk bool) bool
	// budget caps the next prefill batch's tokens; ok is false for no cap.
	budget() (tokens int, ok bool)
	onLaunched(now float64, isPrefill bool, tokens, rows int)
	onFinished(now float64)
	secondsPerToken() float64
}

type inFlight struct {
	isPrefill bool
	tokens    int
	rows      int
}

// newBalancer mirrors python/sglang/srt/managers/scheduler_components/
// prefill_decode_balancer.py on master.
type newBalancer struct {
	burstTokens int
	debt        float64

	unsettledPrefill, unsettledDecode float64
	lastDecode                        float64
	unsettledBatches                  int
	inFlight                          []inFlight
	busySince                         float64
	burstUsed                         int
	extendSeconds                     float64
	extendTokens                      int
}

func (b *newBalancer) secondsPerToken() float64 {
	if b.extendTokens == 0 {
		return 0
	}
	return b.extendSeconds / float64(b.extendTokens)
}

func (b *newBalancer) budget() (int, bool) {
	if b.burstUsed == 0 {
		return 0, false
	}
	return b.burstTokens - b.burstUsed, true
}

func (b *newBalancer) shouldDefer(prefillPending, decodeRunnable, continuesChunk bool) bool {
	if !(prefillPending && decodeRunnable) {
		b.debt = 0
		b.unsettledPrefill, b.unsettledDecode = 0, 0
		b.unsettledBatches = 0
		b.burstUsed = 0
		return false
	}
	if b.unsettledBatches > 0 {
		b.debt = math.Max(b.debt+b.unsettledPrefill-b.unsettledDecode, -b.lastDecode)
		b.unsettledPrefill, b.unsettledDecode = 0, 0
		b.unsettledBatches = 0
	}
	if b.debt <= 0 && !b.prefillInFlight() {
		b.burstUsed = 0
	}
	if continuesChunk {
		return b.burstUsed > 0
	}
	return b.burstUsed >= b.burstTokens
}

func (b *newBalancer) onLaunched(now float64, isPrefill bool, tokens, rows int) {
	if len(b.inFlight) == 0 {
		b.busySince = now
	}
	b.inFlight = append(b.inFlight, inFlight{isPrefill, tokens, rows})
	if isPrefill {
		b.burstUsed += tokens
	}
}

func (b *newBalancer) onFinished(now float64) {
	if len(b.inFlight) == 0 {
		return
	}
	elapsed := now - b.busySince
	b.busySince = now
	f := b.inFlight[0]
	b.inFlight = b.inFlight[1:]
	piggyback := math.Min(float64(f.rows)*b.secondsPerToken(), elapsed)
	if f.tokens > 0 {
		b.extendSeconds += elapsed
		b.extendTokens += f.tokens
	}
	if f.isPrefill {
		b.unsettledPrefill += elapsed - piggyback
	} else {
		b.unsettledDecode += elapsed
		b.lastDecode = elapsed
	}
	b.unsettledBatches++
}

func (b *newBalancer) prefillInFlight() bool {
	for _, f := range b.inFlight {
		if f.isPrefill {
			return true
		}
	}
	return false
}

type prevBalancer struct {
	debt                              float64
	unsettledPrefill, unsettledDecode float64
	unsettledBatches                  int
	inFlight                          []bool
	busySince                         float64
	extendSeconds                     float64
	extendTokens                      int
}

func (b *prevBalancer) secondsPerToken() float64 {
	if b.extendTokens == 0 {
		return 0
	}
	return b.extendSeconds / float64(b.extendTokens)
}

func (b *prevBalancer) budget() (int, bool) { return 0, false }

func (b *prevBalancer) shouldDefer(prefillPending, decodeRunnable, _ bool) bool {
	if !(prefillPending && decodeRunnable) {
		b.debt = 0
		b.unsettledPrefill, b.unsettledDecode = 0, 0
		b.unsettledBatches = 0
		return false
	}
	if b.unsettledBatches > 0 {
		b.debt = math.Max(b.debt+b.unsettledPrefill-b.unsettledDecode, 0)
		b.unsettledPrefill, b.unsettledDecode = 0, 0
		b.unsettledBatches = 0
	}
	return b.debt > 0
}

func (b *prevBalancer) onLaunched(now float64, isPrefill bool, tokens, _ int) {
	if len(b.inFlight) == 0 {
		b.busySince = now
	}
	b.inFlight = append(b.inFlight, isPrefill)
	_ = tokens
}

func (b *prevBalancer) onFinished(now float64) {
	if len(b.inFlight) == 0 {
		return
	}
	elapsed := now - b.busySince
	b.busySince = now
	isPrefill := b.inFlight[0]
	b.inFlight = b.inFlight[1:]
	if isPrefill {
		b.unsettledPrefill += elapsed
	} else {
		b.unsettledDecode += elapsed
	}
	b.unsettledBatches++
}
