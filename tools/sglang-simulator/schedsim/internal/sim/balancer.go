package sim

import (
	"math"
	"os"
	"strconv"
)

// balancer decides whether a prefill batch may run while decode has work.
type balancer interface {
	shouldDefer(prefillPending, decodeRunnable bool) bool
	onLaunched(now float64, isPrefill bool, tokens, rows int)
	onFinished(now float64)
}

type inFlight struct {
	isPrefill bool
	tokens    int
	rows      int
	contended bool
	deferred  bool
}

// newBalancer mirrors python/sglang/srt/managers/scheduler_components/
// prefill_decode_balancer.py on master.
type newBalancer struct {
	debt float64

	unsettledPrefill, unsettledDecode float64
	unsettledBatches                  int
	inFlight                          []inFlight
	busySince                         float64
	extendSeconds                     float64
	extendTokens                      int

	contended   bool
	burstOpen   bool
	lastPrefill float64
	run         float64
	longest     float64
	lastDecode  float64
	target      float64

	deferring          bool
	lastDeferredDecode float64
}

var burstVariant = func() int {
	v, _ := strconv.Atoi(os.Getenv("SCHEDSIM_BURST"))
	return v
}()

var bankVariant, _ = strconv.Atoi(os.Getenv("SCHEDSIM_BANK"))

// projected is the stall the running streams sit in once every prefill in
// flight has finished, estimating each at the last prefill's time.
func (b *newBalancer) projected() float64 {
	p := b.run
	for _, f := range b.inFlight {
		if f.isPrefill {
			p += b.lastPrefill
		}
		if !f.isPrefill || f.rows > 0 {
			p = 0
		}
	}
	return p
}

// fits reports whether extra more prefill batches fit in the longest stall paid.
func (b *newBalancer) fits(extra int) bool {
	return b.lastPrefill > 0 && b.projected()+float64(extra)*b.lastPrefill <= b.longest
}

func (b *newBalancer) secondsPerToken() float64 {
	if b.extendTokens == 0 {
		return 0
	}
	return b.extendSeconds / float64(b.extendTokens)
}

func (b *newBalancer) shouldDefer(prefillPending, decodeRunnable bool) bool {
	b.deferring = b.decide(prefillPending, decodeRunnable)
	return b.deferring
}

func (b *newBalancer) decide(prefillPending, decodeRunnable bool) bool {
	b.contended = prefillPending && decodeRunnable
	if !b.contended {
		b.debt = 0
		b.unsettledPrefill, b.unsettledDecode = 0, 0
		b.unsettledBatches = 0
		return false
	}
	if b.unsettledBatches > 0 {
		floor := 0.0
		switch bankVariant {
		case 1:
			floor = -b.lastDecode
		case 2:
			floor = -b.lastDeferredDecode
		}
		b.debt = math.Max(b.debt+b.unsettledPrefill-b.unsettledDecode, floor)
		b.unsettledPrefill, b.unsettledDecode = 0, 0
		b.unsettledBatches = 0
	}
	switch burstVariant {
	case 1:
		if b.burstOpen {
			return true
		}
	case 2:
		if b.burstOpen {
			return !b.fits(1)
		}
	case 3:
		if b.burstOpen && b.fits(1) {
			return false
		}
	case 4:
		if b.target > 0 && b.burstOpen {
			return !(b.lastPrefill > 0 && b.projected()+b.lastPrefill <= b.target)
		}
	}
	return b.debt > 0
}

// burstContinues reports whether the prefill batch being formed is followed by
// another before decode runs, so it should not carry the decode rows.
func (b *newBalancer) burstContinues(overlap, chunkContinues bool) bool {
	overlapChunk := overlap && chunkContinues && !b.prefillInFlight()
	switch burstVariant {
	case 1:
		return false
	case 2:
		return chunkContinues && b.fits(2)
	case 3:
		return overlapChunk || chunkContinues && b.fits(2)
	case 4:
		if b.target > 0 {
			return chunkContinues && b.lastPrefill > 0 && b.projected()+2*b.lastPrefill <= b.target
		}
	}
	return overlapChunk
}

func (b *newBalancer) onLaunched(now float64, isPrefill bool, tokens, rows int) {
	if len(b.inFlight) == 0 {
		b.busySince = now
	}
	b.inFlight = append(b.inFlight, inFlight{isPrefill, tokens, rows, b.contended, b.deferring && !isPrefill})
	b.burstOpen = isPrefill && rows == 0
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
	switch {
	case !f.contended:
		b.run = 0
	case f.isPrefill:
		b.lastPrefill = elapsed
		b.run += elapsed
		b.longest = math.Max(b.longest, b.run)
		if f.rows > 0 {
			b.run = 0
		}
	default:
		b.run = 0
	}
	if !f.isPrefill {
		b.lastDecode = elapsed
		b.lastDeferredDecode = 0
		if f.deferred {
			b.lastDeferredDecode = elapsed
		}
	}
	if f.isPrefill {
		b.unsettledPrefill += elapsed - piggyback
	} else {
		b.unsettledDecode += elapsed
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
}

func (b *prevBalancer) shouldDefer(prefillPending, decodeRunnable bool) bool {
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

func (b *prevBalancer) onLaunched(now float64, isPrefill bool, _, _ int) {
	if len(b.inFlight) == 0 {
		b.busySince = now
	}
	b.inFlight = append(b.inFlight, isPrefill)
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
