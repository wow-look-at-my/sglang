package sim

import (
	"schedsim/internal/kv"
	"schedsim/internal/trace"
)

// Cost prices a batch from the calibration fitted to the production log. The
// sweep's perturbations are fields here rather than edits to the fit, so every
// variant states which measurement it moved.
type Cost struct {
	Cal trace.Calibration

	// AcceptMean overrides the log's mean accepted length; 0 keeps the fit's.
	AcceptMean float64
	// ReloadPerToken overrides the host-tier copy rate; 0 keeps kv's constant.
	ReloadPerToken float64
}

// NewCost returns the calibrated model with no perturbation.
func NewCost(cal trace.Calibration) Cost { return Cost{Cal: cal} }

// AcceptList is the per-step accepted-token counts the log reported.
func (c Cost) AcceptList() []float64 { return c.Cal.Decode.Accept }

// AcceptLengthMean is the mean accepted tokens per decode step.
func (c Cost) AcceptLengthMean() float64 {
	if c.AcceptMean > 0 {
		return c.AcceptMean
	}
	return c.Cal.Decode.AcceptMean
}

// NumDraft is the speculative depth: drafts verified per decode step.
func (c Cost) NumDraft() int { return c.Cal.Decode.NumDraft }

// ReloadSecondsPerToken is the host-to-device copy cost per prefix token.
func (c Cost) ReloadSecondsPerToken() float64 {
	if c.ReloadPerToken > 0 {
		return c.ReloadPerToken
	}
	return kv.ReloadSecondsPerToken
}

// DeviceTokens is the KV pool capacity the log implies.
func (c Cost) DeviceTokens() int { return c.Cal.DeviceTokens }

// ExtendSeconds prices an extend batch: the per-batch base plus every item at
// its own mid-context.
func (c Cost) ExtendSeconds(items []trace.ExtendItem) float64 {
	return c.Cal.Prefill.PrefillSeconds(items)
}

// DecodeSeconds prices one decode step for a batch of n requests reading sumCtx
// context tokens in total.
func (c Cost) DecodeSeconds(n, sumCtx int) float64 {
	return c.Cal.Decode.StepSeconds(n, sumCtx)
}

// SecondsPerToken is the extend cost at one mid-context, which is what prices a
// mixed batch's decode rows.
func (c Cost) SecondsPerToken(midCtx float64) float64 {
	return c.Cal.Prefill.PrefillSecondsPerToken(midCtx)
}

// ChunkSeconds prices one full chunk of prefill starting at prefix.
func (c Cost) ChunkSeconds(prefix int) float64 {
	return c.Cal.Prefill.PrefillSeconds([]trace.ExtendItem{
		{Tokens: c.Cal.ChunkSize, MidCtx: float64(prefix) + float64(c.Cal.ChunkSize)/2},
	})
}

// AcceptProb returns q with 1+q+q^2+...+q^numDraft = L, the sequential
// acceptance whose expected length is one step's accepted-token count.
func AcceptProb(L float64, numDraft int) float64 {
	lo, hi := 0.0, 1.0
	for i := 0; i < 60; i++ {
		q := (lo + hi) / 2
		s, p := 1.0, 1.0
		for k := 0; k < numDraft; k++ {
			p *= q
			s += p
		}
		if s < L {
			lo = q
		} else {
			hi = q
		}
	}
	return (lo + hi) / 2
}

// WithPrefillBase moves the per-batch overhead to sec while holding a chunk's
// cost at prefix 0 unchanged, so a variant isolates the overhead's effect on
// short batches instead of changing overall prefill throughput.
func WithPrefillBase(p trace.PrefillCost, sec float64) trace.PrefillCost {
	q := p
	if q.ChunkTokens > 0 {
		q.PerToken += (p.Base - sec) / float64(q.ChunkTokens)
	}
	q.Base = sec
	return q
}

// WithContextCost re-prices the decode context term to perCtx, holding the
// fitted step time at the log's context fixed.
func WithContextCost(d trace.DecodeCost, perCtx float64) trace.DecodeCost {
	q := d
	q.Base += (d.PerTokenCtx - perCtx) * d.MeanCtx
	q.PerTokenCtx = perCtx
	return q
}

// WithBatchCost re-prices the decode per-request term to perReq, holding the
// fitted single-request step time fixed.
func WithBatchCost(d trace.DecodeCost, perReq float64) trace.DecodeCost {
	q := d
	q.PerReq = perReq
	return q
}

func clampF(x, lo, hi float64) float64 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}
