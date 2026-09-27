package trace

import (
	"fmt"
	"math"
)

// Calibration is the cost model the scheduler simulation charges batches with.
// Every coefficient here is produced by fitting the logged steps, so a scenario
// run costs GPU time the way the deployment measured it. The residual priors are
// named at their point of use below.

// ExtendItem is one request's contribution to a prefill batch: the tokens it
// computes new, and the mid-context at which it computes them.
type ExtendItem struct {
	Tokens int
	MidCtx float64
}

// PrefillCost prices an extend batch as Base + sum(Items.Tokens *
// (PerToken + PerTokenCtx*MidCtx + PerTokenCtxSq*MidCtx^2)).
type PrefillCost struct {
	ChunkTokens   int
	Base          float64
	PerToken      float64
	PerTokenCtx   float64
	PerTokenCtxSq float64

	// RMSEFit and RMSEHoldout are the relative error of the quadratic fit on the
	// first logged cold run (the fit set) and the second (held out), and
	// MaxErrHoldout its worst single chunk. A linear fit measures ~10% on the
	// holdout; the quadratic form is what the log supports.
	RMSEFit        float64
	RMSEHoldout    float64
	MaxErrHoldout  float64
	Samples        int
	HoldoutSamples int
}

// Decode costs a decode step as Base + PerReq*(batch-1) + PerTokenCtx*sum(contexts).
type DecodeCost struct {
	Base        float64
	PerReq      float64
	PerTokenCtx float64
	// MeanCtx is the context of the steady single-request lines the base term was
	// solved at, so a term can be re-scaled without moving the fitted step time.
	MeanCtx  float64
	NumDraft int
	// Accept carries the accepted-token counts the log reported per step; a run
	// draws a per-request acceptance rate from it.
	Accept     []float64
	AcceptMean float64
	Samples    int
}

// Calibration bundles the two cost models plus the pool occupancy they imply.
type Calibration struct {
	Prefill PrefillCost
	Decode  DecodeCost
	// DeviceTokens is the KV pool's capacity in tokens, from the log's own
	// #full token / full token usage ratio.
	DeviceTokens int
	// ChunkSize is the deployment's chunked_prefill_size, and Page its page size.
	ChunkSize int
	Page      int
}

// PrefillBaseSeconds is charged once per extend batch, before its tokens.
// Arbitrary; a kernel-launch and bookkeeping floor, not a measurement -- the
// per-token terms absorb almost all of a 4096-token chunk's time, and the sweep
// varies 0 ms to 40 ms with no scenario changing its conclusion.
const PrefillBaseSeconds = 10e-3

// DecodePerReqSeconds and DecodePerTokenCtxSeconds are the batch-size and
// context-length terms of a decode step. The log only ever decodes one request
// at one context, so these two cannot be measured from it; they are priors, and
// the sensitivity sweep varies each 0x to 3x.
const (
	DecodePerReqSeconds      = 1.5e-3
	DecodePerTokenCtxSeconds = 6e-9
)

// ColdRuns groups the indices of consecutive full cold chunks: a prefill step
// of exactly chunkSize new tokens with no prefix hit. Two runs in the log are
// isolated cold prompts, the first with nothing running and the second with two
// requests running, which makes the second a genuine holdout for the fit.
func ColdRuns(steps []Step, chunkSize int) [][]int {
	var runs [][]int
	var run []int
	for i, s := range steps {
		if s.Kind == Prefill && s.NewTokens == chunkSize && s.HitTokens == 0 {
			run = append(run, i)
			continue
		}
		if len(run) > 1 {
			runs = append(runs, run)
		}
		run = nil
	}
	if len(run) > 1 {
		runs = append(runs, run)
	}
	return runs
}

// FitPrefill fits the quadratic in mid-context to the logged cold chunks, using
// the first run as the fit set and the second as the holdout. Chunk seconds come
// from StepSeconds, which inverts the log's own input-throughput definition.
func FitPrefill(steps []Step, chunkSize int) PrefillCost {
	return fitPrefill(steps, chunkSize, 2)
}

// FitPrefillLinear fits the same samples with a degree-1 polynomial, for the
// comparison that must show what the quadratic form is worth.
func FitPrefillLinear(steps []Step, chunkSize int) PrefillCost {
	return fitPrefill(steps, chunkSize, 1)
}

func fitPrefill(steps []Step, chunkSize, deg int) PrefillCost {
	runs := ColdRuns(steps, chunkSize)
	if len(runs) == 0 {
		return PrefillCost{ChunkTokens: chunkSize, Base: PrefillBaseSeconds}
	}
	var xs, ys []float64
	for k, i := range runs[0] {
		if k == 0 {
			// The first chunk's gap includes idle time before the prompt arrived.
			continue
		}
		xs = append(xs, float64(k*chunkSize)+float64(chunkSize)/2)
		ys = append(ys, StepSeconds(steps[i]))
	}
	co := polyfit(xs, ys, deg)
	p := PrefillCost{
		ChunkTokens: chunkSize,
		Base:        PrefillBaseSeconds,
		PerToken:    (co[0] - PrefillBaseSeconds) / float64(chunkSize),
		Samples:     len(xs),
	}
	if len(co) > 1 {
		p.PerTokenCtx = co[1] / float64(chunkSize)
	}
	if len(co) > 2 {
		p.PerTokenCtxSq = co[2] / float64(chunkSize)
	}
	p.RMSEFit, _ = fitError(p, steps, runs[0], chunkSize)
	if len(runs) > 1 {
		p.RMSEHoldout, p.MaxErrHoldout = fitError(p, steps, runs[1], chunkSize)
		p.HoldoutSamples = len(runs[1]) - 1
	}
	return p
}

func fitError(p PrefillCost, steps []Step, run []int, chunkSize int) (rmse, maxErr float64) {
	var sum float64
	var n float64
	for k, i := range run {
		if k == 0 {
			continue
		}
		m := float64(k*chunkSize) + float64(chunkSize)/2
		pred := p.PrefillSeconds([]ExtendItem{{Tokens: chunkSize, MidCtx: m}})
		rel := (pred - StepSeconds(steps[i])) / StepSeconds(steps[i])
		sum += rel * rel
		n++
		if math.Abs(rel) > maxErr {
			maxErr = math.Abs(rel)
		}
	}
	if n == 0 {
		return 0, 0
	}
	return math.Sqrt(sum / n), maxErr
}

// PrefillSeconds is the measured cost of an extend batch.
func (p PrefillCost) PrefillSeconds(items []ExtendItem) float64 {
	if len(items) == 0 {
		return 0
	}
	t := p.Base
	for _, it := range items {
		m := it.MidCtx
		t += float64(it.Tokens) * (p.PerToken + p.PerTokenCtx*m + p.PerTokenCtxSq*m*m)
	}
	return t
}

// PrefillSecondsPerToken is the average cost at one context, which the balancer
// measures at run time to price a mixed batch's decode rows.
func (p PrefillCost) PrefillSecondsPerToken(midCtx float64) float64 {
	return p.PerToken + p.PerTokenCtx*midCtx + p.PerTokenCtxSq*midCtx*midCtx
}

// Scaled multiplies every prefill cost term, the sweep's cost perturbation.
func (p PrefillCost) Scaled(k float64) PrefillCost {
	q := p
	q.Base *= k
	q.PerToken *= k
	q.PerTokenCtx *= k
	q.PerTokenCtxSq *= k
	return q
}

// FitDecode solves the step time of the log's steady single-stream decodes,
// where the line before it was also a decode line: step seconds are the accepted
// tokens over the gen throughput that line reports. The base term is what is
// left once the assumed context cost is subtracted at the observed context.
func FitDecode(steps []Step) DecodeCost {
	var (
		stepS  []float64
		ctxS   []float64
		accept []float64
		maxAcc float64
	)
	for i, s := range steps {
		// Steady means the previous line was a decode line too, so the reported
		// gen throughput covers a whole interval of decode steps and not the gap
		// left by an interleaved prefill.
		if i == 0 || s.Kind != Decode || steps[i-1].Kind != Decode {
			continue
		}
		if s.RunningReq != 1 || s.Throughput <= 0 {
			continue
		}
		stepS = append(stepS, s.AcceptLen/s.Throughput)
		ctxS = append(ctxS, float64(s.FullTokens))
		accept = append(accept, s.AcceptLen)
		maxAcc = math.Max(maxAcc, s.AcceptLen)
	}
	d := DecodeCost{
		PerReq:      DecodePerReqSeconds,
		PerTokenCtx: DecodePerTokenCtxSeconds,
		Accept:      accept,
		Samples:     len(stepS),
	}
	if len(stepS) == 0 {
		return d
	}
	var mean, meanCtx float64
	for i := range stepS {
		mean += stepS[i] - d.PerTokenCtx*ctxS[i]
		meanCtx += ctxS[i]
	}
	mean /= float64(len(stepS))
	meanCtx /= float64(len(stepS))
	d.Base = mean
	d.MeanCtx = meanCtx
	for _, a := range accept {
		d.AcceptMean += a
	}
	d.AcceptMean /= float64(len(accept))
	// One accepted token is always the sampled token; the rest are drafts.
	d.NumDraft = int(math.Ceil(maxAcc)) - 1
	return d
}

// Scaled multiplies every decode cost term.
func (d DecodeCost) Scaled(k float64) DecodeCost {
	q := d
	q.Base *= k
	q.PerReq *= k
	q.PerTokenCtx *= k
	return q
}

// StepSeconds is the cost of one decode batch.
func (d DecodeCost) StepSeconds(batch int, sumCtx int) float64 {
	if batch <= 0 {
		return 0
	}
	return d.Base + d.PerReq*float64(batch-1) + d.PerTokenCtx*float64(sumCtx)
}

// DevicePoolTokens derives the KV pool capacity from the log: every decode line
// reports both the tokens held and the fraction of the pool that is, so their
// ratio is the capacity. The median over the lines is the estimate.
func DevicePoolTokens(steps []Step) int {
	var ratios []float64
	for _, s := range steps {
		if s.Kind == Decode && s.FullUsage > 0 && s.FullTokens > 0 {
			ratios = append(ratios, float64(s.FullTokens)/s.FullUsage)
		}
	}
	if len(ratios) == 0 {
		return 0
	}
	return int(median(ratios) + 0.5)
}

// Calibrate runs every fit over the parsed log.
func Calibrate(steps []Step, chunkSize, page int) Calibration {
	return Calibration{
		Prefill:      FitPrefill(steps, chunkSize),
		Decode:       FitDecode(steps),
		DeviceTokens: DevicePoolTokens(steps),
		ChunkSize:    chunkSize,
		Page:         page,
	}
}

// PrefillSeconds and StepSeconds dispatch on the calibration.
func (c Calibration) ExtendSeconds(items []ExtendItem) float64 {
	return c.Prefill.PrefillSeconds(items)
}

func (c Calibration) String() string {
	return fmt.Sprintf(
		"prefill %.1f ms + %.3f us/token + %.3g/token/ctx + %.3g/token/ctx^2 "+
			"(fit %.3f%%, holdout %.3f%%, n=%d/%d); decode %.3f ms + %.1f ms/req + %.3g s/token-ctx "+
			"(n=%d, drafts=%d, accept %.2f); pool %d tokens",
		c.Prefill.Base*1e3, c.Prefill.PerToken*1e6, c.Prefill.PerTokenCtx, c.Prefill.PerTokenCtxSq,
		100*c.Prefill.RMSEFit, 100*c.Prefill.RMSEHoldout, c.Prefill.Samples, c.Prefill.HoldoutSamples,
		c.Decode.Base*1e3, c.Decode.PerReq*1e3, c.Decode.PerTokenCtx,
		c.Decode.Samples, c.Decode.NumDraft, c.Decode.AcceptMean, c.DeviceTokens,
	)
}

// polyfit solves the least-squares coefficients of a degree-d polynomial.
func polyfit(xs, ys []float64, deg int) []float64 {
	n := deg + 1
	m := make([][]float64, n)
	for i := range m {
		m[i] = make([]float64, n+1)
	}
	for k := range xs {
		pow := make([]float64, 2*n)
		pow[0] = 1
		for j := 1; j < 2*n; j++ {
			pow[j] = pow[j-1] * xs[k]
		}
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				m[i][j] += pow[i+j]
			}
			m[i][n] += pow[i] * ys[k]
		}
	}
	for c := 0; c < n; c++ {
		piv := c
		for r := c + 1; r < n; r++ {
			if math.Abs(m[r][c]) > math.Abs(m[piv][c]) {
				piv = r
			}
		}
		m[c], m[piv] = m[piv], m[c]
		for r := 0; r < n; r++ {
			if r == c {
				continue
			}
			f := m[r][c] / m[c][c]
			for j := c; j <= n; j++ {
				m[r][j] -= f * m[c][j]
			}
		}
	}
	out := make([]float64, n)
	for i := range out {
		out[i] = m[i][n] / m[i][i]
	}
	return out
}
