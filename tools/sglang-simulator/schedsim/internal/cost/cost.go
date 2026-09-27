// Package cost holds the GPU cost model the simulator charges and fits it to
// the embedded live log.
package cost

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"schedsim/internal/trace"
)

// Model prices every batch the simulated scheduler launches.
type Model struct {
	// Prefill seconds per token at context x are A + B*x + C*x*x.
	A, B, C float64
	// BatchOverhead is paid once per launched batch.
	BatchOverhead float64

	// A decode step costs DecodeBase + DecodePerReq*(bs-1) + DecodePerCtx*sum(ctx).
	DecodeBase   float64
	DecodePerReq float64
	DecodePerCtx float64
	// AcceptLen is the tokens one speculative decode step yields per request.
	AcceptLen float64

	// ReloadPerToken is the HiCache host-to-device copy cost.
	ReloadPerToken float64

	// DevicePoolTokens is the full-attention KV pool size.
	DevicePoolTokens int
}

// ExtendSeconds integrates the per-token prefill cost over [start, start+n).
func (m Model) ExtendSeconds(start, n int) float64 {
	if n <= 0 {
		return 0
	}
	a, b := float64(start), float64(start+n)
	return m.A*(b-a) + m.B/2*(b*b-a*a) + m.C/3*(b*b*b-a*a*a)
}

// PrefillBatchSeconds prices an extend batch. HiCache loads layer by layer
// under the forward, so a reload only costs what the compute does not hide.
func (m Model) PrefillBatchSeconds(compute float64, reloadTokens int) float64 {
	return m.BatchOverhead + math.Max(compute, m.ReloadPerToken*float64(reloadTokens))
}

// DecodeSeconds is one decode step over bs requests holding ctxSum tokens.
func (m Model) DecodeSeconds(bs int, ctxSum int) float64 {
	if bs <= 0 {
		return 0
	}
	return m.DecodeBase + m.DecodePerReq*float64(bs-1) + m.DecodePerCtx*float64(ctxSum)
}

// Priors the log cannot measure: it only ever decodes one request at one context.
const (
	priorBatchOverhead = 0.010
	priorDecodePerReq  = 0.001
	priorDecodePerCtx  = 1.0e-8
	priorReloadPerTok  = 0.5e-6
)

// Calibration is the fitted model plus how well it reproduces the log.
type Calibration struct {
	Model Model

	// FitChunks are the cold chunks the prefill curve was fitted on.
	FitChunks []ChunkCheck
	// HeldOut are chunks of a second cold prompt that the fit never saw.
	HeldOut []ChunkCheck
	// Mixed is the logged batch that finished the first cold prompt beside.
	Mixed ChunkCheck

	// PoolLo and PoolHi bound the device pool from every decode line's usage.
	PoolLo, PoolHi float64
	// DecodeCtx is the median context the logged decode steps ran at.
	DecodeCtx      int
	DecodeStepSecs float64
}

// ChunkCheck compares one logged batch with the model's price for it.
type ChunkCheck struct {
	Line      int
	Start     int
	Tokens    int
	Measured  float64
	Predicted float64
}

// RelErr is the model's relative error on this batch.
func (c ChunkCheck) RelErr() float64 { return (c.Predicted - c.Measured) / c.Measured }

// Calibrate fits the model to the embedded log.
func Calibrate() (Calibration, error) {
	steps, err := trace.Parse(trace.EmbeddedLog)
	if err != nil {
		return Calibration{}, err
	}
	return CalibrateSteps(steps)
}

// CalibrateSteps fits the prefill curve on the first cold prompt, checks it on
// the second, and reads the decode step and pool size from the decode lines.
func CalibrateSteps(steps []trace.Step) (Calibration, error) {
	const chunk = 4096
	var cal Calibration
	m := Model{BatchOverhead: priorBatchOverhead, DecodePerReq: priorDecodePerReq,
		DecodePerCtx: priorDecodePerCtx, ReloadPerToken: priorReloadPerTok}

	runs := coldRuns(steps, chunk)
	if len(runs) < 2 {
		return cal, errors.New("need two cold prompts in the log")
	}
	// The first chunk of the first prompt carries one-off warm-up work.
	first := runs[0][1:]
	for i, idx := range first {
		s := steps[idx]
		cal.FitChunks = append(cal.FitChunks, ChunkCheck{Line: s.Line, Start: (i + 1) * chunk,
			Tokens: s.NewTokens, Measured: trace.StepSeconds(s)})
	}
	if err := fitPrefill(&m, cal.FitChunks); err != nil {
		return cal, err
	}
	for i := range cal.FitChunks {
		c := &cal.FitChunks[i]
		c.Predicted = m.BatchOverhead + m.ExtendSeconds(c.Start, c.Tokens)
	}

	mixed := steps[runs[0][len(runs[0])-1]+1]
	tail := tailOf(steps, runs[0])
	firstTotal := len(runs[0])*chunk + tail
	follow := firstPendingJump(steps) - mixed.HitTokens
	secondHead := mixed.NewTokens - tail - follow
	for i, idx := range runs[1] {
		s := steps[idx]
		start := secondHead + i*chunk
		cal.HeldOut = append(cal.HeldOut, ChunkCheck{Line: s.Line, Start: start, Tokens: s.NewTokens,
			Measured: trace.StepSeconds(s), Predicted: m.BatchOverhead + m.ExtendSeconds(start, s.NewTokens)})
	}
	compute := m.ExtendSeconds(firstTotal-tail, tail) + m.ExtendSeconds(mixed.HitTokens, follow) +
		m.ExtendSeconds(0, secondHead)
	cal.Mixed = ChunkCheck{Line: mixed.Line, Tokens: mixed.NewTokens, Measured: trace.StepSeconds(mixed),
		Predicted: m.PrefillBatchSeconds(compute, mixed.HitTokens)}

	if err := fitDecode(&m, &cal, steps); err != nil {
		return cal, err
	}
	cal.Model = m
	return cal, nil
}

// coldRuns returns the index runs of full-size, cache-less prefill chunks.
func coldRuns(steps []trace.Step, chunk int) [][]int {
	var runs [][]int
	var cur []int
	for i, s := range steps {
		if s.Kind == trace.Prefill && s.HitTokens == 0 && s.NewTokens == chunk {
			cur = append(cur, i)
			continue
		}
		if len(cur) > 1 {
			runs = append(runs, cur)
		}
		cur = nil
	}
	if len(cur) > 1 {
		runs = append(runs, cur)
	}
	return runs
}

// tailOf is the last partial chunk of a cold prompt: the pending count on its
// last full chunk is what is left after it, over the queue's other requests.
func tailOf(steps []trace.Step, run []int) int {
	last := steps[run[len(run)-1]]
	first := steps[run[0]]
	total := first.Pending + first.NewTokens
	return total - len(run)*last.NewTokens
}

// PendingJumps are the input sizes of requests queued behind consecutive
// prefill lines, read from the rise in #pending-token, with the line that
// first shows each.
func PendingJumps(steps []trace.Step) (lines, inputs []int) {
	for i := 1; i < len(steps); i++ {
		prev, cur := steps[i-1], steps[i]
		if cur.Kind != trace.Prefill || prev.Kind != trace.Prefill || prev.Pending == 0 {
			continue
		}
		if jump := cur.Pending - (prev.Pending - cur.NewTokens); jump > 0 {
			lines = append(lines, i)
			inputs = append(inputs, jump)
		}
	}
	return lines, inputs
}

func firstPendingJump(steps []trace.Step) int {
	_, inputs := PendingJumps(steps)
	if len(inputs) == 0 {
		return 0
	}
	return inputs[0]
}

func fitPrefill(m *Model, chunks []ChunkCheck) error {
	// Least squares on seconds minus overhead against the basis integrals.
	var ata [3][3]float64
	var atb [3]float64
	for _, c := range chunks {
		a, b := float64(c.Start), float64(c.Start+c.Tokens)
		row := [3]float64{b - a, (b*b - a*a) / 2, (b*b*b - a*a*a) / 3}
		y := c.Measured - m.BatchOverhead
		for i := range 3 {
			for j := range 3 {
				ata[i][j] += row[i] * row[j]
			}
			atb[i] += row[i] * y
		}
	}
	x, err := solve3(ata, atb)
	if err != nil {
		return err
	}
	m.A, m.B, m.C = x[0], x[1], x[2]
	if m.A <= 0 {
		return fmt.Errorf("fitted per-token prefill cost %.3g is not positive", m.A)
	}
	return nil
}

func solve3(a [3][3]float64, b [3]float64) ([3]float64, error) {
	// Scale columns so the elimination is well conditioned.
	var scale [3]float64
	for j := range 3 {
		scale[j] = math.Sqrt(a[j][j])
		if scale[j] == 0 {
			return [3]float64{}, errors.New("singular fit")
		}
	}
	for i := range 3 {
		for j := range 3 {
			a[i][j] /= scale[i] * scale[j]
		}
		b[i] /= scale[i]
	}
	for col := range 3 {
		p := col
		for r := col + 1; r < 3; r++ {
			if math.Abs(a[r][col]) > math.Abs(a[p][col]) {
				p = r
			}
		}
		a[col], a[p] = a[p], a[col]
		b[col], b[p] = b[p], b[col]
		if math.Abs(a[col][col]) < 1e-15 {
			return [3]float64{}, errors.New("singular fit")
		}
		for r := col + 1; r < 3; r++ {
			f := a[r][col] / a[col][col]
			for k := col; k < 3; k++ {
				a[r][k] -= f * a[col][k]
			}
			b[r] -= f * b[col]
		}
	}
	var x [3]float64
	for i := 2; i >= 0; i-- {
		s := b[i]
		for k := i + 1; k < 3; k++ {
			s -= a[i][k] * x[k]
		}
		x[i] = s / a[i][i]
	}
	for i := range 3 {
		x[i] /= scale[i]
	}
	return x, nil
}

// fitDecode reads the single-stream decode step and the pool size. A decode
// line right after a prefill line spans that prefill, so it is skipped.
func fitDecode(m *Model, cal *Calibration, steps []trace.Step) error {
	var durs, accepts, ctxs []float64
	cal.PoolLo, cal.PoolHi = 0, math.Inf(1)
	for i, s := range steps {
		if s.Kind != trace.Decode {
			continue
		}
		if s.FullUsage > 0 {
			// Usage is printed to decimals.
			cal.PoolLo = math.Max(cal.PoolLo, float64(s.FullTokens)/(s.FullUsage+0.005))
			cal.PoolHi = math.Min(cal.PoolHi, float64(s.FullTokens)/(s.FullUsage-0.005))
		}
		if i == 0 || steps[i-1].Kind != trace.Decode || s.AcceptLen <= 0 {
			continue
		}
		durs = append(durs, s.AcceptLen/s.Throughput)
		accepts = append(accepts, s.AcceptLen)
		ctxs = append(ctxs, float64(s.FullTokens))
	}
	if len(durs) == 0 || cal.PoolLo > cal.PoolHi {
		return fmt.Errorf("decode lines give no step time or an empty pool interval [%.0f, %.0f]", cal.PoolLo, cal.PoolHi)
	}
	cal.DecodeStepSecs = median(durs)
	cal.DecodeCtx = int(median(ctxs))
	m.AcceptLen = median(accepts)
	m.DecodeBase = cal.DecodeStepSecs - m.DecodePerCtx*float64(cal.DecodeCtx)
	m.DevicePoolTokens = int((cal.PoolLo + cal.PoolHi) / 2)
	return nil
}

func median(xs []float64) float64 {
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}
