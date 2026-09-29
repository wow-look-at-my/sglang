package trace

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// Calibrate pairs the first cold run in a log with the second as fit and
// holdout. On a single-incident log those are two prompts of one deployment.
// On a day-long log they are whatever two runs happen to come first, which
// may be a 6-chunk fragment and an 85-chunk prompt from a boot with a
// different context length: a quadratic fitted to 17 points and evaluated
// far outside their context range reads hundreds of percent off. CalibReport
// fits the largest run instead and holds every other substantial run out,
// per boot, so the error it reports is the model's spread across the prompts
// it will be asked to price.

// HoldoutError is the fit's relative error on one held-out cold run.
type HoldoutError struct {
	Boot   int
	Chunks int
	RMSE   float64
	MaxErr float64
}

// PoolSize is one boot's KV pool capacity, from its own decode lines.
type PoolSize struct {
	Boot   int
	Tokens int
	// Context is the boot's --context-length when its server_args were read.
	Context int
}

// CalibReport is CalibrateBoots' account of where its cost model came from.
type CalibReport struct {
	Boots int
	// FitBoot and FitChunks identify the run the prefill model was fitted on.
	FitBoot   int
	FitChunks int
	Holdouts  []HoldoutError
	// HoldoutMedian is the median holdout RMSE, the number to quote.
	HoldoutMedian float64
	Pools         []PoolSize
	// DecodeSamples counts the steady single-request decode lines pooled
	// across boots for the decode step time.
	DecodeSamples int

	// The wall clock check, available with timestamps. For every cold chunk
	// followed by another batch line, the gap between the two lines is what
	// the chunk actually took, and NewTokens/Throughput is what the model
	// charges. WallRatioMedian is the median of gap/model and
	// WallOverheadMedian the median of gap-model in seconds: the measured
	// counterpart of the assumed PrefillBaseSeconds.
	WallSamples        int
	WallRatioMedian    float64
	WallOverheadMedian float64
}

// CalibrateBoots fits the cost model across process lifetimes. Cold runs
// never cross a boot; the largest run anywhere is the fit set and every other
// run of at least minHoldoutChunks chunks is a holdout. The decode step time
// pools steady single-request lines from every boot, and the pool size is
// reported per boot (boots with different --context-length carry different
// pools) with the fit boot's used for the calibration.
func CalibrateBoots(boots []Boot, chunkSize, page int) (Calibration, CalibReport) {
	const minHoldoutChunks = 8
	rep := CalibReport{Boots: len(boots), FitBoot: -1}
	type run struct {
		boot int
		idx  []int
		ctx0 int
	}
	var runs []run
	for bi, b := range boots {
		for _, r := range b.ColdRunsAll(chunkSize) {
			if r.Chunks() > 1 {
				runs = append(runs, run{bi, r.Indexes(), r.CtxStart})
			}
		}
		if n := DevicePoolTokens(b.Steps); n > 0 {
			rep.Pools = append(rep.Pools, PoolSize{Boot: bi, Tokens: n, Context: b.Args.ContextLength})
		}
	}
	if len(runs) == 0 {
		return Calibration{Prefill: PrefillCost{ChunkTokens: chunkSize, Base: PrefillBaseSeconds}, ChunkSize: chunkSize, Page: page}, rep
	}
	fit := 0
	for i, r := range runs {
		if len(r.idx) > len(runs[fit].idx) {
			fit = i
		}
	}
	rep.FitBoot, rep.FitChunks = runs[fit].boot, len(runs[fit].idx)
	steps := boots[runs[fit].boot].Steps
	p := fitPrefillRun(steps, runs[fit].idx, runs[fit].ctx0, chunkSize, 2)
	var rmses []float64
	for i, r := range runs {
		if i == fit || len(r.idx) < minHoldoutChunks {
			continue
		}
		rmse, maxErr := fitErrorCtx(p, boots[r.boot].Steps, r.idx, r.ctx0, chunkSize)
		rep.Holdouts = append(rep.Holdouts, HoldoutError{Boot: r.boot, Chunks: len(r.idx), RMSE: rmse, MaxErr: maxErr})
		rmses = append(rmses, rmse)
	}
	rep.HoldoutMedian = median(rmses)
	p.RMSEHoldout = rep.HoldoutMedian
	p.HoldoutSamples = len(rmses)
	for _, h := range rep.Holdouts {
		if h.MaxErr > p.MaxErrHoldout {
			p.MaxErrHoldout = h.MaxErr
		}
	}

	// Decode lines are pooled across boots with a prefill sentinel between
	// them, so FitDecode's "previous line was decode" test never pairs the
	// last line of one boot with the first of the next.
	var pooled []Step
	for _, b := range boots {
		pooled = append(pooled, Step{Kind: Prefill})
		pooled = append(pooled, b.Steps...)
	}
	d := FitDecode(pooled)
	rep.DecodeSamples = d.Samples

	var ratios, overheads []float64
	for _, b := range boots {
		for _, r := range b.ColdRunsAll(chunkSize) {
			for i := r.Start; i < r.End && i+1 < len(b.Steps); i++ {
				gap, ok := StepGap(b.Steps[i], b.Steps[i+1])
				if !ok {
					continue
				}
				model := StepSeconds(b.Steps[i])
				ratios = append(ratios, gap/model)
				overheads = append(overheads, gap-model)
			}
		}
	}
	rep.WallSamples = len(ratios)
	rep.WallRatioMedian = median(ratios)
	rep.WallOverheadMedian = median(overheads)

	pool := DevicePoolTokens(steps)
	return Calibration{Prefill: p, Decode: d, DeviceTokens: pool, ChunkSize: chunkSize, Page: page}, rep
}

// fitPrefillRun fits one cold run that extends from context ctx0, skipping
// its first chunk (whose gap holds the idle time before the prompt arrived).
func fitPrefillRun(steps []Step, run []int, ctx0, chunkSize, deg int) PrefillCost {
	var xs, ys []float64
	for k, i := range run {
		if k == 0 {
			continue
		}
		xs = append(xs, float64(ctx0+k*chunkSize)+float64(chunkSize)/2)
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
	p.RMSEFit, _ = fitErrorCtx(p, steps, run, ctx0, chunkSize)
	return p
}

// fitErrorCtx is fitError for a run extending from context ctx0.
func fitErrorCtx(p PrefillCost, steps []Step, run []int, ctx0, chunkSize int) (rmse, maxErr float64) {
	var sum float64
	var n float64
	for k, i := range run {
		if k == 0 {
			continue
		}
		m := float64(ctx0+k*chunkSize) + float64(chunkSize)/2
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

func (r CalibReport) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "calibrated across %d boot(s): prefill fitted on boot %d's %d-chunk cold run, ",
		r.Boots, r.FitBoot, r.FitChunks)
	if len(r.Holdouts) == 0 {
		b.WriteString("no holdout run of 8+ chunks")
	} else {
		worst := 0.0
		for _, h := range r.Holdouts {
			worst = math.Max(worst, h.RMSE)
		}
		fmt.Fprintf(&b, "%d holdout run(s): median error %.1f%%, worst %.1f%%",
			len(r.Holdouts), 100*r.HoldoutMedian, 100*worst)
	}
	fmt.Fprintf(&b, "; decode from %d steady lines", r.DecodeSamples)
	if len(r.Pools) > 0 {
		pools := append([]PoolSize(nil), r.Pools...)
		sort.Slice(pools, func(i, j int) bool { return pools[i].Boot < pools[j].Boot })
		b.WriteString("; pool per boot:")
		for _, p := range pools {
			fmt.Fprintf(&b, " b%d=%d", p.Boot, p.Tokens)
		}
	}
	if r.WallSamples > 0 {
		fmt.Fprintf(&b, "; wall clock over %d cold chunks: gap/model median %.2f, overhead median %.1f ms (PrefillBaseSeconds assumes %.0f ms)",
			r.WallSamples, r.WallRatioMedian, 1e3*r.WallOverheadMedian, 1e3*PrefillBaseSeconds)
	}
	return b.String()
}
