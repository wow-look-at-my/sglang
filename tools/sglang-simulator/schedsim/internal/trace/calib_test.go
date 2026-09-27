package trace

import (
	"math"
	"testing"
)

func cal(t *testing.T) Calibration {
	t.Helper()
	steps, err := Parse(EmbeddedLog)
	if err != nil {
		t.Fatalf("parse embedded log: %v", err)
	}
	c := Calibrate(steps, 4096, 64)
	if c.Prefill.Samples < 10 {
		t.Fatalf("prefill fit has %d samples, want a logged cold run", c.Prefill.Samples)
	}
	return c
}

// The quadratic form is a claim about the log, not a constant: the second cold
// run was never looked at by the fit and must still be predicted closely, and a
// linear form must be visibly worse on it. If a future log makes the quadratic
// unnecessary this test is the place that gets argued.
func TestPrefillFitPredictsTheHeldOutRun(t *testing.T) {
	steps, err := Parse(EmbeddedLog)
	if err != nil {
		t.Fatalf("parse embedded log: %v", err)
	}
	runs := ColdRuns(steps, 4096)
	if len(runs) < 2 {
		t.Fatalf("the log must carry two isolated cold runs to fit and hold out, found %d", len(runs))
	}
	c := cal(t)
	if c.Prefill.RMSEHoldout > 0.03 {
		t.Errorf("holdout relative RMSE %.4f, want under 3%%", c.Prefill.RMSEHoldout)
	}
	if c.Prefill.RMSEFit > 0.03 {
		t.Errorf("fit relative RMSE %.4f, want under 3%%", c.Prefill.RMSEFit)
	}

	// The same least-squares solve at degree 1, scored on the held-out run.
	var xs, ys []float64
	for k, i := range runs[0] {
		if k == 0 {
			continue
		}
		xs = append(xs, float64(k*4096)+2048)
		ys = append(ys, StepSeconds(steps[i]))
	}
	lin := polyfit(xs, ys, 1)
	var sum float64
	var n float64
	for k, i := range runs[1] {
		if k == 0 {
			continue
		}
		m := float64(k*4096) + 2048
		pred := lin[0] + lin[1]*m
		obs := StepSeconds(steps[i])
		sum += ((pred - obs) / obs) * ((pred - obs) / obs)
		n++
	}
	linRMSE := math.Sqrt(sum / n)
	if c.Prefill.RMSEHoldout*3 >= linRMSE {
		t.Errorf("quadratic holdout %.4f is not 3x better than linear %.4f", c.Prefill.RMSEHoldout, linRMSE)
	}
}

// A cost model that is not increasing in the work it prices cannot be a cost
// model; this fails if a fit coefficient ever goes negative on a longer context.
func TestCostsRiseMonotonically(t *testing.T) {
	c := cal(t)
	lastTokens, lastCtx := 0.0, 0.0
	for n := 64; n <= 8192; n *= 2 {
		s := c.Prefill.PrefillSeconds([]ExtendItem{{Tokens: n, MidCtx: 100000}})
		if s <= lastTokens {
			t.Fatalf("prefill seconds did not grow with tokens: %d -> %.6f after %.6f", n, s, lastTokens)
		}
		lastTokens = s
	}
	for m := 1000.0; m <= 500000; m *= 2 {
		s := c.Prefill.PrefillSeconds([]ExtendItem{{Tokens: 4096, MidCtx: m}})
		if s <= lastCtx {
			t.Fatalf("prefill seconds did not grow with context: %.0f -> %.6f after %.6f", m, s, lastCtx)
		}
		lastCtx = s
	}
	last := 0.0
	for bs := 1; bs <= 16; bs *= 2 {
		s := c.Decode.StepSeconds(bs, bs*200000)
		if s <= last {
			t.Fatalf("decode step did not grow with batch: %d -> %.6f after %.6f", bs, s, last)
		}
		last = s
	}
}

// Scaling a calibrated cost must scale the charge and nothing else, so the
// sensitivity sweep perturbs a measurement rather than a hardcoded number.
func TestScaledCostIsExactProportional(t *testing.T) {
	c := cal(t)
	items := []ExtendItem{{Tokens: 4096, MidCtx: 200000}}
	base := c.Prefill.PrefillSeconds(items)
	for _, k := range []float64{0.5, 0.75, 1.33, 2} {
		got := c.Prefill.Scaled(k).PrefillSeconds(items)
		if math.Abs(got-base*k) > 1e-12 {
			t.Errorf("Scaled(%v) gave %.9f, want %v x %.9f", k, got, k, base)
		}
	}
	dbase := c.Decode.StepSeconds(5, 5*200000)
	if got := c.Decode.Scaled(1.4).StepSeconds(5, 5*200000); math.Abs(got-dbase*1.4) > 1e-12 {
		t.Errorf("decode Scaled(1.4) gave %.9f, want %.9f", got, dbase*1.4)
	}
}

// Every decode line reports both the tokens held and the share of the pool that
// is, so their ratios must agree. A spread means the capacity was read off a
// line that does not describe the whole pool.
func TestDevicePoolRatioIsConsistentAcrossLines(t *testing.T) {
	steps, err := Parse(EmbeddedLog)
	if err != nil {
		t.Fatalf("parse embedded log: %v", err)
	}
	capacity := DevicePoolTokens(steps)
	if capacity < 1<<20 {
		t.Fatalf("device pool %d tokens, want at least a million", capacity)
	}
	var n int
	for _, s := range steps {
		if s.Kind != Decode || s.FullUsage <= 0 || s.FullTokens <= 0 {
			continue
		}
		n++
		r := float64(s.FullTokens) / s.FullUsage
		if math.Abs(r-float64(capacity))/float64(capacity) > 0.05 {
			t.Errorf("line %d implies a pool of %.0f, %.1f%% off the median %d", s.Line, r, 100*math.Abs(r-float64(capacity))/float64(capacity), capacity)
		}
	}
	if n < 10 {
		t.Fatalf("only %d decode lines carry both counters", n)
	}
}

// The decode base term is solved from the steady single-stream lines at their
// own context, so the model must reproduce those lines' step times.
func TestDecodeModelReproducesSteadyLines(t *testing.T) {
	steps, err := Parse(EmbeddedLog)
	if err != nil {
		t.Fatalf("parse embedded log: %v", err)
	}
	c := cal(t)
	if c.Decode.Samples < 10 {
		t.Fatalf("only %d steady decode lines to fit", c.Decode.Samples)
	}
	if c.Decode.NumDraft < 1 {
		t.Errorf("drafts %d, want at least one (MTP runs speculative decoding)", c.Decode.NumDraft)
	}
	var worst, n float64
	for i, s := range steps {
		if i == 0 || s.Kind != Decode || steps[i-1].Kind != Decode {
			continue
		}
		if s.RunningReq != 1 || s.Throughput <= 0 {
			continue
		}
		obs := s.AcceptLen / s.Throughput
		pred := c.Decode.StepSeconds(1, s.FullTokens)
		err := math.Abs(pred - obs) / obs
		worst = math.Max(worst, err)
		n++
	}
	if worst > 0.05 {
		t.Errorf("decode model is off by %.1f%% at its worst steady line", 100*worst)
	}
}
