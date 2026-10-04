package trace

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"math"
	"testing"
)

func cal(t *testing.T) Calibration {
	t.Helper()
	c := Calibrate(mustParse(t), 4096, 64)
	require.GreaterOrEqual(t, c.Prefill.Samples, 10)

	return c
}

// The quadratic form is a claim about the log, not a constant: the second cold
// run was never looked at by the fit and must still be predicted closely, and a
// linear form must be visibly worse on it. If a future log makes the quadratic
// unnecessary this test is the place that gets argued.
func TestPrefillFitPredictsTheHeldOutRun(t *testing.T) {
	steps := mustParse(t)
	runs := ColdRuns(steps, 4096)
	require.GreaterOrEqual(t, len(runs), 2)

	c := cal(t)
	assert.False(t, c.Prefill.Samples != incident.C1Chunks-1 || c.Prefill.HoldoutSamples != incident.R3Chunks-1)

	// The log's chunks were priced by a known quadratic, so the fit recovers it.
	want := incident.Model
	for _, f := range []struct {
		name      string
		got, want float64
	}{
		{"per token", c.Prefill.PerToken, want.PerToken},
		{"per token-ctx", c.Prefill.PerTokenCtx, want.PerTokenCtx},
		{"per token-ctx^2", c.Prefill.PerTokenCtxSq, want.PerTokenCtxSq},
	} {
		assert.LessOrEqual(t, math.Abs(f.got-f.want), 0.01*f.want)

	}
	assert.LessOrEqual(t, c.Prefill.RMSEHoldout, 0.03)

	assert.LessOrEqual(t, c.Prefill.RMSEFit, 0.03)

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
	assert.Less(t, c.Prefill.RMSEHoldout*3, linRMSE)

}

// A cost model that is not increasing in the work it prices cannot be a cost
// model; this fails if a fit coefficient ever goes negative on a longer context.
func TestCostsRiseMonotonically(t *testing.T) {
	c := cal(t)
	lastTokens, lastCtx := 0.0, 0.0
	for n := 64; n <= 8192; n *= 2 {
		s := c.Prefill.PrefillSeconds([]ExtendItem{{Tokens: n, MidCtx: 100000}})
		require.Greater(t, s, lastTokens)

		lastTokens = s
	}
	for m := 1000.0; m <= 500000; m *= 2 {
		s := c.Prefill.PrefillSeconds([]ExtendItem{{Tokens: 4096, MidCtx: m}})
		require.Greater(t, s, lastCtx)

		lastCtx = s
	}
	last := 0.0
	for bs := 1; bs <= 16; bs *= 2 {
		s := c.Decode.StepSeconds(bs, bs*200000)
		require.Greater(t, s, last)

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
		assert.LessOrEqual(t, math.Abs(got-base*k), 1e-12)

	}
	dbase := c.Decode.StepSeconds(5, 5*200000)
	got := c.Decode.Scaled(1.4).StepSeconds(5, 5*200000)
	assert.LessOrEqual(t, math.Abs(got-dbase*1.4), 1e-12)

}

// Every decode line reports both the tokens held and the share of the pool that
// is, so their ratios must agree. A spread means the capacity was read off a
// line that does not describe the whole pool.
func TestDevicePoolRatioIsConsistentAcrossLines(t *testing.T) {
	steps := mustParse(t)
	capacity := DevicePoolTokens(steps)
	pool := incident.Model.Pool
	require.LessOrEqual(t, math.Abs(float64(capacity-pool)), 0.01*float64(pool))

	var n int
	for _, s := range steps {
		if s.Kind != Decode || s.FullUsage <= 0 || s.FullTokens <= 0 {
			continue
		}
		n++
		r := float64(s.FullTokens) / s.FullUsage
		assert.LessOrEqual(t, math.Abs(r-float64(capacity))/float64(capacity), 0.05)

	}
	require.GreaterOrEqual(t, n, 10)

}

// The decode base term is solved from the steady single-stream lines at their
// own context, so the model must reproduce those lines' step times.
func TestDecodeModelReproducesSteadyLines(t *testing.T) {
	steps := mustParse(t)
	c := cal(t)
	require.GreaterOrEqual(t, c.Decode.Samples, 10)

	assert.GreaterOrEqual(t, c.Decode.NumDraft, 1)

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
		err := math.Abs(pred-obs) / obs
		worst = math.Max(worst, err)
		n++
	}
	assert.LessOrEqual(t, worst, 0.05)

}
