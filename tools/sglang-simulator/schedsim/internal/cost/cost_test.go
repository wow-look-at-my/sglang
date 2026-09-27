package cost

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func calibrated(t *testing.T) Calibration {
	t.Helper()
	cal, err := Calibrate()
	require.NoError(t, err)
	return cal
}

func rms(cs []ChunkCheck) (float64, float64) {
	var sum, worst float64
	for _, c := range cs {
		e := c.RelErr()
		sum += e * e
		worst = math.Max(worst, math.Abs(e))
	}
	return math.Sqrt(sum / float64(len(cs))), worst
}

func TestPrefillCurveReproducesTheLoggedColdPrompt(t *testing.T) {
	cal := calibrated(t)
	require.Len(t, cal.FitChunks, 104, "the first cold prompt's full chunks after warm-up")
	fit, worst := rms(cal.FitChunks)
	t.Logf("fit rms %.2f%% worst %.2f%%, model %+v", fit*100, worst*100, cal.Model)
	require.Less(t, fit, 0.03)
	require.Less(t, worst, 0.10)

	var measured, predicted float64
	for _, c := range cal.FitChunks {
		measured += c.Measured
		predicted += c.Predicted
	}
	require.InEpsilon(t, measured, predicted, 0.005, "whole-prompt seconds")
}

func TestPrefillCurvePredictsTheHeldOutColdPrompt(t *testing.T) {
	cal := calibrated(t)
	require.Len(t, cal.HeldOut, 57)
	require.Equal(t, 2816, cal.HeldOut[0].Start, "the second prompt's head rode in the mixed batch")
	held, worst := rms(cal.HeldOut)
	t.Logf("held-out rms %.2f%% worst %.2f%%", held*100, worst*100)
	require.Less(t, held, 0.05)
	require.Less(t, worst, 0.08)
}

func TestMixedBatchIsPricedWithinItsLoggedTime(t *testing.T) {
	cal := calibrated(t)
	require.Equal(t, 4000, cal.Mixed.Tokens)
	t.Logf("mixed batch measured %.3f s, predicted %.3f s", cal.Mixed.Measured, cal.Mixed.Predicted)
	require.Less(t, math.Abs(cal.Mixed.RelErr()), 0.15)
}

func TestDecodeStepAndPoolComeFromTheDecodeLines(t *testing.T) {
	cal := calibrated(t)
	m := cal.Model
	t.Logf("pool [%.0f, %.0f], decode %.4f s at ctx %d, accept %.2f", cal.PoolLo, cal.PoolHi,
		cal.DecodeStepSecs, cal.DecodeCtx, m.AcceptLen)
	require.Greater(t, cal.PoolHi, cal.PoolLo)
	require.Less(t, (cal.PoolHi-cal.PoolLo)/cal.PoolLo, 0.01, "usage lines pin the pool to one percent")
	require.InDelta(t, cal.DecodeStepSecs, m.DecodeSeconds(1, cal.DecodeCtx), 1e-12)
	// The logged single stream runs at the rate its own lines report.
	rate := m.AcceptLen / m.DecodeSeconds(1, cal.DecodeCtx)
	require.InDelta(t, 181, rate, 25)
	require.Greater(t, m.DecodeBase, 0.0)
}

func TestExtendSecondsIsAdditive(t *testing.T) {
	m := calibrated(t).Model
	whole := m.ExtendSeconds(1000, 9000)
	parts := m.ExtendSeconds(1000, 4000) + m.ExtendSeconds(5000, 5000)
	require.InDelta(t, whole, parts, 1e-12)
	require.Greater(t, m.ExtendSeconds(400000, 4096), m.ExtendSeconds(0, 4096))
}
