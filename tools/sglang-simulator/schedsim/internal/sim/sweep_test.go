package sim

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"schedsim/internal/trace"
)

// TestSweepCoversThePerturbations pins the sensitivity sweep's breadth: one-at-a-time
// perturbations of the calibrated cost terms, the scheduler's own knobs and the
// workload, each compared across the policies. The first variant is the reference the
// others are read against and perturbs nothing by design; every other row has to change
// something, or it measures the reference twice.
func TestSweepCoversThePerturbations(t *testing.T) {
	steps := mustSteps(t)
	cost := NewCost(calib(t))
	linear := trace.FitPrefillLinear(steps, BaselineChunkSize)
	variants := SweepVariants(linear)
	require.GreaterOrEqual(t, len(variants), 21)

	assert.False(t, variants[0].Cost != nil || variants[0].B != nil || variants[0].Cfg != nil)

	for _, v := range variants[1:] {
		assert.False(t, v.Cost == nil && v.B == nil && v.Cfg == nil)

	}

	rows := RunSweep(cost, linear, testEpisodePtr(), []int64{1, 2, 3}, 7, 0)
	require.Equal(t, len(variants), len(rows))

	var swept, broken int
	for i, r := range rows {
		if r.SweptA {
			swept++
		}
		if len(r.Breaks) > 0 {
			broken++
		}
		t.Logf("%-44s %s", r.Name, ifElse(len(r.Breaks) > 0, "breaks: "+joinBreaks(r.Breaks), "no contract break"))
		assert.Equal(t, variants[i].Name, r.Name)

		assert.False(t, r.B[PolicyIndex(ModeNew)] == (Metrics{}) || r.B[PolicyIndex(ModeOld)] == (Metrics{}))

	}
	assert.NotEqual(t, 0, swept)

	t.Logf("sweep: %d variants, %d also on scenario A, %d where the contract does not hold",
		len(rows), swept, broken)
}

func joinBreaks(b []string) string {
	out := ""
	for i, s := range b {
		if i > 0 {
			out += "; "
		}
		out += s
	}
	return out
}

func ifElse(cond bool, a, b string) string {
	if cond {
		return a
	}
	return b
}
