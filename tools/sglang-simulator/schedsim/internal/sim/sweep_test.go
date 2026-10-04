package sim

import (
	"testing"

	"schedsim/internal/trace"
)

// TestSweepCoversThePerturbations pins the sensitivity sweep's breadth: one-at-a-time
// perturbations of the calibrated cost terms, the scheduler's own knobs and the
// workload, each compared across the three policies. The first variant is the reference
// the others are read against and perturbs nothing by design; every other row has to
// change something, or it measures the reference twice.
func TestSweepCoversThePerturbations(t *testing.T) {
	steps := mustSteps(t)
	cost := NewCost(calib(t))
	linear := trace.FitPrefillLinear(steps, BaselineChunkSize)
	variants := SweepVariants(linear)
	if len(variants) < 21 {
		t.Fatalf("the sweep has %d variants, want at least 21", len(variants))
	}
	if variants[0].Cost != nil || variants[0].B != nil || variants[0].Cfg != nil {
		t.Errorf("the sweep's first variant %q is not the untouched reference", variants[0].Name)
	}
	for _, v := range variants[1:] {
		if v.Cost == nil && v.B == nil && v.Cfg == nil {
			t.Errorf("variant %q perturbs nothing", v.Name)
		}
	}

	rows := RunSweep(cost, linear, testEpisodePtr(), []int64{1, 2, 3}, 7, 0)
	if len(rows) != len(variants) {
		t.Fatalf("%d rows for %d variants", len(rows), len(variants))
	}
	var swept, broken int
	for i, r := range rows {
		if r.SweptA {
			swept++
		}
		if len(r.Breaks) > 0 {
			broken++
		}
		t.Logf("%-44s %s", r.Name, ifElse(len(r.Breaks) > 0, "breaks: "+joinBreaks(r.Breaks), "no contract break"))
		if r.Name != variants[i].Name {
			t.Errorf("row %d is named %q, want %q", i, r.Name, variants[i].Name)
		}
		if r.B[PolicyIndex(ModeNew)] == (Metrics{}) || r.B[PolicyIndex(ModeOld)] == (Metrics{}) {
			t.Errorf("variant %q measured no metrics on scenario B", r.Name)
		}
	}
	if swept == 0 {
		t.Error("no variant reported scenario A, so the episode's numbers are untested")
	}
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
