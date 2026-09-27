package sim

import "testing"

// TestFeaturesTablePinsTheModes guards the comparison itself: each mode's rule set
// is what the Python at its own commit does, and the three must differ only where
// the ports say they do. A silent edit here would compare NEW against an invented
// PREV, which is worse than no comparison.
func TestFeaturesTablePinsTheModes(t *testing.T) {
	old := FeaturesOf(ModeOld)
	if old != (Features{}) {
		t.Errorf("OLD = %+v, want no balancer, no cede, no throttle, no mixed chunk", old)
	}

	prev := FeaturesOf(ModePrev)
	if !prev.Balancer {
		t.Error("PREV must run the balancer")
	}
	if prev.Cede != CedeHalf {
		t.Errorf("PREV cede = %v, want the plain half cap", prev.Cede)
	}
	// arg_groups/mixed_chunk_hook.py does not exist at f15db9ee6a, so nothing
	// resolves --enable-mixed-chunk on: PREV's streams take tokens in pure decode
	// batches only. A PREV that rode a token inside every chunk would report an
	// inflated per-stream decode rate and a shrunken longest stall.
	if prev.MixedChunk {
		t.Error("PREV must not form mixed batches: its commit has no mixed-chunk auto-enable")
	}
	if prev.BurstTokens {
		t.Error("PREV must have no stall bound in prefill tokens")
	}
	if prev.PiggybackCredit {
		t.Error("PREV charges a mixed batch whole to prefill, so it cannot credit its rows")
	}
	if prev.Throttle {
		t.Error("the eviction throttle is not in PREV")
	}

	new := FeaturesOf(ModeNew)
	if !new.Balancer || !new.Throttle || !new.MixedChunk || !new.BurstTokens || !new.PiggybackCredit {
		t.Errorf("NEW = %+v, want balancer, fair cede, throttle, mixed chunk, stall bound, row credit", new)
	}
	if new.Cede != CedeFair {
		t.Errorf("NEW cede = %v, want the amortized fair share", new.Cede)
	}
}

// The debt floors are the line the two balancers differ on besides the stall
// bound: a decode slice longer than the prefill it faces can never become credit
// under PREV, and under NEW is credit up to one decode batch.
func TestBalancerDebtFloorsDiffer(t *testing.T) {
	for _, tc := range []struct {
		name string
		prev bool
		want []float64
	}{
		{"PREV floors at zero", true, []float64{0, 0, 0}},
		// A 2 ms prefill against a 4 ms decode leaves -2 ms each round, and the
		// bank stops at one decode batch rather than accumulating.
		{"NEW banks one decode batch", false, []float64{-0.002, -0.004, -0.004}},
	} {
		b := &Balancer{Prev: tc.prev}
		for round, want := range tc.want {
			b.UnsettledPf, b.UnsettledDc = 0.002, 0.004
			b.LastDecode, b.UnsettledN = 0.004, 1
			b.ShouldDeferPrefill(true, true, false)
			if b.Debt != want {
				t.Errorf("%s round %d: debt = %v, want %v", tc.name, round+1, b.Debt, want)
			}
		}
	}
}

// Without contention nothing is ever deferred, and the balance is cleared rather
// than carried, so a prefill that could not run buys no burst later.
func TestBalancerWithoutContentionClearsState(t *testing.T) {
	b := NewBalancer(FeaturesOf(ModeNew), 4096)
	b.OnLaunch(true, 4096, 0, 0)
	b.OnFinish(0.3)
	b.Debt = 0.2
	b.BurstUsed = 4096
	if b.ShouldDeferPrefill(true, false, true) {
		t.Error("deferred with no decode runnable, which would idle the GPU")
	}
	if b.Debt != 0 || b.BurstUsed != 0 {
		t.Errorf("state survived an uncontended decision: debt %v, burst %d", b.Debt, b.BurstUsed)
	}
}

// A chunk continuation may not run while any prefill tokens are unrepaid, and a
// fresh prefill may run only inside the bound; the token budget then caps what the
// next batch may take.
func TestBurstBoundGatesContinuationsNotFreshWork(t *testing.T) {
	b := NewBalancer(FeaturesOf(ModeNew), 4096)
	b.Debt = 0.001
	b.BurstUsed = 100
	if !b.ShouldDeferPrefill(true, true, true) {
		t.Error("a chunk continuation ran with burst_used > 0")
	}
	if b.ShouldDeferPrefill(true, true, false) {
		t.Error("a fresh prefill was deferred below the bound")
	}
	if got := b.PrefillTokenBudget(); got != 4096-100 {
		t.Errorf("token budget %d, want %d", got, 4096-100)
	}
	b.BurstUsed = 4096
	if !b.ShouldDeferPrefill(true, true, false) {
		t.Error("a fresh prefill ran once the bound was spent")
	}
}

// PREV has no token bound: the balance alone decides, and it never goes negative.
func TestPrevHasNoTokenBound(t *testing.T) {
	b := NewBalancer(FeaturesOf(ModePrev), 0)
	b.Debt = 0
	if b.ShouldDeferPrefill(true, true, true) {
		t.Error("PREV deferred with a repaid balance")
	}
	if got := b.PrefillTokenBudget(); got != -1 {
		t.Errorf("PREV token budget %d, want -1 (Python None)", got)
	}
	b.Debt = 1e-9
	if !b.ShouldDeferPrefill(true, true, true) {
		t.Error("PREV ran a chunk continuation while prefill still owed time")
	}
}
