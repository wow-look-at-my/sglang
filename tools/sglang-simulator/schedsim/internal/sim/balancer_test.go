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
	if got := b.PrefillTokenBudget(false); got != 4096-100 {
		t.Errorf("token budget %d, want %d", got, 4096-100)
	}
	b.BurstUsed = 4096
	if !b.ShouldDeferPrefill(true, true, false) {
		t.Error("a fresh prefill ran once the bound was spent")
	}
}

// Cost per prefill token rises with the context attention reads, so a token bound
// alone lets a chunked prompt's later chunks stall decode for several times the
// seconds one chunk promises. The continuation's cap is the GPU seconds one chunk
// is worth at the measured average rate, priced at the marginal rate of the batch
// that just ran.
func TestContinuationBudgetIsDenominatedInSeconds(t *testing.T) {
	b := NewBalancer(FeaturesOf(ModeNew), 4096)
	// The first prefill measures 200 us per token, and a decode batch follows it.
	b.OnLaunch(true, 4000, 0, 0)
	b.OnFinish(0.8)
	b.OnLaunch(false, 0, 0, 0.8)
	b.OnFinish(1.0)
	if avg := b.PrefillSecondsPerToken(); avg != 0.0002 {
		t.Fatalf("average prefill rate %v, want 0.0002 s/token", avg)
	}
	// The same prompt's next chunk reads a longer context: 2000 tokens cost the
	// same 0.8 s, twice the prefill rate measured so far.
	b.OnLaunch(true, 2000, 0, 1.0)
	b.OnFinish(1.8)
	average, marginal := b.PrefillSecondsPerToken(), b.marginalSecondsPerToken()
	if marginal != 0.0004 {
		t.Fatalf("marginal rate %v, want 0.0004 s/token", marginal)
	}
	got := b.PrefillTokenBudget(true)
	if want := int(4096 * average / marginal); got != want {
		t.Errorf("continuation budget %d, want %d", got, want)
	}
	if float64(got)*marginal > 4096*average {
		t.Errorf("%d tokens at %v s/token cost %v s, more than the %v s one chunk promises",
			got, marginal, float64(got)*marginal, 4096*average)
	}
	if got >= 4096 {
		t.Errorf("budget %d does not bind at twice the measured average rate", got)
	}
}

// The prefill rates are prefill measurements: a decode batch also reports extend
// tokens (one per running request) and costs an order of magnitude more per token,
// so letting one in would price prefill work at decode cost and un-bind every
// continuation cap.
func TestDecodeBatchTokensDoNotEnterThePrefillRates(t *testing.T) {
	b := NewBalancer(FeaturesOf(ModeNew), 4096)
	b.OnLaunch(true, 4000, 0, 0)
	b.OnFinish(0.8)
	b.OnLaunch(false, 8, 0, 0.8) // a decode step of 8 requests, as the scheduler reports it
	b.OnFinish(1.0)
	if avg := b.PrefillSecondsPerToken(); avg != 0.0002 {
		t.Errorf("average prefill rate %v after a decode batch, want 0.0002 s/token", avg)
	}
	if m := b.marginalSecondsPerToken(); m != 0.0002 {
		t.Errorf("marginal rate %v after a decode batch, want 0.0002 s/token", m)
	}
	if got := b.PrefillTokenBudget(true); got != 4096 {
		t.Errorf("continuation budget %d at equal rates, want the full chunk", got)
	}
}

// PREV has no token bound: the balance alone decides, and it never goes negative.
func TestPrevHasNoTokenBound(t *testing.T) {
	b := NewBalancer(FeaturesOf(ModePrev), 0)
	b.Debt = 0
	if b.ShouldDeferPrefill(true, true, true) {
		t.Error("PREV deferred with a repaid balance")
	}
	if got := b.PrefillTokenBudget(true); got != -1 {
		t.Errorf("PREV token budget %d, want -1 (Python None)", got)
	}
	b.Debt = 1e-9
	if !b.ShouldDeferPrefill(true, true, true) {
		t.Error("PREV ran a chunk continuation while prefill still owed time")
	}
}
