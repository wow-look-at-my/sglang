package sim

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

// TestFeaturesTablePinsTheModes guards the comparison itself: each mode's rule set
// is what the Python at its own commit does, and those must differ only where the
// ports say they do. A silent edit here would compare NEW against an invented
// PREV, which is worse than no comparison.
func TestFeaturesTablePinsTheModes(t *testing.T) {
	old := FeaturesOf(ModeOld)
	assert.Equal(t, (Features{}), old)

	prev := FeaturesOf(ModePrev)
	assert.True(t, prev.Balancer)

	assert.Equal(t, CedeHalf, prev.Cede)

	// arg_groups/mixed_chunk_hook.py does not exist at f15db9ee6a.
	assert.False(t, prev.MixedChunk)

	assert.False(t, prev.BurstTokens)

	assert.False(t, prev.PiggybackCredit)

	assert.False(t, prev.Throttle)

	new := FeaturesOf(ModeNew)
	assert.False(t, !new.Balancer || !new.Throttle || !new.MixedChunk || !new.BurstTokens || !new.PiggybackCredit)

	assert.Equal(t, CedeFair, new.Cede)

}

// The debt floors are the line both balancers differ on besides the stall bound:
// a decode slice longer than the prefill it faces can never become credit under
// PREV, and under NEW is credit up to one decode batch.
func TestBalancerDebtFloorsDiffer(t *testing.T) {
	for _, tc := range []struct {
		name string
		prev bool
		want []float64
	}{
		{"PREV floors at zero", true, []float64{0, 0, 0}},
		{"NEW banks one decode batch", false, []float64{-0.002, -0.004, -0.004}},
	} {
		b := &Balancer{Prev: tc.prev}
		for _, want := range tc.want {
			b.UnsettledPf, b.UnsettledDc = 0.002, 0.004
			b.LastDecode, b.UnsettledN = 0.004, 1
			b.ShouldDeferPrefill(true, true, false)
			assert.Equal(t, want, b.Debt)

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
	assert.False(t, b.ShouldDeferPrefill(true, false, true))

	assert.False(t, b.Debt != 0 || b.BurstUsed != 0)

}

// A chunk continuation may not run while any prefill tokens are unrepaid, and a
// fresh prefill may run only inside the bound; the token budget then caps what the
// next batch may take.
func TestBurstBoundGatesContinuationsNotFreshWork(t *testing.T) {
	b := NewBalancer(FeaturesOf(ModeNew), 4096)
	b.Debt = 0.001
	b.BurstUsed = 100
	assert.True(t, b.ShouldDeferPrefill(true, true, true))

	assert.False(t, b.ShouldDeferPrefill(true, true, false))

	got := b.PrefillTokenBudget(false)
	assert.Equal(t, 4096-100, got)

	b.BurstUsed = 4096
	assert.True(t, b.ShouldDeferPrefill(true, true, false))

}

// Cost per prefill token rises with the context attention reads, so a token
// bound alone lets a chunked prompt's later chunks stall decode for several
// times the seconds one chunk promises. The continuation's cap is the GPU
// seconds one chunk is worth at the measured average rate, priced at the
// marginal rate of the batch that ran.
func TestContinuationBudgetIsDenominatedInSeconds(t *testing.T) {
	b := NewBalancer(FeaturesOf(ModeNew), 4096)
	b.OnLaunch(true, 4000, 0, 0)
	b.OnFinish(0.8)
	b.OnLaunch(false, 0, 0, 0.8)
	b.OnFinish(1.0)
	avg := b.PrefillSecondsPerToken()
	require.Equal(t, 0.0002, avg)

	b.OnLaunch(true, 2000, 0, 1.0)
	b.OnFinish(1.8)
	average, marginal := b.PrefillSecondsPerToken(), b.marginalSecondsPerToken()
	require.Equal(t, 0.0004, marginal)

	got := b.PrefillTokenBudget(true)
	want := int(4096 * average / marginal)
	assert.Equal(t, want, got)

	assert.LessOrEqual(t, float64(got)*marginal, 4096*average)

	assert.Less(t, got, 4096)

}

// The prefill rates are prefill measurements: a decode batch also reports extend
// tokens (one per running request) and costs an order of magnitude more per token,
// so letting one in would price prefill work at decode cost and un-bind every
// continuation cap.
func TestDecodeBatchTokensDoNotEnterThePrefillRates(t *testing.T) {
	b := NewBalancer(FeaturesOf(ModeNew), 4096)
	b.OnLaunch(true, 4000, 0, 0)
	b.OnFinish(0.8)
	b.OnLaunch(false, 8, 0, 0.8) // a decode step of multiple requests, as the scheduler
	b.OnFinish(1.0)
	avg := b.PrefillSecondsPerToken()
	assert.Equal(t, 0.0002, avg)

	m := b.marginalSecondsPerToken()
	assert.Equal(t, 0.0002, m)

	got := b.PrefillTokenBudget(true)
	assert.Equal(t, 4096, got)

}

// PREV has no token bound: the balance alone decides, and it never goes negative.
func TestPrevHasNoTokenBound(t *testing.T) {
	b := NewBalancer(FeaturesOf(ModePrev), 0)
	b.Debt = 0
	assert.False(t, b.ShouldDeferPrefill(true, true, true))

	got := b.PrefillTokenBudget(true)
	assert.Equal(t, -1, got)

	b.Debt = 1e-9
	assert.True(t, b.ShouldDeferPrefill(true, true, true))

}
