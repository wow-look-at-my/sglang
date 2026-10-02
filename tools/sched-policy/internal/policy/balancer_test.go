package policy

import (
	"github.com/stretchr/testify/require"
	"math"
	"strings"
	"testing"
)

const (
	chunkTokens = 4096
	prefillSecs = 0.8
	decodeSecs  = 0.02
)

type clock struct{ now float64 }

func balancer() *Balancer {
	b := chunkTokens
	return NewBalancer(&b)
}

type runOpts struct {
	steps          int
	overlap        bool
	prefillPending bool
	decodeRows     int
	continuesChunk *bool
	prefillTokens  int
}

// run drives the scheduler loop shape with a long chunked prompt pending and
// returns the launched classes.
func run(b *Balancer, c *clock, o runOpts) []bool {
	if o.prefillTokens == 0 {
		o.prefillTokens = chunkTokens
	}
	continues := o.prefillPending
	if o.continuesChunk != nil {
		continues = *o.continuesChunk
	}
	var launched []bool
	var inFlight []float64
	gpuFreeAt := 0.0
	for range o.steps {
		defer_ := b.ShouldDeferPrefill(o.prefillPending, true, continues)
		isPrefill := o.prefillPending && !defer_
		rows, tokens, class := 0, 0, ClassDecode
		if isPrefill {
			rows, tokens, class = o.decodeRows, o.prefillTokens+o.decodeRows, ClassPrefill
		}
		b.OnBatchLaunched(class, tokens, rows, c.now)
		start := max(c.now, gpuFreeAt)
		seconds := decodeSecs
		if isPrefill {
			seconds = prefillSecs * float64(tokens) / chunkTokens
		}
		gpuFreeAt = start + seconds
		inFlight = append(inFlight, gpuFreeAt)
		launched = append(launched, isPrefill)
		limit := 0
		if o.overlap {
			limit = 1
		}
		if len(inFlight) > limit {
			c.now = max(c.now, inFlight[0])
			inFlight = inFlight[1:]
			b.OnBatchFinished(c.now)
		}
	}
	return launched
}

func runOne(b *Balancer, c *clock, class BatchClass, seconds float64, tokens int) {
	b.OnBatchLaunched(class, tokens, 0, c.now)
	c.now += seconds
	b.OnBatchFinished(c.now)
}

func prefillShare(launched []bool) float64 {
	prefill, decode := 0.0, 0.0
	for _, p := range launched {
		if p {
			prefill += prefillSecs
		} else {
			decode += decodeSecs
		}
	}
	return prefill / (prefill + decode)
}

func longestPrefillRun(launched []bool) int {
	var sb strings.Builder
	for _, p := range launched {
		if p {
			sb.WriteByte('P')
		} else {
			sb.WriteByte('D')
		}
	}
	longest := 0
	for _, r := range strings.Split(sb.String(), "D") {
		longest = max(longest, len(r))
	}
	return longest
}

func near(t *testing.T, got, want, delta float64) {
	t.Helper()
	require.LessOrEqual(t, math.Abs(got-want), delta)

}

func pending(b *Balancer, continues bool) bool { return b.ShouldDeferPrefill(true, true, continues) }

func ptr(v bool) *bool { return &v }

func TestLongChunkedPrefillDoesNotStarveDecode(t *testing.T) {
	c := &clock{}
	launched := run(balancer(), c, runOpts{steps: 2000, prefillPending: true})
	near(t, prefillShare(launched), 0.5, 0.02)
	require.Equal(t, 1, longestPrefillRun(launched))

}

func TestOverlapStallsDecodeForOneChunkAtMost(t *testing.T) {
	c := &clock{}
	launched := run(balancer(), c, runOpts{steps: 2000, overlap: true, prefillPending: true})
	near(t, prefillShare(launched), 0.5, 0.05)
	require.Equal(t, 1, longestPrefillRun(launched))

}

func TestOvershootOfLastDecodeCountsTowardPrefill(t *testing.T) {
	c := &clock{}
	b := balancer()
	pending(b, false)
	runOne(b, c, ClassPrefill, 0.8, 0)
	runOne(b, c, ClassDecode, 0.5, 0)
	runOne(b, c, ClassDecode, 0.5, 0)
	require.False(t, pending(b, true))

	near(t, b.Debt(), -0.2, 1e-9)
	for range 4 {
		runOne(b, c, ClassDecode, 0.5, 0)
	}
	pending(b, true)
	near(t, b.Debt(), -0.5, 1e-9)
}

func TestPiggybackedDecodeDoesNotReplaceTheDecodeShare(t *testing.T) {
	c := &clock{}
	launched := run(balancer(), c, runOpts{steps: 2000, overlap: true, prefillPending: true, decodeRows: 5})
	near(t, prefillShare(launched), 0.5, 0.05)
	require.Equal(t, 1, longestPrefillRun(launched))

}

func TestMixedBatchIsChargedWithoutItsDecodeRows(t *testing.T) {
	c := &clock{}
	b := balancer()
	runOne(b, c, ClassPrefill, 1.0, 1000)
	b.ShouldDeferPrefill(true, false, false)
	b.OnBatchLaunched(ClassPrefill, 1500, 500, c.now)
	c.now += 1.5
	b.OnBatchFinished(c.now)
	pending(b, false)
	near(t, b.Debt(), 1.0, 1e-9)
}

func TestMeasuresPrefillSecondsPerToken(t *testing.T) {
	c := &clock{}
	b := balancer()
	require.False(t, b.PrefillSecondsPerToken() != 0 || b.MarginalPrefillSecondsPerToken() != 0)

	runOne(b, c, ClassPrefill, 0.5, 1000)
	runOne(b, c, ClassDecode, 0.02, 6)
	runOne(b, c, ClassDecode, 0.03, 8)
	near(t, b.PrefillSecondsPerToken(), 0.5/1000, 1e-12)
	near(t, b.MarginalPrefillSecondsPerToken(), 0.5/1000, 1e-12)
}

func TestContinuationChunkIsCappedByItsMeasuredSeconds(t *testing.T) {
	c := &clock{}
	b := balancer()
	runOne(b, c, ClassPrefill, 0.8, 4000)
	runOne(b, c, ClassDecode, 0.8, 0)
	require.False(t, pending(b, true))

	runOne(b, c, ClassPrefill, 0.8, 2000)
	runOne(b, c, ClassDecode, 1.6, 0)
	require.False(t, pending(b, true))

	near(t, b.MarginalPrefillSecondsPerToken(), 0.0004, 1e-12)
	budget := *b.PrefillTokenBudget(true)
	promised := chunkTokens * b.PrefillSecondsPerToken()
	marginal := b.MarginalPrefillSecondsPerToken()
	require.False(t, float64(budget)*marginal > promised || float64(budget+1)*marginal <= promised || budget >= chunkTokens)

}

func TestFreshPrefillIsCappedInTokens(t *testing.T) {
	c := &clock{}
	b := balancer()
	runOne(b, c, ClassPrefill, 0.02, 1000)
	runOne(b, c, ClassPrefill, 0.8, 2000)
	require.False(t, pending(b, false))

	got := *b.PrefillTokenBudget(false)
	require.Equal(t, chunkTokens-3000, got)

	cont := float64(*b.PrefillTokenBudget(true))
	require.False(t, cont >= chunkTokens || cont*b.MarginalPrefillSecondsPerToken() > chunkTokens*b.PrefillSecondsPerToken())

}

func TestFreshPrefillsBackToBackStopAtTheChunkBound(t *testing.T) {
	c := &clock{}
	launched := run(balancer(), c, runOpts{steps: 500, prefillPending: true, continuesChunk: ptr(false), prefillTokens: chunkTokens / 4})
	require.Equal(t, 4, longestPrefillRun(launched))

}

func TestShortPrefillsShareOneChunkBudget(t *testing.T) {
	c := &clock{}
	b := balancer()
	runOne(b, c, ClassPrefill, 0.3, 1000)
	require.False(t, pending(b, false) || b.Debt() <= 0)

	got := *b.PrefillTokenBudget(false)
	require.Equal(t, chunkTokens-1000, got)

	runOne(b, c, ClassPrefill, 0.9, chunkTokens-1000)
	require.True(t, pending(b, false))

}

func TestChunkedContinuationWaitsForDecode(t *testing.T) {
	c := &clock{}
	b := balancer()
	runOne(b, c, ClassPrefill, 0.3, 1000)
	require.True(t, pending(b, true))

	runOne(b, c, ClassDecode, 0.3, 0)
	require.False(t, pending(b, true))

	got := *b.PrefillTokenBudget(true)
	require.Equal(t, chunkTokens, got)

}

func TestWithoutChunkedPrefillDefersOnDebtAlone(t *testing.T) {
	c := &clock{}
	b := NewBalancer(nil)
	runOne(b, c, ClassPrefill, 0.3, 1000)
	require.False(t, !pending(b, false) || b.PrefillTokenBudget(false) != nil)

}

func TestNeverDefersWithoutRunningDecode(t *testing.T) {
	c := &clock{}
	b := balancer()
	for range 10 {
		require.False(t, b.ShouldDeferPrefill(true, false, true))

		runOne(b, c, ClassPrefill, 1.0, chunkTokens)
	}
	require.Equal(t, float64(0), b.Debt())

}

func TestDecodeDoesNotBankCredit(t *testing.T) {
	c := &clock{}
	b := balancer()
	run(b, c, runOpts{steps: 100})
	for range 100 {
		pending(b, true)
		runOne(b, c, ClassDecode, decodeSecs, 0)
	}
	near(t, b.Debt(), -decodeSecs, 1e-9)
	require.Equal(t, 1, longestPrefillRun(run(b, c, runOpts{steps: 50, prefillPending: true})))

}

func TestIdleGapIsNotCharged(t *testing.T) {
	c := &clock{}
	b := balancer()
	runOne(b, c, ClassPrefill, 0.5, 1)
	c.now += 60
	runOne(b, c, ClassDecode, 0.02, 0)
	pending(b, true)
	near(t, b.Debt(), 0.48, 1e-9)
}

func TestOtherBatchesAndStrayFinishesAreNotCharged(t *testing.T) {
	c := &clock{}
	b := balancer()
	b.OnBatchFinished(c.now)
	runOne(b, c, ClassOther, 2.0, 0)
	pending(b, true)
	require.Equal(t, float64(0), b.Debt())

}
