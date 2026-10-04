package kv

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

// A pool with no host tier loses an evicted prefix outright, so the next turn
// of that conversation recomputes its whole input; a host tier that can hold it
// turns the same eviction into a reload. This is the difference between both
// sides of the thrash collapse, so both directions are asserted on one run.
func TestEvictionWithAndWithoutHostTier(t *testing.T) {
	const (
		convs   = 10
		ctxLen  = 300_000
		turnNew = 800
		out     = 400
	)
	for _, tc := range []struct {
		name       string
		hostMul    float64
		wantRecomp bool
	}{
		{"no host tier", 0, true},
		{"host tier holds the working set", float64(convs*ctxLen) / float64(ctxLen), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dev := 2 * ctxLen
			host := int(float64(dev) * tc.hostMul)
			p := New(dev, host)
			var now float64
			for round := 0; round < 3; round++ {
				for c := 0; c < convs; c++ {
					devHit, hostHit := p.Lookup(c, ctxLen+turnNew)
					p.CountRecompute(c, ctxLen+turnNew, devHit, hostHit)
					p.Admit(c, ctxLen+turnNew, out, devHit, hostHit, now)
					now += 1
					p.Release(c, ctxLen+turnNew+out, now, ctxLen+turnNew+out)
				}
			}
			recomputes := 0
			for _, n := range p.Recomputes {
				recomputes += n
			}
			require.False(t, tc.wantRecomp && recomputes == 0)

			if !tc.wantRecomp {
				require.Equal(t, 0, recomputes)

				require.NotEqual(t, 0, p.ReloadTokens)

			}
		})
	}
}

// An evicted prefix that lands on host is priced as a reload, and one that
// cannot fit the host tier at all as a recompute. A build that classified both
// the same way would report a collapse cost the deployment never pays.
func TestRebuildSecondsPricesHostAndRecompute(t *testing.T) {
	const (
		ctxLen    = 300_000
		recompute = 68.4e-6
	)
	// A 450K-token reservation against a 600K pool that already holds one 300K prefix has to take that prefix.
	admit := func(p *Pool) []Eviction { return p.Admit(2, 450_000, 0, 0, 0, 1) }
	t.Run("spilled to host", func(t *testing.T) {
		p := New(ctxLen*2, ctxLen*4)
		p.Warm(1, ctxLen, 0)
		evs := admit(p)
		require.False(t, len(evs) != 1 || !evs[0].ToHost)

		got := p.RebuildSeconds(evs, recompute)
		want := float64(ctxLen) * ReloadSecondsPerToken
		assert.False(t, got < want*0.99 || got > want*1.01)

		assert.Greater(t, recompute*ctxLen, got)

	})
	t.Run("no host tier", func(t *testing.T) {
		p := New(ctxLen*2, 0)
		p.Warm(1, ctxLen, 0)
		evs := admit(p)
		require.False(t, len(evs) != 1 || evs[0].ToHost)

		got := p.RebuildSeconds(evs, recompute)
		want := float64(ctxLen) * recompute
		assert.False(t, got < want*0.99 || got > want*1.01)

	})
}

// Capacity is the larger tier, because write-through mirrors every cached
// prefix there; the throttle sizes the live set against it.
func TestCapacityIsTheLargerTier(t *testing.T) {
	got := New(100, 400).Capacity()
	assert.Equal(t, 400, got)

	got := New(400, 100).Capacity()
	assert.Equal(t, 400, got)

}
