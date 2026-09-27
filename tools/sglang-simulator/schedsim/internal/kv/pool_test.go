package kv

import "testing"

// A pool with no host tier loses an evicted prefix outright, so the next turn
// of that conversation recomputes its whole input; a host tier that can hold it
// turns the same eviction into a reload. This is the difference between the two
// sides of the thrash collapse, so both directions are asserted on one run.
func TestEvictionWithAndWithoutHostTier(t *testing.T) {
	const (
		convs   = 10
		ctxLen  = 300_000
		turnNew = 800
		out     = 400
	)
	for _, tc := range []struct {
		name      string
		hostMul   float64
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
			if tc.wantRecomp && recomputes == 0 {
				t.Fatalf("with no host tier nothing was recomputed; evicted prefixes must be lost")
			}
			if !tc.wantRecomp {
				if recomputes != 0 {
					t.Fatalf("%d full-prefix recomputes with a host tier that holds the working set", recomputes)
				}
				if p.ReloadTokens == 0 {
					t.Fatalf("no host reloads; the evicted prefixes never came back")
				}
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
	// A 450K-token reservation against a 600K pool that already holds one 300K
	// prefix has to take that prefix.
	admit := func(p *Pool) []Eviction { return p.Admit(2, 450_000, 0, 0, 0, 1) }
	t.Run("spilled to host", func(t *testing.T) {
		p := New(ctxLen*2, ctxLen*4)
		p.Warm(1, ctxLen, 0)
		evs := admit(p)
		if len(evs) != 1 || !evs[0].ToHost {
			t.Fatalf("evictions %+v, want the one prefix spilled to host", evs)
		}
		got := p.RebuildSeconds(evs, recompute)
		want := float64(ctxLen) * ReloadSecondsPerToken
		if got < want*0.99 || got > want*1.01 {
			t.Errorf("rebuild %.3f s, want the %.3f s reload", got, want)
		}
		if recompute*ctxLen <= got {
			t.Errorf("reload costs as much as a recompute; the host tier saves nothing")
		}
	})
	t.Run("no host tier", func(t *testing.T) {
		p := New(ctxLen*2, 0)
		p.Warm(1, ctxLen, 0)
		evs := admit(p)
		if len(evs) != 1 || evs[0].ToHost {
			t.Fatalf("evictions %+v, want the one prefix lost outright", evs)
		}
		got := p.RebuildSeconds(evs, recompute)
		if want := float64(ctxLen) * recompute; got < want*0.99 || got > want*1.01 {
			t.Errorf("rebuild %.3f s, want the %.3f s recompute", got, want)
		}
	})
}

// Capacity is the larger tier, because write-through mirrors every cached
// prefix there; the throttle sizes the live set against it.
func TestCapacityIsTheLargerTier(t *testing.T) {
	if got := New(100, 400).Capacity(); got != 400 {
		t.Errorf("Capacity() = %d with a larger host tier, want 400", got)
	}
	if got := New(400, 100).Capacity(); got != 400 {
		t.Errorf("Capacity() = %d with a smaller host tier, want 400", got)
	}
}
