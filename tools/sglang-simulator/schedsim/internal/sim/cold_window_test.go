package sim

import "testing"

// A cold prompt whose run ended before it reached a first token has no
// arrival-to-first-token interval, and must not be priced as if it did.
func TestColdWindowWithoutFirstTokenIsNotServed(t *testing.T) {
	cost := ScenarioCost()
	for _, mode := range Modes {
		res := Run(ScenarioB(1), DefaultConfig(mode, cost), 4)
		for _, w := range res.Windows {
			if w.Done() && w.FirstTok <= w.Arrival {
				t.Errorf("%v window %s: arrived %.1f s, first token reported at %.1f s, so it counts as "+
					"served with %.1f s to first token", mode, w.Tag, w.Arrival, w.FirstTok, w.FirstTok-w.Arrival)
			}
			if w.Done() {
				continue
			}
			if got := AccountWindow(res, w).Skip; got != Unserved {
				t.Errorf("%v window %s: no first token, but the account excludes it as %v", mode, w.Tag, got)
			}
		}
	}
}
