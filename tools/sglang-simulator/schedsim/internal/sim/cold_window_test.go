package sim

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

// A cold prompt whose run ended before it reached a first token has no
// arrival-to-first-token interval, and must not be priced as if it did.
func TestColdWindowWithoutFirstTokenIsNotServed(t *testing.T) {
	cost := ScenarioCost()
	for _, mode := range Modes {
		res := Run(ScenarioB(1), DefaultConfig(mode, cost), 4)
		for _, w := range res.Windows {
			assert.False(t, w.Done() && w.FirstTok <= w.Arrival)

			if w.Done() {
				continue
			}
			got := AccountWindow(res, w).Skip
			assert.Equal(t, Unserved, got)

		}
	}
}
