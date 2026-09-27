package scenario

import (
	"fmt"
	"strconv"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"schedsim/internal/cost"
	"schedsim/internal/sim"
)

var everyRow = sync.OnceValues(func() ([]Row, sim.Config) {
	cal, err := cost.Calibrate()
	if err != nil {
		panic(err)
	}
	base := Deployment(cal.Model)
	return RunAll(append(Main(), Sensitivity()...), base), base
})

// shown is a metric at the precision the table prints it.
func shown(t *testing.T, m Metric, r sim.Metrics) float64 {
	v, err := strconv.ParseFloat(fmt.Sprintf(m.Format, m.Get(r)), 64)
	require.NoError(t, err)
	return v
}

func notWorse(t *testing.T, m Metric, nw, other sim.Metrics) bool {
	if m.HigherBetter {
		return shown(t, m, nw) >= shown(t, m, other)
	}
	return shown(t, m, nw) <= shown(t, m, other)
}

// tails are the ITL quantiles the cold-prompt bound covers, keyed by metric name.
var tails = map[string]float64{"ITL p99 ms": 0.99, "ITL p99.9 ms": 0.999}

// boundCovered reports whether OLD's edge in this cell is the price of decoding
// during a cold prompt, which TestColdPromptBound checks instead.
func boundCovered(m Metric, r Row, base sim.Config) bool {
	b, ok := BoundFor(r.Scenario, base, r.Results[2])
	if !ok {
		return false
	}
	_, tail := tails[m.Name]
	return tail || m.Name == "cold TTFT s" || m.Name == throughput && b.Throughput > 0
}

const throughput = "throughput tok/s"

func TestNewIsNotWorseThanPrevAnywhere(t *testing.T) {
	rows, _ := everyRow()
	for _, r := range rows {
		for _, m := range Metrics {
			assert.Truef(t, notWorse(t, m, r.Results[2], r.Results[1]),
				"%s %s: NEW %v, PREV %v", r.Scenario.Name, m.Name, shown(t, m, r.Results[2]), shown(t, m, r.Results[1]))
		}
	}
}

func TestNewIsNotWorseThanOldOutsideTheBound(t *testing.T) {
	rows, base := everyRow()
	for _, r := range rows {
		for _, m := range Metrics {
			if boundCovered(m, r, base) {
				continue
			}
			assert.Truef(t, notWorse(t, m, r.Results[2], r.Results[0]),
				"%s %s: NEW %v, OLD %v", r.Scenario.Name, m.Name, shown(t, m, r.Results[2]), shown(t, m, r.Results[0]))
		}
	}
}

// TestColdPromptBound checks NEW against what any schedule that keeps decoding
// during a cold prompt must pay (BOUNDS.md).
func TestColdPromptBound(t *testing.T) {
	rows, base := everyRow()
	for _, r := range rows {
		bs, ok := SeedBounds(r, base)
		if !ok {
			continue
		}
		for i, b := range bs {
			old, nw := r.Runs[0][i], r.Runs[2][i]
			name := fmt.Sprintf("%s seed %d", r.Scenario.Name, Seeds[i])
			// The GPU is never idle while a cold prompt waits, so NEW's TTFT.
			assert.Lessf(t, b.Idle, 0.005, "%s: NEW idles %.2f%% of the cold windows", name, 100*b.Idle)
			assert.GreaterOrEqualf(t, nw.ColdTTFT, b.TTFT-0.05, "%s: NEW's cold TTFT is under its own bound", name)
			if b.Throughput > 0 {
				assert.LessOrEqualf(t, nw.Throughput, b.Throughput+0.05, "%s: NEW's throughput is over its own bound", name)
				if nw.Throughput < old.Throughput {
					assert.Lessf(t, b.Throughput, old.Throughput,
						"%s: NEW %.1f < OLD %.1f tok/s, and decoding as long as NEW while the colds wait still allows %.1f",
						name, nw.Throughput, old.Throughput, b.Throughput)
				}
			}
			for _, m := range Metrics {
				q, tail := tails[m.Name]
				if !tail || notWorse(t, m, r.Results[2], r.Results[0]) || notWorse(t, m, nw, old) {
					continue
				}
				assert.Truef(t, b.Forced(q),
					"%s %s: NEW %v > OLD %v, and stalls at NEW's longest stall need only %.2f%% of gaps",
					name, m.Name, shown(t, m, nw), shown(t, m, old), 100*b.TailShare)
				assert.GreaterOrEqualf(t, m.Get(nw), 1000*b.Chunk, "%s %s: NEW is under the cheapest chunk", name, m.Name)
			}
		}
	}
}
