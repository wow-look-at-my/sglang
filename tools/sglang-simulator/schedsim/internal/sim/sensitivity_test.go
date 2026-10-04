package sim

import (
	"testing"

	"schedsim/internal/trace"
)

// The sweep and every scenario number are only evidence about the real cost
// model if the model is what drives them: perturbing a calibrated cost has to
// move the reported metrics in the direction that cost implies. A run whose
// outcome survived a cost change would mean some number in the tables came from
// somewhere other than the calibration.
//
// Uniform scaling is the perturbation with a known answer. A prefill batch's
// seconds are linear in every prefill term, so the scheduler's own ratios --
// average and marginal seconds per token, which is what the continuation cap and
// the token budget divide -- are unchanged; only the wall clock each prefill
// batch occupies moves, by the factor itself.
func TestCostScaleMovesTheReportedMetrics(t *testing.T) {
	cal := trace.Calibrate(mustSteps(t), BaselineChunkSize, 64)
	sc := ScenarioA(testEpisode())

	var ttft, stall, stallOld []float64
	for _, k := range []float64{0.5, 1, 2} {
		cost := NewCost(cal)
		cost.Cal.Prefill = cost.Cal.Prefill.Scaled(k)
		rows := RunSuite([]Scenario{sc}, cost, DefaultConfig, 0)
		if len(rows) != 1 {
			t.Fatalf("scaled %v: %d rows, want 1", k, len(rows))
		}
		newm := rows[0].Metrics[PolicyIndex(ModeNew)]
		oldm := rows[0].Metrics[PolicyIndex(ModeOld)]
		ttft = append(ttft, newm.Value(MColdTTFT))
		stall = append(stall, newm.LongestStall)
		stallOld = append(stallOld, oldm.LongestStall)
		t.Logf("prefill x%.2f: NEW cold TTFT %.2f s, NEW longest stall %.3f s, OLD longest stall %.2f s",
			k, newm.Value(MColdTTFT), newm.LongestStall, oldm.LongestStall)
	}

	// A cold prompt's own prefill is part of its time to first token, and every
	// chunk a stream waits out is prefill wall clock, so both must grow with the
	// prefill cost. OLD is the cleaner case: it runs prefill whenever one can be
	// formed, so its stall is prefill seconds with no feedback loop to argue with.
	for i := 1; i < len(ttft); i++ {
		if !(ttft[i] > ttft[i-1]) {
			t.Errorf("cold TTFT did not grow with the prefill cost: %.3f s then %.3f s", ttft[i-1], ttft[i])
		}
		if !(stallOld[i] > stallOld[i-1]) {
			t.Errorf("OLD's longest stall did not grow with the prefill cost: %.3f s then %.3f s",
				stallOld[i-1], stallOld[i])
		}
		if !(stall[i] >= stall[i-1]) {
			t.Errorf("NEW's longest stall fell as the prefill cost grew: %.3f s then %.3f s",
				stall[i-1], stall[i])
		}
	}

	// The factor has to reach the metrics at the rate it is applied, not merely
	// agree in sign.
	if !(stallOld[2] > 1.5*stallOld[1]) {
		t.Errorf("doubling the prefill cost moved OLD's longest stall from %.2f s to only %.2f s",
			stallOld[1], stallOld[2])
	}
}
