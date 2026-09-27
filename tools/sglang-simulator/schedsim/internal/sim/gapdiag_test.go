package sim

import (
	"bytes"
	"fmt"
	"testing"

	"schedsim/internal/trace"
)

// ScenarioCost is the calibrated cost model the committed scenarios run on.
func ScenarioCost() Cost {
	steps, err := trace.Parse(trace.EmbeddedLog)
	if err != nil {
		panic("embedded log: " + err.Error())
	}
	return NewCost(trace.Calibrate(steps, BaselineChunkSize, 64))
}

// TestGapDiagnosticScenarioB dumps, for each policy, where scenario B's
// inter-token latency comes from: the gap distribution, the class of batch that
// delivered each token, the cost of a single prefill batch, and the batch
// sequence around the widest gaps. Run it with -v to read the numbers.
func TestGapDiagnosticScenarioB(t *testing.T) {
	cost := ScenarioCost()
	for _, every := range []float64{2, 5} {
		sc := ScenarioB(every)
		for _, mode := range Modes {
			cfg := DefaultConfig(mode, cost)
			var runs []*Result
			for _, seed := range sc.Seeds {
				runs = append(runs, Run(sc, cfg, seed))
			}
			pooled := pooledTrace(runs)
			var buf bytes.Buffer
			fmt.Fprintf(&buf, "%s / %s / pooled over seeds %v\n", sc.Name, mode, sc.Seeds)
			m := Pool(runs, sc.Window)
			fmt.Fprintf(&buf, "  metrics: ITL p99 %s p99.9 %s, raw gap p99 %s, longest stall %s, output %.1f tok/s, cold TTFT %s, turn TTFT p99 %s\n",
				seconds(m.ITLp99), seconds(m.ITLp999), seconds(m.ITLp99Raw),
				seconds(m.LongestStall), m.OutputTokS, seconds(m.ColdTTFTMean),
				seconds(m.TurnTTFTp99))
			WriteDeliveryClassBreakdown(&buf, pooled)
			WritePrefillCostBreakdown(&buf, pooled)
			WriteGapDiagnostic(&buf, runs[0], 5)
			t.Log(buf.String())
		}
	}
}

// pooledTrace concatenates several seeds of one scenario into one trace, so a
// diagnostic can quote the same population the pooled metrics are quantiled over.
func pooledTrace(runs []*Result) *Result {
	merged := &Result{Agents: runs[0].Agents, End: runs[0].End}
	for _, r := range runs {
		merged.Requests = append(merged.Requests, r.Requests...)
		merged.Batches = append(merged.Batches, r.Batches...)
		merged.Windows = append(merged.Windows, r.Windows...)
		merged.PrefillLog = append(merged.PrefillLog, r.PrefillLog...)
		merged.DecodeLog = append(merged.DecodeLog, r.DecodeLog...)
	}
	return merged
}
