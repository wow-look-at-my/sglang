package sim

import (
	"bytes"
	"fmt"
	"sync"
	"testing"

	"schedsim/internal/trace"
	"schedsim/internal/trace/tracetest"
)

// incidentSteps is the generated incident log the scenarios are calibrated on.
var incidentSteps = sync.OnceValue(func() []trace.Step {
	steps, err := trace.Parse(tracetest.Incident(tracetest.DefaultIncident).String())
	if err != nil {
		panic("incident log: " + err.Error())
	}
	return steps
})

// testEpisode is the incident's episode, which scenario A replays.
func testEpisode() Episode {
	ep, err := ExtractEpisode(incidentSteps(), BaselineChunkSize)
	if err != nil {
		panic("incident episode: " + err.Error())
	}
	return ep
}

func testEpisodePtr() *Episode {
	ep := testEpisode()
	return &ep
}

// ScenarioCost is the calibrated cost model the committed scenarios run on.
func ScenarioCost() Cost {
	return NewCost(trace.Calibrate(incidentSteps(), BaselineChunkSize, 64))
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
//
// Each seed is re-identified on the way in. classifyDeliveries keys a delivery by
// (request id, delivery time) because within one run that pair names exactly one batch,
// but ids are assigned per seed, so two seeds can present the same pair with different
// classes and the merged map would label both with whichever batch wrote last. The
// clones exist only in the merged trace; the runs themselves are not touched.
func pooledTrace(runs []*Result) *Result {
	merged := &Result{Agents: runs[0].Agents, End: runs[0].End}
	for ri, r := range runs {
		clones := make(map[*Request]*Request, len(r.Requests))
		reqs := make([]*Request, 0, len(r.Requests))
		for _, req := range r.Requests {
			c := *req
			c.ID = req.ID + (ri+1)*1_000_000
			clones[req] = &c
			reqs = append(reqs, &c)
		}
		remap := func(p *Request) *Request {
			if p == nil {
				return nil
			}
			return clones[p]
		}
		batches := make([]*Batch, 0, len(r.Batches))
		for _, b := range r.Batches {
			nb := *b
			nb.Rows = remapRequests(b.Rows, remap)
			nb.Decode = remapRequests(b.Decode, remap)
			nb.Continuation = remap(b.Continuation)
			nb.Items = make([]Item, len(b.Items))
			for i, it := range b.Items {
				it.Req = remap(it.Req)
				nb.Items[i] = it
			}
			batches = append(batches, &nb)
		}
		merged.Requests = append(merged.Requests, reqs...)
		merged.Batches = append(merged.Batches, batches...)
		merged.Windows = append(merged.Windows, r.Windows...)
		merged.PrefillLog = append(merged.PrefillLog, r.PrefillLog...)
		merged.DecodeLog = append(merged.DecodeLog, r.DecodeLog...)
	}
	return merged
}

// remapRequests rebuilds a batch's request list through the merge's clone table.
func remapRequests(rs []*Request, remap func(*Request) *Request) []*Request {
	if rs == nil {
		return nil
	}
	out := make([]*Request, len(rs))
	for i, r := range rs {
		out[i] = remap(r)
	}
	return out
}

// TestMergedTraceAttributesEverySeed checks the merge against the seeds it joined: the
// per-class sample counts must be the sum of what each run produced on its own, because
// a delivery counted once as "mixed" in one run cannot arrive in the merged trace as
// "decode", and one lost to a key collision cannot reappear anywhere.
func TestMergedTraceAttributesEverySeed(t *testing.T) {
	cost := ScenarioCost()
	for _, sc := range []Scenario{ScenarioA(testEpisode()), ScenarioB(2), ScenarioThrash(4, 600)} {
		var runs []*Result
		want := map[string]int{}
		total := 0
		for _, seed := range sc.Seeds {
			res := Run(sc, DefaultConfig(ModeNew, cost), seed)
			runs = append(runs, res)
			for label, c := range deliveryClassStats(res) {
				want[label] += c.Copies
				total += c.Copies
			}
		}
		got := deliveryClassStats(pooledTrace(runs))
		for label, n := range want {
			if got[label] == nil || got[label].Copies != n {
				t.Errorf("%s / %s: merged trace has %s samples, the seeds summed to %d",
					sc.Name, label, countText(got[label]), n)
			}
		}
	}
}

// countText renders a class's sample count for a failure message, including the case
// where the merged trace has no such class at all.
func countText(c *deliveryClass) string {
	if c == nil {
		return "no"
	}
	return fmt.Sprintf("%d", c.Copies)
}
