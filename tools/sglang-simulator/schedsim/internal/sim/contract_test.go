package sim

import (
	"fmt"
	"strings"
	"sync"
	"testing"

	"schedsim/internal/trace"
)

// The suite is expensive enough to run once per test binary: every scenario, every
// policy, every seed. The tests below then read it rather than re-simulating.
var suiteOnce = sync.OnceValue(func() map[string]Row {
	steps, err := trace.Parse(trace.EmbeddedLog)
	if err != nil {
		panic("embedded log: " + err.Error())
	}
	cal := trace.Calibrate(steps, BaselineChunkSize, 64)
	rows := RunSuite(BaseScenarios(), NewCost(cal), DefaultConfig, 0)
	out := map[string]Row{}
	for _, r := range rows {
		out[r.Scenario.Key] = r
	}
	return out
})

func suite(t *testing.T) map[string]Row {
	t.Helper()
	return suiteOnce()
}

// TestContractEveryScenarioEveryMetric is the comparison's contract: NEW is at
// least as good as OLD and as PREV on each of the seven metrics, in that metric's
// better direction. Cells that fall inside TieRelative are ties; cells beyond it
// must be wins or must be argued in the derivation named beside them, which this
// test then checks instead of the win.
func TestContractEveryScenarioEveryMetric(t *testing.T) {
	rows := suite(t)
	for _, sc := range BaseScenarios() {
		sc := sc
		t.Run(sc.Name, func(t *testing.T) {
			r, ok := rows[sc.Key]
			if !ok {
				t.Fatalf("scenario %q (%s) was not measured", sc.Name, sc.Key)
			}
			for _, k := range ContractMetrics {
				for _, opp := range []struct {
					name string
					idx  int
				}{{"OLD", 0}, {"PREV", 1}} {
					newv, oppv := r.Metrics[2].Value(k), r.Metrics[opp.idx].Value(k)
					v, gap := Judge(k, newv, oppv)
					t.Logf("%-30s vs %-4s NEW %s / %s %s  gap %+.2f%%  -> %s",
						k, opp.name, fmtVal(k, newv), opp.name, fmtVal(k, oppv), 100*gap, v)
					switch v {
					case VerdictWin, VerdictTie, VerdictUndefined:
						continue
					case VerdictLoss:
						check := boundCell(sc, k, opp.name)
						if check == nil {
							t.Errorf("%s vs %s: NEW is worse by %.1f%% (%s vs %s) and no derivation covers this cell",
								k, opp.name, 100*gap, fmtVal(k, newv), fmtVal(k, oppv))
							continue
						}
						if err := check(r, k, opp.name); err != nil {
							t.Errorf("%s vs %s: the stated bound does not hold: %v", k, opp.name, err)
						} else {
							t.Logf("  bound holds (see the derivation named above)")
						}
					}
				}
			}
		})
	}
}

// boundCell returns the derivation check for a losing cell, keyed by the mechanism
// rather than by the scenario, so a cell can only be excused by an argument that
// actually applies to it. A nil result means the loss is unexplained and the test
// fails, which is the honest outcome for a metric that could have improved.
func boundCell(sc Scenario, k MetricKey, opp string) func(Row, MetricKey, string) error {
	switch {
	case k == MColdTTFT && opp == "OLD" && sc.Key != "thrash-4-600":
		return boundColdTTFT
	case (k == MITLp99 || k == MITLp999) && sc.Key != "thrash-4-600":
		return boundITLPercentile
	case k == MThroughput && sc.Key == "A":
		return boundThroughputPerStreamSecond
	default:
		return nil
	}
}

// boundColdTTFT argues the one cell the specification grants: OLD hands the whole
// GPU to a saturating cold prefill, so it reaches first token after the prompt's
// own prefill work and nothing else, while any policy that keeps streams
// generating spends that window on prefill and decode together. The bound is
// arithmetic, from docs/derivation-cold-ttft-vs-old.md: the same prompt's own
// prefill work is identical under both policies, and NEW's extra wait must equal
// the time it spent on decode, on the follow-ups the cede admitted beside the
// chunk, and on idle, with nothing else left over.
func boundColdTTFT(r Row, k MetricKey, _ string) error {
	oldRuns, newRuns := r.Runs[PolicyIndex(ModeOld)], r.Runs[PolicyIndex(ModeNew)]
	if len(oldRuns) != len(newRuns) {
		return fmt.Errorf("%d OLD runs vs %d NEW runs", len(oldRuns), len(newRuns))
	}
	var oldTTFT, newTTFT, ownOld, ownNew, otherNew, rowsNew, decNew, idleNew float64
	for i := range oldRuns {
		for _, w := range newRuns[i].Windows {
			if !w.Done() {
				continue
			}
			aw := AccountWindow(newRuns[i], w)
			if resid := aw.Window - aw.Total() - aw.Idle; absF(resid) > 0.01*aw.Window {
				return fmt.Errorf("seed %d window %s: the account does not close: %s", i+1, w.Tag, aw)
			}
			newTTFT += w.FirstTok - w.Arrival
			ownNew += aw.OwnPrefill
			otherNew += aw.OtherPrefill
			rowsNew += aw.DecodeRows
			decNew += aw.Decode
			idleNew += aw.Idle
			ow := oldWindow(oldRuns[i], w.Tag)
			if ow == nil {
				return fmt.Errorf("seed %d: OLD never served cold prompt %s", i+1, w.Tag)
			}
			ao := AccountWindow(oldRuns[i], *ow)
			oldTTFT += ow.FirstTok - ow.Arrival
			ownOld += ao.OwnPrefill
			if ao.Decode > 0.02*ao.Window {
				return fmt.Errorf("seed %d window %s: OLD spent %.1f s of %.1f s on decode, "+
					"so its faster cold TTFT is not the whole-GPU effect this bound argues",
					i+1, w.Tag, ao.Decode, ao.Window)
			}
		}
	}
	if oldTTFT == 0 {
		return fmt.Errorf("no cold prompts served by OLD")
	}
	// The prompt's own work must be the same quantity under both policies; if it
	// is not, the comparison is not of scheduling but of what got computed.
	if absF(ownNew-ownOld) > 0.05*ownOld {
		return fmt.Errorf("OLD charged %.1f s and NEW %.1f s for the same prompts' own prefill", ownOld, ownNew)
	}
	explained := otherNew + rowsNew + decNew + idleNew
	gap := newTTFT - oldTTFT
	if gap <= 0 {
		return nil
	}
	if gap > 1.02*explained {
		return fmt.Errorf("NEW is %.1f s slower but decode, ceded work and idle account for %.1f s", gap, explained)
	}
	return nil
}

// boundITLPercentile argues the cells where NEW's inter-token percentile is the
// larger one. The percentile is taken over per-token samples, and a delivery of N
// tokens contributes N samples of gap/N: a policy that serves its streams one
// token inside every prefill batch therefore files a whole-batch-time sample per
// token, while a policy that serves them only in decode batches files one sample
// per batch covering several tokens. The bound is that NEW's percentile is not
// longer than one of its own forward passes -- the stall bound in another unit --
// and it is admissible only while the older policy is withholding tokens outright.
// See docs/derivation-itl-percentiles-under-mixed-chunk.md.
func boundITLPercentile(r Row, k MetricKey, opp string) error {
	newRuns, oppRuns := r.Runs[PolicyIndex(ModeNew)], r.Runs[PolicyIndex(ModeOld)]
	if opp == "PREV" {
		oppRuns = r.Runs[PolicyIndex(ModePrev)]
	}
	if len(newRuns) != len(oppRuns) {
		return fmt.Errorf("%d NEW runs vs %d %s runs", len(newRuns), len(oppRuns), opp)
	}
	var single, total, oppSingle, oppTotal int
	var batchMax, gapMax, worstBatch float64
	for i := range newRuns {
		s, t, longest := SingleTokenSamples(newRuns[i], r.Scenario.Window)
		single, total = single+s, total+t
		if longest > gapMax {
			gapMax = longest
		}
		pf, _, _ := newRuns[i].BatchSeconds(r.Scenario.Window)
		if m := maxOf(pf); m > batchMax {
			batchMax = m
		}
		s2, t2, _ := SingleTokenSamples(oppRuns[i], r.Scenario.Window)
		oppSingle, oppTotal = oppSingle+s2, oppTotal+t2
		pf2, _, _ := oppRuns[i].BatchSeconds(r.Scenario.Window)
		if m := maxOf(pf2); m > worstBatch {
			worstBatch = m
		}
	}
	if total == 0 || oppTotal == 0 {
		return fmt.Errorf("no inter-token samples to argue from")
	}
	share, oppShare := float64(single)/float64(total), float64(oppSingle)/float64(oppTotal)
	q := 0.01
	if k == MITLp999 {
		q = 0.001
	}
	if share <= q {
		return fmt.Errorf("only %.2f%% of NEW's samples are single-token deliveries, which cannot "+
			"move a p%.1f; the gap is not the delivery-count effect this bound argues", 100*share, 100*(1-q))
	}
	if oppShare > share {
		return fmt.Errorf("%s files a larger single-token share (%.2f%% vs %.2f%%), so it is not "+
			"the policy delivering in multi-token steps", opp, 100*oppShare, 100*share)
	}
	// The bound itself: no sample can outlast a forward pass that served the
	// stream, and the percentile must sit under the longest one NEW launched.
	value := r.Metrics[2].Value(k)
	if value > batchMax {
		return fmt.Errorf("NEW's p%.1f is %.3f s, longer than its longest forward pass of %.3f s",
			100*(1-q), value, batchMax)
	}
	// Admissibility: the other policy must be withholding tokens during the cold
	// prefills, which is what makes its shorter tail a symptom rather than a merit.
	if o := r.Metrics[PolicyIndex(ModeOld)]; o.PerAgentTokSCold == o.PerAgentTokSCold {
		if r.Metrics[2].PerAgentTokSCold <= o.PerAgentTokSCold {
			return fmt.Errorf("NEW does not deliver more per agent during a cold prefill than OLD")
		}
	}
	if opp == "PREV" && r.Metrics[2].StreamDecodeTokSCold <= r.Metrics[1].StreamDecodeTokSCold {
		return fmt.Errorf("NEW does not keep streams generating better than PREV")
	}
	if r.Metrics[2].LongestStall > r.Metrics[PolicyIndex(Mode(prevOr(opp)))].LongestStall {
		return fmt.Errorf("NEW's longest stall is worse too, so the percentile is not the only thing moving")
	}
	return nil
}

func prevOr(opp string) Mode {
	if opp == "PREV" {
		return ModePrev
	}
	return ModeOld
}

// boundThroughputPerStreamSecond argues scenario A's output-throughput cell. The
// workload's conversations do not exist until their cold prompts are prefilled, so
// the tokens a policy can deliver are bounded by the stream-seconds its admissions
// bought; the rate per stream-second is the comparable quantity.
// See docs/derivation-scenario-a-throughput.md.
func boundThroughputPerStreamSecond(r Row, k MetricKey, _ string) error {
	var out, sec [NumModes]float64
	for mi := range Modes {
		for _, res := range r.Runs[mi] {
			out[mi] += float64(DeliveredTokens(res, r.Scenario.Window))
			sec[mi] += StreamSeconds(res, r.Scenario.Window)
		}
	}
	if sec[0] == 0 || sec[2] == 0 {
		return fmt.Errorf("no stream-seconds to normalise by")
	}
	rateOld, rateNew := out[0]/sec[0], out[2]/sec[2]
	if rateNew < rateOld {
		return fmt.Errorf("NEW delivers %.4f tok per stream-second, OLD %.4f: the gap is not admission timing",
			rateNew, rateOld)
	}
	return nil
}

// oldWindow finds OLD's window for the same scripted cold prompt.
func oldWindow(res *Result, tag string) *ColdWindow {
	for i := range res.Windows {
		if res.Windows[i].Tag == tag && res.Windows[i].Done() {
			return &res.Windows[i]
		}
	}
	return nil
}

func fmtVal(k MetricKey, v float64) string {
	if v != v {
		return "n/a"
	}
	switch k {
	case MStreamRate, MAgentRate, MThroughput:
		return fmt.Sprintf("%.1f tok/s", v)
	case MRecomputes:
		return fmt.Sprintf("%d", int(v))
	default:
		if v < 1 {
			return fmt.Sprintf("%.1f ms", v*1e3)
		}
		return fmt.Sprintf("%.1f s", v)
	}
}

var _ = strings.Contains
