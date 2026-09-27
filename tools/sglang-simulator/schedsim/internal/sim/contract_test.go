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
			logWindowCensus(t, r)
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
						} else if ev, ok := boundEvidence(sc, k, opp.name, r); ok {
							t.Logf("  bound holds, on these numbers: %s", ev)
						} else {
							t.Logf("  bound holds (see the derivation named above)")
						}
					}
				}
			}
		})
	}
}

// bound reports whether a derivation covers a losing cell.
type bound func(Row, MetricKey, string) error

// boundEvidence recomputes the quantities the bound for this cell reads, so a
// passing cell still shows the arithmetic a reader can check against the
// derivation. It returns false for a cell no bound claims.
func boundEvidence(sc Scenario, k MetricKey, opp string, r Row) (string, bool) {
	switch boundName(sc, k, opp) {
	case boundWholeGPU, boundOwnWork:
		n, o, err := coldAccounts(r, opp)
		if err != nil {
			return "", false
		}
		return fmt.Sprintf("%d paired windows; NEW wait %.1f s vs %s %.1f s, own chunks %.1f s vs %.1f"+
			" s in %d passes vs %d, follow-up prefill %.1f s vs %.1f, rows %.1f vs %.1f, overhead %.1f"+
			" vs %.1f, decode %.1f vs %.1f, idle %.1f vs %.1f",
			n.Served, n.TTFT, opp, o.TTFT, n.Own, o.Own, n.OwnBatches, o.OwnBatches,
			n.Other, o.Other, n.Rows, o.Rows, n.Overhead, o.Overhead, n.Decode, o.Decode,
			n.Idle, o.Idle), true
	case boundStreamRate:
		n, o := coldFlowOf(r, ModeNew), coldFlowOf(r, Mode(prevOr(opp)))
		marginal, floor := 0.0, rowTokenFloor(r)
		if n.Secs > o.Secs {
			marginal = (n.Toks - o.Toks) / (n.Secs - o.Secs)
		}
		return fmt.Sprintf("NEW %.0f tok over %.1f stream-s (%.2f tok/s), %s %.0f over %.1f (%.2f);"+
			" the %.1f extra seconds earned %.1f tok/s, over the %.1f tok/s floor of a mixed row, and"+
			" %.0f of NEW's tokens arrived as rows",
			n.Toks, n.Secs, n.rate(), opp, o.Toks, o.Secs, o.rate(), n.Secs-o.Secs, marginal,
			floor, n.Rows), true
	case boundEvictionTail, boundMixedChunkTail:
		oi := PolicyIndex(prevOr(opp))
		over, total := samplesAbove(r, ModeNew, r.Metrics[oi].Value(k))
		rows, batchMax := mixedRowTally(r)
		single, oppSingle := tailShares(r, prevOr(opp))
		return fmt.Sprintf("%d of %d NEW samples (%.3f%%) exceed %s's %s, against %.0f mixed rows;"+
			" single-token deliveries are %.2f%% of NEW's samples and %.2f%% of %s's, and NEW's p%.1f"+
			" of %s sits under its longest forward pass of %s",
			over, total, 100*float64(over)/float64(total), opp, fmtVal(k, r.Metrics[oi].Value(k)),
			rows, 100*single, 100*oppSingle, opp, 100*(1-percentileAt(k)),
			fmtVal(k, r.Metrics[2].Value(k)), fmtVal(k, batchMax)), true
	case boundAdmissionRate:
		var out, sec [NumModes]float64
		for mi := range Modes {
			for _, res := range r.Runs[mi] {
				out[mi] += float64(DeliveredTokens(res, r.Scenario.Window))
				sec[mi] += StreamSeconds(res, r.Scenario.Window)
			}
		}
		return fmt.Sprintf("NEW delivered %.0f tok over %.0f stream-seconds (%.4f tok per stream-second),"+
			" OLD %.0f over %.0f (%.4f)", out[2], sec[2], out[2]/sec[2], out[0], sec[0],
			out[0]/sec[0]), true
	}
	return "", false
}

// thrashPrefix names the scenario family whose turns all miss their prefix. None
// of them has a cold prompt, so the cold-window derivations have nothing to say
// about them and their inter-token tail is argued under eviction instead.
const thrashPrefix = "thrash"

// One losing cell is argued by exactly one derivation. boundName is the single
// table of which, so a bound and the evidence it reports cannot drift apart.
const (
	boundWholeGPU       = "cold TTFT against the policy that withholds decode"
	boundOwnWork        = "cold TTFT against the same balancer"
	boundMixedChunkTail = "inter-token tail under mixed chunked prefill"
	boundEvictionTail   = "inter-token tail under eviction"
	boundStreamRate     = "stream decode rate under mixed chunked prefill"
	boundAdmissionRate  = "output throughput per stream-second"
)

// boundName is the derivation that claims a cell, or "" for one none does: an
// unclaimed loss fails the test, which is the honest outcome for a metric that
// could have improved.
func boundName(sc Scenario, k MetricKey, opp string) string {
	thrash := strings.HasPrefix(sc.Key, thrashPrefix)
	switch {
	case k == MColdTTFT && opp == "OLD" && !thrash:
		return boundWholeGPU
	case k == MColdTTFT && opp == "PREV" && !thrash:
		return boundOwnWork
	case (k == MITLp99 || k == MITLp999) && !thrash:
		return boundMixedChunkTail
	case (k == MITLp99 || k == MITLp999) && thrash:
		return boundEvictionTail
	case k == MStreamRate && opp == "PREV" && !thrash:
		return boundStreamRate
	case k == MThroughput && sc.Key == "A":
		return boundAdmissionRate
	}
	return ""
}

// boundCell returns the check for a losing cell, keyed by the mechanism rather
// than by the scenario, so a cell can only be excused by an argument that actually
// applies to it.
func boundCell(sc Scenario, k MetricKey, opp string) bound {
	switch boundName(sc, k, opp) {
	case boundWholeGPU:
		return boundColdTTFT
	case boundOwnWork:
		return boundColdTTFTvsPrev
	case boundMixedChunkTail:
		return boundITLPercentile
	case boundEvictionTail:
		return boundITLTailUnderEviction
	case boundStreamRate:
		return boundStreamRateVsPrev
	case boundAdmissionRate:
		return boundThroughputPerStreamSecond
	}
	return nil
}

// logWindowCensus reports, per policy, how many cold windows the run priced and
// how many it had to exclude. The cold TTFT bounds below rest only on the priced
// ones, so the count is part of the evidence; a seed that excluded any is listed
// on its own line with the reason.
func logWindowCensus(t *testing.T, r Row) {
	t.Helper()
	for mi, mode := range Modes {
		runs := r.Runs[mi]
		if len(runs) == 0 || len(runs[0].Windows) == 0 {
			continue
		}
		arrived, priced, details := 0, 0, make([]string, 0, len(runs))
		for si, res := range runs {
			c := Census(res)
			arrived += c.Arrived
			priced += c.Priced
			if c.Priced != c.Arrived {
				details = append(details, fmt.Sprintf("seed %d %s", si+1, c))
			}
		}
		if len(details) == 0 {
			t.Logf("%s cold windows: %d/%d priced", mode, priced, arrived)
			continue
		}
		t.Logf("%s cold windows: %d/%d priced, excluded: %s", mode, priced, arrived,
			strings.Join(details, "; "))
	}
}

// ownTolerance bounds how far the two policies may differ on the cold prompt's
// own prefill work before the comparison stops being about scheduling.
const ownTolerance = 0.05

// oldDecodeShare is the share of its cold windows OLD may spend decoding and
// still be the policy that handed the prompt the GPU. It is ownTolerance, not a
// second number: the bound grants that at most that fraction of OLD's cold window
// is anything other than the prompt's own prefill, and OLD's decode seconds are
// exactly such a portion, so giving the same claim two tolerances would let the
// bound keep a term it never accounts for. The asymmetry the derivation rests on
// survives at this size: NEW spends a third to a half of the same windows on
// decode, so the cap keeps OLD an order of magnitude further from sharing.
// See docs/derivation-cold-ttft-vs-old.md.
const oldDecodeShare = ownTolerance

// coldTTFTSlack is the arithmetic tolerance on the account: the buckets are sums
// of floats, not counts, so the bound admits a little over a percent.
const coldTTFTSlack = 1.02

// coldTTFTIdentitySlack bounds the round-off in the account identity: both sides
// are sums of the same five buckets, so a gap that survives subtraction to better
// than a part per million is bookkeeping that still holds.
const coldTTFTIdentitySlack = 1e-6

// coldAccts is one policy's share of the account over a fixed set of cold prompts.
type coldAccts struct {
	TTFT, Own, Other, Rows, Overhead, Decode, Idle, WindowSecs float64
	Served, OwnBatches, OwnTokens                              int
	Excluded                                                   [NumExclusions]int
	WorstDecodeShare                                           float64
	WorstDecodeSeed                                            int
}

// coldAccounts prices the same cold prompts under NEW and under opp: every window
// NEW can price that opp also served, paired by prompt tag. Pairing is what makes
// the two accounts comparable; the difference of their terms is the whole content
// of the bounds below, and a window either policy failed to serve is counted.
func coldAccounts(r Row, opp string) (n, o coldAccts, err error) {
	newRuns := r.Runs[PolicyIndex(ModeNew)]
	oppRuns := r.Runs[PolicyIndex(ModeOld)]
	if opp == "PREV" {
		oppRuns = r.Runs[PolicyIndex(ModePrev)]
	}
	if len(oppRuns) != len(newRuns) {
		return n, o, fmt.Errorf("%d %s runs vs %d NEW runs", len(oppRuns), opp, len(newRuns))
	}
	for i := range newRuns {
		for _, w := range newRuns[i].Windows {
			aw := AccountWindow(newRuns[i], w)
			if !aw.Measurable() {
				n.Excluded[aw.Skip]++
				continue
			}
			if resid := aw.Window - aw.Total() - aw.Idle; absF(resid) > 0.01*absF(aw.Window) {
				return n, o, fmt.Errorf("NEW seed %d window %s: the account does not close: %s",
					i+1, w.Tag, aw)
			}
			ow := servedWindow(oppRuns[i], w.Tag)
			if ow == nil {
				n.Excluded[Unserved]++
				continue
			}
			ao := AccountWindow(oppRuns[i], *ow)
			if !ao.Measurable() {
				n.Excluded[ao.Skip]++
				continue
			}
			if resid := ao.Window - ao.Total() - ao.Idle; absF(resid) > 0.01*absF(ao.Window) {
				return n, o, fmt.Errorf("%s seed %d window %s: the account does not close: %s",
					opp, i+1, w.Tag, ao)
			}
			n.add(aw, w.FirstTok-w.Arrival)
			o.add(ao, ow.FirstTok-ow.Arrival)
			if s := ao.Decode / ao.Window; s > o.WorstDecodeShare {
				o.WorstDecodeShare, o.WorstDecodeSeed = s, i+1
			}
		}
	}
	return n, o, nil
}

func (a *coldAccts) add(a2 WindowAccount, ttft float64) {
	a.Served++
	a.TTFT += ttft
	a.Own += a2.OwnPrefill
	a.Other += a2.OtherPrefill
	a.Rows += a2.DecodeRows
	a.Overhead += a2.Overhead
	a.Decode += a2.Decode
	a.Idle += a2.Idle
	a.WindowSecs += a2.Window
	a.OwnBatches += a2.OwnBatches
	a.OwnTokens += a2.OwnTokens
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
	n, o, err := coldAccounts(r, "OLD")
	if err != nil {
		return err
	}
	if o.Served == 0 {
		return fmt.Errorf("no cold prompt both policies served%s", excludedNote(n.Excluded))
	}
	// The prompt's own work must be the same quantity under both policies; if it
	// is not, the comparison is not of scheduling but of what got computed. The
	// excess is a real term of NEW's wait, so the account below carries it rather
	// than the precondition merely tolerating it.
	if absF(n.Own-o.Own) > ownTolerance*o.Own {
		return fmt.Errorf("OLD charged %.1f s and NEW %.1f s for the same prompts' own prefill, "+
			"a %.2f%% difference against a %.2f%% tolerance, over %d paired windows",
			o.Own, n.Own, 100*absF(n.Own-o.Own)/o.Own, 100*ownTolerance, n.Served)
	}
	// OLD must have spent only a stated share of the cold windows on decode; that
	// is what makes "OLD handed the prompt the whole GPU" the mechanism at work
	// rather than a story told after the fact.
	// See docs/derivation-cold-ttft-vs-old.md.
	if o.Decode > oldDecodeShare*o.WindowSecs {
		return fmt.Errorf("OLD spent %.1f s of %.1f s (%.2f%%) on decode, worst single window %.2f%% "+
			"at seed %d, so its faster cold TTFT is not the whole-GPU effect this bound argues "+
			"(it admits up to %.2f%%)",
			o.Decode, o.WindowSecs, 100*o.Decode/o.WindowSecs, 100*o.WorstDecodeShare,
			o.WorstDecodeSeed, 100*oldDecodeShare)
	}
	explained := n.Other + n.Rows + n.Overhead + n.Decode + n.Idle + (n.Own - o.Own)
	gap := n.TTFT - o.TTFT
	if gap <= 0 {
		return nil
	}
	if gap > coldTTFTSlack*explained {
		return fmt.Errorf("NEW is %.1f s slower but the account gives %.1f s: own-work excess %.1f s,"+
			" ceded prefill %.1f s, decode rows %.1f s, overhead %.1f s, decode %.1f s, idle %.1f s%s",
			gap, explained, n.Own-o.Own, n.Other, n.Rows, n.Overhead, n.Decode, n.Idle,
			excludedNote(n.Excluded))
	}
	return nil
}

// prevIdleSlack bounds how much more GPU time NEW may leave unused inside the
// cold windows than PREV, beyond which its later first token is not work bought
// elsewhere. Expressed as a share of the paired windows; the measured cells idle
// a small fraction of a second in total.
const prevIdleSlack = 0.002

// boundStreamRateVsPrev argues the cells where NEW keeps streams generating at a
// slightly lower tok/s than PREV inside a cold prefill. Metric 1 is a ratio whose
// denominator is the policy's own doing: a second of stream time only exists
// because the policy admitted the turn and reached its first token. So the cell is
// arguable only if NEW delivered at least as many agent tokens in those windows,
// and the loss is then arithmetically the extra seconds those tokens spread over.
// Those seconds are bounded from below by the mixed channel's own price: a stream
// riding a prefill batch earns one token per pass, so no stream-second served that
// way can return more than one token per batch. If NEW's extra seconds returned
// less than the channel's floor, they were life-support and not service, and this
// bound says so rather than excusing the cell.
// See docs/derivation-stream-rate-under-mixed-chunk.md.
func boundStreamRateVsPrev(r Row, k MetricKey, opp string) error {
	n, o := coldFlowOf(r, ModeNew), coldFlowOf(r, Mode(prevOr(opp)))
	if n.Secs <= 0 || o.Secs <= 0 {
		return fmt.Errorf("no stream-seconds inside a cold window to argue from")
	}
	rateN, rateO := n.rate(), o.rate()
	ni, oi := PolicyIndex(ModeNew), PolicyIndex(prevOr(opp))
	if rateN >= rateO {
		return nil
	}
	if n.Toks < o.Toks {
		return fmt.Errorf("NEW delivered %.0f agent tokens inside the cold windows, %s %.0f: the"+
			" slower rate is fewer tokens, not more stream-seconds, and that is a defect",
			n.Toks, opp, o.Toks)
	}
	if n.Secs <= o.Secs {
		return fmt.Errorf("NEW delivered %.0f tokens over %.1f stream-seconds and %s %.0f over %.1f,"+
			" so the rate cannot be the lower one; the metric and the flow disagree",
			n.Toks, n.Secs, opp, o.Toks, o.Secs)
	}
	// The two quantities that do not divide by self-chosen stream time must favour
	// NEW, or the extra seconds bought nothing.
	if !(r.Metrics[ni].PerAgentTokSCold > r.Metrics[oi].PerAgentTokSCold) {
		return fmt.Errorf("NEW's per-agent rate in the cold windows is %.2f tok/s against %s %.2f,"+
			" so the extra stream-seconds delivered nothing", r.Metrics[ni].PerAgentTokSCold,
			opp, r.Metrics[oi].PerAgentTokSCold)
	}
	if r.Metrics[ni].LongestStall > r.Metrics[oi].LongestStall {
		return fmt.Errorf("NEW's longest stall is %.3f s against %s %.3f s, so keeping those streams"+
			" generating cost a stream a longer wait than %s gave it",
			r.Metrics[ni].LongestStall, opp, r.Metrics[oi].LongestStall, opp)
	}
	marginal := (n.Toks - o.Toks) / (n.Secs - o.Secs)
	floor := rowTokenFloor(r)
	if marginal < floor {
		return fmt.Errorf("NEW's %.1f extra stream-seconds returned %.2f tok per stream-second,"+
			" below the %.2f a stream earns riding a prefill batch at its longest: the extra seconds"+
			" held streams that generated nothing", n.Secs-o.Secs, marginal, floor)
	}
	return nil
}

// rowTokenFloor is the highest rate a stream can hold while it is only ever served
// as a mixed batch's decode row: one token per forward pass, so one over the
// longest pass NEW launched inside the cold windows.
func rowTokenFloor(r Row) float64 {
	var longest float64
	for _, res := range r.Runs[PolicyIndex(ModeNew)] {
		wins := coldWindows(res, r.Scenario.Window)
		for _, b := range res.Batches {
			if !b.IsPrefill || len(b.Rows) == 0 || !inWindows(wins, b.End) {
				continue
			}
			if d := b.End - b.Start; d > longest {
				longest = d
			}
		}
	}
	if longest <= 0 {
		return 0
	}
	return 1 / longest
}

// boundColdTTFTvsPrev argues a cell of a different kind: PREV runs the same
// balancer on the same prompt, so there is no whole-GPU asymmetry to invoke. The
// two accounts close by construction, so subtracting them is an identity over the
// six buckets and the bound's whole content is which of them may carry the gap.
// NEW may be later than PREV by the follow-up prefill its cede let in beside the
// chunk, by the decode rows it folded into the cold prompt's batches, by the
// per-pass overhead and reload copy those batch-mates made the pass pay, and by
// the decode those follow-ups kept alive. It may not be later because it computed
// more of the prompt itself, nor because it left the GPU idle where PREV did not.
// See docs/derivation-cold-ttft-vs-prev.md.
func boundColdTTFTvsPrev(r Row, k MetricKey, opp string) error {
	n, o, err := coldAccounts(r, opp)
	if err != nil {
		return err
	}
	if o.Served == 0 {
		return fmt.Errorf("no cold prompt both policies served%s", excludedNote(n.Excluded))
	}
	gap := n.TTFT - o.TTFT
	if gap <= 0 {
		return nil
	}
	ownExcess := n.Own - o.Own
	carried := (n.Other - o.Other) + (n.Rows - o.Rows) + (n.Overhead - o.Overhead) +
		(n.Decode - o.Decode)
	idleExcess := n.Idle - o.Idle
	bought := ownExcess + carried + idleExcess
	if absF(gap-bought) > coldTTFTIdentitySlack*absF(gap) {
		return fmt.Errorf("NEW is %.1f s slower over %d windows but the buckets sum to %.1f s,"+
			" so the account does not close between the two policies", gap, n.Served, bought)
	}
	// The same prompt must be the same computation.
	if n.OwnTokens != o.OwnTokens {
		return fmt.Errorf("NEW prefilled %d tokens of the cold prompt and %s %d, so the two are not"+
			" scheduling the same work", n.OwnTokens, opp, o.OwnTokens)
	}
	// Own is the prompt's chunks priced on their own, with no overhead and no
	// batch-mates in it, so a policy cannot move it by scheduling. One recomputed
	// prefix would move it by more than every cell in the suite.
	if ownExcess > ownTolerance*o.Own {
		return fmt.Errorf("NEW charged %.1f s for the prompt's own chunks against %s %.1f s, %.1f s"+
			" more than the %.0f%% a re-cut of %d identical tokens can cost, over %d passes vs %d,"+
			" so the gap is what got computed and not when",
			n.Own, opp, o.Own, ownExcess-ownTolerance*o.Own, 100*ownTolerance, n.OwnTokens,
			n.OwnBatches, o.OwnBatches)
	}
	// The rest of the wait must be work NEW chose to run for other requests.
	if carried < gap-ownTolerance*o.Own {
		return fmt.Errorf("NEW is %.1f s slower over %d windows than %s, but follow-up prefill %.1f s,"+
			" decode rows %.1f s, overhead %.1f s and decode %.1f s add up to only %.1f s of it, and"+
			" the %.1f s NEW idled is not work either",
			gap, n.Served, opp, n.Other-o.Other, n.Rows-o.Rows, n.Overhead-o.Overhead,
			n.Decode-o.Decode, carried, idleExcess)
	}
	if idleExcess > prevIdleSlack*o.WindowSecs {
		return fmt.Errorf("NEW idled %.1f s more than %s over %.1f s of cold windows, so part of the"+
			" wait is GPU left unused rather than work bought elsewhere",
			idleExcess, opp, o.WindowSecs)
	}
	return nil
}

// excludedNote renders the windows a bound had to leave out, so a number quoted
// without them says how few windows it was actually argued from.
func excludedNote(skipped [NumExclusions]int) string {
	total, parts := 0, []string{}
	for e := Exclusion(1); e < Exclusion(NumExclusions); e++ {
		if skipped[e] > 0 {
			total += skipped[e]
			parts = append(parts, fmt.Sprintf("%d %s", skipped[e], e))
		}
	}
	if total == 0 {
		return ""
	}
	return fmt.Sprintf("; excluded %s", strings.Join(parts, ", "))
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
	// Delivered volume inside the windows is the direct measure of that. The stream
	// rate is not: it divides that volume by stream-seconds the policy chose for
	// itself, so a policy serving strictly more tokens can post a lower rate.
	ni, oi := PolicyIndex(ModeNew), PolicyIndex(prevOr(opp))
	nf, of := coldFlowOf(r, ModeNew), coldFlowOf(r, Mode(prevOr(opp)))
	if nf.Toks < of.Toks {
		return fmt.Errorf("%s delivered %.0f agent tokens during the cold prefills against NEW's %.0f,"+
			" so it is not withholding tokens and NEW's longer tail has no starvation to explain it",
			opp, of.Toks, nf.Toks)
	}
	if pa := r.Metrics[oi].PerAgentTokSCold; !isNaN(pa) && !(r.Metrics[ni].PerAgentTokSCold > pa) {
		return fmt.Errorf("NEW delivers %.2f tok per agent during a cold prefill against %s %.2f, so"+
			" it is not the policy keeping the whole conversation set generating",
			r.Metrics[ni].PerAgentTokSCold, opp, pa)
	}
	if r.Metrics[2].LongestStall > r.Metrics[PolicyIndex(Mode(prevOr(opp)))].LongestStall {
		return fmt.Errorf("NEW's longest stall is worse too, so the percentile is not the only thing moving")
	}
	return nil
}

// coldFlow is metric 1's numerator and denominator for one policy, summed over
// the scenario's seeds, with the mixed-batch rows that carry part of the
// numerator.
type coldFlow struct {
	Toks, Secs, Rows, RowSecs float64
}

func (f coldFlow) rate() float64 { return f.Toks / f.Secs }

// coldFlowOf reduces one policy's runs to metric 1's two quantities.
func coldFlowOf(r Row, mode Mode) coldFlow {
	var f coldFlow
	for _, res := range r.Runs[PolicyIndex(mode)] {
		wins := coldWindows(res, r.Scenario.Window)
		f.Toks += sum(coldTokens(res, wins))
		f.Secs += streamSeconds(res, wins)
		for _, b := range res.Batches {
			if b.IsPrefill && inWindows(wins, b.End) {
				f.Rows += float64(len(b.Rows))
				f.RowSecs += b.End - b.Start
			}
		}
	}
	return f
}

// samplesAbove counts the per-token inter-token samples longer than x, and all
// samples, for one policy over the scenario's seeds. These are the same samples
// Measure takes a percentile of.
func samplesAbove(r Row, mode Mode, x float64) (over, total int) {
	for _, res := range r.Runs[PolicyIndex(mode)] {
		for _, g := range spread(res, func(*Request) bool { return true }) {
			total++
			if g > x {
				over++
			}
		}
	}
	return over, total
}

// mixedRowTally counts the decode rows NEW's prefill batches carried and the
// longest forward pass NEW launched: the population that can file a one-token
// sample at prefill cost, and the ceiling on such a sample.
func mixedRowTally(r Row) (rows, batchMax float64) {
	for _, res := range r.Runs[PolicyIndex(ModeNew)] {
		for _, b := range res.Batches {
			if !b.IsPrefill {
				continue
			}
			rows += float64(len(b.Rows))
			if d := b.End - b.Start; d > batchMax {
				batchMax = d
			}
		}
	}
	return rows, batchMax
}

// tailShares is the single-token share of NEW's and of the other policy's
// inter-token samples, over the same windowed population the bound tests.
func tailShares(r Row, opp Mode) (newShare, oppShare float64) {
	var s, t, s2, t2 int
	for _, res := range r.Runs[PolicyIndex(ModeNew)] {
		a, b, _ := SingleTokenSamples(res, r.Scenario.Window)
		s, t = s+a, t+b
	}
	for _, res := range r.Runs[PolicyIndex(opp)] {
		a, b, _ := SingleTokenSamples(res, r.Scenario.Window)
		s2, t2 = s2+a, t2+b
	}
	if t > 0 {
		newShare = float64(s) / float64(t)
	}
	if t2 > 0 {
		oppShare = float64(s2) / float64(t2)
	}
	return newShare, oppShare
}

// boundITLTailUnderEviction argues the inter-token percentiles of the thrash
// episode, where no cold prompt gives the mixed-chunk story a window to hide in.
// Every turn there misses its prefix, so the prefill batches are long, and NEW
// folds one decode row per running request into each of them: a served token is
// charged the whole batch. OLD and PREV only ever price a token against a decode
// step. The bound is therefore that NEW's percentile is a forward pass it launched
// itself, and that the samples which put it there are no more numerous than the
// rows.
// See docs/derivation-itl-percentiles-under-mixed-chunk.md.
func boundITLTailUnderEviction(r Row, k MetricKey, opp string) error {
	ni, oi := PolicyIndex(ModeNew), PolicyIndex(prevOr(opp))
	value := r.Metrics[ni].Value(k)
	var batchMax, rows float64
	for _, res := range r.Runs[ni] {
		pf, _, _ := res.BatchSeconds(r.Scenario.Window)
		if m := maxOf(pf); !isNaN(m) && m > batchMax {
			batchMax = m
		}
		for _, b := range res.Batches {
			if b.IsPrefill {
				rows += float64(len(b.Rows))
			}
		}
	}
	if isNaN(value) {
		return fmt.Errorf("NEW has no value for %s", k)
	}
	if value > batchMax {
		return fmt.Errorf("NEW's %s of %.3f s outlasts its longest forward pass of %.3f s, so it is"+
			" not a single batch's stall", k, value, batchMax)
	}
	over, total := samplesAbove(r, ModeNew, r.Metrics[oi].Value(k))
	if float64(over)/float64(total) < percentileAt(k) {
		return fmt.Errorf("only %.3f%% of NEW's samples exceed %s's %.3f s, under the %.2f%% tail a"+
			" p%.1f is made of, so the two percentiles are not separated by this population",
			100*float64(over)/float64(total), opp, r.Metrics[oi].Value(k), 100*percentileAt(k),
			100*(1-percentileAt(k)))
	}
	if over > int(rows) {
		return fmt.Errorf("%d of NEW's inter-token samples exceed %s's %.3f s but only %.0f tokens"+
			" were served as a mixed row, so %d samples are not the rows channel",
			over, opp, r.Metrics[oi].Value(k), rows, over-int(rows))
	}
	if r.Metrics[ni].LongestStall > r.Metrics[oi].LongestStall {
		return fmt.Errorf("NEW's longest stall is %.3f s against %s %.3f s, so the tail is not the"+
			" only thing moving", r.Metrics[ni].LongestStall, opp, r.Metrics[oi].LongestStall)
	}
	if !(r.Metrics[ni].ITLp50 <= r.Metrics[oi].ITLp50) {
		return fmt.Errorf("NEW's median inter-token gap is %.4f s against %s %.4f s, so this is a"+
			" worse distribution, not one shifted only at the top", r.Metrics[ni].ITLp50, opp,
			r.Metrics[oi].ITLp50)
	}
	return nil
}

// percentileAt is the sample fraction a metric's name claims, as a fraction.
func percentileAt(k MetricKey) float64 {
	if k == MITLp999 {
		return 0.001
	}
	return 0.01
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

// servedWindow finds the window with this tag that the policy reached a first
// token on, or nil when it never did.
func servedWindow(res *Result, tag string) *ColdWindow {
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
