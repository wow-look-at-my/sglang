package sim

import (
	"fmt"
	"math"
	"sort"
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
	case boundEvictionTail:
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
	case boundMixedChunkTail:
		q := percentileAt(k)
		band, err := itlPassBand(r, Mode(prevOr(opp)), q)
		if err != nil {
			return "", false
		}
		head := fmt.Sprintf("a p%.1f names %d of %d NEW samples; %d ride a mixed batch and %d more waited"+
			" out a pass they were not a row on (%d mixed + %d waited inside the tail itself; of the %d"+
			" waiting samples %d are a stream's first gap and %d mid-stream); the tail's %.1f s of wait"+
			" leaves %.2f%% inside no launched pass and its widest gap spans %d prefill pass(es)",
			100*(1-q), band.Slots, band.Samples, band.Mixed, band.Waited, band.TailMixed,
			band.TailWaited, band.Waited, band.WaitedFirst, band.WaitedMid, band.Wait,
			100*idleShare(band), band.MaxPasses)
		claim := fmt.Sprintf("NEW's own decode-class p%.1f is %s against %s's %s", 100*(1-q),
			fmtVal(k, band.DecodeCut), opp, fmtVal(k, r.Metrics[PolicyIndex(prevOr(opp))].Value(k)))
		if band.Mixed >= band.Slots {
			return head + fmt.Sprintf("; the named sample is a mixed ride, so the bound is NEW's %s of %s"+
				" sitting under its longest forward pass of %s; %s", k.String(),
				fmtVal(k, r.Metrics[2].Value(k)), fmtVal(k, band.PassMax), claim), true
		}
		return head + fmt.Sprintf("; the named sample waited out one prefill pass, so the ceiling is the"+
			" chunk at this run's deepest context %s + %d riding rows %s + reload %s + the calibrated step"+
			" at its largest decode batch %s = %s, over NEW's %s of %s (its longest pass and step were %s"+
			" + %s); %s", fmtVal(k, band.ChunkCeil), band.MaxRows, fmtVal(k, band.RowCeil),
			fmtVal(k, band.ReloadCeil), fmtVal(k, band.StepCeil), fmtVal(k, band.Ceiling), k.String(),
			fmtVal(k, r.Metrics[2].Value(k)), fmtVal(k, band.PassMax), fmtVal(k, band.StepMax),
			claim), true
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
// tokens contributes N samples of gap/N, so which delivery the named sample came
// from is a count, not a guess: the class that fills the p(1-q) tail owns the
// reported number. Two classes can fill it, and the bound each gets is different.
//
// Mixed class first. A mixed delivery is one token carried by a prefill batch, so
// it files that batch's whole wall clock as one sample; a policy that serves its
// streams only in decode batches files one sample per batch covering several
// tokens. When that class alone reaches the tail's slot count the named sample is a
// ride, and the bound is that it is not longer than one of NEW's own forward
// passes -- the stall bound in another unit.
//
// When the mixed class falls short of the slot count, the named sample is the
// largest non-mixed one and a mixed ride is not this cell's mechanism. The tail
// then belongs to the pass-cost class: the samples that waited out a prefill pass
// they were not a row on. Every one of those is a stream's first gap, because the
// pass it waits out was composed one overlap decision before it joined the running
// batch (see Balancer's charge-at-completion note), and a stream that is already a
// row is served by every pass. One gap therefore spans at most one pass, and the
// bound on it is the calibrated wall clock of that pass plus the step that ended
// the gap: a whole chunk plus one extend token per riding request, both priced at
// the deepest mid-context this run's extend batches attended over, the largest
// host-tier copy any pass paid, and the calibrated decode step at the largest
// decode batch NEW launched. Every term is computed from the cost model and the
// run's own trace, never stated as a constant, and if the run's numbers cannot
// produce a ceiling the percentile sits under, the cell is not arguable and fails.
//
// The pass-cost bound carries two more claims. None of the tail's wait may be GPU
// time inside no launched pass: a percentile made of dead time is a defect, not a
// trade. And NEW's pure-decode samples must beat the opponent's reported percentile
// outright -- the improvement claim, which is what trips if a future control law
// makes NEW's own steps the slower ones rather than merely shifting its tail.
//
// Admissibility is shared: the older policy must be withholding tokens outright,
// which is what makes its shorter tail a symptom rather than a merit.
// See docs/derivation-itl-percentiles-under-mixed-chunk.md.
func boundITLPercentile(r Row, k MetricKey, opp string) error {
	newRuns, oppRuns := r.Runs[PolicyIndex(ModeNew)], r.Runs[PolicyIndex(ModeOld)]
	if opp == "PREV" {
		oppRuns = r.Runs[PolicyIndex(ModePrev)]
	}
	if len(newRuns) != len(oppRuns) {
		return fmt.Errorf("%d NEW runs vs %d %s runs", len(newRuns), len(oppRuns), opp)
	}
	q := percentileAt(k)
	band, err := itlPassBand(r, Mode(prevOr(opp)), q)
	if err != nil {
		return err
	}
	if band.Samples == 0 || band.OppSamples == 0 {
		return fmt.Errorf("no inter-token samples to argue from")
	}
	if band.OppMixed >= band.Mixed {
		return fmt.Errorf("%s files %d of %d samples as a mixed delivery against NEW's %d of %d, so it"+
			" is not the policy withholding tokens inside prefill batches", opp, band.OppMixed,
			band.OppSamples, band.Mixed, band.Samples)
	}
	value := r.Metrics[2].Value(k)
	switch {
	case band.Mixed >= band.Slots:
		// The named sample is a mixed delivery: one pass's wall clock, one token.
	case band.Mixed+band.Waited >= band.Slots:
		if err := band.checkPassCost(value, r.Metrics[PolicyIndex(prevOr(opp))].Value(k), q); err != nil {
			return err
		}
	default:
		return fmt.Errorf("%d mixed samples and %d that waited out a prefill pass, of %d: a p%.1f names"+
			" %d, so neither a mixed ride nor one forward pass reaches a percentile this long",
			band.Mixed, band.Waited, band.Samples, 100*(1-q), band.Slots)
	}
	// No sample can outlast a forward pass that served the stream, and the
	// percentile must sit under the longest one NEW launched.
	if value > band.PassMax {
		return fmt.Errorf("NEW's p%.1f is %.3f s, longer than its longest forward pass of %.3f s",
			100*(1-q), value, band.PassMax)
	}
	return boundTailAdmissible(r, k, opp)
}

// itlBand is the population one inter-token percentile is drawn from, split by the
// delivery each sample came from and by the prefill passes its gap waited out.
// Both branches of boundITLPercentile and the evidence line read it, so a cell is
// argued on the numbers it can also show.
type itlBand struct {
	// Samples is the metric's population, one per generated token, and Slots the
	// count a p(1-q) names: the samples at or above the reported percentile.
	Samples, Slots int
	// Mixed counts samples a prefill batch handed its riding request; Waited
	// counts the decode-class samples whose gap spanned a prefill pass, of which
	// WaitedFirst are a stream's first gap and WaitedMid the ones that were not.
	Mixed, Waited, WaitedFirst, WaitedMid int
	// TailMixed and TailWaited are the same two classes within the samples at or
	// above the reported percentile, which is what decides what that percentile is.
	TailMixed, TailWaited int
	// MaxPasses is the most prefill passes any one decode-class gap spanned.
	MaxPasses int
	// Idle and Wait sum the unaccounted and the total GPU-clock seconds over the
	// samples at or above the cut.
	Idle, Wait float64
	// DecodeCut is the p(1-q) of NEW's decode-class samples, token-weighted like
	// the metric itself: the percentile with the pass-cost samples taken out.
	DecodeCut float64
	// PassMax and StepMax are the longest prefill pass and decode step NEW
	// launched. Ceiling is what one such pair may cost at most: a whole chunk of
	// prefill tokens plus one extend token for every request the pass can carry a
	// row for, both priced at the deepest mid-context this run's extend batches
	// attended over, the largest host-tier copy any pass paid, and the calibrated
	// step at the largest decode batch. Every term comes from the cost model the run
	// itself ran on and from the run's own batches.
	PassMax, StepMax, Ceiling float64
	// MaxRows is the most requests any one pass carried as decode rows.
	MaxRows int
	// The terms Ceiling is built from, kept so the evidence line can show the sum.
	ChunkCeil, RowCeil, ReloadCeil, StepCeil float64
	// OppSamples and OppMixed are the same population for the policy compared
	// against, counted the same way.
	OppSamples, OppMixed int
}

// itlPassBand measures the band over NEW's runs and the opponent's, per run: the
// batches of one seed cannot be attributed to another seed's deliveries, so the
// populations are reduced per run and summed, never pooled into one trace.
func itlPassBand(r Row, opp Mode, q float64) (itlBand, error) {
	var band itlBand
	type sample struct {
		v, idle, wait float64
		mixed, waited bool
	}
	var all, decode []sample
	for _, res := range r.Runs[PolicyIndex(ModeNew)] {
		if res == nil {
			return band, fmt.Errorf("a NEW run is missing")
		}
		cost := res.Cfg.Cost
		if cost.Cal.ChunkSize == 0 {
			return band, fmt.Errorf("the run carries no chunk size to price a pass with")
		}
		tl := newGPUTimeline(res)
		cls := classifyDeliveries(res)
		deepest, reloadMax, rowsMax, stepCeil := 0.0, 0.0, 0, 0.0
		for _, b := range res.Batches {
			if b.IsPrefill {
				for _, it := range b.Items {
					if m := float64(it.PrefixBefore) + float64(it.Extend)/2; m > deepest {
						deepest = m
					}
				}
				if d := b.End - b.Start; d > band.PassMax {
					band.PassMax = d
				}
				if b.ReloadSeconds > reloadMax {
					reloadMax = b.ReloadSeconds
				}
				if len(b.Rows) > rowsMax {
					rowsMax = len(b.Rows)
				}
				continue
			}
			sumCtx := 0
			for i := range b.Decode {
				sumCtx += b.Decode[i].Context() - b.DecodeToks[i]
			}
			if c := cost.DecodeSeconds(len(b.Decode), sumCtx); c > stepCeil {
				stepCeil = c
			}
			if d := b.End - b.Start; d > band.StepMax {
				band.StepMax = d
			}
		}
		// One pass can carry a whole chunk of prefill tokens, one extend token per
		// riding request, and the host-tier copy the admission it completes pays.
		// All of it priced at the deepest context this run's prefill attended over.
		perTok := cost.Cal.Prefill.PrefillSecondsPerToken(deepest)
		chunkCeil := float64(res.Cfg.ChunkSize)*perTok + cost.PerBatchSeconds()
		rowCeil := float64(rowsMax) * perTok
		if c := chunkCeil + rowCeil + reloadMax + stepCeil; c > band.Ceiling {
			band.Ceiling = c
			band.ChunkCeil, band.RowCeil = chunkCeil, rowCeil
			band.ReloadCeil, band.StepCeil = reloadMax, stepCeil
			band.MaxRows = rowsMax
		}
		for _, req := range res.Requests {
			for i := 1; i < len(req.Deliveries); i++ {
				d := req.Deliveries[i]
				n := maxI(d.N, 1)
				from := req.Deliveries[i-1].T
				gap := d.T - from
				npf, busy := tl.inside(from, d.T)
				label := cls[deliveryKey{req.ID, int64(math.Round(d.T * 1e6))}]
				s := sample{v: gap / float64(n), idle: gap - busy, wait: gap,
					mixed: label == "mixed", waited: label == "decode" && npf > 0}
				for k := 0; k < n; k++ {
					all = append(all, s)
					if label == "decode" {
						decode = append(decode, s)
					}
				}
				switch {
				case s.mixed:
					band.Mixed += n
				case s.waited:
					band.Waited += n
					if i == 1 {
						band.WaitedFirst += n
					} else {
						band.WaitedMid += n
					}
					if npf > band.MaxPasses {
						band.MaxPasses = npf
					}
				}
			}
		}
	}
	band.Samples = len(all)
	band.Slots = int(math.Ceil(q * float64(len(all))))
	sort.Slice(all, func(i, j int) bool { return all[i].v < all[j].v })
	for _, s := range all[maxI(0, len(all)-band.Slots):] {
		band.Idle += s.idle
		band.Wait += s.wait
		if s.mixed {
			band.TailMixed++
		}
		if s.waited {
			band.TailWaited++
		}
	}
	dv := make([]float64, len(decode))
	for i, s := range decode {
		dv[i] = s.v
	}
	band.DecodeCut = pct(dv, 100*(1-q))
	for _, res := range r.Runs[PolicyIndex(opp)] {
		if res == nil {
			return band, fmt.Errorf("a %s run is missing", opp)
		}
		cls := classifyDeliveries(res)
		for _, req := range res.Requests {
			for i := 1; i < len(req.Deliveries); i++ {
				n := maxI(req.Deliveries[i].N, 1)
				band.OppSamples += n
				if cls[deliveryKey{req.ID,
					int64(math.Round(req.Deliveries[i].T * 1e6))}] == "mixed" {
					band.OppMixed += n
				}
			}
		}
	}
	return band, nil
}

// checkPassCost states the arithmetic of a tail made of samples that waited out one
// prefill pass: the wait is one pass and one step, its whole length is work some
// other request was served, and NEW's own decode steps are still the faster ones.
func (b itlBand) checkPassCost(value, oppValue, q float64) error {
	if b.TailMixed+b.TailWaited < b.Slots {
		return fmt.Errorf("%d mixed and %d pass-waiting samples fill only %d of the %d slots a p%.1f"+
			" names, so the percentile itself is a plain decode sample that no forward pass explains",
			b.TailMixed, b.TailWaited, b.TailMixed+b.TailWaited, b.Slots, 100*(1-q))
	}
	if b.WaitedMid > 0 {
		return fmt.Errorf("%d of NEW's samples are a mid-stream delivery that waited out a prefill pass"+
			" it was not a row on, so the ride is not covering every running request", b.WaitedMid)
	}
	if b.MaxPasses > 1 {
		return fmt.Errorf("NEW's widest gap spans %d back-to-back prefill passes, so the tail is more"+
			" than the one batch's stall this bound prices", b.MaxPasses)
	}
	if b.PassMax+b.StepMax > b.Ceiling {
		return fmt.Errorf("NEW launched a prefill pass of %.3f s and a decode step of %.3f s together"+
			" worth %.3f s, over the %.3f s the deployment's own chunk, rows and reload cost at the"+
			" deepest context this run reached, so the wait a stream can be charged is not bounded by"+
			" one chunk of the deployment's own sizing", b.PassMax, b.StepMax, b.PassMax+b.StepMax,
			b.Ceiling)
	}
	if value > b.Ceiling {
		return fmt.Errorf("NEW's p%.1f of %.3f s exceeds the %.3f s one pass and one step can cost at"+
			" the deepest context this run reached", 100*(1-q), value, b.Ceiling)
	}
	if b.Wait <= 0 {
		return fmt.Errorf("the samples at or above the cut waited out no GPU time at all")
	}
	if b.Idle > itlIdleSlack*b.Wait {
		return fmt.Errorf("%.1f ms of the %.1f ms the tail samples waited out is not inside a launched"+
			" pass (%.2f%%, against a %.2f%% tolerance), so part of NEW's tail is a GPU left idle rather"+
			" than work run for another request", b.Idle*1e3, b.Wait*1e3, 100*b.Idle/b.Wait,
			100*itlIdleSlack)
	}
	if b.DecodeCut > oppValue {
		return fmt.Errorf("NEW's decode-class samples reach %.3f s at their p%.1f against the %.3f s"+
			" percentile it is being compared to, so its own decode steps are the slower ones, which"+
			" no tail's shape explains", b.DecodeCut, 100*(1-q), oppValue)
	}
	return nil
}

// itlIdleSlack bounds the share of a tail sample's wait that no launched forward
// pass accounts for. The engine charges every pass its calibrated seconds and
// starts each at the end of the last one, so a gap with time inside no pass is the
// scheduler holding no work; measured, the tail samples of every cell this bound
// covers account for 100% of their wait.
const itlIdleSlack = 0.01

// boundTailAdmissible is the shared test that the policy being beaten was
// withholding tokens outright: delivered volume inside the cold prefills is the
// direct measure of that. The stream rate is not: it divides that volume by
// stream-seconds the policy chose for itself, so a policy serving strictly more
// tokens can post a lower rate.
func boundTailAdmissible(r Row, k MetricKey, opp string) error {
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
	if r.Metrics[ni].LongestStall > r.Metrics[oi].LongestStall {
		return fmt.Errorf("NEW's longest stall is worse too, so the percentile is not the only thing moving")
	}
	return nil
}

// gpuTimeline indexes one run's batches for a gap query. The engine starts every
// pass at or after the end of the previous one, so the list is ordered and
// non-overlapping and a gap's contents are a binary search plus two prefix sums.
type gpuTimeline struct {
	starts, ends, busy, prefill []float64
}

func newGPUTimeline(res *Result) gpuTimeline {
	tl := gpuTimeline{}
	var busy, pf float64
	for _, b := range res.Batches {
		tl.starts = append(tl.starts, b.Start)
		tl.ends = append(tl.ends, b.End)
		busy += b.End - b.Start
		if b.IsPrefill {
			pf++
		}
		tl.busy = append(tl.busy, busy)
		tl.prefill = append(tl.prefill, pf)
	}
	return tl
}

// inside reports how many prefill passes and how many GPU seconds fall in
// (from, to], the window one inter-token gap covers.
func (tl gpuTimeline) inside(from, to float64) (passes int, secs float64) {
	if len(tl.ends) == 0 {
		return 0, 0
	}
	lo := sort.Search(len(tl.ends), func(i int) bool { return tl.ends[i] > from })
	hi := sort.Search(len(tl.starts), func(i int) bool { return tl.starts[i] >= to })
	if hi <= lo {
		return 0, 0
	}
	prior, priorPf := 0.0, 0.0
	if lo > 0 {
		prior, priorPf = tl.busy[lo-1], tl.prefill[lo-1]
	}
	return int(tl.prefill[hi-1] - priorPf), tl.busy[hi-1] - prior
}

// idleShare is the part of a tail's wait that no launched forward pass covers.
func idleShare(b itlBand) float64 {
	if b.Wait <= 0 {
		return 0
	}
	return b.Idle / b.Wait
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
