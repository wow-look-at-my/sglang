package sim

import (
	"math"
	"testing"

	"schedsim/internal/trace/tracetest"
)

// mixedShare is the weight of mixed-batch deliveries in the metric's population,
// which counts one ITL sample per token.
func mixedShare(res *Result) float64 {
	stats := deliveryClassStats(res)
	total, mixed := 0, 0
	for _, c := range stats {
		total += c.Copies
		if c.Label == "mixed" {
			mixed = c.Copies
		}
	}
	if total == 0 {
		return 0
	}
	return float64(mixed) / float64(total)
}

// TestScenarioBBalanceWinsAgainstPrev pins the scenario-B comparisons the
// balancer decides, over the committed seeds: a change to the control law that
// gives one of them back fails the build instead of costing a re-read of the
// printed table. ITL p99 is absent on purpose: docs/derivation-itl-percentiles-under-mixed-chunk.md
// derives that cell from the delivery accounting rather than from the schedule.
func TestScenarioBBalanceWinsAgainstPrev(t *testing.T) {
	cost := ScenarioCost()
	rows := RunSuite([]Scenario{ScenarioB(2), ScenarioB(5)}, cost, DefaultConfig, 0)
	for _, row := range rows {
		t.Run(row.Scenario.Name, func(t *testing.T) {
			old, prev, neu := row.Metrics[ModeOld], row.Metrics[ModePrev], row.Metrics[ModeNew]
			t.Logf("%s / OLD / PREV / NEW", row.Scenario.Name)
			t.Logf("  stream time in stalls over 1 s: %.2f%% / %.2f%% / %.2f%%",
				100*old.StallFrac1s, 100*prev.StallFrac1s, 100*neu.StallFrac1s)
			t.Logf("  longest stall: %s / %s / %s", seconds(old.LongestStall),
				seconds(prev.LongestStall), seconds(neu.LongestStall))
			t.Logf("  stream decode tok/s in cold: %.1f / %.1f / %.1f",
				old.StreamDecodeTokSCold, prev.StreamDecodeTokSCold, neu.StreamDecodeTokSCold)
			t.Logf("  output tok/s: %.1f / %.1f / %.1f",
				old.OutputTokS, prev.OutputTokS, neu.OutputTokS)
			if neu.LongestStall > prev.LongestStall {
				t.Errorf("longest stall %s, worse than PREV %s",
					seconds(neu.LongestStall), seconds(prev.LongestStall))
			}
			if neu.StallFrac1s > prev.StallFrac1s {
				t.Errorf("stream time in stalls over 1 s %.2f%%, worse than PREV %.2f%%",
					100*neu.StallFrac1s, 100*prev.StallFrac1s)
			}
			if neu.OutputTokS < prev.OutputTokS {
				t.Errorf("output %.1f tok/s, worse than PREV %.1f",
					neu.OutputTokS, prev.OutputTokS)
			}
			if neu.StreamDecodeTokSCold < prev.StreamDecodeTokSCold {
				t.Errorf("stream decode %.1f tok/s in cold, worse than PREV %.1f",
					neu.StreamDecodeTokSCold, prev.StreamDecodeTokSCold)
			}
			if neu.PerAgentTokSCold < prev.PerAgentTokSCold {
				t.Errorf("per-agent %.1f tok/s in cold, worse than PREV %.1f",
					neu.PerAgentTokSCold, prev.PerAgentTokSCold)
			}

			pooled := pooledTrace(row.Runs[ModeNew])
			stats := deliveryClassStats(pooled)
			decode := stats["decode"]
			if decode == nil || len(decode.PerToken) == 0 {
				t.Fatal("no decode-class ITL samples to compare against")
			}
			decodeP99 := pct(decode.PerToken, 99)
			t.Logf("  NEW decode-class p99 %s, mixed share %.2f%% (PREV ITL p99 %s)",
				seconds(decodeP99), 100*mixedShare(pooled), seconds(prev.ITLp99))
			if decodeP99 > prev.ITLp99 {
				t.Errorf("NEW's pure-decode per-token p99 %s, worse than PREV's ITL p99 %s",
					seconds(decodeP99), seconds(prev.ITLp99))
			}

			// The bound holds per batch: at most one chunk of prefill tokens, plus the extend token each riding request adds.
			limit := row.Runs[ModeNew][0].Cfg.ChunkSize + row.Runs[ModeNew][0].Agents
			for _, r := range row.Runs[ModeNew] {
				for _, b := range r.Batches {
					if b.IsPrefill && b.ExtendTokens > limit {
						t.Fatalf("prefill batch carried %d tokens, over the %d allowed",
							b.ExtendTokens, limit)
					}
				}
			}
		})
	}
}

// TestMixedRideIsWhatDecidesTheP99Cell runs scenario B with mixed chunked prefill
// on and off for the interleaving policies, to separate what the control law costs
// from what the delivery form costs. The ride hands each running request one token
// per prefill batch, so it halves the longest stall and, because the metric divides
// a gap by the tokens it carried, raises that policy's own ITL p99. Turning the ride
// off wins p99 outright and gives back the tokens streams generate during a cold
// prompt. Numbers quoted in docs/derivation-itl-percentiles-under-mixed-chunk.md.
//
// The claim needs the regime where mixed deliveries reach PREV's p99 rank: a
// decode step costly enough that fewer decode samples dilute the rides. The
// test's log is the incident priced with that step.
func TestMixedRideIsWhatDecidesTheP99Cell(t *testing.T) {
	model := tracetest.DefaultModel
	model.DecodeBase = 16e-3
	cost := costOf(model)
	sc := ScenarioB(2)
	seeds := []int64{1, 2, 3} // the sensitivity sweep's own seeds
	off := func(mode Mode) Metrics {
		cfg := DefaultConfig(mode, cost)
		cfg.MixedChunk = false
		var runs []*Result
		for _, seed := range seeds {
			runs = append(runs, Run(sc, cfg, seed))
		}
		return Pool(runs, sc.Window)
	}
	on := func(mode Mode) Metrics {
		cfg := DefaultConfig(mode, cost)
		cfg.MixedChunk = true
		var runs []*Result
		for _, seed := range seeds {
			runs = append(runs, Run(sc, cfg, seed))
		}
		return Pool(runs, sc.Window)
	}
	prevOff, prevOn, newOn := off(ModePrev), on(ModePrev), on(ModeNew)
	if prevOn.ITLp99 <= prevOff.ITLp99 {
		t.Errorf("the ride did not raise PREV's ITL p99: %s with mixed chunk, %s without",
			seconds(prevOn.ITLp99), seconds(prevOff.ITLp99))
	}
	if prevOn.LongestStall >= prevOff.LongestStall {
		t.Errorf("the ride did not shorten PREV's longest stall: %s with mixed chunk, %s without",
			seconds(prevOn.LongestStall), seconds(prevOff.LongestStall))
	}
	if newOn.ITLp99 <= prevOn.ITLp99 {
		t.Errorf("NEW's p99 %s is not above PREV's with the same delivery form (%s), so the deficit is not the ride",
			seconds(newOn.ITLp99), seconds(prevOn.ITLp99))
	}
	withoutRide := off(ModeNew)
	if withoutRide.ITLp99 >= prevOff.ITLp99 {
		t.Errorf("p99 without the ride %s did not beat PREV's %s",
			seconds(withoutRide.ITLp99), seconds(prevOff.ITLp99))
	}
	if withoutRide.StreamDecodeTokSCold >= prevOff.StreamDecodeTokSCold {
		t.Errorf("p99 came free without the ride: stream rate %.1f tok/s against PREV's %.1f",
			withoutRide.StreamDecodeTokSCold, prevOff.StreamDecodeTokSCold)
	}
	// The table in docs/derivation-itl-percentiles-under-mixed-chunk.md is these
	// rows; each is printed as PREV / NEW.
	for _, r := range []struct {
		label     string
		prev, neu Metrics
	}{
		{"as shipped: off for PREV, on for NEW", prevOff, newOn},
		{"on for both", prevOn, newOn},
		{"off for both", prevOff, withoutRide},
	} {
		t.Logf("%s: ITL p99 %s / %s, longest stall %s / %s, stream decode tok/s in cold %.1f / %.1f",
			r.label, seconds(r.prev.ITLp99), seconds(r.neu.ITLp99),
			seconds(r.prev.LongestStall), seconds(r.neu.LongestStall),
			r.prev.StreamDecodeTokSCold, r.neu.StreamDecodeTokSCold)
	}
}

// The dense cold cadence and the busy short-chat scenario land above the line, the sparse cadence below
// it, and that is the whole difference between NEW winning p99 in one and losing it in the other.
func TestITLP99BandFollowsTheMixedShare(t *testing.T) {
	cost := ScenarioCost()
	for _, tc := range []struct {
		sc       Scenario
		overLine bool
	}{
		{sc: ScenarioB(2), overLine: true},
		{sc: ScenarioC(2, 16), overLine: true},
		{sc: ScenarioB(5), overLine: false},
	} {
		sc := tc.sc
		var runs []*Result
		for _, seed := range sc.Seeds {
			runs = append(runs, Run(sc, DefaultConfig(ModeNew, cost), seed))
		}
		pooled := pooledTrace(runs)
		share := mixedShare(pooled)
		decode := deliveryClassStats(pooled)["decode"]
		if decode == nil || len(decode.PerToken) == 0 {
			t.Fatalf("%s: no decode-class samples", sc.Name)
		}
		m := Pool(runs, sc.Window)
		decodeTop := pct(decode.PerToken, 99.9)
		decodeP99 := pct(decode.PerToken, 99)
		t.Logf("%s: mixed share %.2f%%, reported ITL p99 %s, decode class p99 %s p99.9 %s",
			sc.Name, 100*share, seconds(m.ITLp99), seconds(decodeP99), seconds(decodeTop))
		switch {
		case tc.overLine && share <= 0.01:
			t.Errorf("%s: mixed share %.2f%% fell under the 1%% line", sc.Name, 100*share)
		case tc.overLine:
			if m.ITLp99 <= decodeTop {
				t.Errorf("%s: p99 %s is still inside the decode band (its p99.9 is %s) at a %.2f%% mixed share",
					sc.Name, seconds(m.ITLp99), seconds(decodeTop), 100*share)
			}
		case share > 0.01:
			t.Errorf("%s: mixed share %.2f%% rose over the 1%% line", sc.Name, 100*share)
		default:
			if math.Abs(m.ITLp99-decodeP99) > 0.002 {
				t.Errorf("%s: p99 %s is not the decode band's %s at a %.2f%% mixed share",
					sc.Name, seconds(m.ITLp99), seconds(decodeP99), 100*share)
			}
		}
	}
}

// TestDeliveryClassShareTable prints the rows
// docs/derivation-itl-percentiles-under-mixed-chunk.md tabulates for the scenarios
// whose ITL cells the tables mark. Counts are summed per run rather than read off a
// merged trace, because request ids repeat across seeds.
func TestDeliveryClassShareTable(t *testing.T) {
	cost := ScenarioCost()
	for _, sc := range []Scenario{
		ScenarioA(testEpisode()), ScenarioC(2, 16), ScenarioC(0.5, 16),
		ScenarioD(400000), ScenarioD(100000), ScenarioThrash(4, 600), ScenarioThrash(4, 1800),
	} {
		for _, mode := range Modes {
			var runs []*Result
			mixed, total := 0, 0
			for _, seed := range sc.Seeds {
				res := Run(sc, DefaultConfig(mode, cost), seed)
				runs = append(runs, res)
				for _, c := range deliveryClassStats(res) {
					total += c.Copies
					if c.Label == "mixed" {
						mixed += c.Copies
					}
				}
			}
			m := Pool(runs, sc.Window)
			t.Logf("%s / %s: mixed %d of %d (%.2f%%), ITL p99 %s p99.9 %s, ITL p50 %s,"+
				" raw gap p99 %s, longest stall %s, stream time in stalls over 1 s %.2f%%,"+
				" GPU busy %.1f%%, decode share %.1f%%, output %.1f tok/s",
				sc.Name, mode, mixed, total, 100*float64(mixed)/float64(total),
				seconds(m.ITLp99), seconds(m.ITLp999), seconds(m.ITLp50),
				seconds(m.ITLp99Raw), seconds(m.LongestStall), 100*m.StallFrac1s,
				100*m.GPUBusyShare, 100*m.DecodeShare, m.OutputTokS)
		}
	}
}
