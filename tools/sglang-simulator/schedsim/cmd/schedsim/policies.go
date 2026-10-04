package main

import (
	"flag"
	"fmt"
	"io"
	"strings"

	"schedsim/internal/sim"
	"schedsim/internal/trace"
)

// policyFlags are the knobs of those-policy comparison.
type policyFlags struct {
	only    string
	workers int
	sweep   bool
	seeds   int
	aSeed   int64
}

func registerPolicyFlags(fs *flag.FlagSet) *policyFlags {
	p := &policyFlags{}
	fs.StringVar(&p.only, "only", "", "comma-separated substrings of scenario names to report (default: all)")
	fs.IntVar(&p.workers, "workers", 0, "parallel run workers (default: the machine's CPU count)")
	fs.BoolVar(&p.sweep, "sweep-policies", true, "print the sensitivity sweep over the three policies")
	fs.IntVar(&p.seeds, "seeds", 0, "seeds per scenario, 0 for each scenario's own list")
	fs.Int64Var(&p.aSeed, "a-seed", 7, "scenario A's seed, which scripts its arrivals")
	return p
}

// printPolicies runs every scenario under OLD, PREV and NEW and writes the
// comparison. Costs come from the same calibration the log gives the other model,
// so both halves of the report charge the GPU the same way.
func printPolicies(w io.Writer, cal trace.Calibration, steps []trace.Step, pf *policyFlags) {
	cost := sim.NewCost(cal)
	linear := trace.FitPrefillLinear(steps, cal.ChunkSize)
	fmt.Fprintf(w, "\n=== Three-policy comparison ===\n\n")
	fmt.Fprintf(w, "Cost model fitted to the log: %s\n", cal.String())
	fmt.Fprintf(w, "Policies: OLD = upstream prefill-priority; PREV = the balancer at f15db9ee6a; "+
		"NEW = the balancer and eviction throttle on this branch.\n")
	fmt.Fprintf(w, "Device pool %d tokens, chunk %d, page %d, %d drafts accepted to %.2f mean per step.\n\n",
		cal.DeviceTokens, cal.ChunkSize, cal.Page, cal.Decode.NumDraft, cal.Decode.AcceptMean)

	scs := sim.BaseScenarios()
	if pf.only != "" {
		var kept []sim.Scenario
		for _, sc := range scs {
			for _, want := range strings.Split(pf.only, ",") {
				if want != "" && strings.Contains(strings.ToLower(sc.Name), strings.ToLower(want)) {
					kept = append(kept, sc)
					break
				}
			}
		}
		scs = kept
	}
	if pf.seeds > 0 {
		for i := range scs {
			scs[i].Seeds = seedList(pf.seeds)
		}
	}
	rows := sim.RunSuite(scs, cost, sim.DefaultConfig, pf.workers)
	sim.WriteTables(w, rows)
	if pf.sweep {
		seeds := []int64{1, 2, 3}
		if pf.seeds > 0 {
			seeds = seedList(pf.seeds)
		}
		fmt.Fprintln(w, strings.Repeat("=", 118))
		sweep := sim.RunSweep(cost, linear, seeds, pf.aSeed, pf.workers)
		sim.WriteSweep(w, sweep)
		fmt.Fprintln(w, "Cells where the contract does not hold are listed per variant above,")
		fmt.Fprintln(w, "and the derivations for the provable ones live beside the scenario tests.")
	}
}

func seedList(n int) []int64 {
	out := make([]int64, n)
	for i := range out {
		out[i] = int64(i + 1)
	}
	return out
}
