// Command schedsim runs every scenario under the OLD, PREV and NEW
// schedulers and prints one markdown table per scenario group.
package main

import (
	"flag"
	"fmt"
	"os"

	"schedsim/internal/cost"
	"schedsim/internal/scenario"
)

func main() {
	logPath := flag.String("log", "", "log to read (default: the embedded live serving log)")
	chunkSize := flag.Int("chunk-size", 4096, "tokens per cold-prefill chunk to look for")
	prefillShare := flag.Float64("prefill-share", 0.5, "prefill's share of contended GPU time under the time-balance policy")
	sweep := flag.Bool("sweep", true, "print the fixed-interval sensitivity table")
	decodePerReq := flag.Float64("decode-per-req", 0.0, "ASSUMED: fractional decode cost added per extra running request")
	interference := flag.Float64("prefill-interference", 0.0, "ASSUMED: fractional prefill slowdown from sharing the GPU with decode")
	policy := registerPolicyFlags(flag.CommandLine)
	flag.Parse()

	cal, err := cost.Calibrate()
	if err != nil {
		fmt.Fprintf(os.Stderr, "schedsim: calibrate: %v\n", err)
		os.Exit(1)
	}
	base := scenario.Deployment(cal.Model)
	fmt.Printf("Cells are OLD / PREV / NEW, each the mean of %d workload seeds (recomputes: their sum). Device pool %d tokens, host tier %d tokens.\n\n",
		len(scenario.Seeds), base.Cost.DevicePoolTokens, base.HostTokens)
	if *only == "bounds" {
		all := append(scenario.Main(), scenario.Sensitivity()...)
		scenario.WriteBounds(os.Stdout, scenario.RunAll(all, base), base)
		return
	}
	if *only != "sensitivity" {
		scenario.WriteTable(os.Stdout, scenario.RunAll(scenario.Main(), base))
		fmt.Println()
	}
	if *only != "main" {
		scenario.WriteTable(os.Stdout, scenario.RunAll(scenario.Sensitivity(), base))
	}

	// The revised scheduler as it resolves by default (mixed chunk on, so the
	// running requests also decode inside every chunk), then the balancer
	// alone, and with the log's peak queue behind the chunks (which cannot
	// join a chunk and so does not count).
	var revisedRes sched.Result
	for _, q := range []struct {
		label string
		queue int
		mixed bool
	}{
		{"mixed chunk (default)", metrics.QueueAtCold, true},
		{"no mixed", metrics.QueueAtCold, false},
		{"no mixed: #queue-req at its peak", metrics.QueuePeakInCold, false},
	} {
		p := work.Params(sched.PolicyQueueBalance)
		p.MixedChunk = q.mixed
		p.DecodePerReqFraction = *decodePerReq
		p.PrefillInterference = *interference
		r := sched.Simulate(work, p, q.queue)
		if len(in.Revised) == 0 {
			revisedRes = r
		}
		in.Revised = append(in.Revised, report.RevisedRun{Queue: q.queue, Label: q.label, Result: r})
	}

	if *sweep {
		for _, n := range []int{1, 2, 4, 8, 16} {
			p := work.Params(sched.PolicyFixedInterval)
			p.Interval = n
			p.DecodePerReqFraction = *decodePerReq
			p.PrefillInterference = *interference
			r := sched.Simulate(work, p, metrics.QueueAtCold)
			in.Sweep = append(in.Sweep, report.SweepPoint{
				Interval:          n,
				DecodeSteps:       r.DecodeSteps,
				EffectiveGenTPS:   r.EffectiveGenTPS,
				PrefillFinishSecs: r.WindowSeconds,
			})
		}
	}

	report.Write(os.Stdout, in)
	fmt.Fprintln(os.Stdout)
	report.Summary(os.Stdout, oldRes, newRes, revisedRes)
	printPolicies(os.Stdout, trace.Calibrate(steps, *chunkSize, PageSize), policy)
}

// PageSize is the deployment's page_size, which the log's server arguments fix at 64.
const PageSize = 64

func splitLines(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "schedsim: "+format+"\n", args...)
	os.Exit(1)
}
