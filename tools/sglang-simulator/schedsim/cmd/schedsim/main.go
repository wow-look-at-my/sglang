// Command schedsim drives two scheduler policies over the same cold-prefill
// workload parsed from a live serving log, and prints what each does.
//
// Usage:
//
//	go run ./cmd/schedsim                 # embedded log at internal/trace/live_log.txt
//	go run ./cmd/schedsim -log FILE       # a different log
//	go run ./cmd/schedsim -chunk-size 2048 -prefill-share 0.5
package main

import (
	"flag"
	"fmt"
	"os"

	"schedsim/internal/report"
	"schedsim/internal/sched"
	"schedsim/internal/trace"
)

func main() {
	logPath := flag.String("log", "", "log to read (default: the embedded live serving log)")
	chunkSize := flag.Int("chunk-size", 4096, "tokens per cold-prefill chunk to look for")
	prefillShare := flag.Float64("prefill-share", 0.5, "prefill's share of contended GPU time under the time-balance policy")
	sweep := flag.Bool("sweep", true, "print the fixed-interval sensitivity table")
	decodePerReq := flag.Float64("decode-per-req", 0.0, "ASSUMED: fractional decode cost added per extra running request")
	interference := flag.Float64("prefill-interference", 0.0, "ASSUMED: fractional prefill slowdown from sharing the GPU with decode")
	flag.Parse()

	logName, text := "internal/trace/live_log.txt (embedded)", trace.EmbeddedLog
	if *logPath != "" {
		b, err := os.ReadFile(*logPath)
		if err != nil {
			fail("read log: %v", err)
		}
		logName, text = *logPath, string(b)
	}

	steps, err := trace.Parse(text)
	if err != nil {
		fail("parse log: %v", err)
	}
	metrics := trace.Summarize(steps, *chunkSize)
	if metrics.ColdStart < 0 {
		fail("no %d-token cold prefill stretch found in %s", *chunkSize, logName)
	}
	work := sched.WorkloadFromLog(steps, metrics, *chunkSize)

	params := work.Params(sched.PolicyTimeBalance)
	params.PrefillShare = *prefillShare
	params.DecodePerReqFraction = *decodePerReq
	params.PrefillInterference = *interference

	oldParams := work.Params(sched.PolicyPrefillPriority)
	oldParams.DecodePerReqFraction = *decodePerReq
	oldParams.PrefillInterference = *interference

	oldRes := sched.Simulate(work, oldParams, metrics.QueueAtCold)
	newRes := sched.Simulate(work, params, metrics.QueueAtCold)

	in := report.Input{
		Metrics:       metrics,
		Old:           oldRes,
		New:           newRes,
		ChunkSize:     *chunkSize,
		Params:        params,
		LogLines:      len(splitLines(text)),
		LogName:       logName,
		ColdFirstLine: steps[metrics.ColdStart].Line,
		ColdLastLine:  steps[metrics.ColdEnd-1].Line,
		Fidelities:    report.DefaultFidelities(metrics),
	}
	if work.RunningReqs != metrics.RunningDecode {
		in.Params.RunningReqs = work.RunningReqs
	}

	// The revised balancer at the queue the log shows when the cold prompt
	// starts, at its peak inside the window, and with a longer queue.
	var revisedRes sched.Result
	for _, q := range []struct {
		label   string
		pending int
	}{
		{"#queue-req at cold start", 1 + metrics.QueueAtCold},
		{"#queue-req peak in window", 1 + metrics.QueuePeakInCold},
		{"two more cold prompts queued", 3},
	} {
		p := work.Params(sched.PolicyQueueBalance)
		p.PendingPrefill = q.pending
		p.DecodePerReqFraction = *decodePerReq
		p.PrefillInterference = *interference
		r := sched.Simulate(work, p, metrics.QueueAtCold)
		if len(in.Revised) == 0 {
			revisedRes = r
		}
		in.Revised = append(in.Revised, report.RevisedRun{PendingPrefill: q.pending, Label: q.label, Result: r})
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
}

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
