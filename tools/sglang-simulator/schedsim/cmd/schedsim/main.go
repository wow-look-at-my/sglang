// Command schedsim drives two scheduler policies over the same cold-prefill
// workload parsed from a live serving log, and prints what each does.
//
// Usage:
//
//	go run ./cmd/schedsim -log FILE       # a serving log, bare or production format
//	go run ./cmd/schedsim -log FILE -chunk-size 2048 -prefill-share 0.5
//	go run ./cmd/schedsim -log FILE -replay-boots 4 -replay-seconds 0
//
// A production log (timestamped lines, several boots) adds three sections:
// the boots it holds, every cold prompt with its measured stall beside the
// policies, and a replay of each stalled boot's own traffic through the
// engine. docs/replaying-a-serving-log.md describes them.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"schedsim/internal/report"
	"schedsim/internal/sched"
	"schedsim/internal/sim"
	"schedsim/internal/trace"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "schedsim: %v\n", err)
		os.Exit(1)
	}
}

// run is the program: it parses args, reads the log and writes every
// section of the report to w. main exits 1 on the error it returns.
func run(args []string, w io.Writer) error {
	fs := flag.NewFlagSet("schedsim", flag.ContinueOnError)
	fs.SetOutput(w)
	logPath := fs.String("log", "", "serving log to read, bare or production format (required)")
	chunkSize := fs.Int("chunk-size", 4096, "tokens per cold-prefill chunk to look for")
	prefillShare := fs.Float64("prefill-share", 0.5, "prefill's share of contended GPU time under the time-balance policy")
	sweep := fs.Bool("sweep", true, "print the fixed-interval sensitivity table")
	decodePerReq := fs.Float64("decode-per-req", 0.0, "ASSUMED: fractional decode cost added per extra running request")
	interference := fs.Float64("prefill-interference", 0.0, "ASSUMED: fractional prefill slowdown from sharing the GPU with decode")
	policy := registerPolicyFlags(fs)
	replay := registerReplayFlags(fs)
	if err := fs.Parse(args); err != nil {
		return err
	}

	if *logPath == "" {
		return fmt.Errorf("-log FILE is required: the serving log to calibrate and replay")
	}
	b, err := os.ReadFile(*logPath)
	if err != nil {
		return fmt.Errorf("read log: %w", err)
	}
	logName, text := *logPath, string(b)

	boots, err := trace.ParseBoots(text)
	if err != nil {
		return fmt.Errorf("parse log: %w", err)
	}
	// The single-window report models the longest cold prompt in the log,
	// taken within one boot so a restart never splices two prompts together.
	anchor := longestRunBoot(boots, *chunkSize)
	if anchor < 0 {
		return fmt.Errorf("no %d-token cold prefill stretch found in %s", *chunkSize, logName)
	}
	steps := boots[anchor].Steps
	metrics := boots[anchor].Summarize(*chunkSize)
	work := sched.WorkloadFromLog(steps, metrics, *chunkSize)
	var incident *report.Incident
	pageSize := PageSize
	if boots[anchor].ArgsKnown && boots[anchor].Args.PageSize > 0 {
		pageSize = boots[anchor].Args.PageSize
	}

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
		Fidelities:    report.DefaultFidelities(metrics, incident),
		Incident:      incident,
	}
	if work.RunningReqs != metrics.RunningDecode {
		in.Params.RunningReqs = work.RunningReqs
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

	report.Write(w, in)
	fmt.Fprintln(w)
	report.Summary(w, oldRes, newRes, revisedRes)

	if len(boots) > 1 || boots[0].Timestamped() {
		printBoots(w, boots, *chunkSize)
	}
	paramsFor := func(w sched.Workload, p sched.Policy) sched.Params {
		q := w.Params(p)
		q.PrefillShare = *prefillShare
		q.DecodePerReqFraction = *decodePerReq
		q.PrefillInterference = *interference
		return q
	}
	eps := collectEpisodes(boots, *chunkSize, paramsFor)
	if len(eps) > 1 {
		printEpisodes(w, boots, eps, *chunkSize)
	}

	cal, calRep := trace.CalibrateBoots(boots, *chunkSize, pageSize)
	fmt.Fprintf(w, "\nCalibration: %s\n", calRep)
	if replay.on && boots[anchor].Timestamped() {
		printReplay(w, boots, sim.NewCost(cal), *chunkSize, replay, policy.workers)
	}
	printPolicies(w, cal, steps, policy)
	return nil
}

// PageSize is the page_size to assume when the log carries no server_args
// line to read it from.
const PageSize = 64

// longestRunBoot is the boot holding the longest run of cold chunks, or -1.
func longestRunBoot(boots []trace.Boot, chunkSize int) int {
	best, bestLen := -1, 0
	for i := range boots {
		for _, r := range boots[i].ColdRunsAll(chunkSize) {
			if r.Chunks() > bestLen {
				best, bestLen = i, r.Chunks()
			}
		}
	}
	return best
}

// printBoots lists the process lifetimes the log holds, with what each one
// measured: its stalls and its serving-time kernel compiles.
func printBoots(w io.Writer, boots []trace.Boot, chunkSize int) {
	fmt.Fprintf(w, "\n=== Boots in the log ===\n\n")
	fmt.Fprintf(w, "  %-4s %-16s %-15s %-15s %-7s %6s %5s %8s %6s %8s %5s %7s\n",
		"boot", "worker", "start", "end", "ended", "steps", "cold", "stalls>5", "sum s", "longest", "jit", "jit s")
	for _, b := range boots {
		stalls := b.Stalls(5)
		var sum, longest, jit float64
		for _, s := range stalls {
			sum += s.Seconds
			longest = maxF(longest, s.Seconds)
		}
		for _, j := range b.JITCompiles {
			jit += j.Seconds
		}
		cold := len(b.PrefillStretches(chunkSize, minEpisodeChunks))
		fmt.Fprintf(w, "  %-4d %-16s %-15s %-15s %-7s %6d %5d %8d %6.0f %8.1f %5d %7.1f\n",
			b.Index, b.Worker, stamp(b.Start), stamp(b.End), b.EndedBy, len(b.Steps), cold,
			len(stalls), sum, longest, len(b.JITCompiles), jit)
	}
	fmt.Fprintf(w, "  'cold' counts prefill stretches of %d+ cold chunks; 'stalls>5' the windows where decode stopped\n", minEpisodeChunks)
	fmt.Fprintf(w, "  for over 5 s with requests running; 'jit' the kernels compiled after serving started.\n")
}

func stamp(t time.Time) string {
	if t.IsZero() {
		return "-"
	}
	return t.Format("01-02T15:04:05")
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
