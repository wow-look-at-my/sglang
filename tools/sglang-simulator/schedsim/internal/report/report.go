// Package report renders the parsed log facts and both models' results as a
// side-by-side before/after, and states which numbers are measured, which are
// derived, and which the log cannot supply at all.
package report

import (
	"fmt"
	"io"
	"strings"

	"schedsim/internal/sched"
	"schedsim/internal/trace"
)

// Source is how strongly the log supports a value.
type Source string

const (
	// Measured: read directly out of a log line.
	Measured Source = "measured"
	// Derived: computed from measured values by definition, not fitted.
	Derived Source = "derived"
	// Assumed: a parameter the log cannot supply.
	Assumed Source = "ASSUMED"
	// NotReproducible: the log does not contain the information at all.
	NotReproducible Source = "NOT IN LOG"
)

// Fidelity records where one number in the report came from.
type Fidelity struct {
	Item   string
	Source Source
	Detail string
}

// Input is everything the report needs.
type Input struct {
	Metrics trace.Metrics
	Old     sched.Result
	New     sched.Result

	ChunkSize  int
	Params     sched.Params
	Sweep      []SweepPoint
	LogLines   int
	LogName    string
	Fidelities []Fidelity

	// ColdFirstLine and ColdLastLine are the log lines of the cold prefill's
	// first and last chunk, so the report can point a reader at the evidence.
	ColdFirstLine int
	ColdLastLine  int

	// Revised holds the revised balancer's runs, one per pending-queue size.
	Revised []RevisedRun
}

// RevisedRun is one queue-balance run, the requests each chunk served and the
// queue length behind it.
type RevisedRun struct {
	ReqsPerChunk int
	Queue        int
	Label        string
	Result       sched.Result
}

// SweepPoint is one row of the decode-interval sensitivity table.
type SweepPoint struct {
	Interval          int
	DecodeSteps       int
	EffectiveGenTPS   float64
	PrefillFinishSecs float64
}

// Write renders the report.
func Write(w io.Writer, in Input) {
	m := in.Metrics
	fmt.Fprintf(w, "Scheduler simulation: prefill/decode contention on a cold long prompt\n")
	fmt.Fprintf(w, "=======================================================================\n\n")

	fmt.Fprintf(w, "INPUT LOG\n")
	fmt.Fprintf(w, "  %s, %d lines, %d scheduler steps (%d prefill, %d decode)\n",
		in.LogName, in.LogLines, m.Steps, m.PrefillSteps, m.DecodeSteps)
	fmt.Fprintf(w, "  cold prefill: %d chunks of %d tokens, log lines %d-%d\n",
		m.ColdChunks, in.ChunkSize, in.ColdFirstLine, in.ColdLastLine)
	fmt.Fprintf(w, "  the first cold chunk reports #pending-token %d\n", m.PendingAtCold)
	fmt.Fprintf(w, "  those chunks compute %d tokens; the rest of the prompt was still pending\n",
		m.ColdTokens)
	fmt.Fprintf(w, "  when the log ends, and #queue-req grew to %d inside the window\n", m.QueuePeakInCold)
	fmt.Fprintf(w, "  #running-req on the cold chunks: %d\n", m.RunningAtCold)
	fmt.Fprintf(w, "  decode stat lines inside the window: %d\n\n", m.DecodeInCold)

	fmt.Fprintf(w, "COST MODEL\n")
	fmt.Fprintf(w, "  a chunk's GPU seconds = its #new-token / its own input throughput, so the\n")
	fmt.Fprintf(w, "  prefill costs are the log's measurements inverted, not a fitted curve\n")
	fmt.Fprintf(w, "  one decode step = %.4f s producing %.2f tokens (measured medians)\n",
		in.Params.DecodeStepSeconds, in.Params.DecodeTokensPerStep)
	fmt.Fprintf(w, "  decode step rate before the collapse: %.1f steps/s; running requests: %d\n\n",
		m.NormalDecodeStepsPerSec(), in.Params.RunningReqs)

	fmt.Fprintf(w, "WHAT THE LOG SHOWS INSIDE THE WINDOW\n")
	fmt.Fprintf(w, "  the decode stat line is emitted every %d decode iterations, so the window\n",
		trace.DecodeLogInterval)
	fmt.Fprintf(w, "  shows no decode line at all. That bounds decode inside the window at %d\n",
		trace.MaxDecodeStepsInColdWindow())
	fmt.Fprintf(w, "  steps; a healthy schedule would have run about %.0f there\n",
		m.NormalDecodeStepsInColdWindow())
	fmt.Fprintf(w, "  (%.2f s of measured prefill at %.1f steps/s)\n", m.ColdSeconds, m.NormalDecodeStepsPerSec())
	fmt.Fprintf(w, "  the operator's ~8-20 tok/s band corresponds to a conversation getting\n")
	fmt.Fprintf(w, "  %.0f-%.0f%% of scheduling steps (rate x step duration / tokens per step),\n",
		100*m.DecodeDutyCycleForRate(8), 100*m.DecodeDutyCycleForRate(20))
	fmt.Fprintf(w, "  i.e. partial starvation, not the total starvation this window shows.\n")
	fmt.Fprintf(w, "  This fragment cannot resolve which: 0 decode lines is consistent with\n")
	fmt.Fprintf(w, "  anything up to %d steps, or %.1f tok/s, and both readings sit below the\n",
		trace.MaxDecodeStepsInColdWindow(), m.GenRateForSteps(trace.MaxDecodeStepsInColdWindow()))
	fmt.Fprintf(w, "  band. It is reported as a band the log cannot pin down, not back-filled.\n")
	fmt.Fprintf(w, "  measured prefill sag: steady %.2f -> bottom %.2f tok/s (peak %.2f)\n",
		m.ColdSteadyTP, m.ColdMinTP, m.ColdPeakTP)
	fmt.Fprintf(w, "  the operator quoted ~13,874 -> ~4,831 tok/s; those are this log's steady\n")
	fmt.Fprintf(w, "  start and its minimum, so the sag is reproduced from the log exactly\n")
	fmt.Fprintf(w, "  cold prefill cost as measured: %.2f s for %d tokens\n\n", m.ColdSeconds, m.ColdTokens)

	fmt.Fprintf(w, "BEFORE / AFTER\n")
	fmt.Fprintf(w, "  %-42s %14s %14s\n", "", "OLD", "NEW")
	rowInt(w, "decode steps during cold prefill", in.Old.DecodeSteps, in.New.DecodeSteps)
	rowF(w, "generation over the window (tok/s)", in.Old.EffectiveGenTPS, in.New.EffectiveGenTPS, 1)
	rowF(w, "longest gap with no decode (s)", in.Old.LongestDecodeGap, in.New.LongestDecodeGap, 2)
	rowF(w, "cold prefill finishes after (s)", in.Old.WindowSeconds, in.New.WindowSeconds, 2)
	rowF(w, "cold prefill as wall throughput (tok/s)", in.Old.PrefillWallTPS, in.New.PrefillWallTPS, 1)
	rowBool(w, "cold prefill completes (no livelock)", in.Old.PrefillCompleted, in.New.PrefillCompleted)
	if in.Old.DecodeSteps > 0 || in.New.DecodeSteps > 0 {
		rowInt(w, "tokens generated over the window", int(in.Old.GeneratedTokens), int(in.New.GeneratedTokens))
	}
	fmt.Fprintf(w, "\n")

	fmt.Fprintf(w, "WHAT THE OLD POLICY DOES\n")
	fmt.Fprintf(w, "  prefill runs whenever a chunk can be formed, decode only when it cannot.\n")
	fmt.Fprintf(w, "  A chunk is always formable while the cold prompt is outstanding, so decode\n")
	fmt.Fprintf(w, "  gets %d steps and the interrupted conversations hold their position for the\n", in.Old.DecodeSteps)
	fmt.Fprintf(w, "  whole %.1f s the prefill occupies the GPU. This is the collapse.\n\n", in.Old.WindowSeconds)

	fmt.Fprintf(w, "WHAT THE NEW POLICY DOES\n")
	fmt.Fprintf(w, "  measured prefill and decode seconds are balanced (prefill share %.2f), so\n",
		in.Params.PrefillShare)
	fmt.Fprintf(w, "  the GPU alternates classes while both have work and decode keeps running:\n")
	fmt.Fprintf(w, "  %d decode steps, %d tokens generated, longest silent gap %.2f s.\n",
		in.New.DecodeSteps, int(in.New.GeneratedTokens), in.New.LongestDecodeGap)
	fmt.Fprintf(w, "  The cold prefill still completes, in %.1f s instead of %.1f s (%.2fx): sharing\n",
		in.New.WindowSeconds, in.Old.WindowSeconds, in.New.WindowSeconds/in.Old.WindowSeconds)
	fmt.Fprintf(w, "  the GPU is the trade the balancer makes, and the share knob sets it.\n\n")

	if len(in.Revised) > 0 {
		fmt.Fprintf(w, "REVISED BALANCER (per-request share of admitted work)\n")
		fmt.Fprintf(w, "  each request a prefill batch serves and the decode batch get equal time,\n")
		fmt.Fprintf(w, "  so a chunk serving k requests gets k/(k+1); a lone chunk is the 50/50\n")
		fmt.Fprintf(w, "  split above. Requests queued behind the chunk cannot join it and do not count.\n")
		fmt.Fprintf(w, "  %-30s %6s %6s %14s %12s %14s\n", "", "reqs", "queue", "decode steps", "gen (tok/s)", "prefill (s)")
		for _, rv := range in.Revised {
			fmt.Fprintf(w, "  %-30s %6d %6d %14d %12.1f %14.1f\n", rv.Label, rv.ReqsPerChunk, rv.Queue,
				rv.Result.DecodeSteps, rv.Result.EffectiveGenTPS, rv.Result.WindowSeconds)
		}
		fmt.Fprintf(w, "\n")
	}

	if len(in.Sweep) > 0 {
		fmt.Fprintf(w, "FIXED-INTERVAL ALTERNATIVE (--prefill-decode-interval N)\n")
		fmt.Fprintf(w, "  %10s %14s %16s %18s\n", "N", "decode steps", "gen (tok/s)", "prefill done (s)")
		for _, p := range in.Sweep {
			fmt.Fprintf(w, "  %10d %14d %16.1f %18.1f\n",
				p.Interval, p.DecodeSteps, p.EffectiveGenTPS, p.PrefillFinishSecs)
		}
		fmt.Fprintf(w, "  a fixed N cannot adapt: the same N buys a different gen rate and a\n")
		fmt.Fprintf(w, "  different prefill wait as chunk cost changes.\n\n")
	}

	fmt.Fprintf(w, "WHERE EACH NUMBER COMES FROM\n")
	for _, f := range in.Fidelities {
		fmt.Fprintf(w, "  [%-12s] %s\n", f.Source, f.Item)
		fmt.Fprintf(w, "  %-14s %s\n", "", f.Detail)
	}
}

func rowInt(w io.Writer, label string, old, new int) {
	fmt.Fprintf(w, "  %-42s %14d %14d\n", label, old, new)
}

func rowF(w io.Writer, label string, old, new float64, prec int) {
	fmt.Fprintf(w, "  %-42s %14.*f %14.*f\n", label, prec, old, prec, new)
}

func rowBool(w io.Writer, label string, old, new bool) {
	fmt.Fprintf(w, "  %-42s %14s %14s\n", label, yn(old), yn(new))
}

func yn(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

// DefaultFidelities builds the provenance section for a run with the default
// knobs.
func DefaultFidelities(m trace.Metrics) []Fidelity {
	return []Fidelity{
		{"input throughput of each cold chunk (the sag)", Measured,
			fmt.Sprintf("the %d chunk lines report it; steady %.2f, bottom %.2f, peak %.2f tok/s, which are the log's own steady start and minimum the operator quoted as ~13,874 -> ~4,831",
				m.ColdChunks, m.ColdSteadyTP, m.ColdMinTP, m.ColdPeakTP)},
		{"#pending-token at the start of the prefill", Measured,
			fmt.Sprintf("%d tokens outstanding on the first cold chunk", m.PendingAtCold)},
		{"a chunk's GPU seconds", Derived,
			"tokens / its own reported throughput; inverts how the log defines input throughput"},
		{"one decode step's duration and tokens", Derived,
			fmt.Sprintf("tokens = measured median accept len %.2f; duration = accept len / that step's reported gen throughput",
				m.DecodeStepToks)},
		{"decode steps inside the cold-prefill window", Derived,
			fmt.Sprintf("the log has 0 decode stat lines there and emits one every %d decode iterations, so under the old policy decode is bounded at %d steps over the window; the old policy's own rule produces exactly 0 because a chunk is always formable",
				trace.DecodeLogInterval, trace.MaxDecodeStepsInColdWindow())},
		{"generation rate the interrupted conversation observed (OLD)", NotReproducible,
			fmt.Sprintf("the ~8-20 tok/s band maps to a decode duty cycle of %.0f-%.0f%% (rate x step duration / tokens per step): a conversation getting that fraction of steps. The log cannot pin it down, because 0 decode stat lines in the window is consistent with anything up to %d steps, or %.1f tok/s. The log's own lines are one conversation plus one cold prompt, so the concurrent figure came from a conversation it was not logging. This is stated as a band the log does not resolve, not back-filled; the old policy this model builds produces 0 tok/s (total starvation). Baseline here: %.1f tok/s median.",
				100*m.DecodeDutyCycleForRate(8), 100*m.DecodeDutyCycleForRate(20),
				trace.MaxDecodeStepsInColdWindow(), m.GenRateForSteps(trace.MaxDecodeStepsInColdWindow()),
				m.BaselineGenTP)},
		{"the log's own low gen lines (not used as a curve)", Measured,
			fmt.Sprintf("%d of %d decode lines report under 100 tok/s, %d of them within two lines of a prefill line. A line aggregates %d decode iterations, so one that caught a prefill chunk spreads too few tokens over too long a window. These are reporting artifacts of the log interval, which is why the model uses each decode line's own tokens/duration and not the line's raw rate as a load curve.",
				m.LowGenLines, m.DecodeSteps, m.LowGenNearPrefill, trace.DecodeLogInterval)},
		{"batch-size scaling of decode cost", Assumed,
			"the log only ever decodes one request, so cost per extra request is a knob (-decode-per-req), default 0 = flat in batch size"},
		{"extra prefill cost from sharing the GPU with decode", Assumed,
			"not measurable from a single-request log; knob -prefill-interference, default 0 = each class pays only its own measured cost"},
		{"prefill share of contended GPU time", Assumed,
			"the balancer's shipped rule is an even split (debt = prefill seconds - decode seconds), so 0.5 is its own value, not a fit to this log"},
	}
}

// Summary prints the one-line-per-model digest.
func Summary(w io.Writer, results ...sched.Result) {
	fmt.Fprintf(w, "SUMMARY\n")
	for _, r := range results {
		fmt.Fprintf(w, "  %s: %d decode steps, %.1f tok/s over %.1f s, prefill done=%v\n",
			strings.TrimSpace(r.Policy), r.DecodeSteps, r.EffectiveGenTPS, r.WindowSeconds, r.PrefillCompleted)
	}
}
