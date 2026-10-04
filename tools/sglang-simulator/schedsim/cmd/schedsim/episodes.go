package main

import (
	"fmt"
	"io"
	"sort"
	"strings"

	"schedsim/internal/sched"
	"schedsim/internal/trace"
)

// One incident is one data point.

// episode is one prefill stretch and what each policy does with it.
type episode struct {
	boot     int
	run      trace.ColdRun
	measured float64
	old, new sched.Result
	revised  sched.Result
}

// minEpisodeChunks keeps the table to prompts long enough to starve decode for whole seconds.
const minEpisodeChunks = 8

func collectEpisodes(boots []trace.Boot, chunkSize int, params func(w sched.Workload, p sched.Policy) sched.Params) []episode {
	var eps []episode
	for bi := range boots {
		b := &boots[bi]
		m := b.Summarize(chunkSize)
		if m.DecodeStepSecs <= 0 {
			continue
		}
		stalls := b.Stalls(0)
		for _, r := range b.PrefillStretches(chunkSize, minEpisodeChunks) {
			w := sched.WorkloadFromRun(b.Steps, r, m)
			queue := b.Steps[r.Start].QueueReq
			ep := episode{boot: bi, run: r}
			for _, st := range stalls {
				if st.Start < r.End && r.Start < st.End && st.Seconds > ep.measured {
					ep.measured = st.Seconds
				}
			}
			ep.old = sched.Simulate(w, params(w, sched.PolicyPrefillPriority), queue)
			ep.new = sched.Simulate(w, params(w, sched.PolicyTimeBalance), queue)
			rp := params(w, sched.PolicyQueueBalance)
			rp.MixedChunk = true
			ep.revised = sched.Simulate(w, rp, queue)
			eps = append(eps, ep)
		}
	}
	return eps
}

func printEpisodes(w io.Writer, boots []trace.Boot, eps []episode, chunkSize int) {
	fmt.Fprintf(w, "\n=== Every cold prompt in the log ===\n\n")
	fmt.Fprintf(w, "Each row is one stretch of consecutive prefill steps in one boot holding %d+ cold\n", minEpisodeChunks)
	fmt.Fprintf(w, "%d-token chunks: every prompt that arrived joined it, and no decode step ran until it\n", chunkSize)
	fmt.Fprintf(w, "ended. 'measured' is the stall the log's timestamps recorded over it (first prefill\n")
	fmt.Fprintf(w, "with requests running to the next decode line; blank when nothing was running or the\n")
	fmt.Fprintf(w, "log has no timestamps).\n")
	fmt.Fprintf(w, "OLD/NEW/REV are the prefill-priority, time-balance and queue-balance (mixed chunk)\n")
	fmt.Fprintf(w, "policies simulated over the run's own measured chunk costs: 'gap' is the longest\n")
	fmt.Fprintf(w, "stretch with no decode step, 'done' when the cold prefill finishes.\n\n")
	fmt.Fprintf(w, "  %-4s %-19s %6s %5s %7s %4s %3s %9s %8s %8s %8s %8s %8s %8s\n",
		"boot", "start", "steps", "cold", "ctx0", "run", "q", "measured", "OLD gap", "NEW gap", "REV gap", "OLD done", "NEW done", "REV done")
	var timestamped bool
	for _, ep := range eps {
		r := ep.run
		start := fmt.Sprintf("line %d", boots[ep.boot].Steps[r.Start].Line)
		if !r.StartAt.IsZero() {
			start = r.StartAt.Format("01-02T15:04:05")
			timestamped = true
		}
		meas := ""
		if ep.measured > 0 {
			meas = fmt.Sprintf("%.1f", ep.measured)
		}
		fmt.Fprintf(w, "  %-4d %-19s %6d %5d %7d %4d %3d %9s %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f\n",
			ep.boot, start, r.Chunks(), r.ColdChunks, r.CtxStart, r.RunningAtStart, r.QueuePeak, meas,
			ep.old.LongestDecodeGap, ep.new.LongestDecodeGap, ep.revised.LongestDecodeGap,
			ep.old.WindowSeconds, ep.new.WindowSeconds, ep.revised.WindowSeconds)
	}
	fmt.Fprintln(w)

	// Aggregates over the prompts that had something to starve.
	var n, measuredN, newOver5, revOver5 int
	var measuredSum, oldSum, newSum, revSum, oldDone, newDone, revDone, oldMax, newMax, revMax float64
	var ratios []float64
	for _, ep := range eps {
		if ep.run.RunningAtStart < 1 {
			continue
		}
		n++
		oldSum += ep.old.LongestDecodeGap
		newSum += ep.new.LongestDecodeGap
		revSum += ep.revised.LongestDecodeGap
		oldDone += ep.old.WindowSeconds
		newDone += ep.new.WindowSeconds
		revDone += ep.revised.WindowSeconds
		oldMax = maxF(oldMax, ep.old.LongestDecodeGap)
		newMax = maxF(newMax, ep.new.LongestDecodeGap)
		revMax = maxF(revMax, ep.revised.LongestDecodeGap)
		if ep.new.LongestDecodeGap > 5 {
			newOver5++
		}
		if ep.revised.LongestDecodeGap > 5 {
			revOver5++
		}
		if ep.measured > 5 {
			measuredN++
			measuredSum += ep.measured
			ratios = append(ratios, ep.old.LongestDecodeGap/ep.measured)
		}
	}
	fmt.Fprintf(w, "  %d prompt(s) started with requests decoding.\n", n)
	if timestamped {
		fmt.Fprintf(w, "  measured: %d of them stalled decode for over 5 s, %.0f s in all\n", measuredN, measuredSum)
		if len(ratios) > 0 {
			sort.Float64s(ratios)
			fmt.Fprintf(w, "  calibration: OLD's simulated gap over the measured stall is %.2f at the median\n", ratios[len(ratios)/2])
			fmt.Fprintf(w, "  (p10 %.2f, p90 %.2f); 1.00 means the model reproduces the stall the log recorded\n",
				ratios[len(ratios)/10], ratios[len(ratios)*9/10])
		}
	}
	fmt.Fprintf(w, "  %-34s %10s %10s %10s\n", "", "OLD", "NEW", "REV")
	fmt.Fprintf(w, "  %-34s %10.0f %10.0f %10.0f\n", "sum of longest gaps (s)", oldSum, newSum, revSum)
	fmt.Fprintf(w, "  %-34s %10.1f %10.1f %10.1f\n", "worst gap (s)", oldMax, newMax, revMax)
	fmt.Fprintf(w, "  %-34s %10d %10d %10d\n", "prompts with a gap over 5 s", n, newOver5, revOver5)
	fmt.Fprintf(w, "  %-34s %10.0f %10.0f %10.0f\n", "sum of cold prefill wall time (s)", oldDone, newDone, revDone)
	fmt.Fprintln(w, strings.Repeat("-", 100))
}

func maxF(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
