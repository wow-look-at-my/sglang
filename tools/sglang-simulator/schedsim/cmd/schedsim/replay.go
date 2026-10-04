package main

import (
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"schedsim/internal/sim"
	"schedsim/internal/trace"
)

// The scenario suite is scripted traffic.

type replayFlags struct {
	on      bool
	seconds float64
	boots   string
	hostMul float64
}

func registerReplayFlags(fs *flag.FlagSet) *replayFlags {
	r := &replayFlags{}
	fs.BoolVar(&r.on, "replay", true, "replay each timestamped boot's own requests under the three policies")
	fs.Float64Var(&r.seconds, "replay-seconds", 3600, "seconds of each boot to replay from its first request (0 = all)")
	fs.StringVar(&r.boots, "replay-boots", "", "comma-separated boot indexes to replay (default: every boot that stalled)")
	fs.Float64Var(&r.hostMul, "replay-host-mul", 4, "host tier for the 'as the fork resolves' row, as a multiple of the device pool")
	return r
}

func printReplay(w io.Writer, boots []trace.Boot, cost sim.Cost, chunkSize int, rf *replayFlags, workers int) {
	fmt.Fprintf(w, "\n=== Replaying the log's own traffic ===\n\n")
	fmt.Fprintf(w, "Each boot's requests, at their logged arrivals, with the prompt sizes and prefix hits\n")
	fmt.Fprintf(w, "the log reported; a conversation's next turn is released at its logged arrival or when\n")
	fmt.Fprintf(w, "its previous reply finishes, whichever is later. Reply lengths come from the next turn's\n")
	fmt.Fprintf(w, "prefix hit. 'measured' is what the log itself recorded for the same window, which\n")
	fmt.Fprintf(w, "opens 30 min before the boot's first stall (or at its first request) and runs %.0f s.\n", rf.seconds)
	fmt.Fprintf(w, "Row 1 of each boot runs the deployment as launched (its max_running and host tier);\n")
	fmt.Fprintf(w, "row 2 runs it as the fork resolves by default (host tier %.0fx the device pool).\n", rf.hostMul)
	fmt.Fprintf(w, "OLD is the policy the log ran, so its column against 'measured' is the engine's fidelity\n")
	fmt.Fprintf(w, "check: its recomputes run high (the pool evicts whole conversations where the server\n")
	fmt.Fprintf(w, "evicts pages), so read recomputes and completed turns as directional and the stall\n")
	fmt.Fprintf(w, "columns as the result; docs/replaying-a-serving-log.md has the derivation.\n\n")

	want := map[int]bool{}
	for _, f := range strings.Split(rf.boots, ",") {
		var i int
		if _, err := fmt.Sscanf(f, "%d", &i); err == nil {
			want[i] = true
		}
	}
	var scs []sim.Scenario
	type anchor struct {
		stalls, over5 int
		stallSum      float64
		longest       float64
		completions   int
		cold          int
		returning     int
		turns         int
		measuredOuts  int
		convs         int
	}
	anchors := map[string]anchor{}
	for bi := range boots {
		b := &boots[bi]
		if !b.Timestamped() || len(b.Steps) == 0 {
			continue
		}
		if len(want) > 0 && !want[bi] {
			continue
		}
		if len(want) == 0 && len(b.Stalls(5)) == 0 {
			continue
		}
		// The window opens half an hour before the boot's first stall.
		skip := 0.0
		if st := b.Stalls(5); len(st) > 0 {
			var first time.Time
			for _, s := range b.Steps {
				if s.Kind == trace.Prefill {
					first = s.At
					break
				}
			}
			skip = maxF(0, st[0].At.Sub(first).Seconds()-1800)
		}
		l, err := sim.BuildLogReplay(b, chunkSize, skip, rf.seconds)
		if err != nil {
			fmt.Fprintf(w, "  boot %d: %v\n", bi, err)
			continue
		}
		a := anchor{turns: len(l.Turns), cold: l.ColdTurns(), returning: l.Returning, measuredOuts: l.MeasuredOuts(), convs: l.Convs}
		end := l.Start.Add(secondsDuration(l.Window))
		for _, st := range b.Stalls(0) {
			if st.At.Before(l.Start) || st.At.After(end) {
				continue
			}
			a.stalls++
			if st.Seconds > 5 {
				a.over5++
				a.stallSum += st.Seconds
			}
			a.longest = maxF(a.longest, st.Seconds)
		}
		for _, c := range b.Completions {
			if !c.Before(l.Start) && !c.After(end) {
				a.completions++
			}
		}
		hostMul := 0.0
		maxRunning := b.Args.MaxRunningRequests
		if b.Args.HierarchicalCache {
			hostMul = rf.hostMul
		}
		name := fmt.Sprintf("boot %d %s from %s, %.0f s, %d requests in %d conversations",
			bi, b.Worker, l.Start.Format("01-02T15:04:05"), l.Window, len(l.Turns), l.Convs)
		note := fmt.Sprintf("as launched: max_running %d, host tier %.0fx", maxRunning, hostMul)
		mk := func(name, note string, hm float64) sim.Scenario {
			return sim.Scenario{
				Name: name, Key: name, Note: note,
				Build:      func(int64, sim.Cost) sim.Workload { return l.Reset() },
				HardStop:   l.Window + 300,
				Window:     l.Window,
				MaxRunning: maxRunning,
				HostMul:    hm,
				Seeds:      []int64{1},
			}
		}
		scs = append(scs, mk(name, note, hostMul))
		anchors[name] = a
		name2 := name + " (fork defaults)"
		scs = append(scs, mk(name2, fmt.Sprintf("as the fork resolves: max_running %d, host tier %.0fx", maxRunning, rf.hostMul), rf.hostMul))
		anchors[name2] = a
	}
	if len(scs) == 0 {
		fmt.Fprintf(w, "  no timestamped boot with a stall to replay\n")
		return
	}
	rows := sim.RunSuite(scs, cost, sim.DefaultConfig, workers)
	for _, r := range rows {
		a := anchors[r.Scenario.Name]
		fmt.Fprintf(w, "%s\n%s\n", r.Scenario.Name, strings.Repeat("-", 100))
		fmt.Fprintf(w, "  %s\n", r.Scenario.Note)
		fmt.Fprintf(w, "  measured in this window: %d stalls over 5 s summing %.0f s, longest %.1f s; %d completions;\n",
			a.over5, a.stallSum, a.longest, a.completions)
		fmt.Fprintf(w, "  %d cold prompts of which %d were conversations returning after eviction; reply length known for %d of %d turns\n\n",
			a.cold, a.returning, a.measuredOuts, a.turns)
		fmt.Fprintf(w, "  %-32s %13s %13s %13s %13s\n", "Metric", "measured", "OLD", "PREV", "NEW")
		row := func(label, measured string, k sim.MetricKey) {
			fmt.Fprintf(w, "  %-32s %13s", label, measured)
			for i := range sim.Modes {
				fmt.Fprintf(w, " %13s", fmtMetric(k, r.Metrics[i].Value(k)))
			}
			fmt.Fprintln(w)
		}
		row("longest stall (s)", fmt.Sprintf("%.1f", a.longest), sim.MLongestStall)
		row("full-prefix recomputes", fmt.Sprintf("%d", a.returning), sim.MRecomputes)
		row("output tok/s over the window", "", sim.MThroughput)
		row("cold TTFT mean (s)", "", sim.MColdTTFT)
		row("stream decode tok/s in cold", "", sim.MStreamRate)
		row("ITL p99 (ms)", "", sim.MITLp99)
		fmt.Fprintf(w, "  %-32s %13d", "completed turns", a.completions)
		for i := range sim.Modes {
			fmt.Fprintf(w, " %13d", r.Metrics[i].CompletedTurns)
		}
		fmt.Fprintln(w)
		fmt.Fprintf(w, "  %-32s %13s", "cold prompts served/arrived", "")
		for i := range sim.Modes {
			fmt.Fprintf(w, " %13s", fmt.Sprintf("%d/%d", r.Metrics[i].ColdServed, r.Metrics[i].ColdArrived))
		}
		fmt.Fprintf(w, "\n\n")
	}
}

func secondsDuration(s float64) time.Duration { return time.Duration(s * float64(time.Second)) }

func fmtMetric(k sim.MetricKey, v float64) string {
	switch k {
	case sim.MRecomputes:
		return fmt.Sprintf("%.0f", v)
	case sim.MITLp99:
		return fmt.Sprintf("%.1f", 1e3*v)
	default:
		return fmt.Sprintf("%.1f", v)
	}
}
