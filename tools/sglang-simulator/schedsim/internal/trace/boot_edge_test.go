package trace

import (
	"strings"
	"testing"
	"time"

	"schedsim/internal/trace/tracetest"
)

func TestLinePrefixToleratesABadTimestamp(t *testing.T) {
	at, worker := linePrefix("2026-13-45Tnot-a-time w1 INFO TP0] x")
	if !at.IsZero() || worker != "w1" {
		t.Errorf("bad timestamp gave %v %q", at, worker)
	}
	if at, worker := linePrefix("TP0] bare"); !at.IsZero() || worker != "" {
		t.Errorf("bare line gave %v %q", at, worker)
	}
	// A batch line with an unparsable timestamp still parses, without At.
	body := tracetest.PrefillBody(tracetest.Prefill{NewTokens: 100})
	s, ok, err := parseStep("2026-13-45Tnot-a-time w1 INFO "+body, 1)
	if err != nil || !ok || !s.At.IsZero() || s.Worker != "w1" {
		t.Errorf("step = %+v ok=%v err=%v", s, ok, err)
	}
}

func TestParseBootsSplitsWorkersAndEndings(t *testing.T) {
	a := tracetest.New("wa", tracetest.Start).ServerArgs(tracetest.DefaultArgs).Steady(2, 1, 100).Completion().JIT("k", 0.5).Crash()
	b := tracetest.New("wb", tracetest.Start.Add(time.Minute)).ServerArgs(tracetest.DefaultArgs).Steady(2, 1, 100)
	// wb's lines interleave with wa's; a line from an unknown worker before
	// any boot is dropped, and a stray bare line joins the current boot.
	text := "2026-09-27T00:00:00Z stray INFO SIGTERM received\n" + a.String() + b.String() +
		tracetest.DecodeBody(tracetest.Decode{Running: 1, FullTokens: 5}) + "\n" +
		"2026-09-27T00:02:00Z wa INFO Scheduler hit an exception\n" +
		"2026-09-27T00:03:00Z wb INFO SIGTERM received\n"
	boots, err := ParseBoots(text)
	if err != nil {
		t.Fatal(err)
	}
	if len(boots) != 2 {
		t.Fatalf("boots = %d", len(boots))
	}
	wa, wb := boots[0], boots[1]
	if wa.Worker != "wa" || wa.EndedBy != "crash" || len(wa.Completions) != 1 || len(wa.JITCompiles) != 1 || wa.JITCompiles[0].Seconds != 0.5 {
		t.Errorf("wa = %+v", wa)
	}
	if wb.Worker != "wb" || wb.EndedBy != "sigterm" || len(wb.Steps) != 3 || wb.Index != 1 {
		t.Errorf("wb = worker %s ended %s steps %d index %d", wb.Worker, wb.EndedBy, len(wb.Steps), wb.Index)
	}
	if !wb.End.Equal(time.Date(2026, 9, 27, 0, 3, 0, 0, time.UTC)) {
		t.Errorf("wb end %v", wb.End)
	}
	if wb.Summarize(4096).DecodeSteps != 3 {
		t.Error("Summarize did not use the boot's steps")
	}
}

func TestParseBootsWithoutServerArgs(t *testing.T) {
	// A worker whose first line is a step opens a boot with ArgsKnown false;
	// a signal-only worker opens nothing that survives.
	text := "2026-09-27T00:00:00Z wx INFO " + tracetest.DecodeBody(tracetest.Decode{FullTokens: 1}) + "\n" +
		"2026-09-27T00:00:01Z wy INFO SIGTERM received\n"
	boots, err := ParseBoots(text)
	if err != nil || len(boots) != 1 || boots[0].ArgsKnown || boots[0].Worker != "wx" {
		t.Fatalf("boots = %+v err %v", boots, err)
	}
	if _, err := ParseBoots("2026-09-27T00:00:01Z wy INFO SIGTERM received\n"); err == nil {
		t.Error("a log with no steps parsed")
	}
	if _, err := ParseBoots(strings.Replace(prodPrefill, "#new-token: 4096", "#new-token: many", 1)); err == nil {
		t.Error("a malformed field parsed")
	}
	// server_args with no steps at all is still a boot.
	only := tracetest.New("wz", tracetest.Start).ServerArgs(tracetest.DefaultArgs).String()
	boots, err = ParseBoots(only)
	if err != nil || len(boots) != 1 || len(boots[0].Steps) != 0 || !boots[0].ArgsKnown || boots[0].Timestamped() {
		t.Errorf("args-only boot = %+v err %v", boots, err)
	}
	var b Boot
	b.touch(time.Time{})
	if !b.End.IsZero() {
		t.Error("touch with a zero time moved End")
	}
}

func TestColdRunsAndStretchesEdges(t *testing.T) {
	l := tracetest.New("w", tracetest.Start)
	// A prompt whose first chunk reused a prefix: the cold chunks after it extend from prefix plus chunk.
	l.Prefill(tracetest.Prefill{NewTokens: 4096, Hit: 8192, Pending: 20000})
	l.ColdPrompt(3*4096, 4096, 0, 0)
	boots, _ := ParseBoots(l.String())
	b := boots[0]
	runs := b.ColdRunsAll(4096)
	if len(runs) != 1 || runs[0].CtxStart != 8192+4096 || runs[0].Chunks() != 3 || !runs[0].EndAt.Equal(b.Steps[3].At) {
		t.Errorf("runs = %+v", runs)
	}
	st := b.PrefillStretches(4096, 1)
	if len(st) != 1 || st[0].Start != 0 || st[0].RunningAtStart != 0 || st[0].ColdChunks != 3 || st[0].CtxStart != 8192+4096 || st[0].End != 4 {
		t.Errorf("stretches = %+v", st)
	}
	if got := b.PrefillStretches(4096, 4); len(got) != 0 {
		t.Errorf("minCold 4 kept %+v", got)
	}
	// A stretch that starts before anyone is running trims to the first step with a request, part way into a cold run.
	l = tracetest.New("w", tracetest.Start).Steady(1, 1, 10)
	l.ColdPrompt(2*4096, 4096, 0, 0)
	l.ColdPrompt(3*4096, 4096, 2, 0).Steady(1, 3, 10)
	boots, _ = ParseBoots(l.String())
	b = boots[0]
	st = b.PrefillStretches(4096, 1)
	if len(st) != 1 || st[0].Start != 3 || st[0].RunningAtStart != 2 || st[0].ColdChunks != 3 || st[0].CtxStart != 2*4096 {
		t.Errorf("trimmed stretch = %+v", st)
	}
	// A stretch holding cold runs (a short request between them) takes its context from the first run it overlaps.
	l = tracetest.New("w", tracetest.Start).Steady(1, 1, 10)
	l.ColdPrompt(2*4096, 4096, 1, 0)
	l.Prefill(tracetest.Prefill{NewTokens: 300, Hit: 1000, Running: 2})
	l.ColdPrompt(2*4096, 4096, 2, 0).Steady(1, 2, 10)
	boots, _ = ParseBoots(l.String())
	st = boots[0].PrefillStretches(4096, 1)
	if len(st) != 1 || st[0].Start != 1 || st[0].CtxStart != 0 || st[0].ColdChunks != 4 || st[0].Tokens != 4*4096+300 {
		t.Errorf("stretch over two runs = %+v", st)
	}
	if st[0].Indexes()[0] != 1 || len(st[0].Indexes()) != 5 {
		t.Errorf("indexes = %v", st[0].Indexes())
	}
}

func TestStallsEdges(t *testing.T) {
	l := tracetest.New("w", tracetest.Start).Steady(1, 2, 10)
	l.Prefill(tracetest.Prefill{NewTokens: 4096, Running: 2, Pending: 4096, Queue: 3})
	l.Prefill(tracetest.Prefill{NewTokens: 500, Hit: 2000, Running: 3})
	l.Steady(1, 4, 10)
	// A bare line in a timestamped boot has no clock and starts no stall.
	l.Raw(tracetest.PrefillBody(tracetest.Prefill{NewTokens: 4096, Running: 2}))
	l.Steady(1, 2, 10)
	l.Prefill(tracetest.Prefill{NewTokens: 4096, Running: 2})
	boots, _ := ParseBoots(l.String())
	st := boots[0].Stalls(0)
	if len(st) != 1 {
		t.Fatalf("stalls = %+v", st)
	}
	s := st[0]
	if s.Start != 1 || s.End != 3 || s.PrefillTokens != 4596 || s.ColdTokens != 4096 || s.MaxHit != 2000 || s.QueuePeak != 3 || s.RunningPeak != 3 || s.Seconds <= 0 {
		t.Errorf("stall = %+v", s)
	}
	if got := boots[0].Stalls(1000); len(got) != 0 {
		t.Errorf("minSeconds filter kept %+v", got)
	}
}

func TestCalibrateBootsWithoutARun(t *testing.T) {
	boots, _ := ParseBoots(tracetest.New("w", tracetest.Start).Steady(3, 1, 140000).String())
	c, rep := CalibrateBoots(boots, 4096, 64)
	if c.Prefill.Base != PrefillBaseSeconds || c.ChunkSize != 4096 || c.Page != 64 || rep.FitBoot != -1 {
		t.Errorf("no-run calibration = %s, report %+v", c, rep)
	}
	if s := rep.String(); !strings.Contains(s, "no holdout run") || !strings.Contains(s, "pool per boot") || strings.Contains(s, "wall clock") {
		t.Errorf("report string = %s", s)
	}
	if rmse, maxErr := fitErrorCtx(PrefillCost{}, nil, []int{0}, 0, 4096); rmse != 0 || maxErr != 0 {
		t.Error("fitErrorCtx over one chunk is not zero")
	}
	// Boots each with a long prompt: one fits, the other holds out.
	two := tracetest.Stalled("a", 10).String() + tracetest.Stalled("b", 9).String()
	boots, _ = ParseBoots(two)
	c, rep = CalibrateBoots(boots, 4096, 64)
	if rep.FitBoot != 0 || rep.FitChunks != 10 || len(rep.Holdouts) != 1 || rep.Holdouts[0].Boot != 1 || c.Prefill.HoldoutSamples != 1 {
		t.Errorf("fit/holdout = %+v", rep)
	}
	if s := rep.String(); !strings.Contains(s, "1 holdout run(s)") || !strings.Contains(s, "wall clock over") {
		t.Errorf("report string = %s", s)
	}
}

func TestMetricsGuardsAndStepHelpers(t *testing.T) {
	var m Metrics
	if m.NormalDecodeStepsPerSec() != 0 || m.StepsForGenRate(10) != 0 || m.GenRateForSteps(3) != 0 || m.DecodeDutyCycleForRate(5) != 0 {
		t.Error("zero metrics did not guard")
	}
	if Decode.String() != "decode" || Prefill.String() != "prefill" {
		t.Error("Kind strings")
	}
	s := Step{NewTokens: 200, Throughput: 100}
	if s.Seconds() != 2 {
		t.Error("Seconds")
	}
	steps := []Step{{Kind: Decode}, {Kind: Decode}, {Kind: Prefill}}
	if nearPrefill(steps, 0, 1) || !nearPrefill(steps, 2, 5) || !nearPrefill(steps, 1, 1) {
		t.Error("nearPrefill bounds")
	}
	if _, err := Parse(strings.Replace(prodDecode, "accept len: 2.77", "accept len: x", 1)); err == nil {
		t.Error("bad float parsed")
	}
	if _, err := Parse(strings.Replace(prodDecode, "gen throughput (token/s): 8.44", "gen throughput (token/s): 0", 1)); err == nil {
		t.Error("zero rate parsed")
	}
	if v, ok, _ := intField("x: 5", "x"); v != 5 || !ok {
		t.Error("intField at end of line")
	}
	if v, _ := floatField("y: 2.5", "y"); v != 2.5 {
		t.Error("floatField at end of line")
	}
	if median([]float64{3, 1, 2, 4}) != 2.5 || minOf(nil) != 0 {
		t.Error("median/minOf")
	}
}
