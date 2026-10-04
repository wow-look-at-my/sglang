package trace

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"schedsim/internal/trace/tracetest"
)

func TestLinePrefixToleratesABadTimestamp(t *testing.T) {
	at, worker := linePrefix("2026-13-45Tnot-a-time w1 INFO TP0] x")
	assert.False(t, !at.IsZero() || worker != "w1")

	at, worker = linePrefix("TP0] bare")
	assert.False(t, !at.IsZero() || worker != "")

	// A batch line with an unparsable timestamp still parses, without At.
	body := tracetest.PrefillBody(tracetest.Prefill{NewTokens: 100})
	s, ok, err := parseStep("2026-13-45Tnot-a-time w1 INFO "+body, 1)
	assert.False(t, err != nil || !ok || !s.At.IsZero() || s.Worker != "w1")

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
	require.Nil(t, err)

	require.Equal(t, 2, len(boots))

	wa, wb := boots[0], boots[1]
	assert.False(t, wa.Worker != "wa" || wa.EndedBy != "crash" || len(wa.Completions) != 1 || len(wa.JITCompiles) != 1 || wa.JITCompiles[0].Seconds != 0.5)

	assert.False(t, wb.Worker != "wb" || wb.EndedBy != "sigterm" || len(wb.Steps) != 3 || wb.Index != 1)

	assert.True(t, wb.End.Equal(time.Date(2026, 9, 27, 0, 3, 0, 0, time.UTC)))

	assert.Equal(t, 3, wb.Summarize(4096).DecodeSteps)

}

func TestParseBootsWithoutServerArgs(t *testing.T) {
	// A worker whose first line is a step opens a boot with ArgsKnown false;
	// a signal-only worker opens nothing that survives.
	text := "2026-09-27T00:00:00Z wx INFO " + tracetest.DecodeBody(tracetest.Decode{FullTokens: 1}) + "\n" +
		"2026-09-27T00:00:01Z wy INFO SIGTERM received\n"
	boots, err := ParseBoots(text)
	require.False(t, err != nil || len(boots) != 1 || boots[0].ArgsKnown || boots[0].Worker != "wx")

	_, err = ParseBoots("2026-09-27T00:00:01Z wy INFO SIGTERM received\n")
	assert.NotNil(t, err)

	_, err = ParseBoots(strings.Replace(prodPrefill, "#new-token: 4096", "#new-token: many", 1))
	assert.NotNil(t, err)

	// server_args with no steps at all is still a boot.
	only := tracetest.New("wz", tracetest.Start).ServerArgs(tracetest.DefaultArgs).String()
	boots, err = ParseBoots(only)
	assert.False(t, err != nil || len(boots) != 1 || len(boots[0].Steps) != 0 || !boots[0].ArgsKnown || boots[0].Timestamped())

	var b Boot
	b.touch(time.Time{})
	assert.True(t, b.End.IsZero())

}

func TestColdRunsAndStretchesEdges(t *testing.T) {
	l := tracetest.New("w", tracetest.Start)
	// A prompt whose first chunk reused a prefix: the cold chunks after it extend from prefix plus chunk.
	l.Prefill(tracetest.Prefill{NewTokens: 4096, Hit: 8192, Pending: 20000})
	l.ColdPrompt(3*4096, 4096, 0, 0)
	boots, _ := ParseBoots(l.String())
	b := boots[0]
	runs := b.ColdRunsAll(4096)
	assert.False(t, len(runs) != 1 || runs[0].CtxStart != 8192+4096 || runs[0].Chunks() != 3 || !runs[0].EndAt.Equal(b.Steps[3].At))

	st := b.PrefillStretches(4096, 1)
	assert.False(t, len(st) != 1 || st[0].Start != 0 || st[0].RunningAtStart != 0 || st[0].ColdChunks != 3 || st[0].CtxStart != 8192+4096 || st[0].End != 4)

	got := b.PrefillStretches(4096, 4)
	assert.Equal(t, 0, len(got))

	// A stretch that starts before anyone is running trims to the first step with a request, part way into a cold run.
	l = tracetest.New("w", tracetest.Start).Steady(1, 1, 10)
	l.ColdPrompt(2*4096, 4096, 0, 0)
	l.ColdPrompt(3*4096, 4096, 2, 0).Steady(1, 3, 10)
	boots, _ = ParseBoots(l.String())
	b = boots[0]
	st = b.PrefillStretches(4096, 1)
	assert.False(t, len(st) != 1 || st[0].Start != 3 || st[0].RunningAtStart != 2 || st[0].ColdChunks != 3 || st[0].CtxStart != 2*4096)

	// A stretch holding cold runs (a short request between them) takes its context from the first run it overlaps.
	l = tracetest.New("w", tracetest.Start).Steady(1, 1, 10)
	l.ColdPrompt(2*4096, 4096, 1, 0)
	l.Prefill(tracetest.Prefill{NewTokens: 300, Hit: 1000, Running: 2})
	l.ColdPrompt(2*4096, 4096, 2, 0).Steady(1, 2, 10)
	boots, _ = ParseBoots(l.String())
	st = boots[0].PrefillStretches(4096, 1)
	assert.False(t, len(st) != 1 || st[0].Start != 1 || st[0].CtxStart != 0 || st[0].ColdChunks != 4 || st[0].Tokens != 4*4096+300)

	assert.False(t, st[0].Indexes()[0] != 1 || len(st[0].Indexes()) != 5)

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
	require.Equal(t, 1, len(st))

	s := st[0]
	assert.False(t, s.Start != 1 || s.End != 3 || s.PrefillTokens != 4596 || s.ColdTokens != 4096 || s.MaxHit != 2000 || s.QueuePeak != 3 || s.RunningPeak != 3 || s.Seconds <= 0)

	got := boots[0].Stalls(1000)
	assert.Equal(t, 0, len(got))

}

func TestCalibrateBootsWithoutARun(t *testing.T) {
	boots, _ := ParseBoots(tracetest.New("w", tracetest.Start).Steady(3, 1, 140000).String())
	c, rep := CalibrateBoots(boots, 4096, 64)
	assert.False(t, c.Prefill.Base != PrefillBaseSeconds || c.ChunkSize != 4096 || c.Page != 64 || rep.FitBoot != -1)

	s := rep.String()
	assert.False(t, !strings.Contains(s, "no holdout run") || !strings.Contains(s, "pool per boot") || strings.Contains(s, "wall clock"))

	rmse, maxErr := fitErrorCtx(PrefillCost{}, nil, []int{0}, 0, 4096)
	assert.False(t, rmse != 0 || maxErr != 0)

	// Boots each with a long prompt: one fits, the other holds out.
	two := tracetest.Stalled("a", 10).String() + tracetest.Stalled("b", 9).String()
	boots, _ = ParseBoots(two)
	c, rep = CalibrateBoots(boots, 4096, 64)
	assert.False(t, rep.FitBoot != 0 || rep.FitChunks != 10 || len(rep.Holdouts) != 1 || rep.Holdouts[0].Boot != 1 || c.Prefill.HoldoutSamples != 1)

	s := rep.String()
	assert.False(t, !strings.Contains(s, "1 holdout run(s)") || !strings.Contains(s, "wall clock over"))

}

func TestMetricsGuardsAndStepHelpers(t *testing.T) {
	var m Metrics
	assert.False(t, m.NormalDecodeStepsPerSec() != 0 || m.StepsForGenRate(10) != 0 || m.GenRateForSteps(3) != 0 || m.DecodeDutyCycleForRate(5) != 0)

	assert.False(t, Decode.String() != "decode" || Prefill.String() != "prefill")

	s := Step{NewTokens: 200, Throughput: 100}
	assert.Equal(t, float64(2), s.Seconds())

	steps := []Step{{Kind: Decode}, {Kind: Decode}, {Kind: Prefill}}
	assert.False(t, nearPrefill(steps, 0, 1) || !nearPrefill(steps, 2, 5) || !nearPrefill(steps, 1, 1))

	_, err := Parse(strings.Replace(prodDecode, "accept len: 2.77", "accept len: x", 1))
	assert.NotNil(t, err)

	_, err := Parse(strings.Replace(prodDecode, "gen throughput (token/s): 8.44", "gen throughput (token/s): 0", 1))
	assert.NotNil(t, err)

	v, ok, _ := intField("x: 5", "x")
	assert.False(t, v != 5 || !ok)

	v, _ := floatField("y: 2.5", "y")
	assert.Equal(t, 2.5, v)

	assert.False(t, median([]float64{3, 1, 2, 4}) != 2.5 || minOf(nil) != 0)

}
