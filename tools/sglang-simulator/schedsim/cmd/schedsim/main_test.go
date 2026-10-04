package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"schedsim/internal/sched"
	"schedsim/internal/trace"
	"schedsim/internal/trace/tracetest"
)

func stalledLog(t *testing.T) string {
	t.Helper()
	first := tracetest.Stalled("w1-first", 24).String()
	second := tracetest.Stalled("w2-second", 20)
	text := first + second.String()
	path := filepath.Join(t.TempDir(), "prod.log")
	require.NoError(t, os.WriteFile(path, []byte(text), 0o644))

	return path
}

// incidentLog writes the generated bare-format incident to a file.
func incidentLog(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "incident.log")
	require.NoError(t, os.WriteFile(path, []byte(tracetest.Incident(tracetest.DefaultIncident).String()), 0o644))

	return path
}

func TestRunOnABareLog(t *testing.T) {
	var out bytes.Buffer
	err := run([]string{"-log", incidentLog(t), "-sweep-policies=false", "-only", "A:"}, &out)
	require.Nil(t, err)

	s := out.String()
	for _, want := range []string{
		"Scheduler simulation", "SUMMARY", "Calibration: calibrated across 1 boot(s)",
		"Three-policy comparison", "A: logged episode",
	} {
		assert.Contains(t, s, want)

	}
	// A bare single-boot log has no boots table, no episodes and no replay.
	for _, absent := range []string{"Boots in the log", "Every cold prompt", "Replaying the log"} {
		assert.NotContains(t, s, absent)

	}
}

func TestRunOnAProductionLog(t *testing.T) {
	var out bytes.Buffer
	err := run([]string{"-log", stalledLog(t), "-sweep-policies=false", "-only", "none-match", "-replay-seconds", "0"}, &out)
	require.Nil(t, err)

	s := out.String()
	for _, want := range []string{
		"=== Boots in the log ===", "w1-first", "w2-second", "sigterm",
		"=== Every cold prompt in the log ===", "2 prompt(s) started with requests decoding",
		"measured: 2 of them stalled decode for over 5 s", "calibration: OLD's simulated gap",
		"Calibration: calibrated across 2 boot(s)", "1 holdout run(s)",
		"=== Replaying the log's own traffic ===", "boot 0 w1-first from", "(fork defaults)",
		"longest stall (s)", "completed turns",
	} {
		assert.Contains(t, s, want)

	}
}

func TestRunReplayFlagsSelectBoots(t *testing.T) {
	log := stalledLog(t)
	var out bytes.Buffer
	require.NoError(t, run([]string{"-log", log, "-sweep-policies=false", "-only", "x", "-replay-boots", "1,junk", "-workers", "1"}, &out))

	s := out.String()
	assert.False(t, !strings.Contains(s, "boot 1 w2-second") || strings.Contains(s, "boot 0 w1-first from"))

	out.Reset()
	require.NoError(t, run([]string{"-log", log, "-sweep-policies=false", "-only", "x", "-replay=false"}, &out))

	assert.NotContains(t, out.String(), "Replaying the log")

}

func TestRunErrors(t *testing.T) {
	var out bytes.Buffer
	err := run(nil, &out)
	assert.False(t, err == nil || !strings.Contains(err.Error(), "-log FILE is required"))

	err := run([]string{"-log", filepath.Join(t.TempDir(), "missing")}, &out)
	assert.False(t, err == nil || !strings.Contains(err.Error(), "read log"))

	empty := filepath.Join(t.TempDir(), "empty.log")
	os.WriteFile(empty, []byte("nothing here\n"), 0o644)
	err := run([]string{"-log", empty}, &out)
	assert.False(t, err == nil || !strings.Contains(err.Error(), "parse log"))

	err := run([]string{"-log", incidentLog(t), "-chunk-size", "12345"}, &out)
	assert.False(t, err == nil || !strings.Contains(err.Error(), "no 12345-token cold prefill stretch"))

	err := run([]string{"-no-such-flag"}, &out)
	assert.NotNil(t, err)

}

func TestRunSweepsWithTheSeedList(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the whole scenario suite")
	}
	var out bytes.Buffer
	require.NoError(t, run([]string{"-log", incidentLog(t), "-seeds", "1", "-only", "A:", "-a-seed", "3"}, &out))

	s := out.String()
	assert.False(t, !strings.Contains(s, "Cells where the contract does not hold") || !strings.Contains(s, "interval"))

}

func TestHelpersOnBoots(t *testing.T) {
	boots, err := trace.ParseBoots(tracetest.Stalled("w", 8).String())
	require.Nil(t, err)

	got := longestRunBoot(boots, 4096)
	assert.Equal(t, 0, got)

	got := longestRunBoot(boots, 99)
	assert.Equal(t, -1, got)

	assert.False(t, stamp(boots[0].Start) != "09-27T00:00:00" || stamp(trace.Boot{}.Start) != "-")

	got := splitLines("a\nb\n")
	assert.False(t, len(got) != 2 || got[1] != "b")

	got := splitLines("a\nb")
	assert.False(t, len(got) != 2 || got[1] != "b")

	assert.False(t, maxF(1, 2) != 2 || maxF(3, 2) != 3)

	got := seedList(3)
	assert.False(t, len(got) != 3 || got[2] != 3)

	assert.False(t, fmtMetric(0, 1.26) != "1.3" || fmtMetric(sim_MRecomputes(), 2.6) != "3" || fmtMetric(sim_MITLp99(), 0.0123) != "12.3")

	assert.Equal(t, 1.5, secondsDuration(1.5).Seconds())

	var out bytes.Buffer
	printBoots(&out, boots, 4096)
	assert.False(t, !strings.Contains(out.String(), "sigterm") || !strings.Contains(out.String(), "1 "))

	// Episodes over a bare log have no timestamps: the measured column is blank and the calibration line is not printed.
	bare, _ := trace.ParseBoots(tracetest.Incident(tracetest.DefaultIncident).String())
	params := func(w sched.Workload, p sched.Policy) sched.Params { return w.Params(p) }
	eps := collectEpisodes(bare, 4096, params)
	require.NotEqual(t, 0, len(eps))

	out.Reset()
	printEpisodes(&out, bare, eps, 4096)
	s := out.String()
	assert.False(t, !strings.Contains(s, "line ") || strings.Contains(s, "calibration:"))

	// A boot with no decode line has no step time and yields no episodes.
	noDecode, _ := trace.ParseBoots(tracetest.New("w", tracetest.Start).ColdPrompt(8*4096, 4096, 2, 0).String())
	eps = collectEpisodes(noDecode, 4096, params)
	assert.Equal(t, 0, len(eps))

}

func TestPrintReplayWithNothingToReplay(t *testing.T) {
	bare, _ := trace.ParseBoots(tracetest.Incident(tracetest.DefaultIncident).String())
	cal, _ := trace.CalibrateBoots(bare, 4096, 64)
	var out bytes.Buffer
	rf := &replayFlags{on: true, seconds: 60, hostMul: 4}
	printReplay(&out, bare, simCost(cal), 4096, rf, 1)
	assert.Contains(t, out.String(), "no timestamped boot with a stall to replay")

	// A timestamped boot whose window holds no request reports the error.
	one, _ := trace.ParseBoots(tracetest.New("w", tracetest.Start).ServerArgs(tracetest.DefaultArgs).Steady(3, 1, 1000).String())
	out.Reset()
	rf.boots = "0"
	printReplay(&out, one, simCost(cal), 4096, rf, 1)
	assert.Contains(t, out.String(), "boot 0: boot 0 has no prefill step")

}
