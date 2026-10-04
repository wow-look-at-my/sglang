package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

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
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// incidentLog writes the generated bare-format incident to a file.
func incidentLog(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "incident.log")
	if err := os.WriteFile(path, []byte(tracetest.Incident(tracetest.DefaultIncident).String()), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestRunOnABareLog(t *testing.T) {
	var out bytes.Buffer
	err := run([]string{"-log", incidentLog(t), "-sweep-policies=false", "-only", "A:"}, &out)
	if err != nil {
		t.Fatal(err)
	}
	s := out.String()
	for _, want := range []string{
		"Scheduler simulation", "SUMMARY", "Calibration: calibrated across 1 boot(s)",
		"Three-policy comparison", "A: logged episode",
	} {
		if !strings.Contains(s, want) {
			t.Errorf("report lacks %q", want)
		}
	}
	// A bare single-boot log has no boots table, no episodes and no replay.
	for _, absent := range []string{"Boots in the log", "Every cold prompt", "Replaying the log"} {
		if strings.Contains(s, absent) {
			t.Errorf("bare log printed %q", absent)
		}
	}
}

func TestRunOnAProductionLog(t *testing.T) {
	var out bytes.Buffer
	err := run([]string{"-log", stalledLog(t), "-sweep-policies=false", "-only", "none-match", "-replay-seconds", "0"}, &out)
	if err != nil {
		t.Fatal(err)
	}
	s := out.String()
	for _, want := range []string{
		"=== Boots in the log ===", "w1-first", "w2-second", "sigterm",
		"=== Every cold prompt in the log ===", "2 prompt(s) started with requests decoding",
		"measured: 2 of them stalled decode for over 5 s", "calibration: OLD's simulated gap",
		"Calibration: calibrated across 2 boot(s)", "1 holdout run(s)",
		"=== Replaying the log's own traffic ===", "boot 0 w1-first from", "(fork defaults)",
		"longest stall (s)", "completed turns",
	} {
		if !strings.Contains(s, want) {
			t.Errorf("report lacks %q", want)
		}
	}
}

func TestRunReplayFlagsSelectBoots(t *testing.T) {
	log := stalledLog(t)
	var out bytes.Buffer
	if err := run([]string{"-log", log, "-sweep-policies=false", "-only", "x", "-replay-boots", "1,junk", "-workers", "1"}, &out); err != nil {
		t.Fatal(err)
	}
	if s := out.String(); !strings.Contains(s, "boot 1 w2-second") || strings.Contains(s, "boot 0 w1-first from") {
		t.Errorf("replay-boots 1 replayed the wrong boot:\n%s", s)
	}
	out.Reset()
	if err := run([]string{"-log", log, "-sweep-policies=false", "-only", "x", "-replay=false"}, &out); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out.String(), "Replaying the log") {
		t.Error("-replay=false still replayed")
	}
}

func TestRunErrors(t *testing.T) {
	var out bytes.Buffer
	if err := run(nil, &out); err == nil || !strings.Contains(err.Error(), "-log FILE is required") {
		t.Errorf("no log: %v", err)
	}
	if err := run([]string{"-log", filepath.Join(t.TempDir(), "missing")}, &out); err == nil || !strings.Contains(err.Error(), "read log") {
		t.Errorf("missing log: %v", err)
	}
	empty := filepath.Join(t.TempDir(), "empty.log")
	os.WriteFile(empty, []byte("nothing here\n"), 0o644)
	if err := run([]string{"-log", empty}, &out); err == nil || !strings.Contains(err.Error(), "parse log") {
		t.Errorf("empty log: %v", err)
	}
	if err := run([]string{"-log", incidentLog(t), "-chunk-size", "12345"}, &out); err == nil || !strings.Contains(err.Error(), "no 12345-token cold prefill stretch") {
		t.Errorf("no stretch: %v", err)
	}
	if err := run([]string{"-no-such-flag"}, &out); err == nil {
		t.Error("bad flag accepted")
	}
}

func TestRunSweepsWithTheSeedList(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the whole scenario suite")
	}
	var out bytes.Buffer
	if err := run([]string{"-log", incidentLog(t), "-seeds", "1", "-only", "A:", "-a-seed", "3"}, &out); err != nil {
		t.Fatal(err)
	}
	if s := out.String(); !strings.Contains(s, "Cells where the contract does not hold") || !strings.Contains(s, "interval") {
		t.Errorf("sweep sections missing:\n%.2000s", s)
	}
}

func TestHelpersOnBoots(t *testing.T) {
	boots, err := trace.ParseBoots(tracetest.Stalled("w", 8).String())
	if err != nil {
		t.Fatal(err)
	}
	if got := longestRunBoot(boots, 4096); got != 0 {
		t.Errorf("longestRunBoot = %d", got)
	}
	if got := longestRunBoot(boots, 99); got != -1 {
		t.Errorf("longestRunBoot with no run = %d", got)
	}
	if stamp(boots[0].Start) != "09-27T00:00:00" || stamp(trace.Boot{}.Start) != "-" {
		t.Errorf("stamp = %q / %q", stamp(boots[0].Start), stamp(trace.Boot{}.Start))
	}
	if got := splitLines("a\nb\n"); len(got) != 2 || got[1] != "b" {
		t.Errorf("splitLines = %q", got)
	}
	if got := splitLines("a\nb"); len(got) != 2 || got[1] != "b" {
		t.Errorf("splitLines without trailing newline = %q", got)
	}
	if maxF(1, 2) != 2 || maxF(3, 2) != 3 {
		t.Error("maxF")
	}
	if got := seedList(3); len(got) != 3 || got[2] != 3 {
		t.Errorf("seedList = %v", got)
	}
	if fmtMetric(0, 1.26) != "1.3" || fmtMetric(sim_MRecomputes(), 2.6) != "3" || fmtMetric(sim_MITLp99(), 0.0123) != "12.3" {
		t.Error("fmtMetric formats")
	}
	if secondsDuration(1.5).Seconds() != 1.5 {
		t.Error("secondsDuration")
	}

	var out bytes.Buffer
	printBoots(&out, boots, 4096)
	if !strings.Contains(out.String(), "sigterm") || !strings.Contains(out.String(), "1 ") {
		t.Errorf("printBoots:\n%s", out.String())
	}

	// Episodes over a bare log have no timestamps: the measured column is blank and the calibration line is not printed.
	bare, _ := trace.ParseBoots(tracetest.Incident(tracetest.DefaultIncident).String())
	params := func(w sched.Workload, p sched.Policy) sched.Params { return w.Params(p) }
	eps := collectEpisodes(bare, 4096, params)
	if len(eps) == 0 {
		t.Fatal("no episodes in the incident log")
	}
	out.Reset()
	printEpisodes(&out, bare, eps, 4096)
	if s := out.String(); !strings.Contains(s, "line ") || strings.Contains(s, "calibration:") {
		t.Errorf("bare-log episodes:\n%s", s)
	}
	// A boot with no decode line has no step time and yields no episodes.
	noDecode, _ := trace.ParseBoots(tracetest.New("w", tracetest.Start).ColdPrompt(8*4096, 4096, 2, 0).String())
	if eps := collectEpisodes(noDecode, 4096, params); len(eps) != 0 {
		t.Errorf("episodes without a decode step time: %d", len(eps))
	}
}

func TestPrintReplayWithNothingToReplay(t *testing.T) {
	bare, _ := trace.ParseBoots(tracetest.Incident(tracetest.DefaultIncident).String())
	cal, _ := trace.CalibrateBoots(bare, 4096, 64)
	var out bytes.Buffer
	rf := &replayFlags{on: true, seconds: 60, hostMul: 4}
	printReplay(&out, bare, simCost(cal), 4096, rf, 1)
	if !strings.Contains(out.String(), "no timestamped boot with a stall to replay") {
		t.Errorf("printReplay:\n%s", out.String())
	}
	// A timestamped boot whose window holds no request reports the error.
	one, _ := trace.ParseBoots(tracetest.New("w", tracetest.Start).ServerArgs(tracetest.DefaultArgs).Steady(3, 1, 1000).String())
	out.Reset()
	rf.boots = "0"
	printReplay(&out, one, simCost(cal), 4096, rf, 1)
	if !strings.Contains(out.String(), "boot 0: boot 0 has no prefill step") {
		t.Errorf("printReplay on a decode-only boot:\n%s", out.String())
	}
}
