package report

import (
	"bytes"
	"strings"
	"testing"

	"schedsim/internal/sched"
	"schedsim/internal/trace"
)

func embedded(t *testing.T) (trace.Metrics, sched.Workload) {
	t.Helper()
	steps, err := trace.Parse(trace.EmbeddedLog)
	if err != nil {
		t.Fatal(err)
	}
	m := trace.Summarize(steps, 4096)
	return m, sched.WorkloadFromLog(steps, m, 4096)
}

func TestWriteRendersEverySection(t *testing.T) {
	m, w := embedded(t)
	old := sched.Simulate(w, w.Params(sched.PolicyPrefillPriority), m.QueueAtCold)
	new := sched.Simulate(w, w.Params(sched.PolicyTimeBalance), m.QueueAtCold)
	rev := sched.Simulate(w, w.Params(sched.PolicyQueueBalance), m.QueueAtCold)
	in := Input{
		Metrics: m, Old: old, New: new, ChunkSize: 4096, Params: w.Params(sched.PolicyTimeBalance),
		LogLines: 10, LogName: "test.log", ColdFirstLine: 1, ColdLastLine: 2,
		Fidelities: DefaultFidelities(m, EmbeddedIncident), Incident: EmbeddedIncident,
		Revised: []RevisedRun{{Queue: 1, Label: "one", Result: rev}},
		Sweep:   []SweepPoint{{Interval: 2, DecodeSteps: 3, EffectiveGenTPS: 4, PrefillFinishSecs: 5}},
	}
	var out bytes.Buffer
	Write(&out, in)
	s := out.String()
	for _, want := range []string{"INPUT LOG", "test.log, 10 lines", "COST MODEL", "operator's ~8-20 tok/s band",
		"WHERE EACH NUMBER COMES FROM", "[measured", "[NOT IN LOG", "one", "13,874 -> ~4,831"} {
		if !strings.Contains(s, want) {
			t.Errorf("report lacks %q", want)
		}
	}
	// Without an incident the operator's words are left out.
	in.Incident = nil
	in.Fidelities = DefaultFidelities(m, nil)
	in.Sweep, in.Revised = nil, nil
	out.Reset()
	Write(&out, in)
	if s := out.String(); strings.Contains(s, "operator") || strings.Contains(s, "13,874") {
		t.Errorf("no-incident report quotes the operator:\n%s", s)
	}
	out.Reset()
	Summary(&out, old, new)
	if s := out.String(); !strings.HasPrefix(s, "SUMMARY\n") || strings.Count(s, "decode steps") != 2 {
		t.Errorf("summary:\n%s", s)
	}
	if commas(1234567) != "1,234,567" || commas(999) != "999" || commas(1000) != "1,000" {
		t.Error("commas")
	}
	if yn(true) != "yes" || yn(false) != "no" {
		t.Error("yn")
	}
}
