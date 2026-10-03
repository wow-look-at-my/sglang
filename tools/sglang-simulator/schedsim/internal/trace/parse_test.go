package trace

import (
	"strings"
	"testing"
)

func mustParse(t *testing.T) []Step {
	t.Helper()
	steps, err := Parse(EmbeddedLog)
	if err != nil {
		t.Fatalf("parse embedded log: %v", err)
	}
	return steps
}

func TestParseReadsEverySchedulerStep(t *testing.T) {
	steps := mustParse(t)
	if len(steps) == 0 {
		t.Fatal("no steps parsed")
	}
	for _, s := range steps {
		if s.Throughput <= 0 {
			t.Fatalf("line %d: %s step has throughput %v", s.Line, s.Kind, s.Throughput)
		}
		if s.Kind == Prefill && s.NewTokens <= 0 {
			t.Fatalf("line %d: prefill step has #new-token %d", s.Line, s.NewTokens)
		}
		if s.Kind == Decode && s.AcceptLen <= 0 {
			t.Fatalf("line %d: decode step has accept len %v", s.Line, s.AcceptLen)
		}
	}
}

func TestParseKeepsPrefillAndDecodeApart(t *testing.T) {
	steps := mustParse(t)
	var prefills, decodes int
	for _, s := range steps {
		if s.Kind == Prefill {
			prefills++
			continue
		}
		decodes++
		// A decode line never carries prefill-only fields.
		if s.NewTokens != 0 || s.Pending != 0 {
			t.Fatalf("line %d: decode step has prefill fields (#new-token %d, #pending-token %d)",
				s.Line, s.NewTokens, s.Pending)
		}
	}
	if prefills == 0 || decodes == 0 {
		t.Fatalf("expected both step kinds, got %d prefill and %d decode", prefills, decodes)
	}
}

// The log must contain the ~426K-token pending prefill the collapse was
// reported against, and the cold stretch must be the isolated one the
// operator described: full chunks, no cache reuse, nothing else on the GPU.
func TestColdPrefillIsTheReportedOne(t *testing.T) {
	steps := mustParse(t)
	m := Summarize(steps, 4096)

	if m.ColdStart < 0 {
		t.Fatal("no cold 4096-token prefill stretch found")
	}
	if m.PendingAtCold < 420000 || m.PendingAtCold > 430000 {
		t.Fatalf("#pending-token at the start of the cold prefill = %d, want ~426K", m.PendingAtCold)
	}
	if m.ColdChunks < 100 {
		t.Fatalf("cold prefill has %d chunks, want ~100 at 4096 tokens each", m.ColdChunks)
	}
	if m.RunningAtCold != 0 {
		t.Fatalf("#running-req on the first cold chunk = %d, want 0 (nothing else running)", m.RunningAtCold)
	}
	if m.DecodeInCold != 0 {
		t.Fatalf("log has %d decode steps inside the cold window, want 0", m.DecodeInCold)
	}
	for i := m.ColdStart; i < m.ColdEnd; i++ {
		s := steps[i]
		if s.Kind != Prefill {
			t.Fatalf("line %d: %s step inside the cold window", s.Line, s.Kind)
		}
		if s.NewTokens != 4096 || s.HitTokens != 0 {
			t.Fatalf("line %d: chunk is %d new / %d cached, want 4096 new / 0 cached",
				s.Line, s.NewTokens, s.HitTokens)
		}
	}
}

// The prefill sag is measured, not invented: the log's own throughput falls from
// the steady start to the bottom the operator quoted.
func TestMeasuredSagMatchesTheReportedNumbers(t *testing.T) {
	steps := mustParse(t)
	m := Summarize(steps, 4096)

	// Operator: "~13,874 tok/s at the start ... down to ~4,831 tok/s".
	if m.ColdSteadyTP < 13700 || m.ColdSteadyTP > 14000 {
		t.Fatalf("steady cold-chunk throughput = %.2f, want ~13,874", m.ColdSteadyTP)
	}
	if m.ColdMinTP < 4800 || m.ColdMinTP > 4860 {
		t.Fatalf("lowest cold-chunk throughput = %.2f, want ~4,831", m.ColdMinTP)
	}
	if m.ColdMinTP >= m.ColdSteadyTP {
		t.Fatalf("no sag: steady %.2f, bottom %.2f", m.ColdSteadyTP, m.ColdMinTP)
	}
}

// Queue growth inside the window is the operator's "#queue-req climbs".
func TestQueueGrowsDuringTheColdPrefill(t *testing.T) {
	steps := mustParse(t)
	m := Summarize(steps, 4096)

	if m.QueuePeakInCold <= m.QueueAtCold {
		t.Fatalf("#queue-req peaked at %d inside the window, started at %d; want growth",
			m.QueuePeakInCold, m.QueueAtCold)
	}
}

// Decode stat lines are emitted every DecodeLogInterval iterations, so the
// absence of decode lines in the window bounds decode there rather than proving
// zero. Pin the arithmetic the report relies on.
func TestDecodeLineIntervalBoundsTheWindow(t *testing.T) {
	steps := mustParse(t)
	m := Summarize(steps, 4096)

	if m.DecodeStepSecs <= 0 || m.DecodeStepToks <= 0 {
		t.Fatalf("decode step medians not derived: %.6f s, %.2f tokens", m.DecodeStepSecs, m.DecodeStepToks)
	}
	normal := m.NormalDecodeStepsInColdWindow()
	bound := float64(MaxDecodeStepsInColdWindow())
	if normal <= bound {
		t.Fatalf("normal decode over the window (%.0f steps) should far exceed the no-line bound (%v)", normal, bound)
	}
	if lo := m.GenRateForSteps(MaxDecodeStepsInColdWindow()); lo >= 8 {
		t.Fatalf("log permits %.1f tok/s in the window, which would reach the reported 8-20 band", lo)
	}
	if got := m.StepsForGenRate(8); got <= bound {
		t.Fatalf("8 tok/s needs %.0f steps, within the %v-step bound", got, bound)
	}
}

// The collapsed rate reads back as a share of steps: the band the operator
// reported is partial starvation, well under the total starvation the log shows.
func TestCollapseBandReadsBackAsADutyCycle(t *testing.T) {
	steps := mustParse(t)
	m := Summarize(steps, 4096)

	lo := m.DecodeDutyCycleForRate(8)
	hi := m.DecodeDutyCycleForRate(20)
	if lo <= 0 || hi <= lo {
		t.Fatalf("duty cycle for 8-20 tok/s = %.4f-%.4f, want a positive increasing pair", lo, hi)
	}
	if hi >= 1 {
		t.Fatalf("20 tok/s would need %.0f%% of steps, which is not starvation", hi*100)
	}
	// A conversation getting the pre-collapse share of steps would run at the
	// baseline rate; the band must be far below it.
	if m.DecodeDutyCycleForRate(m.BaselineGenTP) < hi {
		t.Fatalf("baseline %.1f tok/s maps to a lower duty cycle than the 20 tok/s band",
			m.BaselineGenTP)
	}
}

func TestParseRejectsLogWithoutSteps(t *testing.T) {
	if _, err := Parse("nothing here\n"); err == nil {
		t.Fatal("expected an error for a log with no scheduler steps")
	}
}

func TestParseReportsTheOffendingLine(t *testing.T) {
	_, err := Parse("TP0] Prefill batch, #new-seq: 1, #new-token: 4096, #cached-token: 0, full token usage: 0.01, mamba usage: 0.04, #running-req: 0, #queue-req: 0, #pending-token: 8, cuda graph: False, input throughput (token/s): 0.00\n")
	if err == nil {
		t.Fatal("expected an error for a zero input throughput")
	}
	if !strings.Contains(err.Error(), "line 1") {
		t.Fatalf("error should name the line, got %v", err)
	}
}
