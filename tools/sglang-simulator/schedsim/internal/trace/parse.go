// Package trace parses a live serving log into the sequence of scheduler steps
// it records, so a simulation can be driven by measured timings instead of
// constants chosen to produce a desired answer.
package trace

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const DecodeLogInterval = 40

// Kind distinguishes both batch classes the scheduler alternates between.
type Kind int

const (
	Prefill Kind = iota
	Decode
)

func (k Kind) String() string {
	if k == Prefill {
		return "prefill"
	}
	return "decode"
}

// Step is one scheduler step as the log recorded it.
type Step struct {
	Line       int
	Kind       Kind
	NewSeq     int     // prefill: #new-seq, requests the batch prefilled
	NewTokens  int     // prefill: #new-token, the tokens this forward computed
	HitTokens  int     // prefill: #cached-token, prefix reused without recompute
	Pending    int     // prefill: #pending-token outstanding across the queue
	QueueReq   int     // requests waiting in the queue
	RunningReq int     // requests the scheduler considered running
	FullTokens int     // decode: #full token currently held in the KV pool
	FullUsage  float64 // fraction of the full KV pool occupied
	MambaUsage float64
	AcceptLen  float64 // decode: speculative tokens accepted per step
	Throughput float64 // the step's own reported tok/s (input or gen)

	// At is the line's timestamp in the production format.
	At time.Time
	// Worker is the process id the line carries in the production format.
	Worker string
	// Rank is the tensor-parallel rank that printed the line.
	Rank int
}

// kindRe matches a batch line in either format.
var kindRe = regexp.MustCompile(`^(?:(\d{4}-\d\d-\d\dT\S+) (\S+) (\S+) )?TP(\d+)\] (Prefill|Decode) batch,`)

// Parse reads every scheduler step line out of a log, ignoring the HTTP request
// lines interleaved with them.
func Parse(log string) ([]Step, error) {
	var steps []Step
	for i, raw := range strings.Split(log, "\n") {
		s, ok, err := parseStep(raw, i+1)
		if err != nil {
			return nil, err
		}
		if ok {
			steps = append(steps, s)
		}
	}
	if len(steps) == 0 {
		return nil, errNoSteps
	}
	return steps, nil
}

var errNoSteps = fmt.Errorf("no scheduler step lines in log")

// parseStep parses one line as a batch step. ok is false for any line that is
// not a rank-0 batch line; err reports a batch line with a malformed field.
func parseStep(raw string, lineNo int) (Step, bool, error) {
	m := kindRe.FindStringSubmatch(raw)
	if m == nil {
		return Step{}, false, nil
	}
	rank, _ := strconv.Atoi(m[4])
	if rank != 0 {
		return Step{}, false, nil
	}
	s := Step{Line: lineNo, Worker: m[2], Rank: rank}
	if m[1] != "" {
		if t, err := time.Parse(time.RFC3339Nano, m[1]); err == nil {
			s.At = t
		}
	}
	if m[5] == "Prefill" {
		s.Kind = Prefill
	} else {
		s.Kind = Decode
	}
	{
		i := lineNo - 1
		for _, f := range []struct {
			key string
			dst *int
		}{
			{"new-seq", &s.NewSeq},
			{"new-token", &s.NewTokens},
			{"cached-token", &s.HitTokens},
			{"pending-token", &s.Pending},
			{"queue-req", &s.QueueReq},
			{"running-req", &s.RunningReq},
			{"full token", &s.FullTokens},
		} {
			v, ok, err := intField(raw, f.key)
			if err != nil {
				return Step{}, false, fmt.Errorf("line %d: %w", i+1, err)
			}
			if ok {
				*f.dst = v
			}
		}
		for _, f := range []struct {
			key string
			dst *float64
		}{
			{"full token usage", &s.FullUsage},
			{"mamba usage", &s.MambaUsage},
			{"accept len", &s.AcceptLen},
		} {
			v, err := floatField(raw, f.key)
			if err != nil {
				return Step{}, false, fmt.Errorf("line %d: %w", i+1, err)
			}
			*f.dst = v
		}
		rateKey := "input throughput (token/s)"
		if s.Kind == Decode {
			rateKey = "gen throughput (token/s)"
		}
		v, err := floatField(raw, rateKey)
		if err != nil {
			return Step{}, false, fmt.Errorf("line %d: %w", i+1, err)
		}
		if v <= 0 {
			return Step{}, false, fmt.Errorf("line %d: %s has no positive %s", i+1, s.Kind, rateKey)
		}
		s.Throughput = v
	}
	return s, true, nil
}

// intField pulls an integer field. Metric names are prefixes of one another
// ("full token" vs "full token usage"), so the match is anchored on the exact
// key followed by a colon.
func intField(line, key string) (int, bool, error) {
	idx := strings.Index(line, key+": ")
	if idx < 0 {
		return 0, false, nil
	}
	rest := line[idx+len(key)+2:]
	end := strings.IndexAny(rest, ", ")
	if end < 0 {
		end = len(rest)
	}
	v, err := strconv.Atoi(strings.ReplaceAll(strings.TrimSpace(rest[:end]), ",", ""))
	if err != nil {
		return 0, false, fmt.Errorf("field %q: %w", key, err)
	}
	return v, true, nil
}

func floatField(line, key string) (float64, error) {
	idx := strings.Index(line, key+": ")
	if idx < 0 {
		return 0, nil
	}
	rest := line[idx+len(key)+2:]
	end := strings.IndexAny(rest, ", ")
	if end < 0 {
		end = len(rest)
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(rest[:end]), 64)
	if err != nil {
		return 0, fmt.Errorf("field %q: %w", key, err)
	}
	return v, nil
}

// Metrics is what the log says about the run, with no simulation applied. The
// report quotes these so each model can be checked against the measured run it
// stands in for.
type Metrics struct {
	Steps        int
	PrefillSteps int
	DecodeSteps  int

	// ColdStart and ColdEnd bound the stretch of full-chunk, no-cache-reuse prefill steps that belong to the first cold prompt.
	ColdStart int
	ColdEnd   int
	// PendingAtCold is #pending-token on the first step of that stretch.
	PendingAtCold int
	// DecodeInCold counts decode steps inside the stretch.
	DecodeInCold int
	// QueueAtCold and RunningAtCold are the first step's queue and running counts.
	QueueAtCold   int
	RunningAtCold int

	// QueuePeakInCold is the largest #queue-req seen inside the stretch.
	QueuePeakInCold int

	ColdChunks     int
	ColdTokens     int
	ColdSeconds    float64 // sum of the stretch's measured step durations
	ColdFirstTP    float64 // input tok/s on the stretch's first step
	ColdSteadyTP   float64 // input tok/s on the second step, past cold-start warm-up
	ColdPeakTP     float64
	ColdMinTP      float64 // lowest rate after the first step, i.e. the sag's bottom
	ColdTailTP     float64 // input tok/s on the stretch's last step
	BaselineGenTP  float64 // median gen tok/s over decode steps before the stretch
	MinGenTP       float64
	DecodeStepSecs float64 // median measured duration of one decode step
	DecodeStepToks float64 // median speculative tokens accepted per decode step
	RunningDecode  int     // running requests on the last decode step before the stretch

	LowGenLines       int
	LowGenNearPrefill int
}

// NormalDecodeStepsPerSec is the decode step rate measured before the collapse,
// from the median decode step duration.
func (m Metrics) NormalDecodeStepsPerSec() float64 {
	if m.DecodeStepSecs <= 0 {
		return 0
	}
	return 1 / m.DecodeStepSecs
}

// MaxDecodeStepsInColdWindow bounds the decode steps that could have run inside the cold prefill while still logging no decode line.
func MaxDecodeStepsInColdWindow() int { return DecodeLogInterval - 1 }

// NormalDecodeStepsInColdWindow is how many decode steps would have run over
// the cold window at the pre-collapse rate.
func (m Metrics) NormalDecodeStepsInColdWindow() float64 {
	return m.ColdSeconds * m.NormalDecodeStepsPerSec()
}

// StepsForGenRate is how many decode steps the window will need for an
// interrupted conversation to generate at rate tokens/s.
func (m Metrics) StepsForGenRate(rate float64) float64 {
	if m.DecodeStepToks <= 0 {
		return 0
	}
	return rate * m.ColdSeconds / m.DecodeStepToks
}

// GenRateForSteps is the generation rate achieved over the window when this many
// decode steps run in it.
func (m Metrics) GenRateForSteps(steps int) float64 {
	if m.ColdSeconds <= 0 {
		return 0
	}
	return float64(steps) * m.DecodeStepToks / m.ColdSeconds
}

// DecodeDutyCycleForRate is the fraction of scheduling steps that must be
// decode for a conversation to generate at rate tokens/s.
func (m Metrics) DecodeDutyCycleForRate(rate float64) float64 {
	if m.DecodeStepToks <= 0 || m.DecodeStepSecs <= 0 {
		return 0
	}
	return rate * m.DecodeStepSecs / m.DecodeStepToks
}

// isColdChunk reports whether a step looks like a chunk of a cold long
// prompt: a full-size chunk that reuses no cached prefix.
func isColdChunk(s Step, chunkSize int) bool {
	return s.Kind == Prefill && s.HitTokens == 0 && s.NewTokens == chunkSize
}

// coldWindow finds the longest run of cold chunks. The log can hold more than
// one stretch if the prefill is interrupted, so the longest is the one to model.
func coldWindow(steps []Step, chunkSize int) (start, end int) {
	start, end = -1, -1
	runStart := -1
	for i, s := range steps {
		if isColdChunk(s, chunkSize) {
			if runStart < 0 {
				runStart = i
			}
			if i+1-runStart > end-start {
				start, end = runStart, i+1
			}
			continue
		}
		runStart = -1
	}
	return start, end
}

// Summarize reduces the parsed steps to the observables the models must
// reproduce. Every field is read straight out of the log.
func Summarize(steps []Step, chunkSize int) Metrics {
	m := Metrics{Steps: len(steps), ColdStart: -1, ColdEnd: -1}
	m.ColdStart, m.ColdEnd = coldWindow(steps, chunkSize)
	for _, s := range steps {
		if s.Kind == Prefill {
			m.PrefillSteps++
		} else {
			m.DecodeSteps++
		}
	}
	if m.ColdStart >= 0 {
		first := steps[m.ColdStart]
		m.PendingAtCold = first.Pending
		m.QueueAtCold = first.QueueReq
		m.RunningAtCold = first.RunningReq
		m.ColdChunks = m.ColdEnd - m.ColdStart
		m.ColdFirstTP = first.Throughput
		m.ColdPeakTP = first.Throughput
		m.ColdMinTP = first.Throughput
		chunks := 0
		for i := m.ColdStart; i < m.ColdEnd; i++ {
			s := steps[i]
			if s.QueueReq > m.QueuePeakInCold {
				m.QueuePeakInCold = s.QueueReq
			}
			if s.Kind == Decode {
				m.DecodeInCold++
				continue
			}
			chunks++
			m.ColdTokens += s.NewTokens
			m.ColdSeconds += StepSeconds(s)
			if s.Throughput > m.ColdPeakTP {
				m.ColdPeakTP = s.Throughput
			}
			// The first chunk of a fresh sequence runs cold-start work, so the
			// steady band and the sag's bottom are taken from the chunks after it.
			if chunks == 1 {
				continue
			}
			if chunks == 2 {
				m.ColdSteadyTP = s.Throughput
				m.ColdMinTP = s.Throughput
			}
			if s.Throughput < m.ColdMinTP {
				m.ColdMinTP = s.Throughput
			}
			m.ColdTailTP = s.Throughput
		}
	}
	var baseline, minGen []float64
	for i, s := range steps {
		if s.Kind != Decode {
			continue
		}
		if m.ColdStart >= 0 && i < m.ColdStart {
			baseline = append(baseline, s.Throughput)
		}
		minGen = append(minGen, s.Throughput)
		if m.ColdStart >= 0 && i < m.ColdStart {
			m.RunningDecode = s.RunningReq
		}
	}
	m.BaselineGenTP = median(baseline)
	m.MinGenTP = minOf(minGen)
	for i, s := range steps {
		if s.Kind != Decode || s.Throughput >= 100 {
			continue
		}
		m.LowGenLines++
		if nearPrefill(steps, i, 2) {
			m.LowGenNearPrefill++
		}
	}
	// One decode step's duration is tokens-per-step divided by the rate that produced them.
	var durs, toks []float64
	for _, s := range steps {
		if s.Kind == Decode && s.AcceptLen > 0 {
			durs = append(durs, s.AcceptLen/s.Throughput)
			toks = append(toks, s.AcceptLen)
		}
	}
	m.DecodeStepSecs = median(durs)
	m.DecodeStepToks = median(toks)
	return m
}

// nearPrefill reports whether a step has a prefill step within radius steps of
// it in the log.
func nearPrefill(steps []Step, at, radius int) bool {
	lo, hi := at-radius, at+radius
	if lo < 0 {
		lo = 0
	}
	if hi >= len(steps) {
		hi = len(steps) - 1
	}
	for i := lo; i <= hi; i++ {
		if steps[i].Kind == Prefill {
			return true
		}
	}
	return false
}

// StepSeconds is one prefill step's measured GPU duration: the step's new
// tokens over the input rate the log reports for that same step.
func StepSeconds(s Step) float64 {
	return float64(s.NewTokens) / s.Throughput
}

// Seconds is StepSeconds as a method.
func (s Step) Seconds() float64 { return StepSeconds(s) }

func StepGap(prev, cur Step) (float64, bool) {
	if prev.At.IsZero() || cur.At.IsZero() {
		return 0, false
	}
	return cur.At.Sub(prev.At).Seconds(), true
}

func median(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	sorted := append([]float64(nil), xs...)
	for i := 1; i < len(sorted); i++ {
		for j := i; j > 0 && sorted[j] < sorted[j-1]; j-- {
			sorted[j], sorted[j-1] = sorted[j-1], sorted[j]
		}
	}
	n := len(sorted)
	if n%2 == 1 {
		return sorted[n/2]
	}
	return (sorted[n/2-1] + sorted[n/2]) / 2
}

func minOf(xs []float64) float64 {
	m := 0.0
	for i, v := range xs {
		if i == 0 || v < m {
			m = v
		}
	}
	return m
}
