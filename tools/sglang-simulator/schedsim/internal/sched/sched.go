// Package sched holds the two scheduler policies under comparison and runs both
// over one workload parsed from a live log. The policies are the control logic;
// every cost they charge comes from the log.
package sched

import (
	"fmt"

	"schedsim/internal/trace"
)

// Policy names a scheduling rule.
type Policy int

const (
	// PolicyPrefillPriority is the behavior the collapse was measured under: a
	// prefill batch runs whenever one can be formed, and decode runs only when
	// no prefill is pending.
	PolicyPrefillPriority Policy = iota
	// PolicyFixedInterval runs a fixed number of decode rounds after each
	// prefill batch, the --prefill-decode-interval schedule.
	PolicyFixedInterval
	// PolicyTimeBalance splits contended GPU time between prefill and decode by
	// measured batch duration, which is the shipped balancer.
	PolicyTimeBalance
	// PolicyQueueBalance is the revised balancer: prefill and decode batches get
	// equal time however many requests a prefill batch serves (decode's floor),
	// at most one chunk budget of prefill tokens runs between two points where
	// decode has caught up, and waiting requests that cannot run take nothing.
	PolicyQueueBalance
)

func (p Policy) String() string {
	switch p {
	case PolicyPrefillPriority:
		return "prefill-priority (old)"
	case PolicyFixedInterval:
		return "fixed-interval"
	case PolicyQueueBalance:
		return "queue-balance (revised)"
	default:
		return "time-balance (new)"
	}
}

// Params are the scheduling knobs plus the cost inputs the log supplies.
type Params struct {
	Policy Policy

	// Interval is the number of decode rounds armed after each prefill batch
	// under PolicyFixedInterval.
	Interval int

	// PrefillShare is the fraction of contended GPU time prefill is entitled
	// to under PolicyTimeBalance. 0.5 reproduces the shipped rule: defer prefill
	// while measured prefill seconds exceed measured decode seconds.
	PrefillShare float64

	// DecodeStepSeconds is the measured duration of one decode step and
	// DecodeTokensPerStep the measured tokens it produces.
	DecodeStepSeconds   float64
	DecodeTokensPerStep float64

	// RunningReqs is how many requests are decoding while the prefill runs.
	RunningReqs int

	// DecodePerReqFraction is the share of the decode step cost each additional
	// running request adds. Assumed: the log only ever decodes one request, so
	// this cannot be measured from it. 0 means decode cost is flat in batch size.
	DecodePerReqFraction float64

	// PrefillInterference is the fractional slowdown a prefill batch suffers
	// from sharing a contended window with decode, beyond the time split itself.
	// Assumed: unmeasurable from a single-request log. 0 keeps each class's
	// measured cost intact and lets the time split alone decide.
	PrefillInterference float64

	// MixedChunk makes every chunk under PolicyQueueBalance carry one decode
	// row per running request (mixed chunked prefill, resolved on by default):
	// each row is one more extend token at the chunk's measured per-token cost
	// and yields one token (speculative decoding degrades to a plain decode
	// inside a mixed step). The balancer charges such a chunk as prefill minus
	// its rows, so pure decode keeps its half and the rows' tokens come on top.
	MixedChunk bool
}

// Chunk is one cold-prefill batch as the log measured it.
type Chunk struct {
	Tokens   int
	Usage    float64 // KV-pool occupancy while it ran
	Seconds  float64 // its measured GPU seconds
	InputTPS float64 // its measured input throughput
}

// Workload is the log reduced to what a scheduler needs: the cold prefill's
// chunks in order, plus the decode work running beside them.
type Workload struct {
	Chunks []Chunk

	TotalTokens         int
	RunningReqs         int
	DecodeStepSeconds   float64
	DecodeTokensPerStep float64
	ChunkSize           int

	// Source quotes where the workload came from, for the report header.
	Source string
}

// WorkloadFromLog builds the workload out of the parsed steps. The cold chunks
// are exactly the steps of the isolated stretch, so their costs are measured
// rather than modelled, and the decode cost is the measured median decode step.
func WorkloadFromLog(steps []trace.Step, m trace.Metrics, chunkSize int) Workload {
	w := Workload{
		ChunkSize:           chunkSize,
		RunningReqs:         m.RunningDecode,
		DecodeStepSeconds:   m.DecodeStepSecs,
		DecodeTokensPerStep: m.DecodeStepToks,
	}
	for i := m.ColdStart; i < m.ColdEnd; i++ {
		s := steps[i]
		if s.Kind != trace.Prefill {
			continue
		}
		w.Chunks = append(w.Chunks, Chunk{
			Tokens:   s.NewTokens,
			Usage:    s.FullUsage,
			Seconds:  trace.StepSeconds(s),
			InputTPS: s.Throughput,
		})
		w.TotalTokens += s.NewTokens
	}
	if w.RunningReqs <= 0 {
		w.RunningReqs = 1
	}
	w.Source = fmt.Sprintf(
		"log lines %d-%d: %d cold %d-token chunks, first reports #pending-token %d",
		steps[m.ColdStart].Line, steps[m.ColdEnd-1].Line,
		m.ColdChunks, chunkSize, m.PendingAtCold)
	return w
}

// WorkloadFromRun builds the workload for one cold run of a boot, with the
// decode step and running-request figures the boot's Metrics measured. It is
// WorkloadFromLog for a run other than the longest.
func WorkloadFromRun(steps []trace.Step, r trace.ColdRun, m trace.Metrics) Workload {
	w := Workload{
		ChunkSize:           m.ColdChunks,
		RunningReqs:         r.RunningAtStart,
		DecodeStepSeconds:   m.DecodeStepSecs,
		DecodeTokensPerStep: m.DecodeStepToks,
	}
	if r.Chunks() > 0 {
		w.ChunkSize = steps[r.Start].NewTokens
	}
	for i := r.Start; i < r.End; i++ {
		s := steps[i]
		w.Chunks = append(w.Chunks, Chunk{
			Tokens:   s.NewTokens,
			Usage:    s.FullUsage,
			Seconds:  trace.StepSeconds(s),
			InputTPS: s.Throughput,
		})
		w.TotalTokens += s.NewTokens
	}
	if w.RunningReqs <= 0 {
		w.RunningReqs = 1
	}
	w.Source = fmt.Sprintf(
		"log lines %d-%d: %d cold %d-token chunks, first reports #pending-token %d",
		steps[r.Start].Line, steps[r.End-1].Line, r.Chunks(), w.ChunkSize, r.Pending)
	return w
}

// Params returns the knobs for a policy over this workload.
func (w Workload) Params(policy Policy) Params {
	return Params{
		Policy:              policy,
		PrefillShare:        0.5,
		DecodeStepSeconds:   w.DecodeStepSeconds,
		DecodeTokensPerStep: w.DecodeTokensPerStep,
		RunningReqs:         w.RunningReqs,
	}
}

// Event is one batch the simulation ran.
type Event struct {
	IsPrefill bool
	Tokens    int
	Seconds   float64
	Wall      float64 // wall-clock end time
}

// Result is what one policy did with the workload.
type Result struct {
	Policy string

	PrefillChunksRun int
	PrefillCompleted bool

	// DecodeSteps is how many decode steps ran before the cold prefill finished,
	// which is the window the collapse was measured in.
	DecodeSteps     int
	GeneratedTokens float64

	// WindowSeconds is wall time from the first batch to prefill completion.
	WindowSeconds     float64
	PrefillGPUSeconds float64

	// EffectiveGenTPS is the generation throughput a running request observed
	// across the window: tokens produced divided by window wall time.
	EffectiveGenTPS float64
	// PrefillWallTPS is the cold prefill's tokens over the window wall time.
	PrefillWallTPS float64

	// FirstChunkInputTPS and LastChunkInputTPS are the input rates the log
	// measured for the cold prefill's first and last chunk. They are properties
	// of the chunk, so both policies carry the same values: the log's throughput
	// sag is not something either scheduling rule addresses.
	FirstChunkInputTPS float64
	LastChunkInputTPS  float64

	// LongestDecodeGap is the longest stretch of wall time with no decode step,
	// the direct measure of starvation.
	LongestDecodeGap float64

	// QueueWaits counts requests still queued when the prefill finished.
	QueueWaits int

	Trace []Event
}

// Simulate runs one policy to cold-prefill completion.
func Simulate(w Workload, p Params, queueAtStart int) Result {
	r := Result{
		Policy:             p.Policy.String(),
		QueueWaits:         queueAtStart,
		FirstChunkInputTPS: w.Chunks[0].InputTPS,
		LastChunkInputTPS:  w.Chunks[len(w.Chunks)-1].InputTPS,
	}
	run := &runner{r: &r, w: w, p: p}
	switch p.Policy {
	case PolicyPrefillPriority:
		run.prefillPriority()
	case PolicyFixedInterval:
		run.fixedInterval()
	case PolicyQueueBalance:
		run.queueBalance()
	default:
		run.timeBalance()
	}
	run.finish()
	return r
}

type runner struct {
	r *Result
	w Workload
	p Params

	wall           float64
	lastDecodeWall float64
	lastCharge     float64
	hasLastCharge  bool
}

// prefillPriority is the old rule: a prefill batch wins the step whenever one is
// pending, so decode runs only once the prefill is exhausted. The cold prefill
// therefore completes with no decode step at all.
func (x *runner) prefillPriority() {
	for _, c := range x.w.Chunks {
		x.prefill(c)
	}
}

// fixedInterval implements --prefill-decode-interval N: each prefill batch arms
// N decode rounds before the next prefill batch may run.
func (x *runner) fixedInterval() {
	for i, c := range x.w.Chunks {
		x.prefill(c)
		if i == len(x.w.Chunks)-1 {
			break
		}
		for n := 0; n < x.p.Interval && x.p.RunningReqs > 0; n++ {
			x.decode(1)
		}
	}
}

// timeBalance implements the shipped balancer: it keeps a running balance of
// measured prefill seconds minus measured decode seconds, settled only while
// both classes contend, and defers prefill while the balance is positive. A
// positive balance means prefill has had more than its share, so decode runs
// until the split is even. The balance is floored at zero, so neither class can
// bank credit for a burst later.
func (x *runner) timeBalance() {
	// weight converts decode seconds into prefill-equivalent seconds, so the
	// balance settles to zero exactly when the split matches PrefillShare.
	weight := x.p.PrefillShare / (1 - x.p.PrefillShare)
	var debt float64
	for i := 0; i < len(x.w.Chunks); {
		x.settle(&debt, 0)
		if debt > 0 && x.p.RunningReqs > 0 && x.decodeSeconds() > 0 {
			x.decode(weight)
			continue
		}
		x.prefill(x.w.Chunks[i])
		i++
	}
}

// queueBalance implements the revised balancer
// (python/sglang/srt/managers/scheduler_components/prefill_decode_balancer.py).
// A prefill batch's seconds are charged in full (a mixed chunk's minus its
// decode rows), so decode keeps half of contended time; the queue behind the
// chunk does not count. burst counts prefill tokens since decode last caught
// up; a chunk continuation runs only once it is zero, and no batch runs once it
// reaches a chunk. The balance may go negative by one decode step, the
// overshoot the balancer carries. This model is sequential (no overlap), so
// nothing is in flight at a decision and the chunks are whole.
func (x *runner) queueBalance() {
	var debt float64
	burst := 0
	for i := 0; i < len(x.w.Chunks); {
		x.settle(&debt, -x.decodeSeconds())
		if x.p.RunningReqs <= 0 || x.decodeSeconds() <= 0 {
			// No contention: the balancer resets and never defers.
			debt, burst = 0, 0
		}
		if debt <= 0 {
			burst = 0
		}
		continuesChunk := i > 0
		exhausted := burst >= x.w.ChunkSize || (continuesChunk && burst > 0)
		if exhausted && x.p.RunningReqs > 0 && x.decodeSeconds() > 0 {
			x.decode(1)
			continue
		}
		if x.p.MixedChunk && x.p.RunningReqs > 0 {
			x.mixedChunk(x.w.Chunks[i])
		} else {
			x.prefill(x.w.Chunks[i])
		}
		burst += x.w.Chunks[i].Tokens
		i++
	}
}

// settle folds the last completed batch's measured duration into the balance,
// which is what the balancer does at the next decision after a completion.
// floor is how far below zero the balance may go.
func (x *runner) settle(debt *float64, floor float64) {
	if !x.hasLastCharge {
		return
	}
	*debt = maxFloat(*debt+x.lastCharge, floor)
	x.lastCharge = 0
	x.hasLastCharge = false
}

func (x *runner) prefill(c Chunk) {
	x.r.PrefillChunksRun++
	x.r.PrefillGPUSeconds += c.Seconds
	secs := c.Seconds * (1 + x.p.PrefillInterference)
	x.wall += secs
	x.r.Trace = append(x.r.Trace, Event{IsPrefill: true, Tokens: c.Tokens, Seconds: secs, Wall: x.wall})
	x.lastCharge = secs
	x.hasLastCharge = true
	x.noteGap()
}

// mixedChunk runs one chunk with the running requests' decode rows inside it.
func (x *runner) mixedChunk(c Chunk) {
	rows := float64(x.p.RunningReqs)
	rowSecs := rows * c.Seconds / float64(c.Tokens)
	x.r.PrefillChunksRun++
	x.r.PrefillGPUSeconds += c.Seconds
	secs := (c.Seconds + rowSecs) * (1 + x.p.PrefillInterference)
	x.wall += secs
	x.r.Trace = append(x.r.Trace, Event{IsPrefill: true, Tokens: c.Tokens, Seconds: secs, Wall: x.wall})
	x.r.DecodeSteps++
	x.r.GeneratedTokens += rows
	x.noteGap()
	x.lastDecodeWall = x.wall
	x.lastCharge = secs - rowSecs*(1+x.p.PrefillInterference)
	x.hasLastCharge = true
}

func (x *runner) decode(weight float64) {
	x.r.DecodeSteps++
	// One decode step advances every running request, so the batch's tokens are
	// the measured per-request acceptance times the number decoding.
	x.r.GeneratedTokens += x.p.DecodeTokensPerStep * float64(x.p.RunningReqs)
	secs := x.decodeSeconds()
	x.wall += secs
	x.r.Trace = append(x.r.Trace, Event{Seconds: secs, Wall: x.wall})
	x.lastDecodeWall = x.wall
	x.lastCharge = -secs * weight
	x.hasLastCharge = true
	x.noteGap()
}

func (x *runner) decodeSeconds() float64 {
	extra := float64(x.p.RunningReqs-1) * x.p.DecodePerReqFraction
	return x.p.DecodeStepSeconds * (1 + extra)
}

// noteGap tracks the longest interval with no decode step.
func (x *runner) noteGap() {
	if g := x.wall - x.lastDecodeWall; g > x.r.LongestDecodeGap {
		x.r.LongestDecodeGap = g
	}
}

func (x *runner) finish() {
	x.r.WindowSeconds = x.wall
	x.r.PrefillCompleted = x.r.PrefillChunksRun == len(x.w.Chunks)
	if x.r.WindowSeconds <= 0 {
		return
	}
	x.r.EffectiveGenTPS = x.r.GeneratedTokens / x.r.WindowSeconds
	x.r.PrefillWallTPS = float64(x.w.TotalTokens) / x.r.WindowSeconds
}

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
