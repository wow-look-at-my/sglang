package sim

import (
	"math"
	"sort"
)

// Metrics is the seven measurements the comparison is judged on, plus the
// supporting columns the spec asks to report alongside them. Every one is a pure
// function of the run trace, so two runs with the same trace report the same
// numbers.
type Metrics struct {
	// 1: tokens per second per stream, over the seconds each agent stream spent
	// past its first token inside a cold prefill's window.
	StreamDecodeTokSCold float64
	// 2: the same tokens over the cold window times every agent conversation, so
	// a follow-up stuck in the queue shows up as a lost rate.
	PerAgentTokSCold float64
	// 3: the longest gap between consecutive deliveries of any stream.
	LongestStall float64
	// 4: time to first token of the cold prompts.
	ColdTTFTMean, ColdTTFTMax float64
	// 5 and 6: inter-token latency percentiles, each gap spread over the tokens
	// its chunk carried, which is what a per-token claim compares against.
	ITLp99, ITLp999 float64
	// 7: output tokens per second over the window, and full-prefix recomputes.
	OutputTokS   float64
	Recomputes   int

	// Supporting columns.
	StallFrac1s    float64
	LowGenLines    int
	DecodeLogLines int
	CompletedTurns int
	GPUBusyShare   float64
	DecodeShare    float64
	TurnTTFTp50    float64
	TurnTTFTp99    float64
	ITLp50         float64
	ITLp99Raw      float64
	ColdArrived    int
	ColdServed     int
	ShortTTFTp50   float64
	ShortTTFTp99   float64
}

// MetricKey names one of the seven metrics the contract is written over.
type MetricKey int

const (
	MStreamRate MetricKey = iota
	MAgentRate
	MLongestStall
	MColdTTFT
	MITLp99
	MITLp999
	MThroughput
	MRecomputes
)

// ContractMetrics is the ordered list of the seven metrics.
var ContractMetrics = []MetricKey{MStreamRate, MAgentRate, MLongestStall, MColdTTFT,
	MITLp99, MITLp999, MThroughput, MRecomputes}

func (k MetricKey) String() string {
	switch k {
	case MStreamRate:
		return "stream decode tok/s in cold"
	case MAgentRate:
		return "per-agent tok/s in cold"
	case MLongestStall:
		return "longest stall"
	case MColdTTFT:
		return "cold TTFT mean"
	case MITLp99:
		return "ITL p99"
	case MITLp999:
		return "ITL p99.9"
	case MThroughput:
		return "output tok/s"
	default:
		return "full-prefix recomputes"
	}
}

// HigherIsBetter states the metric's direction: rates and completed work want to
// rise, stalls and recomputes want to fall.
func (k MetricKey) HigherIsBetter() bool {
	switch k {
	case MStreamRate, MAgentRate, MThroughput:
		return true
	default:
		return false
	}
}

// Value reads one metric out of a measurement.
func (m Metrics) Value(k MetricKey) float64 {
	switch k {
	case MStreamRate:
		return m.StreamDecodeTokSCold
	case MAgentRate:
		return m.PerAgentTokSCold
	case MLongestStall:
		return m.LongestStall
	case MColdTTFT:
		return m.ColdTTFTMean
	case MITLp99:
		return m.ITLp99
	case MITLp999:
		return m.ITLp999
	case MThroughput:
		return m.OutputTokS
	default:
		return float64(m.Recomputes)
	}
}

// interval is a merged stretch of cold-prefill time.
type interval struct{ lo, hi float64 }

// Measure reduces a run's trace to the metrics, over [0, window].
func Measure(res *Result, window float64) Metrics {
	var m Metrics
	if window <= 0 {
		window = res.End
	}
	wins := coldWindows(res, window)
	m.ColdArrived = len(res.Windows)
	for _, w := range res.Windows {
		if w.Done() {
			m.ColdServed++
		}
	}
	// Cold TTFT over the prompts that reached a first token; the arrived/served
	// count is reported beside it so an unserved prompt cannot hide a number.
	var ttft []float64
	for _, w := range res.Windows {
		if w.Done() {
			ttft = append(ttft, w.FirstTok-w.Arrival)
		}
	}
	m.ColdTTFTMean, m.ColdTTFTMax = meanOf(ttft), maxOf(ttft)

	var agent []float64
	agent = coldTokens(res, wins)
	if secs := streamSeconds(res, wins); secs > 0 {
		m.StreamDecodeTokSCold = sum(agent) / secs
	}
	if n := res.Agents; n > 0 && len(wins) > 0 {
		m.PerAgentTokSCold = sum(agent) / (spanOf(wins) * float64(n))
	}

	m.LongestStall = maxOf(gaps(res, func(*Request) bool { return true }))
	itl := spread(res, func(*Request) bool { return true })
	m.ITLp50, m.ITLp99, m.ITLp999 = pct(itl, 50), pct(itl, 99), pct(itl, 99.9)
	m.ITLp99Raw = pct(gaps(res, func(*Request) bool { return true }), 99)
	m.StallFrac1s = stalledFraction(res, window)

	out := 0
	for _, r := range res.Requests {
		for _, d := range r.Deliveries {
			if d.T <= window {
				out += d.N
			}
		}
		if r.Finish >= 0 && r.Finish <= window {
			m.CompletedTurns++
		}
	}
	if window > 0 {
		m.OutputTokS = float64(out) / window
	}
	recomp := 0
	for _, n := range res.Pool.Recomputes {
		recomp += n
	}
	m.Recomputes = recomp

	for _, l := range res.DecodeLog {
		if l.T <= window {
			m.DecodeLogLines++
			if l.Gen < 25 {
				m.LowGenLines++
			}
		}
	}
	busy, dec := busySeconds(res, window)
	m.GPUBusyShare = busy / window
	if busy > 0 {
		m.DecodeShare = dec / busy
	}
	turn := ttfts(res, func(r *Request) bool { return r.Kind == KindAgent })
	m.TurnTTFTp50, m.TurnTTFTp99 = pct(turn, 50), pct(turn, 99)
	short := ttfts(res, func(r *Request) bool { return r.Kind == KindShort })
	m.ShortTTFTp50, m.ShortTTFTp99 = pct(short, 50), pct(short, 99)
	return m
}

// coldWindows merges the cold prompts' prefill stretches, clipped to the run
// window. A prompt that never reached its first token keeps a window that runs
// to the end of the measurement, so the streams it crowded out are still
// counted against the policy that left it unserved.
func coldWindows(res *Result, window float64) []interval {
	var raw []interval
	for _, w := range res.Windows {
		hi := w.FirstTok
		if !w.Done() {
			hi = minF(window, res.End)
		}
		hi = minF(hi, window)
		if hi > w.Arrival {
			raw = append(raw, interval{w.Arrival, hi})
		}
	}
	sort.Slice(raw, func(i, j int) bool { return raw[i].lo < raw[j].lo })
	var out []interval
	for _, in := range raw {
		if n := len(out); n > 0 && in.lo <= out[n-1].hi {
			if in.hi > out[n-1].hi {
				out[n-1].hi = in.hi
			}
			continue
		}
		out = append(out, in)
	}
	return out
}

func inWindows(wins []interval, t float64) bool {
	for _, w := range wins {
		if t >= w.lo && t <= w.hi {
			return true
		}
	}
	return false
}

func spanOf(wins []interval) float64 {
	s := 0.0
	for _, w := range wins {
		s += w.hi - w.lo
	}
	return s
}

// coldTokens counts the agent streams' delivered tokens inside the cold windows.
func coldTokens(res *Result, wins []interval) []float64 {
	var out []float64
	for _, r := range res.Requests {
		if r.Kind != KindAgent {
			continue
		}
		for _, d := range r.Deliveries {
			if inWindows(wins, d.T) {
				out = append(out, float64(d.N))
			}
		}
	}
	return out
}

// streamSeconds is the time each agent stream spent past its first token inside
// the cold windows, summed over streams: metric 1's denominator.
func streamSeconds(res *Result, wins []interval) float64 {
	total := 0.0
	for _, r := range res.Requests {
		if r.Kind != KindAgent {
			continue
		}
		for _, w := range wins {
			lo := maxF(w.lo, r.FirstTok)
			if r.FirstTok < 0 || lo >= w.hi {
				continue
			}
			hi := w.hi
			if r.Finish >= 0 && r.Finish < hi {
				hi = r.Finish
			}
			if hi > lo {
				total += hi - lo
			}
		}
	}
	return total
}

// gaps lists the interval between consecutive deliveries of each selected
// stream after its first token.
func gaps(res *Result, sel func(*Request) bool) []float64 {
	var out []float64
	for _, r := range res.Requests {
		if !sel(r) || len(r.Deliveries) < 2 {
			continue
		}
		for i := 1; i < len(r.Deliveries); i++ {
			out = append(out, r.Deliveries[i].T-r.Deliveries[i-1].T)
		}
	}
	return out
}

// spread lists per-token inter-token gaps: a chunk that carried N tokens over
// T seconds contributes N samples of T/N.
func spread(res *Result, sel func(*Request) bool) []float64 {
	var out []float64
	for _, r := range res.Requests {
		if !sel(r) || len(r.Deliveries) < 2 {
			continue
		}
		for i := 1; i < len(r.Deliveries); i++ {
			n := r.Deliveries[i].N
			if n < 1 {
				n = 1
			}
			g := (r.Deliveries[i].T - r.Deliveries[i-1].T) / float64(n)
			for k := 0; k < n; k++ {
				out = append(out, g)
			}
		}
	}
	return out
}

// stalledFraction is the share of stream time spent inside a stall longer than
// a second, which separates one unlucky stream from a system that stalls always.
func stalledFraction(res *Result, window float64) float64 {
	inStall, live := 0.0, 0.0
	for _, r := range res.Requests {
		if len(r.Deliveries) < 2 {
			continue
		}
		end := r.Finish
		if end < 0 || end > window {
			end = window
		}
		for i := 1; i < len(r.Deliveries); i++ {
			gap := r.Deliveries[i].T - r.Deliveries[i-1].T
			if r.Deliveries[i-1].T >= end {
				break
			}
			live += minF(gap, end-r.Deliveries[i-1].T)
			if gap > 1 {
				inStall += minF(gap, end-r.Deliveries[i-1].T)
			}
		}
	}
	if live <= 0 {
		return math.NaN()
	}
	return inStall / live
}

func busySeconds(res *Result, window float64) (busy, decode float64) {
	for _, b := range res.Batches {
		if b.Start >= window {
			continue
		}
		end := minF(b.End, window)
		if end <= b.Start {
			continue
		}
		busy += end - b.Start
		if !b.IsPrefill {
			decode += end - b.Start
		}
	}
	return busy, decode
}

func ttfts(res *Result, sel func(*Request) bool) []float64 {
	var out []float64
	for _, r := range res.Requests {
		if sel(r) && r.FirstTok >= 0 {
			out = append(out, r.FirstTok-r.Arrival)
		}
	}
	return out
}

// FindTag returns the scripted request with this tag, or nil.
func FindTag(res *Result, tag string) *Request {
	for _, r := range res.Requests {
		if r.Tag == tag {
			return r
		}
	}
	return nil
}

// FindWindow returns the cold window a tagged prompt opened.
func FindWindow(res *Result, tag string) (ColdWindow, bool) {
	for _, w := range res.Windows {
		if w.Tag == tag {
			return w, true
		}
	}
	return ColdWindow{}, false
}

func pct(v []float64, p float64) float64 {
	if len(v) == 0 {
		return math.NaN()
	}
	c := append([]float64(nil), v...)
	sort.Float64s(c)
	idx := p / 100 * float64(len(c)-1)
	lo := int(math.Floor(idx))
	hi := int(math.Ceil(idx))
	return c[lo] + (c[hi]-c[lo])*(idx-float64(lo))
}

func maxOf(v []float64) float64 {
	m := math.NaN()
	for _, x := range v {
		if math.IsNaN(m) || x > m {
			m = x
		}
	}
	return m
}

func sum(v []float64) float64 {
	t := 0.0
	for _, x := range v {
		t += x
	}
	return t
}
