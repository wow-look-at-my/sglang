package sim

import (
	"math"
	"sort"
)

type delivery struct {
	t   float64
	gap float64
	n   float64
}

type reqLog struct {
	req        *Request
	arrival    float64
	firstToken float64
	finish     float64
	deliveries []delivery
	recompute  bool
}

// Recorder collects what the metrics are computed from.
type Recorder struct {
	logs  map[*Request]*reqLog
	order []*reqLog

	End         float64
	Stuck       bool
	Retractions int

	PrefillSeconds, DecodeSeconds float64
	spans                         []span
}

type span struct {
	start, end float64
	isPrefill  bool
}

func newRecorder() *Recorder { return &Recorder{logs: map[*Request]*reqLog{}} }

func (rc *Recorder) arrived(r *Request) {
	if _, ok := rc.logs[r]; ok {
		return
	}
	l := &reqLog{req: r, arrival: r.Arrival, firstToken: math.NaN(), finish: math.NaN()}
	rc.logs[r] = l
	rc.order = append(rc.order, l)
}

// matched counts a returning conversation that lost at least half its
// previous context from both tiers.
func (rc *Recorder) matched(r *Request, hit int) {
	if r.Conv.Finished > 0 && 2*hit < r.Conv.Len {
		rc.logs[r].recompute = true
	}
}

func (rc *Recorder) launched(b *batch) {
	if b.isPrefill {
		rc.PrefillSeconds += b.end - b.start
	} else {
		rc.DecodeSeconds += b.end - b.start
	}
	rc.spans = append(rc.spans, span{b.start, b.end, b.isPrefill})
}

func (rc *Recorder) delivered(r *Request, t, n float64) {
	l := rc.logs[r]
	if math.IsNaN(l.firstToken) {
		l.firstToken = t
		r.lastToken = t
		l.deliveries = append(l.deliveries, delivery{t: t, n: n})
		return
	}
	l.deliveries = append(l.deliveries, delivery{t: t, gap: t - r.lastToken, n: n})
	r.lastToken = t
}

func (rc *Recorder) finished(r *Request) { rc.logs[r].finish = r.finish }

// Metrics are one run's outcome.
type Metrics struct {
	// StreamRate is tokens per second a decoding request receives while a cold.
	StreamRate float64
	// Stall is the longest gap between tokens of one stream.
	Stall float64
	// ColdTTFT is the mean cold-prompt TTFT, or the mean TTFT of all requests when the workload has no cold prompt.
	ColdTTFT float64
	ITLp99   float64
	ITLp999  float64
	// Throughput is output tokens per second over the run.
	Throughput float64
	// Recomputes counts returning turns whose cached context was lost.
	Recomputes int

	TurnTTFTp50, TurnTTFTp99 float64
	Completed                int
	Stuck                    bool

	// Gaps counts inter-token gaps over all streams.
	Gaps float64
	// Cold TTFT windows: length, decode and prefill seconds inside.
	WindowSeconds, WindowDecode, WindowPrefill, WindowStreams float64
}

// Metrics reduces the recorded run.
func (rc *Recorder) Metrics() Metrics {
	var m Metrics
	m.Stuck = rc.Stuck
	end := rc.End
	var windows [][2]float64
	var coldTTFT, allTTFT, turnTTFT []float64
	for _, l := range rc.order {
		ft := l.firstToken
		if math.IsNaN(ft) {
			ft = end
		}
		allTTFT = append(allTTFT, ft-l.arrival)
		if l.req.Kind == Cold {
			coldTTFT = append(coldTTFT, ft-l.arrival)
			windows = append(windows, [2]float64{l.arrival, ft})
		} else {
			turnTTFT = append(turnTTFT, ft-l.arrival)
		}
		if l.recompute {
			m.Recomputes++
		}
		if !math.IsNaN(l.finish) {
			m.Completed++
		}
	}
	if len(coldTTFT) > 0 {
		m.ColdTTFT = mean(coldTTFT)
	} else {
		m.ColdTTFT = mean(allTTFT)
	}
	m.TurnTTFTp50 = quantile(turnTTFT, 0.5)
	m.TurnTTFTp99 = quantile(turnTTFT, 0.99)
	if len(windows) == 0 {
		windows = [][2]float64{{0, end}}
	}
	windows = mergeWindows(windows)

	var gaps []float64
	var total, streamTokens, streamSecs, decodeSecs float64
	for _, l := range rc.order {
		for _, d := range l.deliveries {
			if d.t <= end {
				total += d.n
			}
			if d.gap > 0 {
				gaps = append(gaps, d.gap)
				m.Stall = math.Max(m.Stall, d.gap)
			}
		}
		if math.IsNaN(l.firstToken) {
			continue
		}
		stop := end
		if !math.IsNaN(l.finish) {
			stop = l.finish
		} else {
			// A stream still waiting at the end is stalled until then.
			m.Stall = math.Max(m.Stall, end-l.req.lastToken)
		}
		decodeSecs += overlap(windows, l.firstToken, stop)
		if l.req.Kind == Cold {
			continue
		}
		streamSecs += overlap(windows, l.firstToken, stop)
		for _, d := range l.deliveries[1:] {
			if inWindows(windows, d.t) {
				streamTokens += d.n
			}
		}
	}
	if streamSecs > 0 {
		m.StreamRate = streamTokens / streamSecs
	}
	if end > 0 {
		m.Throughput = total / end
	}
	m.ITLp99 = quantile(gaps, 0.99)
	m.ITLp999 = quantile(gaps, 0.999)
	m.Gaps = float64(len(gaps))
	if len(coldTTFT) > 0 {
		for _, w := range windows {
			m.WindowSeconds += w[1] - w[0]
		}
		for _, s := range rc.spans {
			if s.isPrefill {
				m.WindowPrefill += overlap(windows, s.start, s.end)
			} else {
				m.WindowDecode += overlap(windows, s.start, s.end)
			}
		}
		m.WindowStreams = decodeSecs / m.WindowSeconds
	}
	return m
}

// Mean averages every metric over runs; Stuck is true if any run stuck.
func Mean(runs []Metrics) Metrics {
	var m Metrics
	n := float64(len(runs))
	for _, r := range runs {
		m.StreamRate += r.StreamRate / n
		m.Stall += r.Stall / n
		m.ColdTTFT += r.ColdTTFT / n
		m.ITLp99 += r.ITLp99 / n
		m.ITLp999 += r.ITLp999 / n
		m.Throughput += r.Throughput / n
		m.Recomputes += r.Recomputes
		m.TurnTTFTp50 += r.TurnTTFTp50 / n
		m.TurnTTFTp99 += r.TurnTTFTp99 / n
		m.Completed += r.Completed
		m.Stuck = m.Stuck || r.Stuck
		m.Gaps += r.Gaps / n
		m.WindowSeconds += r.WindowSeconds / n
		m.WindowDecode += r.WindowDecode / n
		m.WindowPrefill += r.WindowPrefill / n
		m.WindowStreams += r.WindowStreams / n
	}
	return m
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func quantile(xs []float64, q float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	i := int(math.Ceil(q*float64(len(s)))) - 1
	return s[max(0, min(i, len(s)-1))]
}

func mergeWindows(ws [][2]float64) [][2]float64 {
	sort.Slice(ws, func(i, j int) bool { return ws[i][0] < ws[j][0] })
	out := [][2]float64{ws[0]}
	for _, w := range ws[1:] {
		last := &out[len(out)-1]
		if w[0] <= last[1] {
			last[1] = math.Max(last[1], w[1])
			continue
		}
		out = append(out, w)
	}
	return out
}

func overlap(ws [][2]float64, a, b float64) float64 {
	s := 0.0
	for _, w := range ws {
		lo, hi := math.Max(a, w[0]), math.Min(b, w[1])
		if hi > lo {
			s += hi - lo
		}
	}
	return s
}

func inWindows(ws [][2]float64, t float64) bool {
	for _, w := range ws {
		if t > w[0] && t <= w[1] {
			return true
		}
	}
	return false
}
