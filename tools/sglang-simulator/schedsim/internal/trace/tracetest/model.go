package tracetest

import "math"

// Model is the cost the generated batch lines report.
type Model struct {
	PrefillBase, PerToken, PerTokenCtx, PerTokenCtxSq float64
	DecodeBase, DecodePerReq, DecodePerTokenCtx       float64
	Pool                                              int
}

// DefaultModel is the cost of the long-context deployment the scenarios describe.
var DefaultModel = Model{
	PrefillBase: 10e-3, PerToken: 68.403e-6, PerTokenCtx: 4.72e-11, PerTokenCtxSq: 6.05e-16,
	DecodeBase: 13e-3, DecodePerReq: 1.5e-3, DecodePerTokenCtx: 6e-9,
	Pool: 1_406_118,
}

// Extend is one request's share of a prefill batch: Tokens computed new on
// top of Ctx tokens already in the KV cache.
type Extend struct {
	Tokens, Ctx int
}

// PrefillSeconds is the modelled duration of a batch of extends.
func (m Model) PrefillSeconds(items ...Extend) float64 {
	t := m.PrefillBase
	for _, it := range items {
		mid := float64(it.Ctx) + float64(it.Tokens)/2
		t += float64(it.Tokens) * (m.PerToken + m.PerTokenCtx*mid + m.PerTokenCtxSq*mid*mid)
	}
	return t
}

// PrefillTPS is the input throughput a batch of extends reports.
func (m Model) PrefillTPS(items ...Extend) float64 {
	n := 0
	for _, it := range items {
		n += it.Tokens
	}
	return float64(n) / m.PrefillSeconds(items...)
}

// DecodeSeconds is the modelled duration of one decode step.
func (m Model) DecodeSeconds(running, fullTokens int) float64 {
	return m.DecodeBase + m.DecodePerReq*float64(running-1) + m.DecodePerTokenCtx*float64(fullTokens)
}

// DecodeAt writes a decode line for running requests holding fullTokens,
// priced by m. The server prints usage to decimals, so FullTokens is the
// count that rounded usage implies, which keeps every line's ratio at the
// pool. Gen throughput is every request's accepted tokens over the step, and
// the clock advances by one log interval of steps.
func (l *Log) DecodeAt(m Model, running, fullTokens int, accept float64) *Log {
	usage := math.Round(100*float64(fullTokens)/float64(m.Pool)) / 100
	if usage < 0.01 {
		usage = 0.01
	}
	full := int(usage*float64(m.Pool) + 0.5)
	step := m.DecodeSeconds(running, full)
	l.Line(DecodeBody(Decode{Running: running, FullTokens: full, Usage: usage, Accept: accept, TPS: float64(running) * accept / step}))
	return l.Advance(40 * step)
}

// ColdChunks writes a prompt of tokens prefilled from context ctx0 in chunks:
// every full chunk alone, with no prefix hit, and a final partial chunk when
// tokens is not a multiple of chunk. pendingAfter is what the queue holds
// besides this prompt; each line reports it plus what is left of the prompt.
// Chunk durations come from m, and the clock advances by each.
func (l *Log) ColdChunks(m Model, tokens, ctx0, chunk, running, queue, pendingAfter int) *Log {
	done := 0
	for done < tokens {
		n := chunk
		if tokens-done < n {
			n = tokens - done
		}
		left := tokens - done - n
		tps := m.PrefillTPS(Extend{Tokens: n, Ctx: ctx0 + done})
		l.Line(PrefillBody(Prefill{NewTokens: n, Running: running, Queue: queue, Pending: left + pendingAfter, Usage: 0.5, TPS: tps}))
		l.Advance(float64(n) / tps)
		done += n
	}
	return l
}

// Accepts cycles through the accepted-token counts a speculative decoder with drafts reports.
func Accepts(i int) float64 {
	return []float64{2.70, 2.74, 2.78, 2.81, 2.76}[i%5]
}
