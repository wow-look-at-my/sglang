package sim

import (
	"fmt"

	"schedsim/internal/trace"
)

// Exclusion names why a cold window cannot be priced at all, as opposed to being priced and coming out unfavourable.
type Exclusion int

const (
	// Priced: the window has an arrival, a first token and a matching request.
	Priced Exclusion = iota
	// Unserved: the run ended before the prompt reached its first token.
	Unserved
	// Orphan: no request in the run opened this window.
	Orphan
	// Empty: the first token is not after the arrival, which no served prompt can produce.
	Empty
)

func (e Exclusion) String() string {
	switch e {
	case Unserved:
		return "unserved"
	case Orphan:
		return "orphan"
	case Empty:
		return "empty"
	default:
		return "priced"
	}
}

// NumExclusions is one past the last Exclusion, for an array indexed by them.
const NumExclusions = int(Empty) + 1

// WindowAccount splits one cold prompt's wait into the GPU work and idle time
// that filled it.
type WindowAccount struct {
	Window     float64
	OwnPrefill float64
	// OtherPrefill is every other request's extend work in the window's batches.
	OtherPrefill float64
	// DecodeRows is the work a mixed batch does for the running streams.
	DecodeRows float64
	// Overhead is what a forward pass costs without computing anything.
	Overhead float64
	Decode   float64
	Idle     float64
	// Batches counts the forward passes that ran inside the window.
	Batches int
	// OwnBatches and OwnTokens describe the prompt's own prefill: how many forward passes carried its chunks.
	OwnBatches int
	OwnTokens  int
	// Skip is Unserved, Orphan or Empty when the window has no interval to price.
	Skip Exclusion
}

// Measurable reports whether the account covers a window that exists.
func (a WindowAccount) Measurable() bool { return a.Skip == Priced }

// Total is the charged GPU time, the complement of Idle.
func (a WindowAccount) Total() float64 {
	return a.OwnPrefill + a.OtherPrefill + a.DecodeRows + a.Overhead + a.Decode
}

// String renders the account with its arithmetic closed.
func (a WindowAccount) String() string {
	if !a.Measurable() {
		return fmt.Sprintf("%s cold prompt, no interval to price", a.Skip)
	}
	return fmt.Sprintf("window %.2f s = own prefill %.2f (%d passes, %d tokens) + other prefill %.2f"+
		" + decode rows %.2f + overhead %.2f + decode %.2f + idle %.2f"+
		" (residual %.3f, %d batches)",
		a.Window, a.OwnPrefill, a.OwnBatches, a.OwnTokens,
		a.OtherPrefill, a.DecodeRows, a.Overhead, a.Decode, a.Idle,
		a.Window-a.Total()-a.Idle, a.Batches)
}

// AccountWindow prices the stretch from a cold prompt's arrival to its first
// token. The prompt's own prefill work is the part no policy can remove: the
// same chunks at the same contexts cost the same under every rule. A window the
// run never served is excluded with its reason rather than priced from a
// first-token value the run never recorded.
func AccountWindow(res *Result, win ColdWindow) WindowAccount {
	if !win.Done() {
		return WindowAccount{Skip: Unserved}
	}
	ac := WindowAccount{Window: win.FirstTok - win.Arrival}
	own := res.ColdRequest(win)
	if own == nil {
		return WindowAccount{Window: ac.Window, Skip: Orphan}
	}
	if ac.Window <= 0 {
		return WindowAccount{Window: ac.Window, Skip: Empty}
	}
	for _, b := range res.Batches {
		if b.End <= win.Arrival || b.Start >= win.FirstTok {
			continue
		}
		lo, hi := maxF(b.Start, win.Arrival), minF(b.End, win.FirstTok)
		secs := hi - lo
		ac.Batches++
		if !b.IsPrefill {
			ac.Decode += secs
			continue
		}
		ownCost, otherCost, rowCost := 0.0, 0.0, 0.0
		ownTokens := 0
		for _, it := range b.Items {
			c := float64(it.Extend) * res.Cfg.Cost.SecondsPerToken(float64(it.PrefixBefore)+float64(it.Extend)/2)
			if it.Req == own {
				ownCost += c
				ownTokens += it.Extend
			} else {
				otherCost += c
			}
		}
		if ownTokens > 0 {
			ac.OwnBatches++
			ac.OwnTokens += ownTokens
		}
		for _, r := range b.Rows {
			rowCost += res.Cfg.Cost.SecondsPerToken(float64(r.Context()))
		}
		total := ownCost + otherCost + rowCost
		if total <= 0 {
			otherCost, total = 1, 1
		}
		// A pass costs its tokens plus a fixed overhead and any reload copy it made before its kernels.
		overhead := b.ReloadSeconds + res.Cfg.Cost.PerBatchSeconds()
		if k := secs / (total + overhead); k > 0 {
			ac.OwnPrefill += k * ownCost
			ac.OtherPrefill += k * otherCost
			ac.DecodeRows += k * rowCost
			ac.Overhead += k * overhead
		}
	}
	ac.Idle = ac.Window - ac.Total()
	if ac.Idle < 0 {
		ac.Idle = 0
	}
	return ac
}

// ColdRequest returns the request that opened a cold window.
func (r *Result) ColdRequest(win ColdWindow) *Request {
	for _, req := range r.Requests {
		if req.Kind == KindCold && req.Arrival == win.Arrival && req.Tag == win.Tag {
			return req
		}
	}
	return nil
}

// WindowCensus tallies one run's cold windows by whether they can be priced.
type WindowCensus struct {
	Arrived int
	Priced  int
	Counts  [4]int
}

// Census tallies the cold windows of one run.
func Census(res *Result) WindowCensus {
	c := WindowCensus{Arrived: len(res.Windows)}
	for _, w := range res.Windows {
		a := AccountWindow(res, w)
		c.Counts[a.Skip]++
		if a.Measurable() {
			c.Priced++
		}
	}
	return c
}

// String renders the census with each exclusion reason beside its count.
func (c WindowCensus) String() string {
	s := fmt.Sprintf("%d/%d windows priced", c.Priced, c.Arrived)
	for _, e := range []Exclusion{Unserved, Orphan, Empty} {
		if c.Counts[e] > 0 {
			s += fmt.Sprintf(", %d %s", c.Counts[e], e)
		}
	}
	return s
}

// StreamSeconds is the time every stream in the run spent past its first token
// and inside the window: the denominator a throughput has to be divided by before
// it can be compared between policies that admit streams at different times.
func StreamSeconds(res *Result, window float64) float64 {
	total := 0.0
	for _, r := range res.Requests {
		if r.FirstTok < 0 || r.FirstTok >= window {
			continue
		}
		end := r.Finish
		if end < 0 || end > window {
			end = window
		}
		if end > r.FirstTok {
			total += end - r.FirstTok
		}
	}
	return total
}

// DeliveredTokens counts the tokens streamed inside the window.
func DeliveredTokens(res *Result, window float64) int {
	out := 0
	for _, r := range res.Requests {
		for _, d := range r.Deliveries {
			if d.T <= window {
				out += d.N
			}
		}
	}
	return out
}

// BatchSeconds lists the GPU duration of every batch that finished inside the
// window, which is what bounds a stream's longest possible inter-token gap.
func (r *Result) BatchSeconds(window float64) (prefill, decode, rows []float64) {
	for _, b := range r.Batches {
		if b.End > window {
			continue
		}
		if b.IsPrefill {
			prefill = append(prefill, b.End-b.Start)
			if len(b.Rows) > 0 {
				rows = append(rows, b.End-b.Start)
			}
			continue
		}
		decode = append(decode, b.End-b.Start)
	}
	return prefill, decode, rows
}

// ChunkCostSeconds is the calibrated cost of the deployment's full chunk at one
// prefix, the number a single forward pass can hold a stream silent for.
func (c Cost) ChunkCostSeconds(prefix int) float64 {
	return c.Cal.Prefill.PrefillSeconds([]trace.ExtendItem{
		{Tokens: c.Cal.ChunkSize, MidCtx: float64(prefix) + float64(c.Cal.ChunkSize)/2},
	})
}

// SingleTokenSamples counts the per-token inter-token samples that came from a
// one-token delivery, and the total sample count, for the streams the ITL
// percentiles are measured over: with speculative decoding a delivery of N tokens
// contributes N samples of gap/N, so a delivery's share of a percentile depends on
// how many tokens it carried, not only on how long the wait was.
func SingleTokenSamples(res *Result, window float64) (single, total int, longest float64) {
	for _, r := range res.Requests {
		if len(r.Deliveries) < 2 {
			continue
		}
		for i := 1; i < len(r.Deliveries); i++ {
			d := r.Deliveries[i]
			if d.T > window {
				break
			}
			n := d.N
			if n < 1 {
				n = 1
			}
			total += n
			if n == 1 {
				single++
				if g := d.T - r.Deliveries[i-1].T; g > longest {
					longest = g
				}
			}
		}
	}
	return single, total, longest
}
