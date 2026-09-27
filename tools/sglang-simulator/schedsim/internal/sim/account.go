package sim

import (
	"fmt"

	"schedsim/internal/trace"
)

// WindowAccount splits one cold prompt's wait into the GPU work and idle time
// that filled it. Every batch's seconds are charged in full and attributed
// proportionally to what the batch priced, so the five buckets sum to the window
// by construction; what the account proves is the *split*, which is why a longer
// wait is or is not something the policy bought on purpose.
type WindowAccount struct {
	Window       float64
	OwnPrefill   float64
	OtherPrefill float64
	DecodeRows   float64
	Decode       float64
	Idle         float64
	// Batches counts the forward passes that ran inside the window.
	Batches int
}

// Total is the charged GPU time, the complement of Idle.
func (a WindowAccount) Total() float64 {
	return a.OwnPrefill + a.OtherPrefill + a.DecodeRows + a.Decode
}

// String renders the account with its arithmetic closed.
func (a WindowAccount) String() string {
	return fmt.Sprintf("window %.2f s = own prefill %.2f + other prefill %.2f + decode rows %.2f"+
		" + decode %.2f + idle %.2f (residual %.3f, %d batches)",
		a.Window, a.OwnPrefill, a.OtherPrefill, a.DecodeRows, a.Decode, a.Idle,
		a.Window-a.Total()-a.Idle, a.Batches)
}

// AccountWindow prices the stretch from a cold prompt's arrival to its first
// token. The prompt's own prefill work is the part no policy can remove: the
// same chunks at the same contexts cost the same under every rule.
func AccountWindow(res *Result, win ColdWindow) WindowAccount {
	ac := WindowAccount{Window: win.FirstTok - win.Arrival}
	own := res.ColdRequest(win)
	if own == nil || ac.Window <= 0 {
		return ac
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
		for _, it := range b.Items {
			c := float64(it.Extend) * res.Cfg.Cost.SecondsPerToken(float64(it.PrefixBefore)+float64(it.Extend)/2)
			if it.Req == own {
				ownCost += c
			} else {
				otherCost += c
			}
		}
		for _, r := range b.Rows {
			rowCost += res.Cfg.Cost.SecondsPerToken(float64(r.Context()))
		}
		total := ownCost + otherCost + rowCost
		if total <= 0 {
			otherCost, total = 1, 1
		}
		// The batch's per-batch overhead and its reload copy ride with the work
		// that caused the batch to run at all, in proportion to their tokens.
		ac.OwnPrefill += secs * ownCost / total
		ac.OtherPrefill += secs * otherCost / total
		ac.DecodeRows += secs * rowCost / total
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

