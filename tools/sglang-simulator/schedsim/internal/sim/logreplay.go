package sim

import (
	"fmt"
	"sort"
	"time"

	"schedsim/internal/trace"
)

// A serving log records what every client sent and when, one prefill line per
// arrival.

// LogTurn is one request as the log recorded it.
type LogTurn struct {
	Conv int
	// At is the arrival in seconds after the replay's start.
	At    float64
	Input int
	// Out is the reply length: measured from the next turn's prefix when there is one, the median reply otherwise.
	Out      int
	Hit      int
	Kind     Kind
	Line     int
	Measured bool // Out came from the next turn, not the median
	// Returning marks a cold request that matched an idle conversation.
	Returning bool
}

// LogReplay is a Workload built from one boot.
type LogReplay struct {
	Turns []LogTurn
	// Window is the replayed span in seconds and Start its wall-clock origin.
	Window float64
	Start  time.Time
	Shared int
	Convs  int
	// Returning counts the cold turns that matched an idle conversation: the log's own count of full-prefix recomputes.
	Returning int

	byConv map[int][]int
	next   map[int]int
}

// The matching tolerances.
const (
	maxReply  = 8192
	returnTol = 0.03
)

// Requests before the window are not replayed, so a conversation that began
// there looks new at its first turn inside it.
func BuildLogReplay(b *trace.Boot, chunkSize int, skipSeconds, maxSeconds float64) (*LogReplay, error) {
	skip := skipSeconds
	if !b.Timestamped() {
		return nil, fmt.Errorf("boot %d has no timestamps to replay against", b.Index)
	}
	steps := b.Steps
	var start time.Time
	for _, s := range steps {
		if s.Kind == trace.Prefill {
			start = s.At
			break
		}
	}
	if start.IsZero() {
		return nil, fmt.Errorf("boot %d has no prefill step", b.Index)
	}
	l := &LogReplay{Start: start.Add(time.Duration(skip * float64(time.Second))), byConv: map[int][]int{}, next: map[int]int{}}
	l.Shared = sharedPrefix(steps)

	// convs holds each conversation's latest turn (an index into l.Turns).
	var convs []*struct{ lastTurn int }
	var outs []int
	// A prompt longer than one chunk is logged as its first line followed by full chunks and a final partial one.
	chain := false
	prefill := prefillSteps(steps)
	ctxCap := b.Args.ContextLength
	for pi, s := range prefill {
		at := s.At.Sub(start).Seconds()
		if at < skip {
			chain = s.Pending > 0 && s.NewTokens == chunkSize
			continue
		}
		if maxSeconds > 0 && at-skip > maxSeconds {
			break
		}
		if chain && s.NewSeq == 1 && s.HitTokens == 0 {
			chain = s.Pending > 0 && s.NewTokens == chunkSize
			continue
		}
		chain = s.Pending > 0 && s.NewTokens == chunkSize
		rest := chainTokens(prefill, pi, chunkSize)
		arrivals := splitArrivals(s, chunkSize, rest)
		// A line admitting several requests does not itemise their prefix hits, but
		// each follow-up is some conversation's next turn: the most recently idle
		// conversations whose last prompts add up to the line's hit are the ones,
		// and each's hit is its own last prompt.
		if len(arrivals) > 1 {
			var followUps []int
			for k, arr := range arrivals {
				if arr.pending == 0 {
					followUps = append(followUps, k)
				}
			}
			idle := idleConvs(l, convs, at, len(followUps))
			sum := 0
			for _, ci := range idle {
				sum += l.Turns[convs[ci].lastTurn].Input
			}
			hit := s.HitTokens
			if len(arrivals) > len(followUps) {
				hit -= arrivals[0].hit
			}
			if len(idle) == len(followUps) && absF(float64(sum-hit)) <= 0.15*float64(hit)+float64(chunkSize) {
				for j, k := range followUps {
					last := l.Turns[convs[idle[j]].lastTurn].Input
					arrivals[k].hit = last
					arrivals[k].conv = idle[j]
				}
			}
		}
		for _, arr := range arrivals {
			t := LogTurn{At: at - skip, Input: arr.hit + arr.nw + arr.pending, Hit: arr.hit, Line: s.Line, Conv: -1}
			if ctxCap > 0 && t.Input > ctxCap {
				t.Input = ctxCap
			}
			if t.Input <= 0 {
				continue
			}
			if arr.conv >= 0 {
				// Assigned above: its reply is the line's own new tokens' share less nothing we can see, so the median stands in.
				t.Conv = arr.conv
			}
			// A prefix hit beyond the shared prompt names the conversation
			// whose last prompt it extends; the overshoot is that turn's reply.
			if t.Conv < 0 && t.Hit > l.Shared {
				best, bestD := -1, maxReply+1
				for ci, c := range convs {
					last := l.Turns[c.lastTurn]
					d := t.Hit - last.Input
					if d >= 0 && d < bestD && last.At < at {
						best, bestD = ci, d
					}
				}
				if best >= 0 {
					last := &l.Turns[convs[best].lastTurn]
					last.Out, last.Measured = bestD, true
					outs = append(outs, bestD)
					t.Conv = best
				}
			}
			// A cold request the size of an idle conversation is that
			// conversation back after its prefix was evicted. Its size is the
			// last prompt plus a reply, which the log has not stated yet, so
			// the match allows a reply's worth of slack.
			if t.Conv < 0 && t.Hit <= l.Shared && t.Input >= 8*chunkSize {
				best, bestD := -1, 0.0
				for ci, c := range convs {
					last := l.Turns[c.lastTurn]
					if last.At >= at-2 {
						continue
					}
					d := float64(t.Input - last.Input)
					if d < 0 {
						d = -d * 4 // a shrunken prompt is a different one
					} else if d > maxReply {
						d -= maxReply
					} else {
						d = 0
					}
					rel := d / float64(t.Input)
					if rel <= returnTol && (best < 0 || rel < bestD) {
						best, bestD = ci, rel
					}
				}
				if best >= 0 {
					last := &l.Turns[convs[best].lastTurn]
					if d := t.Input - last.Input; d >= 0 && d <= maxReply {
						last.Out, last.Measured = d, true
						outs = append(outs, d)
					}
					t.Conv, t.Returning = best, true
					l.Returning++
				}
			}
			if t.Conv < 0 {
				t.Conv = len(convs)
				convs = append(convs, &struct{ lastTurn int }{})
			}
			if t.Hit <= l.Shared && t.Input >= 8*chunkSize {
				t.Kind = KindCold
			} else {
				t.Kind = KindAgent
			}
			convs[t.Conv].lastTurn = len(l.Turns)
			l.Turns = append(l.Turns, t)
			l.byConv[t.Conv] = append(l.byConv[t.Conv], convs[t.Conv].lastTurn)
		}
	}
	if len(l.Turns) == 0 {
		return nil, fmt.Errorf("boot %d has no requests in the window", b.Index)
	}
	med := medianInt(outs, 300)
	for i := range l.Turns {
		if !l.Turns[i].Measured {
			l.Turns[i].Out = med
		}
		if l.Turns[i].Out < 1 {
			l.Turns[i].Out = 1
		}
	}
	l.Convs = len(convs)
	l.Window = l.Turns[len(l.Turns)-1].At
	if maxSeconds > 0 && maxSeconds < l.Window {
		l.Window = maxSeconds
	}
	return l, nil
}

type arrival struct {
	hit, nw, pending int
	conv             int
}

// splitArrivals divides a batch line among the requests it admitted. One
// line is one request; a line with #new-seq n and pending tokens holds one
// chunked prompt that took the chunk and n-1 follow-ups that took the rest,
// which the log does not itemise, so they share it evenly. rest is the
// chunked prompt's remaining tokens, summed from the lines that follow.
func splitArrivals(s trace.Step, chunkSize, rest int) []arrival {
	if s.NewSeq <= 1 {
		return []arrival{{s.HitTokens, s.NewTokens, rest, -1}}
	}
	big := arrival{nw: minI(s.NewTokens, chunkSize), pending: rest, conv: -1}
	restNew := s.NewTokens - big.nw
	restHit := s.HitTokens
	if s.Pending == 0 {
		// No chunked prompt: n requests sharing the line evenly.
		out := make([]arrival, s.NewSeq)
		for i := range out {
			out[i] = arrival{s.HitTokens / s.NewSeq, s.NewTokens / s.NewSeq, 0, -1}
		}
		return out
	}
	out := []arrival{big}
	for i := 1; i < s.NewSeq; i++ {
		out = append(out, arrival{restHit / (s.NewSeq - 1), restNew / (s.NewSeq - 1), 0, -1})
	}
	return out
}

// idleConvs picks the n conversations whose last turn is the most recent
// among those that arrived before at, in that order.
func idleConvs(l *LogReplay, convs []*struct{ lastTurn int }, at float64, n int) []int {
	type cand struct {
		ci int
		at float64
	}
	var cs []cand
	for ci, c := range convs {
		if t := l.Turns[c.lastTurn]; t.At < at {
			cs = append(cs, cand{ci, t.At})
		}
	}
	sort.Slice(cs, func(i, j int) bool { return cs[i].at > cs[j].at })
	var out []int
	for i := 0; i < len(cs) && i < n; i++ {
		out = append(out, cs[i].ci)
	}
	return out
}

// prefillSteps is the boot's prefill lines alone.
func prefillSteps(steps []trace.Step) []trace.Step {
	var out []trace.Step
	for _, s := range steps {
		if s.Kind == trace.Prefill {
			out = append(out, s)
		}
	}
	return out
}

// chainTokens sums the chunks that continue the prompt started at prefill[i]:
// the following lines with one sequence and no prefix hit, up to and
// including the first that is not a full chunk or leaves nothing pending.
func chainTokens(prefill []trace.Step, i, chunkSize int) int {
	if prefill[i].Pending == 0 || prefill[i].NewTokens != chunkSize {
		return 0
	}
	sum := 0
	for j := i + 1; j < len(prefill); j++ {
		s := prefill[j]
		if s.NewSeq != 1 || s.HitTokens != 0 {
			break
		}
		sum += s.NewTokens
		if s.Pending == 0 || s.NewTokens != chunkSize {
			break
		}
	}
	return sum
}

// sharedPrefix estimates the system prompt every conversation starts with:
// the most common small prefix hit, which is what a brand-new conversation
// reuses.
func sharedPrefix(steps []trace.Step) int {
	count := map[int]int{}
	for _, s := range steps {
		if s.Kind == trace.Prefill && s.HitTokens > 0 && s.HitTokens <= 32768 {
			count[s.HitTokens]++
		}
	}
	best, n := 0, 0
	for v, c := range count {
		if c > n || (c == n && v < best) {
			best, n = v, c
		}
	}
	if n < 3 {
		return 0
	}
	return best
}

func medianInt(xs []int, fallback int) int {
	if len(xs) == 0 {
		return fallback
	}
	s := append([]int(nil), xs...)
	sort.Ints(s)
	return s[len(s)/2]
}

// Initial releases each conversation's first turn at its measured arrival.
func (l *LogReplay) Initial() []*Request {
	var out []*Request
	for i, t := range l.Turns {
		if l.byConv[t.Conv][0] == i {
			out = append(out, l.request(i, t.At))
			l.next[t.Conv] = 1
		}
	}
	return out
}

// OnFinish releases the conversation's next turn at its measured arrival, or
// now if the reply it follows finished later than the log's own run did: a
// client cannot send its next turn before it has the answer.
func (l *LogReplay) OnFinish(r *Request, now float64) []*Request {
	k := l.next[r.Conv]
	turns := l.byConv[r.Conv]
	if k >= len(turns) {
		return nil
	}
	l.next[r.Conv] = k + 1
	i := turns[k]
	at := l.Turns[i].At
	if at < now {
		at = now
	}
	return []*Request{l.request(i, at)}
}

func (l *LogReplay) request(i int, at float64) *Request {
	t := l.Turns[i]
	tag := ""
	if t.Kind == KindCold {
		tag = fmt.Sprintf("line%d", t.Line)
	}
	return NewRequest(i+1, t.Kind, t.Conv, at, t.Input, t.Out, tag)
}

// Agents counts the conversations the replay holds.
func (l *LogReplay) Agents() int { return l.Convs }

// SharedPrefix is the system prompt resident for the whole run.
func (l *LogReplay) SharedPrefix() int { return l.Shared }

// Reset returns a fresh copy whose turn cursors start over, so one
// reconstruction can drive several runs.
func (l *LogReplay) Reset() *LogReplay {
	c := *l
	c.next = map[int]int{}
	return &c
}

// ColdTurns counts the turns replayed as cold prompts.
func (l *LogReplay) ColdTurns() int {
	n := 0
	for _, t := range l.Turns {
		if t.Kind == KindCold {
			n++
		}
	}
	return n
}

// MeasuredOuts counts the turns whose reply length the log itself gave.
func (l *LogReplay) MeasuredOuts() int {
	n := 0
	for _, t := range l.Turns {
		if t.Measured {
			n++
		}
	}
	return n
}
