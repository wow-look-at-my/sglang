package tracetest

import "time"

// Prompt is one request the log prefilled in chunks, as the generator wrote it.
type Prompt struct {
	At time.Time
	// Hit is the prefix the first chunk reused, Tokens the work the chunks computed.
	Hit, Tokens, Chunks int
	Running             int
	// Seconds is the modelled time of every chunk, which is the wall clock the log advanced by.
	Seconds float64
}

// ColdChunks counts the prompt's full chunks that reused no prefix: all of
// them when the first chunk hit nothing.
func (p Prompt) ColdChunks(chunk int) int {
	if p.Hit > 0 {
		return (p.Tokens - chunk) / chunk
	}
	return p.Tokens / chunk
}

// CtxStart is the context the prompt's run of cold chunks extends from.
func (p Prompt) CtxStart(chunk int) int {
	if p.Hit > 0 {
		return p.Hit + chunk
	}
	return 0
}

// Prompt writes a request of tokens new tokens on top of a hit-token cached
// prefix. Up to a chunk it is one line; past that the first line carries the
// hit and a full chunk and every later chunk reuses nothing, with #pending-token
// counting what is left. A prompt of more than one chunk is recorded in
// l.Prompts.
func (l *Log) Prompt(m Model, hit, tokens, chunk, running int) *Log {
	if tokens <= chunk {
		tps := m.PrefillTPS(Extend{Tokens: tokens, Ctx: hit})
		return l.Prefill(Prefill{NewTokens: tokens, Hit: hit, Running: running, Usage: 0.4, TPS: tps})
	}
	p := Prompt{At: l.Now, Hit: hit, Tokens: tokens, Chunks: (tokens + chunk - 1) / chunk, Running: running}
	start := l.Now
	tps := m.PrefillTPS(Extend{Tokens: chunk, Ctx: hit})
	l.Prefill(Prefill{NewTokens: chunk, Hit: hit, Running: running, Pending: tokens - chunk, Usage: 0.4, TPS: tps})
	l.ColdChunks(m, tokens-chunk, hit+chunk, chunk, running, 0, 0)
	p.Seconds = l.Now.Sub(start).Seconds()
	l.Prompts = append(l.Prompts, p)
	return l
}

// TrafficSpec lays out one boot of agent traffic in the production format.
// Convs conversations open on a Shared-token system prompt, conversation c's
// first turn adding Open+c*OpenStep tokens.
type TrafficSpec struct {
	Worker string
	Start  time.Time
	Args   Args
	Model  Model
	// JIT lists the seconds of each kernel compile logged after server_args.
	JIT []float64

	Convs, Shared, Open, OpenStep int
	Rounds, TurnNew, Reply        int
	Think                         float64
	// Returns lists rounds in which conversation round%Convs comes back with its prefix evicted: a cold prompt of its whole context.
	Returns []int
	// Long lists rounds in which the last conversation's turn is LongTurn tokens, chunked behind its cached prefix.
	Long     []int
	LongTurn int
	// Colds maps a round to a fresh cold prompt of that many tokens sent after it.
	Colds map[int]int
	// End is how the boot ends: "sigterm", "crash" or "" for a log that stops.
	End string
}

// Traffic is a boot the generator wrote, with what it wrote.
type Traffic struct {
	*Log
	Spec TrafficSpec
	// Turns counts every request; Convs the conversations; Returns the evicted conversations that came back.
	Turns, Convs, Returns, FreshColds int
	// ColdTurns counts requests at most the system prompt reused that are chunks or longer.
	ColdTurns int
	// Measured counts turns a later turn of the same conversation followed.
	Measured int
}

// NewTraffic writes the boot the spec describes.
func NewTraffic(s TrafficSpec) *Traffic {
	m := s.Model
	chunk := s.Args.ChunkedPrefillSize
	t := &Traffic{Log: New(s.Worker, s.Start).ServerArgs(s.Args), Spec: s, Convs: s.Convs}
	l := t.Log
	l.Advance(1)
	for _, sec := range s.JIT {
		l.JIT("_fwd_kernel", sec)
	}
	ctx := make([]int, s.Convs)
	running := 0
	held := func() int {
		sum := 0
		for _, c := range ctx {
			sum += c
		}
		return sum
	}
	has := func(xs []int, r int) bool {
		for _, x := range xs {
			if x == r {
				return true
			}
		}
		return false
	}
	request := func(hit, tokens int) {
		t.Turns++
		if hit <= s.Shared && hit+tokens >= 8*chunk {
			t.ColdTurns++
		}
		l.Prompt(m, hit, tokens, chunk, running)
	}
	decode := func() {
		for i := 0; i < 2; i++ {
			l.DecodeAt(m, running, held(), Accepts(t.Turns+i))
		}
		l.Completion()
	}
	for r := 0; r < s.Rounds; r++ {
		for c := 0; c < s.Convs; c++ {
			switch {
			case r == 0:
				n := s.Open + c*s.OpenStep
				request(s.Shared, n)
				ctx[c] = s.Shared + n
				running++
			case has(s.Returns, r) && c == r%s.Convs:
				n := ctx[c] + s.Reply + s.TurnNew
				request(0, n)
				ctx[c] = n
				t.Returns++
				t.Measured++
			case has(s.Long, r) && c == s.Convs-1:
				request(ctx[c]+s.Reply, s.LongTurn)
				ctx[c] += s.Reply + s.LongTurn
				t.Measured++
			default:
				request(ctx[c]+s.Reply, s.TurnNew)
				ctx[c] += s.Reply + s.TurnNew
				t.Measured++
			}
			decode()
		}
		if n, ok := s.Colds[r]; ok {
			request(0, n)
			t.FreshColds++
			decode()
		}
		l.Advance(s.Think)
	}
	switch s.End {
	case "sigterm":
		l.Sigterm()
	case "crash":
		l.Crash()
	}
	return t
}

// DefaultTraffic is a boot of conversations whose contexts stay far apart, so
// every turn names its conversation unambiguously: a returning prompt, a long
// turn on a cached prefix and fresh cold prompts larger than any conversation.
func DefaultTraffic(worker string, at time.Time) TrafficSpec {
	args := DefaultArgs
	args.ContextLength = 524288
	return TrafficSpec{
		Worker: worker, Start: at, Args: args, Model: DefaultModel,
		Convs: 6, Shared: 12288, Open: 28000, OpenStep: 40000,
		Rounds: 6, TurnNew: 1200, Reply: 400, Think: 6,
		Returns: []int{2, 4}, Long: []int{3}, LongTurn: 36000,
		Colds: map[int]int{1: 360000, 3: 380000},
		End:   "sigterm",
	}
}

// Corpus is a deployment's log across restarts: a boot at the shorter context
// length that crashes, a boot that compiles kernels while serving and drains,
// a hierarchical-cache boot carrying the longest cold prompt, and a boot the
// log stops in the middle of. Each boot starts after the last one's final line.
func Corpus() []*Traffic {
	crash := DefaultTraffic("w0-crash", Start)
	crash.Args.ContextLength = 262144
	crash.Rounds, crash.Returns, crash.Long, crash.Colds = 3, []int{2}, nil, map[int]int{1: 300000}
	crash.End = "crash"

	drain := DefaultTraffic("w1-drain", Start)
	drain.JIT = []float64{2.1, 0.8}

	hicache := DefaultTraffic("w2-hicache", Start)
	hicache.Args.HierarchicalCache = true
	hicache.Rounds, hicache.Returns = 8, []int{2, 4, 7}
	hicache.Colds = map[int]int{1: 360000, 5: 440000}

	stops := DefaultTraffic("w3-stops", Start)
	stops.Rounds, stops.Returns, stops.Long, stops.Colds, stops.End = 2, nil, nil, nil, ""

	var out []*Traffic
	at := Start
	for _, s := range []TrafficSpec{crash, drain, hicache, stops} {
		s.Start = at
		t := NewTraffic(s)
		out = append(out, t)
		at = t.Now.Add(30 * time.Second)
	}
	return out
}

// Text joins the boots into one log.
func Text(boots []*Traffic) string {
	s := ""
	for _, b := range boots {
		s += b.String()
	}
	return s
}
