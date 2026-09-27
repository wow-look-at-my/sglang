package sim

// Kind separates the traffic classes the scenarios mix, because the metrics are
// defined per class: cold prompts measure time to first token, agent turns
// measure the stall an in-flight stream suffers.
type Kind int

const (
	// KindAgent is one turn of a closed-loop conversation: its previous context
	// is in the prefix cache unless the pool thrashed it away.
	KindAgent Kind = iota
	// KindCold is a one-shot prompt with no cached prefix at all.
	KindCold
	// KindShort is a one-shot short-chat prompt.
	KindShort
)

func (k Kind) String() string {
	switch k {
	case KindAgent:
		return "agent"
	case KindCold:
		return "cold"
	default:
		return "short"
	}
}

// Delivery is one streamed chunk: N tokens observed at time T. With MTP a chunk
// carries 1 to NumDraft+1 tokens, so inter-token gaps must be spread over the
// token count to compare against a per-token latency claim.
type Delivery struct {
	T float64
	N int
}

// Request is one request's life through the simulation.
type Request struct {
	ID   int
	Kind Kind
	// Conv is the conversation the request belongs to; the prefix cache is keyed
	// by it, so a one-shot prompt owns its own id.
	Conv int

	Arrival   float64
	InputLen  int
	OutTarget int

	// Prefix counts the tokens already computed for this request: its cache hit
	// at arrival plus every chunk of its own prefill since.
	Prefix int
	// OutDone counts the tokens this request has streamed.
	OutDone int
	// Reserved is the pool footprint admission took.
	Reserved int
	// ReloadToks are the input tokens that came back from the host tier.
	ReloadToks int

	// AcceptQ is the per-draft acceptance probability drawn for this request.
	AcceptQ float64

	PrefillStart float64
	FirstTok     float64
	Finish       float64
	Deliveries   []Delivery

	// ArrivedDuringCold is set at queueing: a cold prefill was in flight or
	// queued when this request joined the waiting queue.
	ArrivedDuringCold bool

	// Win indexes this request's entry in the run's cold windows, -1 for a
	// request that is not a cold prompt.
	Win int

	// Tag names a scripted request so a test can point at it.
	Tag string
}

// NewRequest builds a request at its arrival time.
func NewRequest(id int, kind Kind, conv int, at float64, input, out int, tag string) *Request {
	return &Request{ID: id, Kind: kind, Conv: conv, Arrival: at, InputLen: input, OutTarget: out,
		Tag: tag, PrefillStart: -1, FirstTok: -1, Finish: -1, Win: -1}
}

// Context is the length the request's attention reads on its next decode step.
func (r *Request) Context() int { return r.InputLen + r.OutDone }

// Work is the prefill work left for this request.
func (r *Request) Work() int {
	w := r.InputLen - r.Prefix
	if w < 1 {
		return 1
	}
	return w
}

// InDecodePhase reports whether the stream was past its first token and not yet
// finished at time t, the window the per-stream decode rate is measured over.
func (r *Request) InDecodePhase(t float64) bool {
	if r.FirstTok < 0 || t < r.FirstTok {
		return false
	}
	if r.Finish >= 0 {
		return t < r.Finish
	}
	return true
}
