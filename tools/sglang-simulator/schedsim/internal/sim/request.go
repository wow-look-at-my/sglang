package sim

// Kind labels a request for the metrics.
type Kind int

const (
	// Turn is a conversation's follow-up: an agent turn or a chat message.
	Turn Kind = iota
	// Cold is a long prompt with nothing cached.
	Cold
)

// Conv is a conversation: the segments its context is cached as.
type Conv struct {
	ID    int
	Chain []*Segment
	// Len is the tokens the chain holds.
	Len      int
	Finished int
	// LastFinish is when its last turn finished.
	LastFinish float64
}

// Request is one generation request.
type Request struct {
	ID      int
	Conv    *Conv
	Kind    Kind
	Arrival float64
	// NewTokens are appended to the conversation's context.
	NewTokens int
	OutputLen int
	MaxNew    int

	chain     []*Segment
	prefixLen int
	target    int
	done      int
	prefixIdx int
	locked    []*Segment
	rest      []*Segment
	own       int
	matched   bool

	generated  float64
	inFlight   float64
	allocated  int
	firstToken float64
	lastToken  float64
	finish     float64
	started    bool
	queuedAt   float64
	retracted  bool
}

// InputLen is the prompt length the request prefills.
func (r *Request) InputLen() int { return r.prefixLen + r.NewTokens }

// ctx is the request's current KV length.
func (r *Request) ctx() int { return r.target + int(r.generated) }

func (r *Request) remainingOutput() float64 {
	return float64(r.OutputLen) - r.generated - r.inFlight
}
