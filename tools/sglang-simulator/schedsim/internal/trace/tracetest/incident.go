package tracetest

// IncidentSpec lays out a single-boot incident in the bare format. One
// conversation decodes alone and finishes its reply.
type IncidentSpec struct {
	Model       Model
	Chunk, Page int
	// C1 is C1Chunks full chunks and a C1Tail-token partial one.
	C1Chunks, C1Tail int
	// R2 extends the conversation's R2Cached-token prefix by R2Work tokens.
	R2Cached, R2Work int
	// R3 and R4 are their heads, then the full chunks and tails named here.
	R3Chunks, R3Tail int
	R4Chunks, R4Tail int
	// R2AfterK, R3AfterK and R4AfterK count C1's chunks done when each request queued.
	R2AfterK, R3AfterK, R4AfterK int
	// Steady is how many decode lines a lone conversation writes before C1 and after the episode.
	Steady int
}

// DefaultIncident is the long-context collapse the simulator models.
var DefaultIncident = IncidentSpec{
	Model: DefaultModel, Chunk: 4096, Page: 64,
	C1Chunks: 105, C1Tail: 263,
	R2Cached: 248320, R2Work: 921,
	R3Chunks: 57, R3Tail: 1500,
	R4Chunks: 20, R4Tail: 700,
	R2AfterK: 4, R3AfterK: 30, R4AfterK: 70,
	Steady: 16,
}

func ceilPage(n, page int) int { return (n + page - 1) / page * page }

// C1Len is the first cold prompt's input length.
func (s IncidentSpec) C1Len() int { return s.C1Chunks*s.Chunk + s.C1Tail }

// R2Len is the follow-up's whole input, cached prefix included.
func (s IncidentSpec) R2Len() int { return s.R2Cached + s.R2Work }

// R3Head is what is left of the batch that finishes C1 once C1's tail and
// R2's work take their pages.
func (s IncidentSpec) R3Head() int {
	return s.Chunk - ceilPage(s.C1Tail, s.Page) - ceilPage(s.R2Work, s.Page)
}

// R3Len is R3's whole input.
func (s IncidentSpec) R3Len() int { return s.R3Head() + s.R3Chunks*s.Chunk + s.R3Tail }

// R4Head is what is left of the batch that finishes R3.
func (s IncidentSpec) R4Head() int { return s.Chunk - ceilPage(s.R3Tail, s.Page) }

// R4Len is R4's whole input.
func (s IncidentSpec) R4Len() int { return s.R4Head() + s.R4Chunks*s.Chunk + s.R4Tail }

// MixedTokens is the batch that finishes C1: its tail, R2's work and R3's head.
func (s IncidentSpec) MixedTokens() int { return s.C1Tail + s.R2Work + s.R3Head() }

// C1ChunkTPS is the input throughput of C1's chunk k, counted from zero.
func (s IncidentSpec) C1ChunkTPS(k int) float64 {
	return s.Model.PrefillTPS(Extend{Tokens: s.Chunk, Ctx: k * s.Chunk})
}

// Incident writes the log the spec describes.
func Incident(s IncidentSpec) *Log {
	m := s.Model
	l := New("", Start)
	l.Bare = true
	for i := 0; i < s.Steady; i++ {
		l.DecodeAt(m, 1, s.R2Cached-(s.Steady-i)*120, Accepts(i))
	}
	l.Completion()

	queued := []struct{ after, tokens int }{{s.R2AfterK, s.R2Len()}, {s.R3AfterK, s.R3Len()}, {s.R4AfterK, s.R4Len()}}
	for k := 1; k <= s.C1Chunks; k++ {
		queue, pending := 0, 0
		for _, q := range queued {
			if q.after < k {
				queue++
				pending += q.tokens
			}
		}
		l.Prefill(Prefill{NewTokens: s.Chunk, Queue: queue, Pending: s.C1Len() - k*s.Chunk + pending, Usage: 0.31, TPS: s.C1ChunkTPS(k - 1)})
	}
	mixed := []Extend{{s.C1Tail, s.C1Chunks * s.Chunk}, {s.R2Work, s.R2Cached}, {s.R3Head(), 0}}
	l.Prefill(Prefill{NewSeq: 3, NewTokens: s.MixedTokens(), Hit: s.R2Cached, Queue: 1, Pending: s.R3Len() - s.R3Head() + s.R4Len(),
		Usage: 0.49, TPS: m.PrefillTPS(mixed...)})
	l.ColdChunks(m, s.R3Chunks*s.Chunk, s.R3Head(), s.Chunk, 2, 1, s.R3Tail+s.R4Len())
	finish := []Extend{{s.R3Tail, s.R3Len() - s.R3Tail}, {s.R4Head(), 0}}
	l.Prefill(Prefill{NewSeq: 2, NewTokens: s.R3Tail + s.R4Head(), Running: 2, Pending: s.R4Len() - s.R4Head(), Usage: 0.66, TPS: m.PrefillTPS(finish...)})
	l.ColdChunks(m, s.R4Len()-s.R4Head(), s.R4Head(), s.Chunk, 3, 0, 0)

	all := s.C1Len() + s.R2Len() + s.R3Len() + s.R4Len()
	// The first decode lines after the prefills average their own window, which held the prefills.
	l.Decode(Decode{Running: 4, FullTokens: all, Usage: 0.71, Accept: 2.77, TPS: 8.44})
	l.Decode(Decode{Running: 4, FullTokens: all + 300, Usage: 0.71, Accept: 2.74, TPS: 9.71})
	for i := 0; i < s.Steady; i++ {
		l.DecodeAt(m, 4, all+600+i*400, Accepts(i))
	}
	l.Completion().Completion().Completion()
	for i := 0; i < s.Steady; i++ {
		l.DecodeAt(m, 1, s.C1Len()+600+i*120, Accepts(i+2))
	}
	return l.Completion()
}
