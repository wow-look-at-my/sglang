package sim

import (
	"fmt"

	"schedsim/internal/trace"
)

// Episode is the logged incident reduced to the arrivals a replay needs. Every
// number is derived from a counter in the log rather than typed in, so a rebuilt
// log moves the scenario with it; TestEpisodeMatchesTheLog pins the derivation
// to the values the incident report quotes.
type Episode struct {
	// ChunkSize is the deployment's chunked_prefill_size.
	ChunkSize int
	// C1Len is the first cold prompt's input length: #pending-token on its first
	// chunk counts what that chunk left over.
	C1Len int
	// C1Chunks counts C1's full-size chunks; its tail is C1Len mod ChunkSize.
	C1Chunks int

	// R2Len is the follow-up that queued during C1's prefill, and R2Cached the
	// prefix it matched. R3Len and R4Len are the two cold prompts behind it.
	R2Len, R2Cached int
	R3Len, R4Len    int

	// R2AfterK and the rest count how many of C1's chunks had completed when the
	// queue-depth counter first showed that request waiting.
	R2AfterK, R3AfterK, R4AfterK int

	// MixedNewSeq, MixedNewTokens and MixedHit describe the batch that finished
	// C1, which the log printed.
	MixedNewSeq, MixedNewTokens, MixedHit int
	MixedLine                             int

	// ChunkSeconds is the log's measured duration of each of C1's chunks. The
	// first chunk's gap includes the idle time before the prompt arrived, so the
	// arrival clock substitutes the model's cost for it.
	ChunkSeconds []float64
}

// R2Arrival is when the follow-up queued, measured from C1's arrival: the
// incident's own statement of it, in seconds.
const R2Arrival = 1.3

// TailLen is C1's last, short chunk.
func (e Episode) TailLen() int { return e.C1Len % e.ChunkSize }

// R2Work is the follow-up's uncached input.
func (e Episode) R2Work() int { return e.R2Len - e.R2Cached }

// Arrival is the wall time a request that queued after k of C1's chunks reached
// the scheduler: the log's own chunk durations summed from the second chunk,
// since the first line's gap includes idle time before the prompt arrived.
func (e Episode) Arrival(k int, firstChunkSeconds float64) float64 {
	t := firstChunkSeconds
	for i := 1; i < k && i < len(e.ChunkSeconds); i++ {
		t += e.ChunkSeconds[i]
	}
	return t
}

// ExtractEpisode reads the logged incident out of the parsed steps.
func ExtractEpisode(steps []trace.Step, chunkSize int) (Episode, error) {
	ep := Episode{ChunkSize: chunkSize}
	first := -1
	for i, s := range steps {
		if s.Kind == trace.Prefill && s.HitTokens == 0 && s.NewTokens == chunkSize {
			first = i
			break
		}
	}
	if first < 0 {
		return ep, fmt.Errorf("no %d-token cold chunk in the log", chunkSize)
	}
	ep.C1Len = steps[first].Pending + steps[first].NewTokens

	k := 0
	prevQueue := 0
	last := first
	// #pending-token counts C1's remainder plus every queued request's whole
	// input, so the length of each arrival is the step the residual takes; the
	// queue-depth counter says which chunk it arrived behind.
	residualPrev := 0
	var queued []int
	for i := first; i < len(steps); i++ {
		s := steps[i]
		if s.Kind != trace.Prefill || s.NewTokens != chunkSize || s.HitTokens != 0 {
			break
		}
		k++
		ep.ChunkSeconds = append(ep.ChunkSeconds, trace.StepSeconds(s))
		left := ep.C1Len - k*chunkSize
		if delta := s.Pending - left - residualPrev; delta > 0 {
			queued = append(queued, delta)
		}
		residualPrev = s.Pending - left
		if s.QueueReq > prevQueue {
			switch s.QueueReq {
			case 1:
				ep.R2AfterK = k - 1
			case 2:
				ep.R3AfterK = k - 1
			case 3:
				ep.R4AfterK = k - 1
			}
		}
		prevQueue, last = s.QueueReq, i
	}
	ep.C1Chunks = k
	if len(queued) != 3 {
		return ep, fmt.Errorf("the cold run queued %d requests, want R2, R3 and R4", len(queued))
	}
	ep.R2Len, ep.R3Len, ep.R4Len = queued[0], queued[1], queued[2]

	for i := last + 1; i < len(steps); i++ {
		s := steps[i]
		if s.Kind != trace.Prefill {
			continue
		}
		if s.NewSeq >= 2 {
			ep.MixedNewSeq, ep.MixedNewTokens, ep.MixedHit, ep.MixedLine = s.NewSeq, s.NewTokens, s.HitTokens, s.Line
			// The follow-up is the request that matched a prefix in that batch: the
			// two cold prompts in it matched none, and the cold run's tail is known.
			ep.R2Cached = s.HitTokens
			break
		}
	}
	if ep.MixedNewSeq == 0 {
		return ep, fmt.Errorf("no multi-request prefill batch after the cold run")
	}
	if ep.R2Len <= ep.R2Cached || ep.R3Len <= 0 || ep.R4Len <= 0 {
		return ep, fmt.Errorf("episode lengths out of range: R2 %d/%d, R3 %d, R4 %d",
			ep.R2Len, ep.R2Cached, ep.R3Len, ep.R4Len)
	}
	return ep, nil
}

// MixedItems is the composition of the batch that finished C1: its tail, the
// follow-up's uncached work, and what is left of the chunk budget for the next
// cold prompt. This is the batch-formation rule the log reproduced.
func (e Episode) MixedItems() (tail, follow, next int) {
	tail = e.TailLen()
	follow = e.R2Work()
	page := 64
	next = e.ChunkSize - ceilPage(tail, page) - ceilPage(follow, page)
	return tail, follow, next
}
