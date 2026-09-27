// Package sim runs the scheduler as a discrete-event simulation: requests
// arrive, batches are formed by the shipped batch-formation rules, each batch
// costs the GPU time the production log calibrated, and the three scheduler
// policies choose between prefill and decode under their own control laws.
//
// The policies are the only difference between two runs of the same scenario:
// arrivals, batch formation, admission against the KV pool, the cost model and
// the acceptance draws are shared, so a metric that moves is a policy effect.
package sim

// Mode names the scheduler under test.
type Mode int

const (
	// ModeOld is upstream prefill-priority: a prefill batch runs whenever one
	// can be formed and decode only otherwise. No balancer, no cede, no
	// throttle, and prefill batches carry no decode rows.
	ModeOld Mode = iota
	// ModePrev is the balancer at f15db9ee6a: the balance is floored at zero, a
	// mixed batch would be charged whole to prefill, and there is no token bound.
	// Mixed chunked prefill is not resolved on at that commit, so its streams
	// take tokens in pure decode batches only.
	ModePrev
	// ModeNew is the balancer at HEAD: the balance banks one decode batch, a
	// mixed batch is charged only its prefill part, chunked_prefill_size tokens
	// bound the stall, and the eviction throttle gates admission.
	ModeNew
)

// Modes lists every policy in comparison order.
var Modes = []Mode{ModeOld, ModePrev, ModeNew}

func (m Mode) String() string {
	switch m {
	case ModeOld:
		return "OLD"
	case ModePrev:
		return "PREV"
	default:
		return "NEW"
	}
}

// Cede is the chunk-sharing rule a mode applies to an in-progress chunk.
type Cede int

const (
	// CedeNone leaves the chunked request the whole budget.
	CedeNone Cede = iota
	// CedeHalf gives a ceding request at most half of each chunk (FCFS).
	CedeHalf
	// CedeFair amortizes the half over the chunked request's progress since it
	// first ceded, so a chunk it ran alone banks room for a whole chunk later.
	CedeFair
)

// Features are the per-mode rule selections the engine reads.
type Features struct {
	Balancer bool
	Cede     Cede
	Throttle bool
	// MixedChunk folds one decode row per running request into each prefill batch.
	MixedChunk bool
	// PiggybackCredit subtracts a mixed batch's decode rows from its prefill charge.
	PiggybackCredit bool
	// BurstTokens is chunked_prefill_size for the stall bound, 0 for none.
	BurstTokens bool
}

// FeaturesOf returns the rule set of one mode. Mixed chunking is a deployment
// feature, not part of the control law: arg_groups/mixed_chunk_hook.py exists at
// HEAD and resolves it on for this deployment, and exists at neither of the two
// commits OLD and PREV are read from, so a prefill batch there carries prefill
// rows only. TestFeaturesTablePinsTheModes guards the table.
func FeaturesOf(m Mode) Features {
	switch m {
	case ModeOld:
		return Features{}
	case ModePrev:
		return Features{Balancer: true, Cede: CedeHalf}
	default:
		return Features{
			Balancer:        true,
			Cede:            CedeFair,
			Throttle:        true,
			MixedChunk:      true,
			PiggybackCredit: true,
			BurstTokens:     true,
		}
	}
}
