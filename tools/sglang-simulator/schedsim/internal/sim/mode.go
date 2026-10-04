// Package sim runs the scheduler as a discrete-event simulation: requests
// arrive, batches are formed by the shipped batch-formation rules, each batch
// costs the GPU time the production log calibrated, and those scheduler
// policies choose between prefill and decode under their own control laws.
//
// The policies are the only difference between multiple runs of the same
// scenario: arrivals, batch formation, admission against the KV pool, the cost
// model and the acceptance draws are shared, so a metric that moves is a policy effect.
package sim

// Mode names the scheduler under test.
type Mode int

const (
	// ModeOld is upstream prefill-priority: a prefill batch runs whenever one can be formed and decode only otherwise.
	ModeOld Mode = iota
	// ModePrev is the balancer at f15db9ee6a: the balance is floored at zero.
	ModePrev
	// ModeNew is the balancer at HEAD: the balance banks one decode batch, a mixed batch is charged only its prefill part.
	ModeNew
)

// NumModes is the policy count; Modes must hold exactly this many entries.
const NumModes = 3

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
	// CedeFair amortizes the half over the chunked request's progress since it first ceded.
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
	BurstTokens     bool
}

// FeaturesOf returns the rule set of one mode.
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
