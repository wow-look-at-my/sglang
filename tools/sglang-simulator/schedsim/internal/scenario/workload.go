// Package scenario defines the workloads the policies are compared on.
package scenario

import (
	"math/rand"

	"schedsim/internal/sim"
)

// Span is a closed range a workload draws from uniformly.
type Span struct{ Lo, Hi float64 }

func (s Span) draw(rng *rand.Rand) float64 { return s.Lo + rng.Float64()*(s.Hi-s.Lo) }

func (s Span) drawInt(rng *rand.Rand) int { return int(s.draw(rng)) }

// ColdPrompt is a long uncached one-shot prompt.
type ColdPrompt struct {
	At     float64
	Tokens int
	Output int
}

// Agents is a closed loop of long-context conversations, each alternating a
// turn and a think time, plus cold prompts at fixed times.
type Agents struct {
	Seed    int64
	N       int
	Ctx0    Span
	Turn    Span
	Output  Span
	Think   Span
	MaxNew  int
	Colds   []ColdPrompt
	Horizon float64

	// Per-agent draws keep each agent's n-th turn the same under every policy.
	convs map[*sim.Conv]*rand.Rand
}

// Start seeds every agent's context in the cache and queues its first turn.
func (a *Agents) Start(e *sim.Engine) {
	a.convs = map[*sim.Conv]*rand.Rand{}
	for i := range a.N {
		rng := rand.New(rand.NewSource(a.Seed*1000 + int64(i)))
		ctx := a.Ctx0.drawInt(rng)
		seg := &sim.Segment{Tokens: ctx}
		e.Cache.Seed(seg, true, -float64(a.N-i))
		conv := &sim.Conv{ID: i, Chain: []*sim.Segment{seg}, Len: ctx, Finished: 1}
		a.convs[conv] = rng
		a.submitTurn(e, conv, a.Think.draw(rng)/2)
	}
	for i, c := range a.Colds {
		conv := &sim.Conv{ID: 1000 + i}
		e.Submit(&sim.Request{Conv: conv, Kind: sim.Cold, Arrival: c.At, NewTokens: c.Tokens,
			OutputLen: c.Output, MaxNew: a.MaxNew})
	}
}

func (a *Agents) submitTurn(e *sim.Engine, conv *sim.Conv, at float64) {
	if at >= a.Horizon {
		return
	}
	rng := a.convs[conv]
	e.Submit(&sim.Request{Conv: conv, Kind: sim.Turn, Arrival: at, NewTokens: a.Turn.drawInt(rng),
		OutputLen: a.Output.drawInt(rng), MaxNew: a.MaxNew})
}

// OnFinish queues the agent's next turn after its think time.
func (a *Agents) OnFinish(e *sim.Engine, r *sim.Request) {
	if rng := a.convs[r.Conv]; rng != nil {
		a.submitTurn(e, r.Conv, e.Now()+a.Think.draw(rng))
	}
}

// Chat is open-loop short chat: Poisson arrivals over a few shared system
// prompts.
type Chat struct {
	Seed    int64
	Rate    float64
	System  int
	Prompts int
	User    Span
	Output  Span
	MaxNew  int
	Horizon float64
}

// Start queues every arrival up front.
func (c *Chat) Start(e *sim.Engine) {
	rng := rand.New(rand.NewSource(c.Seed))
	var sys []*sim.Segment
	for range c.Prompts {
		sys = append(sys, &sim.Segment{Tokens: c.System})
	}
	t := 0.0
	for id := 0; ; id++ {
		t += rng.ExpFloat64() / c.Rate
		if t >= c.Horizon {
			return
		}
		s := sys[rng.Intn(len(sys))]
		conv := &sim.Conv{ID: id, Chain: []*sim.Segment{s}, Len: s.Tokens}
		e.Submit(&sim.Request{Conv: conv, Kind: sim.Turn, Arrival: t, NewTokens: c.User.drawInt(rng),
			OutputLen: c.Output.drawInt(rng), MaxNew: c.MaxNew})
	}
}

// OnFinish ends the chat: each is one turn.
func (c *Chat) OnFinish(*sim.Engine, *sim.Request) {}

// Replay submits a fixed request list against seeded conversations.
type Replay struct {
	Seeded   []*sim.Conv
	Requests []*sim.Request
}

// Start seeds the conversations' contexts and queues the requests.
func (p *Replay) Start(e *sim.Engine) {
	for _, c := range p.Seeded {
		for _, s := range c.Chain {
			e.Cache.Seed(s, true, -1)
		}
	}
	for _, r := range p.Requests {
		e.Submit(r)
	}
}

// OnFinish does nothing: the replay has no follow-ups.
func (p *Replay) OnFinish(*sim.Engine, *sim.Request) {}
