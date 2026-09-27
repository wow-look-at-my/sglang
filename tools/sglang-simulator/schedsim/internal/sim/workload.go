package sim

import (
	"math/rand"
	"sort"
)

// AgentParams describes one closed-loop conversation's turn distribution. The
// defaults are the deployment's agent traffic: a turn adds 200-1400 tokens of
// tool output to the context, asks for 200-800 tokens back, and returns 1-5 s
// later.
type AgentParams struct {
	NewMin, NewMax       int
	OutMin, OutMax       int
	ThinkMin, ThinkMax   float64
}

// DefaultAgent is the agent turn model the scenarios are built on.
var DefaultAgent = AgentParams{NewMin: 200, NewMax: 1400, OutMin: 200, OutMax: 800, ThinkMin: 1, ThinkMax: 5}

// ThrashAgent is the ten-conversation episode's turn model: longer idle gaps and
// a shorter answer, over contexts no cache can hold.
var ThrashAgent = AgentParams{NewMin: 200, NewMax: 1400, OutMin: 200, OutMax: 600, ThinkMin: 2, ThinkMax: 8}

// Workload produces arrivals. A conversation is closed-loop: its next turn
// arrives a think time after the previous turn's last token.
type Workload interface {
	Initial() []*Request
	// OnFinish returns the request's conversation's next turn, if any.
	OnFinish(r *Request, now float64) []*Request
	// Agents is the number of closed-loop conversations, the denominator of the
	// per-agent decode rate.
	Agents() int
}

// Mix generates closed-loop agent conversations plus open-loop one-shot prompts.
type Mix struct {
	rng    *rand.Rand
	cost   Cost
	nextID int
	nextCh int

	// Shared is the system-prompt prefix every turn of the conversation carries;
	// it stays resident while whole conversation prefixes are evicted.
	Shared int
	// StopAt ends the closed loop: no turn arrives after this time.
	StopAt float64

	streams map[int]*Stream
	initial []*Request
}

// Stream is one conversation's evolving context.
type Stream struct {
	ID int
	// Ctx is the conversation's own length in tokens, excluding Shared.
	Ctx int
	P   AgentParams
}

// NewMix returns a workload generator for one seed.
func NewMix(seed int64, cost Cost, shared int) *Mix {
	return &Mix{rng: rand.New(rand.NewSource(seed)), cost: cost, nextCh: 1000,
		Shared: shared, streams: map[int]*Stream{}}
}

// Cost exposes the calibrated model to scripted scenario builders.
func (m *Mix) Cost() Cost { return m.cost }

// Rand exposes the workload's generator so a scripted scenario draws its own
// turn parameters from the same deterministic stream.
func (m *Mix) Rand() *rand.Rand { return m.rng }

// NextRequestID allocates an id that no other request in this run holds.
func (m *Mix) NextRequestID() int { m.nextID++; return m.nextID }

// AddStream registers a closed-loop conversation whose next turn arrives at
// firstArrival with a context of ctx tokens (excluding Shared).
func (m *Mix) AddStream(ctx int, firstArrival float64, p AgentParams) *Stream {
	st := &Stream{ID: len(m.streams), Ctx: ctx, P: p}
	m.streams[st.ID] = st
	m.initial = append(m.initial, m.Turn(st, firstArrival))
	return st
}

// RegisterStream adds a conversation whose first turn is scripted rather than
// drawn, so a scenario can hand an episode's requests a continuing stream.
func (m *Mix) RegisterStream(ctx int, p AgentParams) *Stream {
	st := &Stream{ID: len(m.streams), Ctx: ctx, P: p}
	m.streams[st.ID] = st
	return st
}

// Turn builds one turn of a conversation: its whole context plus new tokens.
func (m *Mix) Turn(st *Stream, at float64) *Request {
	nc := m.rng
	n := UniInt(nc, st.P.NewMin, st.P.NewMax)
	return NewRequest(m.NextRequestID(), KindAgent, st.ID, at, m.Shared+st.Ctx+n,
		UniInt(nc, st.P.OutMin, st.P.OutMax), "")
}

// AddScripted queues a request the scenario names itself, bound to an existing
// conversation. Scenario A's episode is scripted this way: its follow-up and cold
// prompts each belong to a conversation that keeps going afterwards.
func (m *Mix) AddScripted(kind Kind, conv int, at float64, input, out int, tag string) *Request {
	r := NewRequest(m.NextRequestID(), kind, conv, at, input, out, tag)
	m.initial = append(m.initial, r)
	return r
}

// AddCold queues a one-shot prompt with no cached prefix at all.
func (m *Mix) AddCold(at float64, input, out int, tag string) *Request {
	r := NewRequest(m.NextRequestID(), KindCold, m.nextConv(), at, input, out, tag)
	m.initial = append(m.initial, r)
	return r
}

// AddShort queues a one-shot short-chat prompt.
func (m *Mix) AddShort(at float64, input, out int) *Request {
	r := NewRequest(m.NextRequestID(), KindShort, m.nextConv(), at, input, out, "")
	m.initial = append(m.initial, r)
	return r
}

func (m *Mix) nextConv() int {
	m.nextCh++
	return m.nextCh
}

// Initial returns the arrivals scripted up to this point.
func (m *Mix) Initial() []*Request { return m.initial }

// OnFinish advances the conversation and schedules its next turn.
func (m *Mix) OnFinish(r *Request, now float64) []*Request {
	st, ok := m.streams[r.Conv]
	if !ok {
		return nil
	}
	st.Ctx = r.InputLen - m.Shared + r.OutDone
	at := now + UniFloat(m.rng, st.P.ThinkMin, st.P.ThinkMax)
	if m.StopAt > 0 && at > m.StopAt {
		return nil
	}
	return []*Request{m.Turn(st, at)}
}

// Agents counts the closed-loop conversations.
func (m *Mix) Agents() int { return len(m.streams) }

// SharedPrefix is the system prompt every turn carries, resident in the pool for
// the whole run.
func (m *Mix) SharedPrefix() int { return m.Shared }

// WarmSeeds lists the conversations whose context starts out cached, with the
// length in tokens of each conversation's own prefix.
func (m *Mix) WarmSeeds() map[int]int {
	if len(m.streams) == 0 {
		return nil
	}
	out := map[int]int{}
	for id, st := range m.streams {
		out[id] = st.Ctx
	}
	return out
}

// UniInt draws an inclusive integer uniform.
func UniInt(rng *rand.Rand, a, b int) int {
	if b <= a {
		return a
	}
	return a + rng.Intn(b-a+1)
}

// UniFloat draws a continuous uniform.
func UniFloat(rng *rand.Rand, a, b float64) float64 { return a + rng.Float64()*(b-a) }

// BuildAgentsPlusCold: n closed-loop conversations plus a cold prompt every
// everyMin minutes from t=60 s, which is scenario B (and, with a single length
// and one arrival, scenario D).
func BuildAgentsPlusCold(seed int64, cost Cost, n, ctxMin, ctxMax int, everyMin float64, coldLen int, window float64, coldTimes []float64) *Mix {
	m := NewMix(seed, cost, 0)
	m.StopAt = window
	for i := 0; i < n; i++ {
		m.AddStream(UniInt(m.rng, ctxMin, ctxMax), UniFloat(m.rng, 0, 5), DefaultAgent)
	}
	times := coldTimes
	if times == nil && everyMin > 0 {
		for t := 60.0; t < window; t += everyMin * 60 {
			times = append(times, t)
		}
	}
	for i, t := range times {
		m.AddCold(t, coldLen, 400, itoa(i))
	}
	return m
}

// BuildShortChat: Poisson arrivals of short prompts, scenario C.
func BuildShortChat(seed int64, cost Cost, rate, window float64) *Mix {
	m := NewMix(seed, cost, 0)
	m.StopAt = window
	rng := m.rng
	t := 0.0
	for {
		t += rng.ExpFloat64() / rate
		if t > window {
			break
		}
		m.AddShort(t, UniInt(rng, 200, 3000), UniInt(rng, 50, 600))
	}
	return m
}

// BuildThrash: ten long conversations whose working set is far larger than the
// cache, so every turn pays for a prefix the pool could not keep.
func BuildThrash(seed int64, cost Cost, window float64, ctxMin, ctxMax int, p AgentParams) *Mix {
	m := NewMix(seed, cost, SharedSystemPrompt)
	m.StopAt = window
	for i := 0; i < ThrashConversations; i++ {
		m.AddStream(UniInt(m.rng, ctxMin, ctxMax)-SharedSystemPrompt, UniFloat(m.rng, 0, 60), p)
	}
	return m
}

// sortRequests orders arrivals by time, ids breaking ties deterministically.
func sortRequests(rs []*Request) {
	sort.SliceStable(rs, func(i, j int) bool {
		if rs[i].Arrival != rs[j].Arrival {
			return rs[i].Arrival < rs[j].Arrival
		}
		return rs[i].ID < rs[j].ID
	})
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	s, neg := "", i < 0
	if neg {
		i = -i
	}
	for i > 0 {
		s = string(rune('0'+i%10)) + s
		i /= 10
	}
	if neg {
		return "-" + s
	}
	return s
}
