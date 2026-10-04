package sim

import (
	"sync"

	"schedsim/internal/trace"
)

// SharedSystemPrompt is the prefix every conversation of the thrash episode carries.
const SharedSystemPrompt = 11_584

// ThrashConversations is the logged thrash episode's conversation count.
const ThrashConversations = 10

// BaselineChunkSize is the deployment's chunked_prefill_size in the log.
const BaselineChunkSize = 4096

// The context and prompt sizes the scenarios are specified with.
const (
	AgentContextMin = 100_000
	AgentContextMax = 250_000
	// ColdPromptLen is the deployment's long cold prompt, 400K-class.
	ColdPromptLen = 400_000
)

// Scenario is a committed workload plus the window its metrics use.
type Scenario struct {
	Name string
	// Key shares runs between scenarios whose workload is identical, so the comparison never measures the same run twice.
	Key   string
	Note  string
	Build func(seed int64, cost Cost) Workload
	// Window is the interval the throughput and rate metrics use.
	HardStop   float64
	Window     float64
	MaxRunning int
	HostMul    float64
	Seeds      []int64
}

var episodeOnce = sync.OnceValue(func() Episode {
	steps, err := trace.Parse(trace.EmbeddedLog)
	if err != nil {
		panic("embedded log: " + err.Error())
	}
	ep, err := ExtractEpisode(steps, BaselineChunkSize)
	if err != nil {
		panic("embedded log episode: " + err.Error())
	}
	return ep
})

// EpisodeOf returns the logged episode, derived once from the embedded log.
func EpisodeOf() Episode { return episodeOnce() }

func ScenarioA() Scenario {
	return Scenario{
		Name: "A: logged episode",
		Key:  "A",
		Note: "C1 430K cold, R2 follow-up at +1.3 s, R3/R4 cold behind it, then 4 agent streams to +300 s",
		Build: func(seed int64, cost Cost) Workload {
			ep := EpisodeOf()
			m := NewMix(seed, cost, 0)
			m.StopAt = 300
			// Conversation X already holds R2's matched prefix.
			x := m.RegisterStream(ep.R2Cached, DefaultAgent)
			y := m.RegisterStream(0, DefaultAgent)
			z := m.RegisterStream(0, DefaultAgent)
			w := m.RegisterStream(0, DefaultAgent)
			first := cost.ChunkSeconds(0)
			m.AddScripted(KindCold, z.ID, 0, ep.C1Len, 600, "C1")
			m.AddScripted(KindAgent, x.ID, R2Arrival, ep.R2Len, 400, "R2")
			m.AddScripted(KindCold, y.ID, ep.Arrival(ep.R3AfterK, first), ep.R3Len, 400, "R3")
			m.AddScripted(KindCold, w.ID, ep.Arrival(ep.R4AfterK, first), ep.R4Len, 400, "R4")
			return m
		},
		Window:     300,
		HardStop:   420,
		MaxRunning: 6,
		HostMul:    4,
		Seeds:      []int64{7},
	}
}

// The 1-minute cadence is the overload case: prefill demand above half the GPU,
// where cold prompts must queue rather than starve decode forever.
func ScenarioB(everyMin float64) Scenario {
	name := "B: 5 agents + cold 400K every " + itoa(int(everyMin)) + " min"
	note := "5 agent streams at 100K-250K context, one cold 400K prompt every " + itoa(int(everyMin)) + " min, 15 min"
	if everyMin <= 1 {
		name = "overload: 5 agents + cold 400K every 1 min"
		note = "prefill demand above half the GPU: cold prompts must queue, decode must not starve forever"
	}
	p := DefaultBParams()
	p.EveryMin = everyMin
	return ScenarioFromB(p, name, "B-"+itoa(int(everyMin)), note)
}

// ScenarioC: short chat at a fixed arrival rate, which is where the scheduler is
// judged on throughput rather than on any one stream.
func ScenarioC(rate float64, maxRunning int) Scenario {
	return Scenario{
		Name: "C: short chat " + ftoa(rate) + " req/s, max_running " + itoa(maxRunning),
		Key:  "C-" + ftoa(rate) + "-" + itoa(maxRunning),
		Note: "Poisson arrivals, 200-3000 input tokens, 50-600 output",
		Build: func(seed int64, cost Cost) Workload {
			return BuildShortChat(seed, cost, rate, 600)
		},
		Window:     600,
		HardStop:   900,
		MaxRunning: maxRunning,
		HostMul:    4,
		Seeds:      []int64{1, 2, 3},
	}
}

// ScenarioD: one cold prompt at t=60 s against agents, sweeping the cold
// prompt's length, which is the "how large a prompt is too large" question.
func ScenarioD(coldLen int) Scenario {
	p := DefaultBParams()
	p.ColdLen = coldLen
	p.CadenceOff = true
	p.ColdTimes = []float64{60}
	p.Window = 420
	return ScenarioFromB(p, "D: one cold prompt, "+itoa(coldLen/1000)+"K", "D-"+itoa(coldLen),
		"5 agent streams, a single cold prompt at t=60 s, 420 s")
}

// ScenarioThrash: those-conversation episode whose working set is far past the
// cache, at each host tier. The tier decides whether an evicted prefix reloads or
// must be recomputed, which is the difference between a stall and a collapse.
func ScenarioThrash(hostMul, window float64) Scenario {
	return Scenario{
		Name: "thrash: host " + ftoa(hostMul) + "x, " + itoa(int(window)) + " s",
		Key:  "thrash-" + ftoa(hostMul) + "-" + itoa(int(window)),
		Note: "10 conversations, contexts 150K-430K against a 1.4M-token pool; a thrashed turn hits only the shared system prompt",
		Build: func(seed int64, cost Cost) Workload {
			return BuildThrash(seed, cost, window, 150_000, 430_000, ThrashAgent)
		},
		Window:     window,
		HardStop:   window + 300,
		MaxRunning: ThrashConversations,
		HostMul:    hostMul,
		Seeds:      []int64{1, 2, 3},
	}
}

// BaseScenarios lists every scenario the comparison reports, in the order the
// specification defines them.
func BaseScenarios() []Scenario {
	out := []Scenario{ScenarioA()}
	for _, every := range []float64{1, 2, 5} {
		out = append(out, ScenarioB(every))
	}
	for _, rate := range []float64{0.5, 1, 2, 3, 5} {
		out = append(out, ScenarioC(rate, 16))
	}
	for _, rate := range []float64{1, 5} {
		out = append(out, ScenarioC(rate, 6))
	}
	for _, coldLen := range []int{25_000, 100_000, 200_000, 400_000, 500_000} {
		out = append(out, ScenarioD(coldLen))
	}
	for _, hostMul := range []float64{0, 1.5, 4} {
		out = append(out, ScenarioThrash(hostMul, 600))
	}
	for _, hostMul := range []float64{0, 1.5, 4} {
		out = append(out, ScenarioThrash(hostMul, 1800))
	}
	return out
}

// ftoa prints a float as a short decimal, for scenario keys and labels.
func ftoa(f float64) string {
	if float64(int(f)) == f {
		return itoa(int(f))
	}
	tenths := int(f*10 + 0.5)
	return itoa(tenths/10) + "." + itoa(tenths%10)
}
