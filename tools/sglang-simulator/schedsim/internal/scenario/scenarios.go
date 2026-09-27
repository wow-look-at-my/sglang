package scenario

import (
	"fmt"
	"math"

	"schedsim/internal/cost"
	"schedsim/internal/sim"
	"schedsim/internal/trace"
)

// Scenario is one workload on one deployment.
type Scenario struct {
	Name    string
	Build   func(seed int64) (sim.Workload, float64)
	Adjust  func(*sim.Config)
	Summary string
	// Colds are the prompt lengths of the workload's cold prompts.
	Colds []int
}

func Deployment(m cost.Model) sim.Config {
	return sim.Config{Cost: m, ChunkSize: 4096, PageSize: 64, MaxRunning: 16,
		HostTokens: 2 * m.DevicePoolTokens, Overlap: true, NewTokenRatio: 0.3, ClipMaxNew: 4096}
}

// Seeds are the workload draws every scenario is averaged over.
var Seeds = func() []int64 {
	s := make([]int64, 16)
	for i := range s {
		s[i] = int64(7 + i)
	}
	return s
}()

// Run simulates one scenario under one policy, averaged over Seeds.
func Run(s Scenario, p sim.Policy, base sim.Config) sim.Metrics {
	cfg := base
	cfg.Policy = p
	if s.Adjust != nil {
		s.Adjust(&cfg)
	}
	runs := make([]sim.Metrics, len(Seeds))
	for i, seed := range Seeds {
		w, horizon := s.Build(seed)
		e := sim.NewEngine(cfg)
		e.Run(w, horizon)
		runs[i] = e.Rec.Metrics()
	}
	return sim.Mean(runs)
}

var (
	agentTurn   = Span{200, 2000}
	agentOutput = Span{150, 800}
	agentThink  = Span{3, 15}
)

func agents(seed int64, n int, ctx0 Span, colds []ColdPrompt, horizon float64) *Agents {
	return &Agents{Seed: seed, N: n, Ctx0: ctx0, Turn: agentTurn, Output: agentOutput, Think: agentThink,
		MaxNew: 16384, Colds: colds, Horizon: horizon}
}

func periodicColds(every, horizon float64, tokens int) []ColdPrompt {
	var cs []ColdPrompt
	for t := 30.0; t < horizon; t += every {
		cs = append(cs, ColdPrompt{At: t, Tokens: tokens, Output: 1000})
	}
	return cs
}

// B is agents with a cold 400K prompt every period.
func B(minutes float64, horizon float64, tweak func(*Agents)) Scenario {
	var colds []int
	for range periodicColds(60*minutes, horizon, 400000) {
		colds = append(colds, 400000)
	}
	return Scenario{
		Name:  fmt.Sprintf("B-%g (cold every %g min)", minutes, minutes),
		Colds: colds,
		Build: func(seed int64) (sim.Workload, float64) {
			a := agents(seed, 5, Span{60000, 180000}, periodicColds(60*minutes, horizon, 400000), horizon)
			if tweak != nil {
				tweak(a)
			}
			return a, horizon
		},
	}
}

// D is agents and one cold prompt of the given size.
func D(tokens int) Scenario {
	horizon := 90 + float64(tokens)/1500
	return Scenario{
		Name:  fmt.Sprintf("D-%dK", tokens/1000),
		Colds: []int{tokens},
		Build: func(seed int64) (sim.Workload, float64) {
			return agents(seed, 5, Span{60000, 180000}, []ColdPrompt{{At: 30, Tokens: tokens, Output: 1000}}, horizon), horizon
		},
	}
}

// Thrash is long conversations whose contexts overflow the device pool.
func Thrash(hostRatio float64, horizon float64) Scenario {
	return Scenario{
		Name: fmt.Sprintf("Thrash host=%gx %gs", hostRatio, horizon),
		Build: func(seed int64) (sim.Workload, float64) {
			return agents(seed, 10, Span{150000, 280000}, nil, horizon), horizon
		},
		Adjust: func(c *sim.Config) { c.HostTokens = int(hostRatio * float64(c.Cost.DevicePoolTokens)) },
	}
}

// C is short chat at a fixed arrival rate.
func C(rate float64, maxRunning int) Scenario {
	const horizon = 300
	return Scenario{
		Name: fmt.Sprintf("C rate=%g max_running=%d", rate, maxRunning),
		Build: func(seed int64) (sim.Workload, float64) {
			return &Chat{Seed: seed + 4, Rate: rate, System: 2000, Prompts: 4, User: Span{200, 1500},
				Output: Span{100, 600}, MaxNew: 4096, Horizon: horizon}, horizon
		},
		Adjust: func(c *sim.Config) { c.MaxRunning = maxRunning },
	}
}

// A replays the logged episode: a cold prompt arrives with nothing running, a
// follow-up of the idle conversation arrives during it, then more cold
// prompts, each at the time the log first queues it.
func A() Scenario {
	steps, err := trace.Parse(trace.EmbeddedLog)
	if err != nil {
		panic(err)
	}
	m := trace.Summarize(steps, 4096)
	lines, inputs := cost.PendingJumps(steps)
	first := steps[m.ColdStart]
	c1 := first.Pending + first.NewTokens
	mixed := steps[m.ColdEnd]
	arrival := func(idx int) float64 {
		t := 0.0
		// The queue count on a line was read when the batch before it launched.
		for i := m.ColdStart; i < idx-1; i++ {
			t += trace.StepSeconds(steps[i])
		}
		return t
	}
	followCtx := mixed.HitTokens
	return Scenario{
		Name:  "A (logged episode)",
		Colds: []int{c1, inputs[1], inputs[2]},
		Build: func(int64) (sim.Workload, float64) {
			seg := &sim.Segment{Tokens: followCtx}
			r1 := &sim.Conv{ID: 1, Chain: []*sim.Segment{seg}, Len: followCtx, Finished: 1}
			reqs := []*sim.Request{
				{Conv: &sim.Conv{ID: 2}, Kind: sim.Cold, Arrival: 0, NewTokens: c1, OutputLen: 1000, MaxNew: 16384},
				{Conv: r1, Kind: sim.Turn, Arrival: arrival(lines[0]), NewTokens: inputs[0] - followCtx,
					OutputLen: 600, MaxNew: 16384},
				{Conv: &sim.Conv{ID: 3}, Kind: sim.Cold, Arrival: arrival(lines[1]), NewTokens: inputs[1],
					OutputLen: 1000, MaxNew: 16384},
				{Conv: &sim.Conv{ID: 4}, Kind: sim.Cold, Arrival: arrival(lines[2]), NewTokens: inputs[2],
					OutputLen: 1000, MaxNew: 16384},
			}
			return &Replay{Seeded: []*sim.Conv{r1}, Requests: reqs}, math.Inf(1)
		},
	}
}

// Main is every scenario the requirement is checked on.
func Main() []Scenario {
	s := []Scenario{A(), B(1, 600, nil), B(2, 600, nil), B(5, 900, nil)}
	for _, mr := range []int{16, 6} {
		for _, rate := range []float64{0.5, 1, 2, 3} {
			s = append(s, C(rate, mr))
		}
	}
	for _, k := range []int{25, 100, 200, 400, 500} {
		s = append(s, D(k*1000))
	}
	for _, h := range []float64{0, 1.5, 2, 4} {
		for _, d := range []float64{600, 1800} {
			s = append(s, Thrash(h, d))
		}
	}
	return s
}

// Sensitivity perturbs one assumption at a time around B-2.
func Sensitivity() []Scenario {
	scale := func(name string, f func(*sim.Config)) Scenario {
		s := B(2, 600, nil)
		s.Name = "B-2 " + name
		s.Adjust = f
		return s
	}
	agentsTweak := func(name string, t func(*Agents), f func(*sim.Config)) Scenario {
		s := B(2, 600, t)
		s.Name = "B-2 " + name
		s.Adjust = f
		return s
	}
	return []Scenario{
		scale("prefill x0.75", func(c *sim.Config) { c.Cost.A *= 0.75; c.Cost.B *= 0.75; c.Cost.C *= 0.75 }),
		scale("prefill x1.33", func(c *sim.Config) { c.Cost.A *= 1.33; c.Cost.B *= 1.33; c.Cost.C *= 1.33 }),
		scale("batch overhead 0", func(c *sim.Config) { c.Cost.BatchOverhead = 0 }),
		scale("batch overhead 40ms", func(c *sim.Config) { c.Cost.BatchOverhead = 0.04 }),
		scale("decode base x0.7", func(c *sim.Config) { c.Cost.DecodeBase *= 0.7 }),
		scale("decode per-req x3", func(c *sim.Config) { c.Cost.DecodePerReq *= 3 }),
		scale("decode per-ctx x0", func(c *sim.Config) { c.Cost.DecodePerCtx = 0 }),
		scale("decode per-ctx x3", func(c *sim.Config) { c.Cost.DecodePerCtx *= 3 }),
		scale("accept 2.2", func(c *sim.Config) { c.Cost.AcceptLen = 2.2 }),
		scale("accept 3.5", func(c *sim.Config) { c.Cost.AcceptLen = 3.5 }),
		scale("overlap off", func(c *sim.Config) { c.Overlap = false }),
		scale("chunk 2048", func(c *sim.Config) { c.ChunkSize = 2048 }),
		scale("chunk 8192", func(c *sim.Config) { c.ChunkSize = 8192 }),
		scale("shortest-prefill-first", func(c *sim.Config) { c.ShortestFirst = true }),
		scale("no host tier", func(c *sim.Config) { c.HostTokens = 0 }),
		scale("host 1.5x", func(c *sim.Config) { c.HostTokens = c.Cost.DevicePoolTokens * 3 / 2 }),
		scale("reload x10", func(c *sim.Config) { c.Cost.ReloadPerToken *= 10 }),
		scale("mixed chunk off", func(c *sim.Config) { c.NoMixedChunk = true }),
		agentsTweak("faster agents", func(a *Agents) { a.Think = Span{1, 5} }, nil),
		agentsTweak("10 agents", func(a *Agents) { a.N = 10 }, nil),
		agentsTweak("follow-ups 2.5-4K", func(a *Agents) { a.Turn = Span{2500, 4000} }, nil),
		agentsTweak("outputs 1-3K", func(a *Agents) { a.Output = Span{1000, 3000} }, nil),
	}
}
