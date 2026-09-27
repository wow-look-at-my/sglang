package sim

import (
	"fmt"
	"io"
	"strings"

	"schedsim/internal/trace"
)

// BParams are the scenario-B traffic knobs, which the sensitivity sweep moves one
// at a time. Scenario B is also the template for scenario D (one cold prompt) and
// the overload case (a one-minute cadence), so the sweep reaches those shapes too.
type BParams struct {
	Agents   int
	CtxMin   int
	CtxMax   int
	Turn     AgentParams
	EveryMin float64
	ColdLen  int
	// ColdTimes overrides the cadence when a scenario scripts its arrivals.
	ColdTimes []float64
	Window    float64
	// CadenceOff suppresses the periodic cold prompt; ColdTimes then drives it.
	CadenceOff bool
	MaxRunning int
	HostMul    float64
}

// DefaultBParams is scenario B as specified: five agent streams at 100K-250K
// context, a cold 400K prompt every two minutes, a 15 minute run.
func DefaultBParams() BParams {
	return BParams{
		Agents: 5, CtxMin: AgentContextMin, CtxMax: AgentContextMax, Turn: DefaultAgent,
		EveryMin: 2, ColdLen: ColdPromptLen, Window: 900, MaxRunning: 6, HostMul: 4,
	}
}

// ScenarioFromB builds a committed workload out of the traffic parameters.
func ScenarioFromB(p BParams, name, key, note string) Scenario {
	return Scenario{
		Name: name, Key: key, Note: note,
		Build: func(seed int64, cost Cost) Workload {
			m := NewMix(seed, cost, 0)
			m.StopAt = p.Window
			for i := 0; i < p.Agents; i++ {
				m.AddStream(UniInt(m.rng, p.CtxMin, p.CtxMax), UniFloat(m.rng, 0, 5), p.Turn)
			}
			times := p.ColdTimes
			if !p.CadenceOff && p.EveryMin > 0 {
				for t := 60.0; t < p.Window; t += p.EveryMin * 60 {
					times = append(times, t)
				}
			}
			for i, t := range times {
				m.AddCold(t, p.ColdLen, 400, itoa(i))
			}
			return m
		},
		Window:     p.Window,
		HardStop:   p.Window + 300,
		MaxRunning: p.MaxRunning,
		HostMul:    p.HostMul,
		Seeds:      []int64{1, 2, 3, 4, 5},
	}
}

// Variant is one sensitivity perturbation. Each moves a single measurement or
// setting, so a conclusion that survives every variant is not resting on it.
type Variant struct {
	Name string
	Cost func(*Cost)
	Cfg  func(*Config)
	// B moves the traffic parameters of the swept scenario.
	B func(*BParams)
	// SkipA leaves scenario A at its baseline for variants that only make sense
	// against a steady agent mix.
	SkipA bool
}

// SweepVariants returns the one-at-a-time perturbations: the calibrated cost
// terms, the speculative and overlap settings, the chunk size, the queue policy,
// the traffic shape, and the cache tiers. `linear` is the same log refitted with
// a degree-1 polynomial, which is the alternative form the fit must beat.
func SweepVariants(linear trace.PrefillCost) []Variant {
	return []Variant{
		{Name: "baseline"},
		{Name: "prefill cost x0.75", Cost: func(c *Cost) { c.Cal.Prefill = c.Cal.Prefill.Scaled(0.75) }},
		{Name: "prefill cost x1.33", Cost: func(c *Cost) { c.Cal.Prefill = c.Cal.Prefill.Scaled(1.33) }},
		{Name: "prefill fit linear", Cost: func(c *Cost) { c.Cal.Prefill = linear }, SkipA: true},
		{Name: "per-batch overhead 0 ms", Cost: func(c *Cost) { c.Cal.Prefill = WithPrefillBase(c.Cal.Prefill, 0) }},
		{Name: "per-batch overhead 40 ms", Cost: func(c *Cost) { c.Cal.Prefill = WithPrefillBase(c.Cal.Prefill, 0.040) }},
		{Name: "decode D0 x0.7", Cost: func(c *Cost) { c.Cal.Decode.Base *= 0.7 }},
		{Name: "decode D0 x1.4", Cost: func(c *Cost) { c.Cal.Decode.Base *= 1.4 }},
		{Name: "decode DCtx 0", Cost: func(c *Cost) { c.Cal.Decode = WithContextCost(c.Cal.Decode, 0) }},
		{Name: "decode DCtx x3", Cost: func(c *Cost) { c.Cal.Decode = WithContextCost(c.Cal.Decode, 3*c.Cal.Decode.PerTokenCtx) }},
		{Name: "decode DBS 0", Cost: func(c *Cost) { c.Cal.Decode = WithBatchCost(c.Cal.Decode, 0) }},
		{Name: "decode DBS x3", Cost: func(c *Cost) { c.Cal.Decode = WithBatchCost(c.Cal.Decode, 3*c.Cal.Decode.PerReq) }},
		{Name: "MTP accept 2.2", Cost: func(c *Cost) { c.AcceptMean = 2.2 }},
		{Name: "MTP accept 3.5", Cost: func(c *Cost) { c.AcceptMean = 3.5 }},
		{Name: "overlap scheduler off", Cfg: func(c *Config) { c.Overlap = false }},
		{Name: "chunked_prefill_size 2048", Cfg: func(c *Config) { c.ChunkSize = 2048 }},
		{Name: "chunked_prefill_size 8192", Cfg: func(c *Config) { c.ChunkSize = 8192 }},
		{Name: "schedule policy shortest-prefill-first", Cfg: func(c *Config) { c.Policy = PolicySPF }},
		{Name: "mixed chunked prefill off", Cfg: func(c *Config) { c.MixedChunk = false }},
		{Name: "agents think 3x faster", B: func(p *BParams) {
			p.Turn.ThinkMin /= 3
			p.Turn.ThinkMax /= 3
		}},
		{Name: "10 agents, max_running 16", B: func(p *BParams) { p.Agents, p.MaxRunning = 10, 16 }},
		{Name: "follow-ups 2.5K-4K new tokens", B: func(p *BParams) { p.Turn.NewMin, p.Turn.NewMax = 2500, 4000 }},
		{Name: "outputs 1K-3K tokens", B: func(p *BParams) { p.Turn.OutMin, p.Turn.OutMax = 1000, 3000 }},
		{Name: "host tier 0x", B: func(p *BParams) { p.HostMul = 0 }},
		{Name: "host tier 1.5x", B: func(p *BParams) { p.HostMul = 1.5 }},
		{Name: "host tier 8x", B: func(p *BParams) { p.HostMul = 8 }},
		{Name: "host reload 10x slower", Cost: func(c *Cost) { c.ReloadPerToken = 10 * c.ReloadSecondsPerToken() }},
	}
}

// SweepRow is one variant measured on the swept scenario B and, unless the variant
// only makes sense against a steady agent mix, on scenario A as well.
type SweepRow struct {
	Name    string
	B       [NumModes]Metrics
	A       [NumModes]Metrics
	SweptA bool
	// Breaks lists the seven-metric cells where NEW is worse than the policy
	// named, so the sweep reports where the contract does not hold rather than
	// only where it does.
	Breaks []string
}

// RunSweep measures every variant. The seeds are the sweep's own, typically fewer
// than a scenario's, since the point is the direction of each perturbation.
func RunSweep(base Cost, linear trace.PrefillCost, seeds []int64, aSeed int64, workers int) []SweepRow {
	variants := SweepVariants(linear)
	rows := make([]SweepRow, len(variants))
	var jobs []func()
	for i, v := range variants {
		cost := base
		if v.Cost != nil {
			v.Cost(&cost)
		}
		p := DefaultBParams()
		if v.B != nil {
			v.B(&p)
		}
		sc := ScenarioFromB(p, "sweep "+v.Name, "sweep-"+itoa(i), "sensitivity variant "+v.Name)
		sc.Seeds = seeds
		i, v, cost, sc := i, v, cost, sc
		jobs = append(jobs, func() {
			rows[i].Name = v.Name
			for mi, mode := range Modes {
				cfg := DefaultConfig(mode, cost)
				if v.Cfg != nil {
					v.Cfg(&cfg)
				}
				var runs []*Result
				for _, seed := range sc.Seeds {
					runs = append(runs, Run(sc, cfg, seed))
				}
				rows[i].B[mi] = Pool(runs, sc.Window)
			}
			if v.SkipA {
				return
			}
			rows[i].SweptA = true
			asc := ScenarioA()
			asc.Seeds = []int64{aSeed}
			for mi, mode := range Modes {
				cfg := DefaultConfig(mode, cost)
				if v.Cfg != nil {
					v.Cfg(&cfg)
				}
				res := Run(asc, cfg, aSeed)
				rows[i].A[mi] = Measure(res, asc.Window)
			}
		})
	}
	runJobs(jobs, workers)
	for i := range rows {
		rows[i].Breaks = ContractBreaks(rows[i].B)
	}
	return rows
}

// ContractBreaks lists the cells where the third policy is worse than the first.
func ContractBreaks(m [NumModes]Metrics) []string {
	var out []string
	for _, k := range ContractMetrics {
		newv := m[2].Value(k)
		if !Better(k, newv, m[0].Value(k)) {
			out = append(out, k.String()+" vs OLD")
		}
		if !Better(k, newv, m[1].Value(k)) {
			out = append(out, k.String()+" vs PREV")
		}
	}
	return out
}

// WriteSweep prints the sweep with each cell as OLD / PREV / NEW.
func WriteSweep(w io.Writer, rows []SweepRow) {
	fmt.Fprintln(w, "Sensitivity sweep (scenario B at a 2-minute cold cadence; each variant moves one thing)")
	fmt.Fprintln(w, "Every cell is OLD / PREV / NEW. 'contract' lists the seven metrics where NEW is worse.")
	fmt.Fprintln(w)
	fmt.Fprintf(w, "  %-44s %-24s %-22s %-22s %-20s %s\n",
		"Variant", "B longest stall", "B ITL p99", "B cold TTFT mean", "B output tok/s", "contract")
	for _, r := range rows {
		fmt.Fprintf(w, "  %-44s %-24s %-22s %-22s %-20s %s\n", r.Name,
			triple(r.B, func(m Metrics) string { return seconds(m.LongestStall) }),
			triple(r.B, func(m Metrics) string { return seconds(m.ITLp99) }),
			triple(r.B, func(m Metrics) string { return seconds(m.ColdTTFTMean) }),
			triple(r.B, func(m Metrics) string { return fmt.Sprintf("%.0f", m.OutputTokS) }),
			strings.Join(r.Breaks, ", "))
	}
	fmt.Fprintln(w)
	fmt.Fprintln(w, "  Scenario A under the same perturbations (the logged episode, one seed):")
	fmt.Fprintf(w, "  %-44s %-24s %-24s %s\n", "Variant", "A longest stall", "A cold TTFT mean", "A stream tok/s")
	for _, r := range rows {
		if !r.SweptA {
			fmt.Fprintf(w, "  %-44s %s\n", r.Name, "not swept")
			continue
		}
		fmt.Fprintf(w, "  %-44s %-24s %-24s %s\n", r.Name,
			triple(r.A, func(m Metrics) string { return seconds(m.LongestStall) }),
			triple(r.A, func(m Metrics) string { return seconds(m.ColdTTFTMean) }),
			triple(r.A, func(m Metrics) string { return fmt.Sprintf("%.1f", m.StreamDecodeTokSCold) }))
	}
	fmt.Fprintln(w)
}

func triple(m [NumModes]Metrics, f func(Metrics) string) string {
	return f(m[0]) + " / " + f(m[1]) + " / " + f(m[2])
}

