package sim

import (
	"math"
	"runtime"
	"sync"

	"schedsim/internal/kv"
)

// Row is one scenario measured under each policy, pooled over its seeds.
type Row struct {
	Scenario Scenario
	Runs     [NumModes][]*Result
	Metrics  [NumModes]Metrics
}

// SuiteConfig returns the run configuration for a mode, given the calibration.
type SuiteConfig func(mode Mode, cost Cost) Config

// RunSuite runs every scenario under every mode. Runs are independent and each
// carries its own generator, so parallelism cannot reorder a batch: a run's
// result depends on its config and seed alone.
func RunSuite(scs []Scenario, cost Cost, cfgOf SuiteConfig, workers int) []Row {
	rows := make([]Row, len(scs))
	var jobs []func()
	for i, sc := range scs {
		rows[i].Scenario = sc
		for mi, mode := range Modes {
			for _, seed := range sc.Seeds {
				i, mi, sc, mode, seed := i, mi, sc, mode, seed
				jobs = append(jobs, func() {
					res := Run(sc, cfgOf(mode, cost), seed)
					rows[i].Runs[mi] = append(rows[i].Runs[mi], res)
				})
			}
		}
	}
	runJobs(jobs, workers)
	for i, sc := range scs {
		for mi := range Modes {
			// Runs were appended in job order, which is seed order per mode.
			rows[i].Runs[mi] = orderRuns(rows[i].Runs[mi], sc.Seeds)
			rows[i].Metrics[mi] = Pool(rows[i].Runs[mi], sc.Window)
		}
	}
	return rows
}

// orderRuns puts a mode's runs back in seed order, since the workers finished
// them in whatever order they were scheduled.
func orderRuns(runs []*Result, seeds []int64) []*Result {
	out := make([]*Result, len(seeds))
	for _, r := range runs {
		for i, s := range seeds {
			if r.Cfg.Seed == s {
				out[i] = r
			}
		}
	}
	return out
}

func runJobs(jobs []func(), workers int) {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	var wg sync.WaitGroup
	ch := make(chan func())
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for f := range ch {
				f()
			}
		}()
	}
	for _, f := range jobs {
		ch <- f
	}
	close(ch)
	wg.Wait()
}

// Pool reduces several seeds of one scenario into one measurement: latency
// samples are pooled across seeds, while rates and shares are averaged per seed.
func Pool(runs []*Result, window float64) Metrics {
	if len(runs) == 0 {
		return Metrics{}
	}
	if len(runs) == 1 {
		return Measure(runs[0], window)
	}
	merged := &Result{Agents: runs[0].Agents, End: window, Pool: kv.New(0, 0)}
	var out, util, share, stream, agent, stalled float64
	for _, r := range runs {
		merged.Requests = append(merged.Requests, r.Requests...)
		merged.Windows = append(merged.Windows, r.Windows...)
		merged.Batches = append(merged.Batches, r.Batches...)
		merged.DecodeLog = append(merged.DecodeLog, r.DecodeLog...)
		merged.PrefillLog = append(merged.PrefillLog, r.PrefillLog...)
		for id, n := range r.Pool.Recomputes {
			merged.Pool.Recomputes[id] += n
		}
		m := Measure(r, window)
		out += m.OutputTokS
		util += m.GPUBusyShare
		share += m.DecodeShare
		stream += m.StreamDecodeTokSCold
		agent += m.PerAgentTokSCold
		stalled += m.StallFrac1s
	}
	m := Measure(merged, window)
	n := float64(len(runs))
	// These are per-run ratios: their denominators are that run's own cold windows and conversation count.
	m.OutputTokS, m.GPUBusyShare, m.DecodeShare = out/n, util/n, share/n
	m.StreamDecodeTokSCold, m.PerAgentTokSCold = stream/n, agent/n
	if !isNaN(stalled) {
		m.StallFrac1s = stalled / n
	}
	return m
}

// PolicyIndex is a mode's position in a Row's arrays.
func PolicyIndex(m Mode) int {
	for i, x := range Modes {
		if x == m {
			return i
		}
	}
	return -1
}

// SeedValues lists one metric's value for each seed of one policy, in seed
// order, so a comparison can be read against the spread across seeds rather
// than against a single pooled number.
func (r Row) SeedValues(mode Mode, k MetricKey) []float64 {
	out := make([]float64, len(r.Runs[PolicyIndex(mode)]))
	for i, res := range r.Runs[PolicyIndex(mode)] {
		out[i] = Measure(res, r.Scenario.Window).Value(k)
	}
	return out
}

// Spread reports the full range of a list of values as a fraction of its mean.
func Spread(v []float64) float64 {
	if len(v) < 2 {
		return 0
	}
	lo, hi, total := v[0], v[0], 0.0
	for _, x := range v {
		lo, hi = minF(lo, x), maxF(hi, x)
		total += x
	}
	mean := total / float64(len(v))
	if mean == 0 {
		return 0
	}
	return (hi - lo) / mean
}

// Better reports whether a is at least as good as b under the metric's direction.
// Equal values, including both undefined, count as not worse.
func Better(k MetricKey, a, b float64) bool {
	if isNaN(a) || isNaN(b) {
		return true
	}
	if k.HigherIsBetter() {
		return a >= b
	}
	return a <= b
}

// TieRelative is the relative gap below which a cell's ordering is set by which requests happened to fall inside the measurement window.
const TieRelative = 0.005

// Verdict is the outcome of comparing one metric across policies.
type Verdict int

const (
	// VerdictUndefined: at least one side has no value (no cold prompt in the workload), so no claim is made either way.
	VerdictUndefined Verdict = iota
	VerdictWin
	VerdictTie
	VerdictLoss
)

func (v Verdict) String() string {
	switch v {
	case VerdictWin:
		return "win"
	case VerdictTie:
		return "tie"
	case VerdictLoss:
		return "loss"
	default:
		return "undefined"
	}
}

// Judge compares one metric between the new policy and another, returning the
// shortfall as a fraction of the other value: positive only when the new policy
// is worse by more than TieRelative.
func Judge(k MetricKey, newv, other float64) (Verdict, float64) {
	if isNaN(newv) || isNaN(other) {
		return VerdictUndefined, 0
	}
	gap := Deficient(k, newv, other)
	switch {
	case gap > TieRelative:
		return VerdictLoss, gap
	case -gap > TieRelative:
		return VerdictWin, gap
	default:
		return VerdictTie, gap
	}
}

// Deficient reports how far a falls short of b under the metric's direction,
// as a fraction of b: positive when a is worse, negative when a is better.
func Deficient(k MetricKey, a, b float64) float64 {
	if isNaN(a) || isNaN(b) {
		return 0
	}
	if b == 0 {
		switch {
		case a == 0:
			return 0
		case k.HigherIsBetter():
			return -math.Inf(1)
		default:
			return math.Inf(1)
		}
	}
	if k.HigherIsBetter() {
		return (b - a) / absF(b)
	}
	return (a - b) / absF(b)
}

func isNaN(f float64) bool { return f != f }

func absF(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}
