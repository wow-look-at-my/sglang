package sim

import (
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
	var out, util, share float64
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
	}
	m := Measure(merged, window)
	n := float64(len(runs))
	m.OutputTokS, m.GPUBusyShare, m.DecodeShare = out/n, util/n, share/n
	return m
}

// Cell compares one metric between two policies. HigherIsBetter states which
// direction is an improvement; NaN means the metric is undefined for the
// scenario, for example a cold-prefill rate in a workload with no cold prompt.
type Cell struct {
	Key   MetricKey
	Old   float64
	New   float64
	Worse bool
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

// Deficient reports the shortfall of a relative to b, signed so that a positive
// number always means a is worse: the excess of a bad latency, or the shortfall
// of a missing rate. Scale is the metric's own magnitude.
func Deficient(k MetricKey, a, b float64) float64 {
	if isNaN(a) || isNaN(b) || b == 0 {
		return 0
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
