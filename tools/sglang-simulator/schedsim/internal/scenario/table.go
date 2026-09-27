package scenario

import (
	"fmt"
	"io"
	"runtime"
	"sync"

	"schedsim/internal/sim"
)

// Policies are the schedulers every scenario compares.
var Policies = []sim.Policy{sim.Old, sim.Prev, sim.New}

// Row is one scenario's metrics under each policy: the seed mean, and each
// seed's run in the order of Seeds.
type Row struct {
	Scenario Scenario
	Results  [3]sim.Metrics
	Runs     [3][]sim.Metrics
}

// RunAll simulates every scenario under every policy in parallel.
func RunAll(scenarios []Scenario, base sim.Config) []Row {
	rows := make([]Row, len(scenarios))
	type job struct{ i, p int }
	jobs := make(chan job)
	var wg sync.WaitGroup
	for range runtime.NumCPU() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				runs := Run(scenarios[j.i], Policies[j.p], base)
				rows[j.i].Runs[j.p] = runs
				rows[j.i].Results[j.p] = sim.Mean(runs)
			}
		}()
	}
	for i, s := range scenarios {
		rows[i].Scenario = s
		for p := range Policies {
			jobs <- job{i, p}
		}
	}
	close(jobs)
	wg.Wait()
	return rows
}

// Metric names one column, its unit, and which way is better.
type Metric struct {
	Name         string
	Get          func(sim.Metrics) float64
	HigherBetter bool
	Format       string
}

// Metrics are the columns the requirement is asserted on.
var Metrics = []Metric{
	{"stream tok/s", func(m sim.Metrics) float64 { return m.StreamRate }, true, "%.1f"},
	{"longest stall s", func(m sim.Metrics) float64 { return m.Stall }, false, "%.2f"},
	{"cold TTFT s", func(m sim.Metrics) float64 { return m.ColdTTFT }, false, "%.1f"},
	{"ITL p99 ms", func(m sim.Metrics) float64 { return m.ITLp99 * 1000 }, false, "%.0f"},
	{"ITL p99.9 ms", func(m sim.Metrics) float64 { return m.ITLp999 * 1000 }, false, "%.0f"},
	{"throughput tok/s", func(m sim.Metrics) float64 { return m.Throughput }, true, "%.1f"},
	{"recomputes", func(m sim.Metrics) float64 { return float64(m.Recomputes) }, false, "%.0f"},
}

// WriteTable prints OLD / PREV / NEW per metric, one line per scenario.
func WriteTable(w io.Writer, rows []Row) {
	fmt.Fprint(w, "| scenario |")
	for _, m := range Metrics {
		fmt.Fprintf(w, " %s |", m.Name)
	}
	fmt.Fprint(w, "\n|---|")
	for range Metrics {
		fmt.Fprint(w, "---|")
	}
	fmt.Fprintln(w)
	for _, r := range rows {
		fmt.Fprintf(w, "| %s |", r.Scenario.Name)
		for _, m := range Metrics {
			fmt.Fprint(w, " ")
			for p := range Policies {
				if p > 0 {
					fmt.Fprint(w, " / ")
				}
				fmt.Fprintf(w, m.Format, m.Get(r.Results[p]))
			}
			fmt.Fprint(w, " |")
		}
		fmt.Fprintln(w)
	}
}
