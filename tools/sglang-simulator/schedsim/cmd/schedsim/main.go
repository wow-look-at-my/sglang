// Command schedsim runs every scenario under the OLD, PREV and NEW
// schedulers and prints one markdown table per scenario group.
package main

import (
	"flag"
	"fmt"
	"os"

	"schedsim/internal/cost"
	"schedsim/internal/scenario"
)

func main() {
	only := flag.String("only", "", "run only the main table, the sensitivity sweep, or the cold-prompt bounds: main, sensitivity, bounds")
	flag.Parse()

	cal, err := cost.Calibrate()
	if err != nil {
		fmt.Fprintf(os.Stderr, "schedsim: calibrate: %v\n", err)
		os.Exit(1)
	}
	base := scenario.Deployment(cal.Model)
	fmt.Printf("Cells are OLD / PREV / NEW, each the mean of %d workload seeds (recomputes: their sum). Device pool %d tokens, host tier %d tokens.\n\n",
		len(scenario.Seeds), base.Cost.DevicePoolTokens, base.HostTokens)
	if *only == "bounds" {
		all := append(scenario.Main(), scenario.Sensitivity()...)
		scenario.WriteBounds(os.Stdout, scenario.RunAll(all, base), base)
		return
	}
	if *only != "sensitivity" {
		scenario.WriteTable(os.Stdout, scenario.RunAll(scenario.Main(), base))
		fmt.Println()
	}
	if *only != "main" {
		scenario.WriteTable(os.Stdout, scenario.RunAll(scenario.Sensitivity(), base))
	}
}
