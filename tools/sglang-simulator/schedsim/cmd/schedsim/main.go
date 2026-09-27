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
	only := flag.String("only", "", "run only the main table or only the sensitivity sweep: main, sensitivity")
	flag.Parse()

	cal, err := cost.Calibrate()
	if err != nil {
		fmt.Fprintf(os.Stderr, "schedsim: calibrate: %v\n", err)
		os.Exit(1)
	}
	base := scenario.Deployment(cal.Model)
	fmt.Printf("Cells are OLD / PREV / NEW. Device pool %d tokens, host tier %d tokens.\n\n",
		base.Cost.DevicePoolTokens, base.HostTokens)
	if *only != "sensitivity" {
		scenario.WriteTable(os.Stdout, scenario.RunAll(scenario.Main(), base))
		fmt.Println()
	}
	if *only != "main" {
		scenario.WriteTable(os.Stdout, scenario.RunAll(scenario.Sensitivity(), base))
	}
}
