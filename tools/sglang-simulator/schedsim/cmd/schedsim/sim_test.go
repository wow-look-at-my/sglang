package main

import (
	"schedsim/internal/sim"
	"schedsim/internal/trace"
)

func sim_MRecomputes() sim.MetricKey       { return sim.MRecomputes }
func sim_MITLp99() sim.MetricKey           { return sim.MITLp99 }
func simCost(c trace.Calibration) sim.Cost { return sim.NewCost(c) }
