package scenario

import (
	"fmt"
	"io"
	"math"

	"schedsim/internal/sim"
)

// Bound is the least a schedule that keeps decoding pays against OLD on one
// cold-prompt scenario. BOUNDS.md derives each field.
type Bound struct {
	// Prefill is the mean time of a cold prompt's own chunks run back to back.
	Prefill float64
	// DecodeShare is NEW's decode time over the time the cold prompts wait.
	DecodeShare float64
	TTFT        float64
	// Idle is the share of the cold windows NEW leaves the GPU idle.
	Idle float64
	// Chunk is the cheapest chunk batch among the cold prompts.
	Chunk float64
	// TailShare is the least share of gaps that stall on the colds at NEW's longest stall.
	TailShare float64
}

// coldPrefill prices one prompt of n tokens as chunk batches run back to back.
func coldPrefill(cfg sim.Config, n int) (total, cheapest float64) {
	cheapest = math.Inf(1)
	for start := 0; start < n; start += cfg.ChunkSize {
		c := cfg.Cost.BatchOverhead + cfg.Cost.ExtendSeconds(start, min(cfg.ChunkSize, n-start))
		total += c
		cheapest = math.Min(cheapest, c)
	}
	return total, cheapest
}

// BoundFor derives the bound from NEW's run of s. It returns false for a
// workload without cold prompts.
func BoundFor(s Scenario, base sim.Config, newRun sim.Metrics) (Bound, bool) {
	if len(s.Colds) == 0 || newRun.WindowSeconds == 0 {
		return Bound{}, false
	}
	cfg := base
	if s.Adjust != nil {
		s.Adjust(&cfg)
	}
	var b Bound
	b.Chunk = math.Inf(1)
	var stalls float64
	for _, n := range s.Colds {
		p, c := coldPrefill(cfg, n)
		b.Prefill += p / float64(len(s.Colds))
		b.Chunk = math.Min(b.Chunk, c)
		stalls += math.Ceil(p / newRun.Stall)
	}
	b.DecodeShare = newRun.WindowDecode / newRun.WindowSeconds
	b.TTFT = b.Prefill / (1 - b.DecodeShare)
	b.Idle = 1 - (newRun.WindowDecode+newRun.WindowPrefill)/newRun.WindowSeconds
	b.TailShare = newRun.WindowStreams * stalls / newRun.Gaps
	return b, true
}

// WriteBounds prints each cold-prompt scenario's bound beside OLD and NEW.
func WriteBounds(w io.Writer, rows []Row, base sim.Config) {
	fmt.Fprintln(w, "| scenario | prefill alone s | NEW decode share | TTFT bound s | TTFT OLD / NEW s | NEW idle | cheapest chunk ms | tail share | ITL p99 OLD / NEW ms | ITL p99.9 OLD / NEW ms |")
	fmt.Fprintln(w, "|---|---|---|---|---|---|---|---|---|---|")
	for _, r := range rows {
		old, nw := r.Results[0], r.Results[2]
		b, ok := BoundFor(r.Scenario, base, nw)
		if !ok {
			continue
		}
		fmt.Fprintf(w, "| %s | %.1f | %.3f | %.1f | %.1f / %.1f | %.2f%% | %.0f | %.2f%% | %.0f / %.0f | %.0f / %.0f |\n",
			r.Scenario.Name, b.Prefill, b.DecodeShare, b.TTFT, old.ColdTTFT, nw.ColdTTFT, 100*b.Idle,
			1000*b.Chunk, 100*b.TailShare, 1000*old.ITLp99, 1000*nw.ITLp99, 1000*old.ITLp999, 1000*nw.ITLp999)
	}
}
