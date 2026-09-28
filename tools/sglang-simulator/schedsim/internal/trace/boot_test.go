package trace

import (
	"strings"
	"testing"
	"time"
)

const prodPrefill = `2026-09-27T00:15:24.123Z 5ylyr21v-rhkt6 INFO TP0] Prefill batch, #new-seq: 1, #new-token: 4096, #cached-token: 0, full token usage: 0.49, mamba usage: 0.33, #running-req: 4, #queue-req: 2, #pending-token: 118201, cuda graph: False, input throughput (token/s): 13874.93`
const prodDecode = `2026-09-27T00:15:25.500Z 5ylyr21v-rhkt6 INFO TP0] Decode batch, #running-req: 10, #full token: 685000, full token usage: 0.49, mamba num: 12, mamba usage: 0.67, accept len: 2.77, accept rate: 0.69, cuda graph: True, gen throughput (token/s): 8.44, #queue-req: 0`
const bareDecode = `TP0] Decode batch, #running-req: 1, #full token: 1000, full token usage: 0.10, mamba num: 1, mamba usage: 0.04, accept len: 2.70, accept rate: 0.70, cuda graph: True, gen throughput (token/s): 180.00, #queue-req: 0`

func TestParseProductionFormat(t *testing.T) {
	steps, err := Parse(prodPrefill + "\n" + prodDecode + "\n")
	if err != nil {
		t.Fatal(err)
	}
	if len(steps) != 2 {
		t.Fatalf("steps = %d, want 2", len(steps))
	}
	p := steps[0]
	if p.Kind != Prefill || p.NewTokens != 4096 || p.HitTokens != 0 || p.RunningReq != 4 || p.QueueReq != 2 || p.Pending != 118201 {
		t.Errorf("prefill fields = %+v", p)
	}
	if p.Worker != "5ylyr21v-rhkt6" || p.Rank != 0 {
		t.Errorf("worker/rank = %q/%d", p.Worker, p.Rank)
	}
	want := time.Date(2026, 9, 27, 0, 15, 24, 123e6, time.UTC)
	if !p.At.Equal(want) {
		t.Errorf("At = %v, want %v", p.At, want)
	}
	d := steps[1]
	if d.Kind != Decode || d.RunningReq != 10 || d.AcceptLen != 2.77 || d.Throughput != 8.44 {
		t.Errorf("decode fields = %+v", d)
	}
	if gap, ok := StepGap(p, d); !ok || gap < 1.376 || gap > 1.378 {
		t.Errorf("StepGap = %v, %v", gap, ok)
	}
}

func TestParseKeepsOnlyRankZero(t *testing.T) {
	tp1 := strings.Replace(prodPrefill, "TP0]", "TP1]", 1)
	steps, err := Parse(prodPrefill + "\n" + tp1 + "\n")
	if err != nil {
		t.Fatal(err)
	}
	if len(steps) != 1 {
		t.Fatalf("steps = %d, want the rank-0 line only", len(steps))
	}
}

func TestBareFormatHasNoTimestamp(t *testing.T) {
	steps, err := Parse(bareDecode + "\n")
	if err != nil {
		t.Fatal(err)
	}
	if !steps[0].At.IsZero() || steps[0].Worker != "" {
		t.Errorf("bare line carried %v %q", steps[0].At, steps[0].Worker)
	}
	if _, ok := StepGap(steps[0], steps[0]); ok {
		t.Error("StepGap reported ok without timestamps")
	}
}

func TestParseServerArgs(t *testing.T) {
	line := `2026-09-26T19:55:58.468Z zw21lesh-67pt5 INFO server_args={'model_path': '/repository', 'context_length': 262144, 'page_size': 64, 'chunked_prefill_size': 4096, 'decode_log_interval': 40, 'max_running_requests': 16, 'max_mamba_cache_size': 48, 'enable_hierarchical_cache': False, 'enable_mixed_chunk': True}`
	a, ok := ParseServerArgs(line)
	if !ok {
		t.Fatal("not recognised")
	}
	if a.PageSize != 64 || a.ContextLength != 262144 || a.ChunkedPrefillSize != 4096 || a.DecodeLogInterval != 40 ||
		a.MaxRunningRequests != 16 || a.MaxMambaCacheSize != 48 || a.HierarchicalCache || !a.MixedChunk {
		t.Errorf("args = %+v", a)
	}
	if _, ok := ParseServerArgs(prodPrefill); ok {
		t.Error("a batch line parsed as server args")
	}
}

func TestEmbeddedLogIsOneBoot(t *testing.T) {
	boots, err := ParseBoots(EmbeddedLog)
	if err != nil {
		t.Fatal(err)
	}
	steps, _ := Parse(EmbeddedLog)
	if len(boots) != 1 || boots[0].ArgsKnown || len(boots[0].Steps) != len(steps) {
		t.Fatalf("boots = %d, args known %v, steps %d vs %d", len(boots), boots[0].ArgsKnown, len(boots[0].Steps), len(steps))
	}
	if boots[0].Timestamped() || boots[0].Stalls(1) != nil {
		t.Error("bare log reported timestamps")
	}
	runs := boots[0].ColdRunsAll(4096)
	longest := ColdRun{}
	for _, r := range runs {
		if r.Chunks() > longest.Chunks() {
			longest = r
		}
	}
	start, end := coldWindow(steps, 4096)
	if longest.Start != start || longest.End != end {
		t.Errorf("longest run [%d,%d) != coldWindow [%d,%d)", longest.Start, longest.End, start, end)
	}
}

func TestLog2Boots(t *testing.T) {
	boots, err := ParseBoots(Log2())
	if err != nil {
		t.Fatal(err)
	}
	if len(boots) != 11 {
		t.Fatalf("boots = %d, want 11", len(boots))
	}
	var crashes, drains []string
	for _, b := range boots {
		if !b.ArgsKnown || b.Args.PageSize != 64 {
			t.Errorf("boot %d args %+v", b.Index, b.Args)
		}
		switch b.EndedBy {
		case "crash":
			crashes = append(crashes, b.End.Format("01-02T15:04"))
		case "sigterm":
			drains = append(drains, b.Worker)
		}
	}
	if got, want := strings.Join(crashes, ","), "09-26T19:58,09-27T00:26,09-27T20:08"; got != want {
		t.Errorf("crashes = %s, want %s", got, want)
	}
	if len(drains) != 5 {
		t.Errorf("drains = %v, want 5", drains)
	}
	if boots[2].Args.ContextLength != 262144 || boots[3].Args.ContextLength != 524288 {
		t.Errorf("context lengths %d %d", boots[2].Args.ContextLength, boots[3].Args.ContextLength)
	}
	if boots[8].Worker != "0liw8vzv-75dvg" || !boots[8].Args.HierarchicalCache {
		t.Errorf("boot 8 = %s hicache %v", boots[8].Worker, boots[8].Args.HierarchicalCache)
	}
	if n := len(boots[8].Completions); n < 4000 {
		t.Errorf("boot 8 completions = %d", n)
	}
	if n := len(boots[2].JITCompiles); n == 0 {
		t.Error("boot 2 logged no JIT compiles")
	}
}

func TestLog2Stalls(t *testing.T) {
	boots, _ := ParseBoots(Log2())
	var all []Stall
	for _, b := range boots {
		all = append(all, b.Stalls(5)...)
	}
	if len(all) < 45 || len(all) > 55 {
		t.Errorf("stalls >= 5 s: %d, want about 49", len(all))
	}
	longest := Stall{}
	for _, s := range all {
		if s.Seconds > longest.Seconds {
			longest = s
		}
	}
	if longest.Seconds < 140 || longest.Seconds > 155 {
		t.Errorf("longest stall %.1f s, want about 149", longest.Seconds)
	}
	if got := longest.At.Format("2006-01-02T15:04"); got != "2026-09-27T00:15" {
		t.Errorf("longest stall at %s", got)
	}
	if longest.RunningPeak < 10 {
		t.Errorf("longest stall peaked at %d running", longest.RunningPeak)
	}
}

func TestColdRunCarriesStartingContext(t *testing.T) {
	boots, _ := ParseBoots(Log2())
	continued, fresh := 0, 0
	for _, b := range boots {
		for _, r := range b.ColdRunsAll(4096) {
			if r.Chunks() < 8 {
				continue
			}
			if r.CtxStart > 0 {
				continued++
			} else {
				fresh++
			}
		}
	}
	if continued == 0 || fresh == 0 {
		t.Errorf("continued=%d fresh=%d; the corpus has both", continued, fresh)
	}
}

func TestCalibrateBootsHoldsOutEveryRun(t *testing.T) {
	boots, _ := ParseBoots(Log2())
	c, rep := CalibrateBoots(boots, 4096, 64)
	if rep.FitChunks < 100 {
		t.Errorf("fit run has %d chunks; the corpus holds a 109-chunk prompt", rep.FitChunks)
	}
	if len(rep.Holdouts) < 100 {
		t.Errorf("holdouts = %d", len(rep.Holdouts))
	}
	if rep.HoldoutMedian > 0.10 {
		t.Errorf("median holdout error %.1f%%", 100*rep.HoldoutMedian)
	}
	if len(rep.Pools) < 5 {
		t.Errorf("pools = %v", rep.Pools)
	}
	if rep.WallSamples == 0 || rep.WallRatioMedian < 0.9 || rep.WallRatioMedian > 1.1 {
		t.Errorf("wall check: %d samples, ratio %.2f", rep.WallSamples, rep.WallRatioMedian)
	}
	if c.DeviceTokens < 1_000_000 || c.Decode.Samples < 1000 {
		t.Errorf("calibration %s", c)
	}
	// A bare single-incident log still calibrates to what Calibrate gives.
	one, _ := ParseBoots(EmbeddedLog)
	c1, _ := CalibrateBoots(one, 4096, 64)
	steps, _ := Parse(EmbeddedLog)
	c0 := Calibrate(steps, 4096, 64)
	if c1.Prefill.PerToken != c0.Prefill.PerToken || c1.Decode.Base != c0.Decode.Base || c1.DeviceTokens != c0.DeviceTokens {
		t.Errorf("embedded log: CalibrateBoots %s\n!= Calibrate %s", c1, c0)
	}
}

func TestPrefillStretchesMatchStalls(t *testing.T) {
	boots, _ := ParseBoots(Log2())
	var stretches, matched int
	var ratios []float64
	for _, b := range boots {
		stalls := b.Stalls(5)
		for _, r := range b.PrefillStretches(4096, 8) {
			stretches++
			if r.ColdChunks < 8 || r.ColdChunks > r.Chunks() {
				t.Errorf("stretch %+v has %d cold chunks over %d steps", r, r.ColdChunks, r.Chunks())
			}
			for k := r.Start; k < r.End; k++ {
				if b.Steps[k].Kind != Prefill {
					t.Fatalf("stretch [%d,%d) holds a decode step at %d", r.Start, r.End, k)
				}
			}
			if r.RunningAtStart > 0 && b.Steps[r.Start].RunningReq != r.RunningAtStart {
				t.Errorf("stretch starts at a step with %d running, recorded %d", b.Steps[r.Start].RunningReq, r.RunningAtStart)
			}
			for _, st := range stalls {
				if st.Start == r.Start {
					matched++
					// The stall's wall clock and the stretch's summed
					// chunk costs measure the same stretch two ways.
					var model float64
					for k := r.Start; k < r.End; k++ {
						model += StepSeconds(b.Steps[k])
					}
					ratios = append(ratios, model/st.Seconds)
				}
			}
		}
	}
	if stretches < 60 || matched < 40 {
		t.Errorf("stretches=%d matched stalls=%d", stretches, matched)
	}
	if m := median(ratios); m < 0.9 || m > 1.2 {
		t.Errorf("chunk-cost sum over measured stall: median %.2f", m)
	}
}
