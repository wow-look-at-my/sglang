package trace

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"schedsim/internal/trace/tracetest"
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

func TestBareLogIsOneBoot(t *testing.T) {
	text := tracetest.Incident(incident).String()
	boots, err := ParseBoots(text)
	if err != nil {
		t.Fatal(err)
	}
	steps, _ := Parse(text)
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

// corpus parses the generated multi-boot log, one boot per generated boot.
func corpus(t *testing.T) ([]*tracetest.Traffic, []Boot) {
	t.Helper()
	gen := tracetest.Corpus()
	boots, err := ParseBoots(tracetest.Text(gen))
	if err != nil {
		t.Fatal(err)
	}
	if len(boots) != len(gen) {
		t.Fatalf("boots = %d, the log was written with %d", len(boots), len(gen))
	}
	return gen, boots
}

// expectedRuns lists, per boot, the cold runs of at least minChunks chunks the
// generator wrote, as chunk count and starting context.
func expectedRuns(gen []*tracetest.Traffic, minChunks int) [][][2]int {
	out := make([][][2]int, len(gen))
	for i, g := range gen {
		for _, p := range g.Prompts {
			if n := p.ColdChunks(4096); n >= minChunks {
				out[i] = append(out[i], [2]int{n, p.CtxStart(4096)})
			}
		}
	}
	return out
}

func TestCorpusBoots(t *testing.T) {
	gen, boots := corpus(t)
	var ended []string
	for i, b := range boots {
		g := gen[i]
		if !b.ArgsKnown || b.Args.PageSize != 64 || b.Worker != g.Worker || !b.Start.Equal(g.Spec.Start) {
			t.Errorf("boot %d = %s from %v, args %+v; written as %s from %v", i, b.Worker, b.Start, b.Args, g.Worker, g.Spec.Start)
		}
		if b.Args.ContextLength != g.Spec.Args.ContextLength || b.Args.HierarchicalCache != g.Spec.Args.HierarchicalCache {
			t.Errorf("boot %d context %d hicache %v, written with %d %v", i, b.Args.ContextLength, b.Args.HierarchicalCache,
				g.Spec.Args.ContextLength, g.Spec.Args.HierarchicalCache)
		}
		if n := len(b.Completions); n != g.Turns {
			t.Errorf("boot %d completions = %d, the log completed %d requests", i, n, g.Turns)
		}
		if n := len(b.JITCompiles); n != len(g.Spec.JIT) {
			t.Errorf("boot %d logged %d JIT compiles, written with %d", i, n, len(g.Spec.JIT))
		}
		ended = append(ended, b.EndedBy)
	}
	if got, want := strings.Join(ended, ","), "crash,sigterm,sigterm,eof"; got != want {
		t.Errorf("ended by %s, want %s", got, want)
	}
	if !boots[0].End.Equal(gen[0].Now) {
		t.Errorf("crash at %v, the log crashed at %v", boots[0].End, gen[0].Now)
	}
	if j := boots[1].JITCompiles; len(j) != 2 || j[0].Seconds != 2.1 || j[0].Kernel != "_fwd_kernel" {
		t.Errorf("boot 1 JIT compiles = %+v", j)
	}
}

// Every prompt that held the GPU for a few seconds with requests running is
// a stall, measured from its first chunk to the decode step after its last.
func TestCorpusStalls(t *testing.T) {
	gen, boots := corpus(t)
	var longest, longestWant Stall
	for i, b := range boots {
		var want []tracetest.Prompt
		for _, p := range gen[i].Prompts {
			if p.Running >= 1 && p.Seconds >= 5 {
				want = append(want, p)
			}
		}
		got := b.Stalls(5)
		if len(got) != len(want) {
			t.Errorf("boot %d: %d stalls, the log holds %d prompts that stalled decode for 5 s", i, len(got), len(want))
			continue
		}
		for k, s := range got {
			p := want[k]
			if !s.At.Equal(p.At) || math.Abs(s.Seconds-p.Seconds) > 1e-6 || s.RunningPeak != p.Running || s.ColdTokens != p.Tokens-chunkIfHit(p) {
				t.Errorf("boot %d stall %d = %+v; the prompt at %v ran %.3f s behind %d running", i, k, s, p.At, p.Seconds, p.Running)
			}
			if s.Seconds > longest.Seconds {
				longest = s
			}
			if p.Seconds > longestWant.Seconds {
				longestWant = Stall{At: p.At, Seconds: p.Seconds}
			}
		}
	}
	if longest.Seconds == 0 || !longest.At.Equal(longestWant.At) {
		t.Errorf("longest stall %.1f s at %v, the longest prompt ran %.1f s at %v", longest.Seconds, longest.At, longestWant.Seconds, longestWant.At)
	}
}

// chunkIfHit is the part of a prompt that reused a prefix: its first chunk, when it hit one.
func chunkIfHit(p tracetest.Prompt) int {
	if p.Hit > 0 {
		return 4096
	}
	return 0
}

func TestColdRunCarriesStartingContext(t *testing.T) {
	gen, boots := corpus(t)
	want := expectedRuns(gen, 8)
	continued, fresh := 0, 0
	for i, b := range boots {
		var got [][2]int
		for _, r := range b.ColdRunsAll(4096) {
			if r.Chunks() < 8 {
				continue
			}
			got = append(got, [2]int{r.Chunks(), r.CtxStart})
			if r.CtxStart > 0 {
				continued++
			} else {
				fresh++
			}
		}
		if fmt.Sprint(got) != fmt.Sprint(want[i]) {
			t.Errorf("boot %d cold runs (chunks, ctx0) = %v, the log was written with %v", i, got, want[i])
		}
	}
	if continued == 0 || fresh == 0 {
		t.Errorf("continued=%d fresh=%d; the corpus has both", continued, fresh)
	}
}

func TestCalibrateBootsHoldsOutEveryRun(t *testing.T) {
	gen, boots := corpus(t)
	c, rep := CalibrateBoots(boots, 4096, 64)
	runs, fitBoot, fitChunks := 0, -1, 0
	for i, rs := range expectedRuns(gen, 8) {
		for _, r := range rs {
			runs++
			if r[0] > fitChunks {
				fitBoot, fitChunks = i, r[0]
			}
		}
	}
	if rep.FitBoot != fitBoot || rep.FitChunks != fitChunks {
		t.Errorf("fit on boot %d's %d-chunk run, the longest the log holds is boot %d's %d", rep.FitBoot, rep.FitChunks, fitBoot, fitChunks)
	}
	if len(rep.Holdouts) != runs-1 {
		t.Errorf("holdouts = %d, want every other run of 8+ chunks, %d", len(rep.Holdouts), runs-1)
	}
	// The log priced every chunk with one model, so the fit predicts every run, cached prefix or not.
	if rep.HoldoutMedian > 0.01 || c.Prefill.MaxErrHoldout > 0.01 {
		t.Errorf("holdout error median %.2f%%, worst %.2f%%", 100*rep.HoldoutMedian, 100*c.Prefill.MaxErrHoldout)
	}
	pool := tracetest.DefaultModel.Pool
	if len(rep.Pools) != len(boots) {
		t.Errorf("pools = %v, want one per boot", rep.Pools)
	}
	for _, p := range rep.Pools {
		if math.Abs(float64(p.Tokens-pool)) > 0.01*float64(pool) || p.Context != boots[p.Boot].Args.ContextLength {
			t.Errorf("boot %d pool %d at context %d, the log printed against %d", p.Boot, p.Tokens, p.Context, pool)
		}
	}
	if rep.WallSamples == 0 || rep.WallRatioMedian < 0.99 || rep.WallRatioMedian > 1.01 {
		t.Errorf("wall check: %d samples, ratio %.3f", rep.WallSamples, rep.WallRatioMedian)
	}
	// Steady single-request decode lines pool across boots: the sum of what each boot alone gives.
	steady := 0
	for _, b := range boots {
		steady += FitDecode(b.Steps).Samples
	}
	if c.Decode.Samples != steady || steady == 0 || math.Abs(float64(c.DeviceTokens-pool)) > 0.01*float64(pool) {
		t.Errorf("calibration %s; %d steady lines across the boots", c, steady)
	}
	// A bare single-incident log still calibrates to what Calibrate gives.
	text := tracetest.Incident(incident).String()
	one, _ := ParseBoots(text)
	c1, _ := CalibrateBoots(one, 4096, 64)
	steps, _ := Parse(text)
	c0 := Calibrate(steps, 4096, 64)
	if c1.Prefill.PerToken != c0.Prefill.PerToken || c1.Decode.Base != c0.Decode.Base || c1.DeviceTokens != c0.DeviceTokens {
		t.Errorf("bare log: CalibrateBoots %s\n!= Calibrate %s", c1, c0)
	}
}

func TestPrefillStretchesMatchStalls(t *testing.T) {
	gen, boots := corpus(t)
	wantStretches, wantMatched := 0, 0
	for _, g := range gen {
		for _, p := range g.Prompts {
			if p.ColdChunks(4096) < 8 {
				continue
			}
			wantStretches++
			if p.Running >= 1 && p.Seconds >= 5 {
				wantMatched++
			}
		}
	}
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
	if stretches != wantStretches || matched != wantMatched {
		t.Errorf("stretches=%d matched stalls=%d, the log holds %d and %d", stretches, matched, wantStretches, wantMatched)
	}
	if m := median(ratios); m < 0.99 || m > 1.01 {
		t.Errorf("chunk-cost sum over measured stall: median %.2f", m)
	}
}
