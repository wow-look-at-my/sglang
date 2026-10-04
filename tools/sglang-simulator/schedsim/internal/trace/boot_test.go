package trace

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"schedsim/internal/trace/tracetest"
)

const prodPrefill = `2026-09-27T00:15:24.123Z 5ylyr21v-rhkt6 INFO TP0] Prefill batch, #new-seq: 1, #new-token: 4096, #cached-token: 0, full token usage: 0.49, mamba usage: 0.33, #running-req: 4, #queue-req: 2, #pending-token: 118201, cuda graph: False, input throughput (token/s): 13874.93`
const prodDecode = `2026-09-27T00:15:25.500Z 5ylyr21v-rhkt6 INFO TP0] Decode batch, #running-req: 10, #full token: 685000, full token usage: 0.49, mamba num: 12, mamba usage: 0.67, accept len: 2.77, accept rate: 0.69, cuda graph: True, gen throughput (token/s): 8.44, #queue-req: 0`
const bareDecode = `TP0] Decode batch, #running-req: 1, #full token: 1000, full token usage: 0.10, mamba num: 1, mamba usage: 0.04, accept len: 2.70, accept rate: 0.70, cuda graph: True, gen throughput (token/s): 180.00, #queue-req: 0`

func TestParseProductionFormat(t *testing.T) {
	steps, err := Parse(prodPrefill + "\n" + prodDecode + "\n")
	require.Nil(t, err)

	require.Equal(t, 2, len(steps))

	p := steps[0]
	assert.False(t, p.Kind != Prefill || p.NewTokens != 4096 || p.HitTokens != 0 || p.RunningReq != 4 || p.QueueReq != 2 || p.Pending != 118201)

	assert.False(t, p.Worker != "5ylyr21v-rhkt6" || p.Rank != 0)

	want := time.Date(2026, 9, 27, 0, 15, 24, 123e6, time.UTC)
	assert.True(t, p.At.Equal(want))

	d := steps[1]
	assert.False(t, d.Kind != Decode || d.RunningReq != 10 || d.AcceptLen != 2.77 || d.Throughput != 8.44)

	gap, ok := StepGap(p, d)
	assert.False(t, !ok || gap < 1.376 || gap > 1.378)

}

func TestParseKeepsOnlyRankZero(t *testing.T) {
	tp1 := strings.Replace(prodPrefill, "TP0]", "TP1]", 1)
	steps, err := Parse(prodPrefill + "\n" + tp1 + "\n")
	require.Nil(t, err)

	require.Equal(t, 1, len(steps))

}

func TestBareFormatHasNoTimestamp(t *testing.T) {
	steps, err := Parse(bareDecode + "\n")
	require.Nil(t, err)

	assert.False(t, !steps[0].At.IsZero() || steps[0].Worker != "")

	_, ok := StepGap(steps[0], steps[0])
	assert.False(t, ok)

}

func TestParseServerArgs(t *testing.T) {
	line := `2026-09-26T19:55:58.468Z zw21lesh-67pt5 INFO server_args={'model_path': '/repository', 'context_length': 262144, 'page_size': 64, 'chunked_prefill_size': 4096, 'decode_log_interval': 40, 'max_running_requests': 16, 'max_mamba_cache_size': 48, 'enable_hierarchical_cache': False, 'enable_mixed_chunk': True}`
	a, ok := ParseServerArgs(line)
	require.True(t, ok)

	assert.False(t, a.PageSize != 64 || a.ContextLength != 262144 || a.ChunkedPrefillSize != 4096 || a.DecodeLogInterval != 40 || a.MaxRunningRequests != 16 || a.MaxMambaCacheSize != 48 || a.HierarchicalCache || !a.MixedChunk)

	_, ok = ParseServerArgs(prodPrefill)
	assert.False(t, ok)

}

func TestBareLogIsOneBoot(t *testing.T) {
	text := tracetest.Incident(incident).String()
	boots, err := ParseBoots(text)
	require.Nil(t, err)

	steps, _ := Parse(text)
	require.False(t, len(boots) != 1 || boots[0].ArgsKnown || len(boots[0].Steps) != len(steps))

	assert.False(t, boots[0].Timestamped() || boots[0].Stalls(1) != nil)

	runs := boots[0].ColdRunsAll(4096)
	longest := ColdRun{}
	for _, r := range runs {
		if r.Chunks() > longest.Chunks() {
			longest = r
		}
	}
	start, end := coldWindow(steps, 4096)
	assert.False(t, longest.Start != start || longest.End != end)

}

// corpus parses the generated multi-boot log, one boot per generated boot.
func corpus(t *testing.T) ([]*tracetest.Traffic, []Boot) {
	t.Helper()
	gen := tracetest.Corpus()
	boots, err := ParseBoots(tracetest.Text(gen))
	require.Nil(t, err)

	require.Equal(t, len(gen), len(boots))

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
		assert.False(t, !b.ArgsKnown || b.Args.PageSize != 64 || b.Worker != g.Worker || !b.Start.Equal(g.Spec.Start))

		assert.False(t, b.Args.ContextLength != g.Spec.Args.ContextLength || b.Args.HierarchicalCache != g.Spec.Args.HierarchicalCache)

		n := len(b.Completions)
		assert.Equal(t, g.Turns, n)

		n := len(b.JITCompiles)
		assert.Equal(t, len(g.Spec.JIT), n)

		ended = append(ended, b.EndedBy)
	}
	got, want := strings.Join(ended, ","), "crash,sigterm,sigterm,eof"
	assert.Equal(t, want, got)

	assert.True(t, boots[0].End.Equal(gen[0].Now))

	j := boots[1].JITCompiles
	assert.False(t, len(j) != 2 || j[0].Seconds != 2.1 || j[0].Kernel != "_fwd_kernel")

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
		assert.Equal(t, len(want), len(got))

		for k, s := range got {
			p := want[k]
			assert.False(t, !s.At.Equal(p.At) || math.Abs(s.Seconds-p.Seconds) > 1e-6 || s.RunningPeak != p.Running || s.ColdTokens != p.Tokens-chunkIfHit(p))

			if s.Seconds > longest.Seconds {
				longest = s
			}
			if p.Seconds > longestWant.Seconds {
				longestWant = Stall{At: p.At, Seconds: p.Seconds}
			}
		}
	}
	assert.False(t, longest.Seconds == 0 || !longest.At.Equal(longestWant.At))

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
		assert.Equal(t, fmt.Sprint(want[i]), fmt.Sprint(got))

	}
	assert.False(t, continued == 0 || fresh == 0)

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
	assert.False(t, rep.FitBoot != fitBoot || rep.FitChunks != fitChunks)

	assert.Equal(t, runs-1, len(rep.Holdouts))

	// The log priced every chunk with one model, so the fit predicts every run, cached prefix or not.
	assert.False(t, rep.HoldoutMedian > 0.01 || c.Prefill.MaxErrHoldout > 0.01)

	pool := tracetest.DefaultModel.Pool
	assert.Equal(t, len(boots), len(rep.Pools))

	for _, p := range rep.Pools {
		assert.False(t, math.Abs(float64(p.Tokens-pool)) > 0.01*float64(pool) || p.Context != boots[p.Boot].Args.ContextLength)

	}
	assert.False(t, rep.WallSamples == 0 || rep.WallRatioMedian < 0.99 || rep.WallRatioMedian > 1.01)

	// Steady single-request decode lines pool across boots: the sum of what each boot alone gives.
	steady := 0
	for _, b := range boots {
		steady += FitDecode(b.Steps).Samples
	}
	assert.False(t, c.Decode.Samples != steady || steady == 0 || math.Abs(float64(c.DeviceTokens-pool)) > 0.01*float64(pool))

	// A bare single-incident log still calibrates to what Calibrate gives.
	text := tracetest.Incident(incident).String()
	one, _ := ParseBoots(text)
	c1, _ := CalibrateBoots(one, 4096, 64)
	steps, _ := Parse(text)
	c0 := Calibrate(steps, 4096, 64)
	assert.False(t, c1.Prefill.PerToken != c0.Prefill.PerToken || c1.Decode.Base != c0.Decode.Base || c1.DeviceTokens != c0.DeviceTokens)

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
			assert.False(t, r.ColdChunks < 8 || r.ColdChunks > r.Chunks())

			for k := r.Start; k < r.End; k++ {
				require.Equal(t, Prefill, b.Steps[k].Kind)

			}
			assert.False(t, r.RunningAtStart > 0 && b.Steps[r.Start].RunningReq != r.RunningAtStart)

			for _, st := range stalls {
				if st.Start == r.Start {
					matched++
					// The stall's wall clock and the stretch's summed chunk costs measure the same stretch ways.
					var model float64
					for k := r.Start; k < r.End; k++ {
						model += StepSeconds(b.Steps[k])
					}
					ratios = append(ratios, model/st.Seconds)
				}
			}
		}
	}
	assert.False(t, stretches != wantStretches || matched != wantMatched)

	m := median(ratios)
	assert.False(t, m < 0.99 || m > 1.01)

}
