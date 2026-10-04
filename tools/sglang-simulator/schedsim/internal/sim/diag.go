package sim

import (
	"fmt"
	"io"
	"math"
	"sort"
)

// GapRun is one stretch of prefill batches launched back to back.
type GapRun struct {
	First, Last   int
	Batches       []*Batch
	Tokens        int
	Seconds       float64
	Continuations int
	// DecodeWait is the time from the end of the decode batch to the start of the next one.
	DecodeWait float64
}

// PrefillRuns lists every back-to-back prefill stretch of a run, in time order.
func PrefillRuns(res *Result) []GapRun {
	var out []GapRun
	bs := res.Batches
	for i := 0; i < len(bs); i++ {
		if !bs[i].IsPrefill {
			continue
		}
		j := i
		for j+1 < len(bs) && bs[j+1].IsPrefill {
			j++
		}
		r := GapRun{First: i, Last: j}
		for _, b := range bs[i : j+1] {
			r.Batches = append(r.Batches, b)
			r.Tokens += b.ExtendTokens
			r.Seconds += b.End - b.Start
			if b.Continuation != nil {
				r.Continuations++
			}
		}
		r.DecodeWait = decodeWait(bs, i, j)
		out = append(out, r)
		i = j
	}
	return out
}

// decodeWait is the GPU time held between the decode batch before the stretch
// and the decode batch after it: the whole window a stream sees no decode step.
func decodeWait(bs []*Batch, lo, hi int) float64 {
	before := -1.0
	for i := lo - 1; i >= 0; i-- {
		if !bs[i].IsPrefill {
			before = bs[i].End
			break
		}
	}
	after := -1.0
	for i := hi + 1; i < len(bs); i++ {
		if !bs[i].IsPrefill {
			after = bs[i].Start
			break
		}
	}
	switch {
	case before < 0 && after < 0:
		return bs[hi].End - bs[lo].Start
	case before < 0:
		return after - bs[lo].Start
	case after < 0:
		return bs[hi].End - before
	default:
		return after - before
	}
}

// gapSample is one per-token inter-token latency and the delivery it came from.
type gapSample struct {
	Gap  float64
	Req  *Request
	From float64
	To   float64
	Toks int
}

// GapSamples lists every per-token gap of the selected requests, each gap
// spread over the tokens its delivery carried, as the ITL metric defines it.
func GapSamples(res *Result, sel func(*Request) bool) []gapSample {
	var out []gapSample
	for _, r := range res.Requests {
		if !sel(r) || len(r.Deliveries) < 2 {
			continue
		}
		for i := 1; i < len(r.Deliveries); i++ {
			n := r.Deliveries[i].N
			if n < 1 {
				n = 1
			}
			out = append(out, gapSample{
				Gap:  (r.Deliveries[i].T - r.Deliveries[i-1].T) / float64(n),
				Req:  r,
				From: r.Deliveries[i-1].T,
				To:   r.Deliveries[i].T,
				Toks: n,
			})
		}
	}
	return out
}

// batchesIn returns the batches whose GPU time overlaps (from, to].
func batchesIn(res *Result, from, to float64) []*Batch {
	var out []*Batch
	for _, b := range res.Batches {
		if b.End > from && b.Start < to {
			out = append(out, b)
		}
	}
	return out
}

// deliveryKey identifies the delivery a gap sample ended at, so it can be
// matched to the batch that produced it.
type deliveryKey struct {
	Req int
	T   int64
}

// deliveryClass labels a batch by what it handed the stream.
type deliveryClass struct {
	Label   string
	Samples int
	Copies  int
	Tokens  int
	// PerToken has one entry per delivery, Values one per token: the metric quantiles over tokens.
	PerToken  []float64
	Values    []float64
	RawGaps   []float64
	Over100ms int
	Over300ms int
}

// classifyDeliveries labels every delivery with the batch that emitted it. The
// engine records a row's token at its mixed batch's end and a decode step's
// accepted tokens at that batch's end, so (request, time) names one batch.
func classifyDeliveries(res *Result) map[deliveryKey]string {
	out := map[deliveryKey]string{}
	key := func(r *Request, t float64) deliveryKey { return deliveryKey{r.ID, int64(math.Round(t * 1e6))} }
	for _, b := range res.Batches {
		if !b.IsPrefill {
			for i, r := range b.Decode {
				if i < len(b.DecodeToks) && b.DecodeToks[i] > 0 {
					out[key(r, b.End)] = "decode"
				}
			}
			continue
		}
		for _, r := range b.Rows {
			out[key(r, b.End)] = "mixed"
		}
	}
	return out
}

// deliveryClassStats groups every ITL sample by the class of batch that
// delivered its token, counting one sample per token.
func deliveryClassStats(res *Result) map[string]*deliveryClass {
	cls := classifyDeliveries(res)
	byLabel := map[string]*deliveryClass{}
	for _, r := range res.Requests {
		for i := 1; i < len(r.Deliveries); i++ {
			label := cls[deliveryKey{r.ID, int64(math.Round(r.Deliveries[i].T * 1e6))}]
			if label == "" {
				label = "other"
			}
			c := byLabel[label]
			if c == nil {
				c = &deliveryClass{Label: label}
				byLabel[label] = c
			}
			n := r.Deliveries[i].N
			if n < 1 {
				n = 1
			}
			gap := r.Deliveries[i].T - r.Deliveries[i-1].T
			perToken := gap / float64(n)
			c.Samples++
			c.Copies += n
			c.Tokens += n
			c.PerToken = append(c.PerToken, perToken)
			for k := 0; k < n; k++ {
				c.Values = append(c.Values, perToken)
			}
			c.RawGaps = append(c.RawGaps, gap)
			if perToken > 0.1 {
				c.Over100ms++
			}
			if perToken > 0.3 {
				c.Over300ms++
			}
		}
	}
	return byLabel
}

// WriteDeliveryClassBreakdown prints, per class of delivering batch, how many
// tokens it handed out and what per-token gap it therefore contributed. A
// mixed batch hands one token per riding request where a decode step hands
// the sampled accept length, so the same wall-clock wait is divided
// differently.
//
// The tail lines name the class the reported p99 and p99.9 fall in, which is
// decided by each class's weight in samples, not its delivery count.
func WriteDeliveryClassBreakdown(w io.Writer, res *Result) {
	type tokenSample struct {
		value float64
		label string
	}
	byLabel := deliveryClassStats(res)
	var labels []string
	for l := range byLabel {
		labels = append(labels, l)
	}
	sort.Strings(labels)
	// Built in label order and sorted stably, so a tie between classes names the same label every run.
	var all []tokenSample
	for _, l := range labels {
		for _, perToken := range byLabel[l].Values {
			all = append(all, tokenSample{perToken, l})
		}
	}
	deliveries, copies := 0, 0
	for _, l := range labels {
		deliveries += byLabel[l].Samples
		copies += byLabel[l].Copies
	}
	fmt.Fprintf(w, "  per-token gap by the class of batch that delivered the token (%d deliveries, %d samples as the metric counts one per token):\n",
		deliveries, copies)
	for _, l := range labels {
		c := byLabel[l]
		fmt.Fprintf(w, "    %-7s deliveries %d, %d samples (%.2f%% of all), %.2f tokens per delivery on average; per-token p50 %s p99 %s p99.9 %s max %s; %d samples over 100ms, %d over 300ms; raw gap p50 %s p99 %s\n",
			c.Label, c.Samples, c.Copies, 100*float64(c.Copies)/float64(copies),
			float64(c.Tokens)/float64(c.Samples),
			seconds(pct(c.PerToken, 50)), seconds(pct(c.PerToken, 99)),
			seconds(pct(c.PerToken, 99.9)), seconds(pct(c.PerToken, 100)),
			c.Over100ms, c.Over300ms,
			seconds(pct(c.RawGaps, 50)), seconds(pct(c.RawGaps, 99)))
	}
	if len(all) == 0 {
		return
	}
	sort.SliceStable(all, func(i, j int) bool { return all[i].value < all[j].value })
	for _, p := range []float64{99, 99.9} {
		at := int(p / 100 * float64(len(all)-1))
		above := 0
		for i := at; i < len(all); i++ {
			if all[i].label == all[at].label {
				above++
			}
		}
		fmt.Fprintf(w, "    ITL p%s %s comes from a %s delivery: %d %s samples sit above the p%s mark, %.1f%% of that class\n",
			trimFloat(p), seconds(all[at].value), all[at].label, above, all[at].label,
			trimFloat(p), 100*float64(above)/float64(byLabel[all[at].label].Copies))
	}
}

func trimFloat(p float64) string {
	if p == math.Trunc(p) {
		return fmt.Sprintf("%.0f", p)
	}
	return fmt.Sprintf("%.1f", p)
}

func WritePrefillCostBreakdown(w io.Writer, res *Result) {
	var cont, fresh []float64
	var contToks, freshToks []float64
	for _, b := range res.Batches {
		if !b.IsPrefill {
			continue
		}
		secs := b.End - b.Start
		if b.Continuation != nil {
			cont = append(cont, secs)
			contToks = append(contToks, float64(b.ExtendTokens))
		} else {
			fresh = append(fresh, secs)
			freshToks = append(freshToks, float64(b.ExtendTokens))
		}
	}
	line := func(label string, secs, toks []float64) {
		if len(secs) == 0 {
			fmt.Fprintf(w, "    %-14s none\n", label)
			return
		}
		usPerTok := 0.0
		for i := range secs {
			if toks[i] > 0 {
				usPerTok += secs[i] / toks[i]
			}
		}
		fmt.Fprintf(w, "    %-14s batches %d; seconds p50 %s p99 %s max %s; tokens p50 %.0f p99 %.0f max %.0f; mean %s per token\n",
			label, len(secs), seconds(pct(secs, 50)), seconds(pct(secs, 99)),
			seconds(pct(secs, 100)), pct(toks, 50), pct(toks, 99),
			pct(toks, 100), seconds(usPerTok/float64(len(secs))))
	}
	fmt.Fprintf(w, "  prefill batch cost, continuation of a chunked prompt vs fresh work:\n")
	line("continuation", cont, contToks)
	line("fresh", fresh, freshToks)
}

// WriteGapDiagnostic prints the inter-token gap distribution and the batch
// sequence around the widest gaps, so a latency difference can be traced to the
// batches that caused it rather than argued about.
func WriteGapDiagnostic(w io.Writer, res *Result, top int) {
	samples := GapSamples(res, func(r *Request) bool { return true })
	fmt.Fprintf(w, "  per-token ITL samples %d", len(samples))
	if len(samples) == 0 {
		fmt.Fprintln(w, ": none")
		return
	}
	gaps := make([]float64, len(samples))
	for i, s := range samples {
		gaps[i] = s.Gap
	}
	fmt.Fprintf(w, ": p50 %s p90 %s p99 %s p99.9 %s max %s\n",
		seconds(pct(gaps, 50)), seconds(pct(gaps, 90)), seconds(pct(gaps, 99)),
		seconds(pct(gaps, 99.9)), seconds(pct(gaps, 100)))
	fmt.Fprintf(w, "  gap buckets      %s\n", bucketLine(gaps))

	runs := PrefillRuns(res)
	shapes := make([]int, 0, len(runs))
	tok := make([]float64, 0, len(runs))
	sec := make([]float64, 0, len(runs))
	wait := make([]float64, 0, len(runs))
	multiFresh, multiToks := 0, 0
	for _, r := range runs {
		shapes = append(shapes, len(r.Batches))
		tok = append(tok, float64(r.Tokens))
		sec = append(sec, r.Seconds)
		wait = append(wait, r.DecodeWait)
		if len(r.Batches) > 1 {
			multiFresh += len(r.Batches) - r.Continuations
			multiToks += r.Tokens
		}
	}
	fmt.Fprintf(w, "  prefill runs %d: batches per run p50 %d p99 %d max %d; run tokens max %.0f; run seconds max %s; decode wait p99 %s max %s\n",
		len(runs), pctInt(shapes, 50), pctInt(shapes, 99), maxInt(shapes),
		maxOf(tok), seconds(pct(sec, 100)), seconds(pct(wait, 99.9)), seconds(pct(wait, 100)))
	fmt.Fprintf(w, "  runs by length %s\n", histInt(shapes))
	if n := countGT(shapes, 1); n > 0 {
		fmt.Fprintf(w, "  of %d runs longer than one batch, %d prefill batches were fresh (not a chunk continuation) and they carried %d tokens\n",
			n, multiFresh, multiToks)
	} else {
		fmt.Fprintln(w, "  no run is longer than one prefill batch")
	}

	sort.Slice(samples, func(i, j int) bool { return samples[i].Gap > samples[j].Gap })
	if top > len(samples) {
		top = len(samples)
	}
	fmt.Fprintf(w, "  the %d widest per-token gaps, and the batches inside each:\n", top)
	for _, s := range samples[:top] {
		in := batchesIn(res, s.From, s.To)
		pf, dc, ptok, psec, cont, mixed := 0, 0, 0, 0.0, 0, 0
		for _, b := range in {
			if b.IsPrefill {
				pf++
				ptok += b.ExtendTokens
				psec += b.End - b.Start
				if b.Continuation != nil {
					cont++
				}
				if len(b.Rows) > 0 {
					mixed++
				}
				continue
			}
			dc++
		}
		fmt.Fprintf(w, "    %8.1f s req %d %s: %s over %d token(s); %d prefill batch(es) (%d continuation, %d mixed, %d tokens, %s) and %d decode batch(es)\n",
			s.From, s.Req.ID, s.Req.Kind, seconds(s.Gap), s.Toks, pf, cont, mixed, ptok,
			seconds(psec), dc)
	}
}

func bucketLine(v []float64) string {
	edges := []float64{0.01, 0.03, 0.1, 0.3, 1.0}
	labels := []string{"<10ms", "10-30ms", "30-100ms", "100-300ms", "300ms-1s", ">1s"}
	counts := make([]int, len(labels))
	for _, x := range v {
		k := len(edges)
		for i, e := range edges {
			if x < e {
				k = i
				break
			}
		}
		counts[k]++
	}
	out := ""
	for i, l := range labels {
		if counts[i] == 0 {
			continue
		}
		out += fmt.Sprintf("%s %d (%.2f%%)  ", l, counts[i],
			100*float64(counts[i])/float64(len(v)))
	}
	return out
}

func histInt(v []int) string {
	counts := map[int]int{}
	for _, x := range v {
		counts[x]++
	}
	keys := make([]int, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	out := ""
	for _, k := range keys {
		out += fmt.Sprintf("%d:%d  ", k, counts[k])
	}
	return out
}

func countGT(v []int, n int) int {
	c := 0
	for _, x := range v {
		if x > n {
			c++
		}
	}
	return c
}

func pctInt(v []int, p float64) int {
	if len(v) == 0 {
		return 0
	}
	c := make([]float64, len(v))
	for i, x := range v {
		c[i] = float64(x)
	}
	return int(pct(c, p))
}

func maxInt(v []int) int {
	m := 0
	for _, x := range v {
		if x > m {
			m = x
		}
	}
	return m
}
