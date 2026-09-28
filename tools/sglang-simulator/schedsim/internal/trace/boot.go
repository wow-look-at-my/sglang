package trace

import (
	"regexp"
	"strconv"
	"strings"
	"time"
)

// A production log is many process lifetimes in one file: every restart prints
// a new server_args line, workers overlap during a rolling update, and a cold
// run of chunks that straddles a restart is two prompts, not one. Boot is one
// such lifetime, and every analysis that walks steps in order walks one Boot.

// ServerArgs is the subset of the server_args line the simulator prices with.
type ServerArgs struct {
	PageSize           int
	ChunkedPrefillSize int
	DecodeLogInterval  int
	MaxRunningRequests int
	MaxMambaCacheSize  int
	ContextLength      int
	HierarchicalCache  bool
	MixedChunk         bool
	// Raw is the whole dict text, for anything the fields above leave out.
	Raw string
}

// JITCompile is one kernel the deployment compiled after serving started.
type JITCompile struct {
	Kernel  string
	Seconds float64
	At      time.Time
}

// Boot is one process lifetime: its arguments, its rank-0 batch steps, the
// requests it completed and how it ended.
type Boot struct {
	Index  int
	Worker string
	// Start is the server_args line's timestamp and End the last line the
	// worker printed; both are zero in the bare format.
	Start, End time.Time
	Args       ServerArgs
	ArgsKnown  bool
	Steps      []Step
	// Completions are the timestamps of "POST /v1/chat/completions ... 200".
	Completions []time.Time
	JITCompiles []JITCompile
	// EndedBy is "sigterm" (a drain), "crash" (a scheduler exception or a
	// child's SIGQUIT), or "eof" (the log ends with the process still up).
	EndedBy string
}

var (
	prefixRe     = regexp.MustCompile(`^(\d{4}-\d\d-\d\dT\S+) (\S+) (\S+) `)
	argRe        = regexp.MustCompile(`'([a-z_]+)': (\d+|True|False)`)
	jitRe        = regexp.MustCompile(`Triton kernel '([^']+)' took ([0-9.]+) s to compile`)
	completionRe = regexp.MustCompile(`"POST /v1/chat/completions HTTP/1\.[01]" 200`)
)

// ParseServerArgs reads the fields ServerArgs carries out of a server_args
// line; ok is false for any other line.
func ParseServerArgs(line string) (ServerArgs, bool) {
	idx := strings.Index(line, "server_args=")
	if idx < 0 {
		return ServerArgs{}, false
	}
	a := ServerArgs{Raw: line[idx+len("server_args="):]}
	for _, m := range argRe.FindAllStringSubmatch(a.Raw, -1) {
		n, _ := strconv.Atoi(m[2])
		b := m[2] == "True"
		switch m[1] {
		case "page_size":
			a.PageSize = n
		case "chunked_prefill_size":
			a.ChunkedPrefillSize = n
		case "decode_log_interval":
			a.DecodeLogInterval = n
		case "max_running_requests":
			a.MaxRunningRequests = n
		case "max_mamba_cache_size":
			a.MaxMambaCacheSize = n
		case "context_length":
			a.ContextLength = n
		case "enable_hierarchical_cache":
			a.HierarchicalCache = b
		case "enable_mixed_chunk":
			a.MixedChunk = b
		}
	}
	return a, true
}

// linePrefix splits the production prefix off a line. Bare lines return a zero
// time and an empty worker.
func linePrefix(raw string) (at time.Time, worker string) {
	m := prefixRe.FindStringSubmatch(raw)
	if m == nil {
		return time.Time{}, ""
	}
	t, err := time.Parse(time.RFC3339Nano, m[1])
	if err != nil {
		return time.Time{}, m[2]
	}
	return t, m[2]
}

// ParseBoots splits a log into process lifetimes. A server_args line opens a
// boot for the worker that printed it; every later line from that worker
// belongs to it until the worker prints another server_args line. Lines with
// no worker (the bare format) belong to the most recently opened boot, and a
// log with no server_args line at all is one boot with ArgsKnown=false, whose
// Steps are exactly what Parse returns.
func ParseBoots(log string) ([]Boot, error) {
	var boots []*Boot
	byWorker := map[string]*Boot{}
	var current *Boot
	open := func(worker string, at time.Time) *Boot {
		b := &Boot{Index: len(boots), Worker: worker, Start: at, EndedBy: "eof"}
		boots = append(boots, b)
		if worker != "" {
			byWorker[worker] = b
		}
		current = b
		return b
	}
	owner := func(worker string, at time.Time) *Boot {
		if worker != "" {
			if b, ok := byWorker[worker]; ok {
				return b
			}
		}
		if current == nil {
			return open(worker, at)
		}
		return current
	}
	for i, raw := range strings.Split(log, "\n") {
		at, worker := linePrefix(raw)
		if args, ok := ParseServerArgs(raw); ok {
			b := open(worker, at)
			b.Args, b.ArgsKnown = args, true
			b.End = at
			continue
		}
		s, ok, err := parseStep(raw, i+1)
		if err != nil {
			return nil, err
		}
		if ok {
			b := owner(worker, at)
			b.Steps = append(b.Steps, s)
			b.touch(at)
			continue
		}
		if worker == "" && current == nil {
			continue
		}
		switch {
		case strings.Contains(raw, "SIGTERM received"):
			b := owner(worker, at)
			b.EndedBy = "sigterm"
			b.touch(at)
		case strings.Contains(raw, "Received sigquit"),
			strings.Contains(raw, "SIGQUIT received"),
			strings.Contains(raw, "Scheduler hit an exception"):
			b := owner(worker, at)
			b.EndedBy = "crash"
			b.touch(at)
		case completionRe.MatchString(raw):
			b := owner(worker, at)
			b.Completions = append(b.Completions, at)
			b.touch(at)
		default:
			if m := jitRe.FindStringSubmatch(raw); m != nil {
				sec, _ := strconv.ParseFloat(m[2], 64)
				b := owner(worker, at)
				b.JITCompiles = append(b.JITCompiles, JITCompile{Kernel: m[1], Seconds: sec, At: at})
				b.touch(at)
			}
		}
	}
	out := make([]Boot, 0, len(boots))
	for _, b := range boots {
		if len(b.Steps) == 0 && !b.ArgsKnown {
			continue
		}
		out = append(out, *b)
	}
	if len(out) == 0 {
		return nil, errNoSteps
	}
	for i := range out {
		out[i].Index = i
	}
	return out, nil
}

func (b *Boot) touch(at time.Time) {
	if !at.IsZero() && at.After(b.End) {
		b.End = at
	}
}

// Timestamped reports whether this boot's lines carry wall-clock times.
func (b *Boot) Timestamped() bool {
	return len(b.Steps) > 0 && !b.Steps[0].At.IsZero()
}

// Summarize is Summarize over this boot's steps alone.
func (b *Boot) Summarize(chunkSize int) Metrics {
	return Summarize(b.Steps, chunkSize)
}

// ColdRun is one maximal run of consecutive cold chunks: [Start, End) indexes
// into the boot's Steps.
type ColdRun struct {
	Start, End int
	// Tokens is the run's prefill work; Pending is #pending-token on its first
	// chunk, the log's own statement of how much prompt was still outstanding.
	Tokens  int
	Pending int
	// RunningAtStart is #running-req on the first chunk: the conversations
	// that were decoding when the prompt took the GPU.
	RunningAtStart int
	QueuePeak      int
	// StartAt is the first chunk's timestamp and EndAt the timestamp of the
	// step after the run (the last chunk's own when there is none); zero in
	// the bare format.
	StartAt, EndAt time.Time
	// ColdChunks counts the cold chunks in the run. For a run from
	// ColdRunsAll it equals Chunks(); for a stretch from PrefillStretches
	// the other steps are the arrivals and follow-ups the stretch also ran.
	ColdChunks int
	// CtxStart is the context the run's first chunk extends from. A cold
	// prompt starts at 0; a prompt whose first chunk reused a cached prefix
	// logs that chunk with #cached-token > 0, so the cold chunks that follow
	// it start at prefix plus that chunk, and cost what that context costs.
	CtxStart int
}

// Chunks is the number of cold chunks in the run.
func (r ColdRun) Chunks() int { return r.End - r.Start }

// Indexes lists the run's step indexes, the shape ColdRuns uses.
func (r ColdRun) Indexes() []int {
	idx := make([]int, 0, r.Chunks())
	for i := r.Start; i < r.End; i++ {
		idx = append(idx, i)
	}
	return idx
}

// ColdRunsAll lists every maximal run of consecutive cold chunks in this
// boot, in order, one chunk or longer. coldWindow keeps only the longest of
// these; a day-long log holds many prompts, and each one starves decode on
// its own.
func (b *Boot) ColdRunsAll(chunkSize int) []ColdRun {
	var runs []ColdRun
	steps := b.Steps
	for i := 0; i < len(steps); {
		if !isColdChunk(steps[i], chunkSize) {
			i++
			continue
		}
		r := ColdRun{Start: i, Pending: steps[i].Pending, RunningAtStart: steps[i].RunningReq, StartAt: steps[i].At}
		if i > 0 {
			if p := steps[i-1]; p.Kind == Prefill && p.NewSeq == 1 && p.HitTokens > 0 && p.NewTokens == chunkSize {
				r.CtxStart = p.HitTokens + p.NewTokens
			}
		}
		j := i
		for ; j < len(steps) && isColdChunk(steps[j], chunkSize); j++ {
			r.Tokens += steps[j].NewTokens
			if steps[j].QueueReq > r.QueuePeak {
				r.QueuePeak = steps[j].QueueReq
			}
		}
		r.End = j
		r.ColdChunks = j - i
		if j < len(steps) {
			r.EndAt = steps[j].At
		} else {
			r.EndAt = steps[j-1].At
		}
		runs = append(runs, r)
		i = j
	}
	return runs
}

// PrefillStretches lists every maximal run of consecutive prefill steps
// holding at least minCold cold chunks. A stretch is what a stall measures:
// the GPU ran nothing but prefill from its first step to the decode step
// after it, however many prompts arrived and joined in the meantime. Its
// Tokens count every prefill step's work, its RunningAtStart is #running-req
// on the first step that had any, and its CtxStart is the first cold run's.
func (b *Boot) PrefillStretches(chunkSize, minCold int) []ColdRun {
	var out []ColdRun
	steps := b.Steps
	cold := b.ColdRunsAll(chunkSize)
	for i := 0; i < len(steps); {
		if steps[i].Kind != Prefill {
			i++
			continue
		}
		j := i
		for ; j < len(steps) && steps[j].Kind == Prefill; j++ {
		}
		end := j
		// Prefill with nothing decoding starves no one, so the stretch
		// starts at the first step that had a request running; a stretch
		// that never did keeps its first step and RunningAtStart 0.
		first := i
		for k := i; k < end; k++ {
			if steps[k].RunningReq > 0 {
				first = k
				break
			}
		}
		r := ColdRun{Start: first, End: end, Pending: steps[first].Pending, StartAt: steps[first].At,
			RunningAtStart: steps[first].RunningReq, CtxStart: -1}
		for k := first; k < end; k++ {
			s := steps[k]
			r.Tokens += s.NewTokens
			if s.QueueReq > r.QueuePeak {
				r.QueuePeak = s.QueueReq
			}
		}
		for _, c := range cold {
			if c.End <= first || c.Start >= end {
				continue
			}
			from := c.Start
			if from < first {
				from = first
			}
			r.ColdChunks += c.End - from
			if r.CtxStart < 0 {
				r.CtxStart = c.CtxStart + (from-c.Start)*chunkSize
			}
		}
		if r.CtxStart < 0 {
			r.CtxStart = 0
		}
		if j < len(steps) {
			r.EndAt = steps[j].At
		} else {
			r.EndAt = steps[j-1].At
		}
		if r.ColdChunks >= minCold {
			out = append(out, r)
		}
		i = j
	}
	return out
}

// Stall is a stretch where requests were decoding, a prefill took the GPU,
// and no decode step ran again for Seconds: the log's direct measurement of
// decode starvation, available only with timestamps.
type Stall struct {
	// Start is the first prefill step logged with running requests, End the
	// next decode step; [Start, End) indexes into the boot's Steps.
	Start, End int
	Seconds    float64
	// PrefillTokens is all prefill work in the window and ColdTokens the part
	// with no prefix hit.
	PrefillTokens int
	ColdTokens    int
	// RunningReq is #running-req on the first prefill and RunningPeak the
	// most any prefill in the window reported (arrivals join the count).
	RunningReq  int
	RunningPeak int
	QueuePeak   int
	// MaxHit is the largest #cached-token any prefill in the window reused.
	MaxHit int
	At     time.Time
}

// Stalls lists every stall of at least minSeconds in this boot. Without
// timestamps there is no wall clock to measure against and it returns nil.
func (b *Boot) Stalls(minSeconds float64) []Stall {
	if !b.Timestamped() {
		return nil
	}
	steps := b.Steps
	var out []Stall
	for i := 0; i < len(steps); i++ {
		s := steps[i]
		if s.Kind != Prefill || s.RunningReq < 1 || s.At.IsZero() {
			continue
		}
		j := i + 1
		st := Stall{Start: i, RunningReq: s.RunningReq, At: s.At}
		for ; j < len(steps) && steps[j].Kind == Prefill; j++ {
		}
		for k := i; k < j; k++ {
			p := steps[k]
			st.PrefillTokens += p.NewTokens
			if p.HitTokens == 0 {
				st.ColdTokens += p.NewTokens
			}
			if p.HitTokens > st.MaxHit {
				st.MaxHit = p.HitTokens
			}
			if p.QueueReq > st.QueuePeak {
				st.QueuePeak = p.QueueReq
			}
			if p.RunningReq > st.RunningPeak {
				st.RunningPeak = p.RunningReq
			}
		}
		if j >= len(steps) {
			break
		}
		st.End = j
		st.Seconds = steps[j].At.Sub(s.At).Seconds()
		if st.Seconds >= minSeconds {
			out = append(out, st)
		}
		i = j
	}
	return out
}
