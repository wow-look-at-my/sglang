// Package tracetest writes serving logs in the production format for tests:
// timestamped, worker-prefixed lines with the server_args header, batch
// lines, completions, kernel compiles and the signals that end a boot. A
// test states what the deployment did and reads back what the parser and
// the models make of it, without carrying a corpus around.
package tracetest

import (
	"fmt"
	"strings"
	"time"
)

// Log accumulates lines. The clock advances as the test says it did; each
// line carries the clock's value when it was written.
type Log struct {
	Worker string
	Level  string
	Now    time.Time
	// Bare writes every line in the bare format, with no timestamp, worker or level.
	Bare bool
	// Prompts lists every prompt Prompt wrote in more than one chunk.
	Prompts []Prompt
	lines   []string
}

// Start is the wall clock the canned logs begin at.
var Start = time.Date(2026, 9, 27, 0, 0, 0, 0, time.UTC)

// New opens a log for worker at the given time.
func New(worker string, at time.Time) *Log {
	return &Log{Worker: worker, Level: "INFO", Now: at}
}

// Advance moves the clock by seconds.
func (l *Log) Advance(seconds float64) *Log {
	l.Now = l.Now.Add(time.Duration(seconds * float64(time.Second)))
	return l
}

// Line writes body behind the production prefix.
func (l *Log) Line(body string) *Log {
	if l.Bare {
		return l.Raw(body)
	}
	l.lines = append(l.lines, fmt.Sprintf("%s %s %s %s", l.Now.Format(time.RFC3339Nano), l.Worker, l.Level, body))
	return l
}

// Raw writes a line with no prefix, the bare format.
func (l *Log) Raw(body string) *Log {
	l.lines = append(l.lines, body)
	return l
}

// Args is the subset of server_args a log states.
type Args struct {
	PageSize, ChunkedPrefillSize, DecodeLogInterval, MaxRunningRequests, MaxMambaCacheSize, ContextLength int
	HierarchicalCache, MixedChunk                                                                         bool
}

// DefaultArgs is the deployment the corpus ran.
var DefaultArgs = Args{PageSize: 64, ChunkedPrefillSize: 4096, DecodeLogInterval: 40, MaxRunningRequests: 16, MaxMambaCacheSize: 48, ContextLength: 262144}

func pyBool(b bool) string {
	if b {
		return "True"
	}
	return "False"
}

// ServerArgs writes the boot header.
func (l *Log) ServerArgs(a Args) *Log {
	return l.Line(fmt.Sprintf("server_args={'model_path': '/repository', 'context_length': %d, 'page_size': %d, "+
		"'chunked_prefill_size': %d, 'decode_log_interval': %d, 'max_running_requests': %d, 'max_mamba_cache_size': %d, "+
		"'enable_hierarchical_cache': %s, 'enable_mixed_chunk': %s}",
		a.ContextLength, a.PageSize, a.ChunkedPrefillSize, a.DecodeLogInterval, a.MaxRunningRequests, a.MaxMambaCacheSize,
		pyBool(a.HierarchicalCache), pyBool(a.MixedChunk)))
}

// Prefill is one prefill batch line's fields.
type Prefill struct {
	NewSeq, NewTokens, Hit, Running, Queue, Pending int
	Usage                                           float64
	TPS                                             float64
}

// PrefillBody is the bare body of a prefill line. A zero TPS is written as
// 13874.93, the corpus's steady rate, and NewSeq 0 as 1.
func PrefillBody(p Prefill) string {
	if p.TPS == 0 {
		p.TPS = 13874.93
	}
	if p.NewSeq == 0 {
		p.NewSeq = 1
	}
	return fmt.Sprintf("TP0] Prefill batch, #new-seq: %d, #new-token: %d, #cached-token: %d, full token usage: %.2f, "+
		"mamba usage: 0.33, #running-req: %d, #queue-req: %d, #pending-token: %d, cuda graph: False, input throughput (token/s): %.2f",
		p.NewSeq, p.NewTokens, p.Hit, p.Usage, p.Running, p.Queue, p.Pending, p.TPS)
}

// Prefill writes a prefill line and advances the clock by the time the line
// itself says the batch took (tokens over its rate), so wall clock and model
// agree the way they do in the corpus.
func (l *Log) Prefill(p Prefill) *Log {
	l.Line(PrefillBody(p))
	tps := p.TPS
	if tps == 0 {
		tps = 13874.93
	}
	return l.Advance(float64(p.NewTokens) / tps)
}

// Decode is one decode batch line's fields.
type Decode struct {
	Running, FullTokens, Queue int
	Usage, Accept, TPS         float64
}

// DecodeBody is the bare body of a decode line. Zero Accept and TPS are
// written as 2.78 and 190.0, one healthy single-request step.
func DecodeBody(d Decode) string {
	if d.Accept == 0 {
		d.Accept = 2.78
	}
	if d.TPS == 0 {
		d.TPS = 190.0
	}
	if d.Running == 0 {
		d.Running = 1
	}
	return fmt.Sprintf("TP0] Decode batch, #running-req: %d, #full token: %d, full token usage: %.2f, mamba num: %d, "+
		"mamba usage: 0.04, accept len: %.2f, accept rate: 0.70, cuda graph: True, gen throughput (token/s): %.2f, #queue-req: %d",
		d.Running, d.FullTokens, d.Usage, d.Running, d.Accept, d.TPS, d.Queue)
}

// Decode writes a decode line and advances the clock by one log interval of
// steps at the line's own rate (40 steps of accept/TPS seconds each).
func (l *Log) Decode(d Decode) *Log {
	l.Line(DecodeBody(d))
	accept, tps := d.Accept, d.TPS
	if accept == 0 {
		accept = 2.78
	}
	if tps == 0 {
		tps = 190.0
	}
	return l.Advance(40 * accept / tps)
}

// Completion writes one finished request.
func (l *Log) Completion() *Log {
	return l.Line(`10.41.31.0:58753 - "POST /v1/chat/completions HTTP/1.1" 200 OK`)
}

// JIT writes a serving-time kernel compile.
func (l *Log) JIT(kernel string, seconds float64) *Log {
	return l.Line(fmt.Sprintf("TP0] Triton kernel '%s' took %.2f s to compile after serving started.", kernel, seconds))
}

// Sigterm ends the boot with a drain.
func (l *Log) Sigterm() *Log { return l.Line("SIGTERM received. Shutting down.") }

// Crash ends the boot with a child failure.
func (l *Log) Crash() *Log {
	return l.Line("Received sigquit from a child process. It usually means the child failed.")
}

// String is the log text.
func (l *Log) String() string { return strings.Join(l.lines, "\n") + "\n" }

// Steady writes n decode lines with running requests decoding.
func (l *Log) Steady(n, running, fullTokens int) *Log {
	for i := 0; i < n; i++ {
		l.Decode(Decode{Running: running, FullTokens: fullTokens, Usage: float64(fullTokens) / 1.4e6})
	}
	return l
}

// ColdPrompt writes a chunked cold prompt of tokens, running behind
// running decoding requests: the first chunk reports the whole prompt
// pending, each later chunk what is left.
func (l *Log) ColdPrompt(tokens, chunk, running, queue int) *Log {
	left := tokens
	for left > 0 {
		n := chunk
		if left < n {
			n = left
		}
		l.Prefill(Prefill{NewTokens: n, Running: running, Queue: queue, Pending: left, Usage: 0.5})
		left -= n
	}
	return l
}

// Stalled is a canned boot: server_args, a warm-up, a few conversations
// decoding, a cold prompt of chunks*4096 tokens that stalls them, and a
// return to decoding, ending in a drain. It exercises every section a
// timestamped log adds to the report.
func Stalled(worker string, chunks int) *Log {
	l := New(worker, Start).ServerArgs(DefaultArgs)
	l.Advance(1).JIT("_fwd_kernel", 1.5)
	// Four conversations arrive with a shared 12k prefix and decode.
	for i := 0; i < 4; i++ {
		l.Prefill(Prefill{NewTokens: 1500, Hit: 12288, Running: i, Usage: 0.2})
	}
	l.Steady(6, 4, 300000)
	l.ColdPrompt(chunks*4096, 4096, 4, 1)
	l.Steady(6, 5, 700000)
	// A follow-up turn on the first conversation, then the answers land.
	l.Prefill(Prefill{NewTokens: 800, Hit: 14000, Running: 4, Usage: 0.5})
	l.Steady(6, 5, 700000)
	for i := 0; i < 5; i++ {
		l.Completion()
	}
	l.Steady(3, 1, 20000)
	return l.Sigterm()
}
