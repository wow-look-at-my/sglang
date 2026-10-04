# Replaying a serving log

`go run ./cmd/schedsim -log FILE` accepts a production log as the server writes it. Every line carries an RFC3339 timestamp, the worker id and the level before the rank tag (`2026-09-27T00:15:24.123Z 5ylyr21v-rhkt6 INFO TP0] Prefill batch, ...`). One file holds every process lifetime of the deployment. The bare `TP0] ...` format, with no prefix, also parses. `-log` is required: the command reads no log of its own, and no serving log is kept in the repository. The tests generate their logs with `internal/trace/tracetest`.

## What the log gives that one incident could not

**Boots.** `trace.ParseBoots` splits the file at every `server_args=` line. It attributes each later line to the worker that printed it, so a rolling update's overlap and a crash's auto-relaunch are separate lifetimes. This is with their own arguments, steps, HTTP completions, JIT-compile lines and end (`sigterm`, `crash`, `eof`). Nothing that walks steps in order ever crosses a boot, which is what kept the old fit/holdout pairing. This is from splicing prompts of different context lengths into one run.

**Timestamps.** `Step.At` is the line's wall clock. `Boot.Stalls(min)` reads decode starvation straight off it: a prefill line logged with requests running, followed by no decode line for at least `min` seconds. The day-long corpus records stalls of 5 s or more, the longest 148.8 s.

**Every prompt.** `Boot.ColdRunsAll` lists every run of cold chunks and `Boot.PrefillStretches` every stretch of consecutive prefill steps holding at least eight of them. This is starting at the first step that had a request decoding. A stretch is what a stall measures: however many prompts arrived and joined in, no decode step ran until it ended.

## The three sections a timestamped log adds

**Boots in the log.** One row per lifetime: how it ended, its steps, cold stretches, stalls over 5 s, and the kernels it compiled after serving started.

**Every cold prompt in the log.** One row per stretch, with the measured stall beside the single-window policies (`OLD` prefill-priority, `NEW` time-balance, `REV` queue-balance with mixed chunk) run over that stretch's own measured chunk costs. The aggregate's calibration line is the check to read first: `OLD`'s simulated gap over the measured stall, at the median and the 10th. This is 90th percentiles. On the corpus it is 1.07 (0.99 to 1.41): the old policy's model reproduces the stalls the log recorded, so the `NEW` column beside it is worth reading. Over the stretches that began with requests decoding, the sum of longest gaps falls from 1,791 s to 225 s and the worst from 151.8 s to 21.5 s. This is while each cold prompt takes about twice as long to finish (the `done` columns).

**Replaying the log's own traffic.** Each boot that stalled is rebuilt as a workload (`sim.BuildLogReplay`) and run through the full engine under `OLD`, `PREV` and `NEW`. This is every request at its logged arrival, with the prompt size and prefix hit the log reported, over a window that opens many minutes. This is before the boot's first stall and runs `-replay-seconds` (3,600 s by default). A conversation's next turn is released at its logged arrival or when its previous reply finishes, whichever is later. Row 1 runs the deployment as launched. Row 2 as the fork resolves by default (host tier `-replay-host-mul` times the device pool).

Conversations and reply lengths are not in the log. Both come from one observation. An agent's next turn carries the whole previous turn as its prefix, so a request whose `#cached-token` equals a known conversation's last prompt plus. Some tokens is that conversation's next turn. The excess is the reply (known this way for 85–90% of turns. The median stands in for the rest). A cold request the size of an idle conversation is that conversation returning after eviction. A line admitting several requests does not itemise them. Its follow-ups go to the most recently idle conversations whose last prompts add up to the line's hit. `#pending-token` counts the queue behind a chunked prompt too, so a prompt's size is the sum of its own chunk lines, not the pending figure.

## Reading the replay's anchors

The `measured` column is the log's own record of the same window: its stalls, completions, and cold prompts that were conversations returning. The `OLD` column beside it is the fidelity check for the engine, since `OLD` is the policy the log ran.

- Longest stall: `OLD` lands within the measured range on most boots but not all (boot 4's 148.8 s pile-up simulates at 47 s). This is because the replay is closed-loop and a turn that finishes at a different time shifts the arrivals behind it.
- Full-prefix recomputes: `OLD` overstates them (124 against 33 on boot 4, 109 against 12 on boot 7). The engine's pool evicts whole conversations, while the server's radix cache evicts pages least-recently-used across every conversation. As a result, a real eviction usually leaves a partial hit the next turn reuses. Until the pool models page-level eviction, read the recompute and completed-turn columns as directional and the stall columns as the result. The per-stretch table above them carries the calibrated stall comparison.
- Completed turns under `NEW` can trail the measurement on a boot that ran as launched (no host tier): cold prompts finish later under the balancer. As a result, this is their conversations' later turns fall outside the window. With the fork's host tier the same boots complete every logged turn.

## Calibration across boots

`trace.CalibrateBoots` fits the prefill cost on the largest cold run in the log. It holds out every other run of multiple chunks or more, per boot, reporting the median and worst holdout error. On the corpus the median is 4.7% many runs. A run that continues a partly cached prompt is priced from that context, not zero (`ColdRun.CtxStart`). The wall clock checks the model too: the gap between a cold chunk's line. The next batch line is what the chunk took, and its ratio to `#new-token / input throughput` is 1.01 at the median with a 2 ms base overhead. This is against the assumed `PrefillBaseSeconds` of 10 ms. The page size comes from the boot's `server_args` when the log has one.

## Tests

`scripts/coverage.sh` runs the module's tests with a coverage profile and fails if any file of this work (the trace layer, `WorkloadFromRun`, the log replay, the report, the command) is under 95% statement coverage. Fork CI runs it on every push. `internal/trace/tracetest` writes production-format logs for the tests, so a test states the traffic (a boot header, decode lines, a chunked cold prompt behind running requests, completions, a drain). The test checks what the parser, the stall measurement, the calibration and the replay make of it, without carrying a corpus around.
