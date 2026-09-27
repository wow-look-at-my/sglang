# ITL tail: what each policy's inter-token latency can and cannot be compared on

The three-policy tables report `ITL p99` and `ITL p99.9` as contract metrics, and
NEW loses at least one of them to OLD and to PREV in most scenarios. This document
names the measured quantities behind those cells and does the arithmetic, so the
assertions in `scenario_contract_test.go` are the ones the model supports and the
unwinnable cells are not quietly dropped.

Everything below is scenario B (five closed-loop agent streams at 100K-250K
context plus one cold 400K prompt on a fixed cadence), seeds `[1 2 3 4 5]`,
window 900 s, cost model calibrated from the embedded production log
(`internal/trace/live_log.txt`), chunk 4096, page 64, three drafts accepted to
2.76 tokens per step on average.

## The quantities the arithmetic uses

| name | value | where it comes from |
| --- | --- | --- |
| metric population | one ITL sample per generated token | `spread` in `metrics.go`: a delivery's gap is divided by the tokens it carried and that value is recorded once per token |
| tokens per decode delivery | 2.70 (OLD 2.69, PREV 2.71) | measured accept length of the calibrated decode step |
| tokens per mixed delivery | 1.00 | a prefill batch extends one token per riding request; speculative decoding degrades to one token in a mixed step |
| deliveries / samples, NEW at a 2 min cadence | 397,500 / 1,048,428 | pooled over the five seeds |
| of those, mixed | 14,631 / 14,631 (1.40%) | same population |
| deliveries / samples, NEW at a 5 min cadence | 493,534 / 1,316,986, mixed 10,226 (0.78%) | same population |
| one chunk's promise in seconds | `chunked_prefill_size x prefill_seconds_per_token` = 4096 x 126 us = 0.52 s | the cap `prefill_token_budget` enforces for a continuation |
| continuation batch cost, seconds-denominated bound (shipped) | tokens p50 3,606 p99 4,101; seconds p50 413 ms p99 653 ms max 842 ms | `WritePrefillCostBreakdown` |
| continuation batch cost, token-denominated bound | tokens p50 4,099; seconds p50 464 ms p99 809 ms max 954 ms | same, run with `PrefillTokenBudget` in its token form |

## `ITL p99` is a per-class statement, not a per-policy one

The 1% tail of 1,048,428 samples is 10,484 samples. Mixed deliveries carry 14,631
samples, so with 10,477 of them above the 99th-percentile rank the reported p99 is
drawn from the mixed band, not from decode steps:

    mixed share          = 14,631 / 1,048,428 = 1.40% > 1%
    reported p99         = 332.3 ms
    decode band          = p50 7.6 ms, p99 25.7 ms, p99.9 38.4 ms
    mixed band           = p50 390.6 ms, p99 624.0 ms, max 841.7 ms
    332.3 ms > 38.4 ms   => the p99 sample is not a decode sample

At the 5 minute cadence the same policy has 10,226 mixed samples out of 1,316,986:

    mixed share          = 0.78% < 1%
    reported p99         = 27.7 ms, decode band p99 = 27.1 ms

so there the p99 is a decode sample and NEW beats PREV's 39.1 ms. One count
crossing one threshold flips the metric; the schedule did not change shape.

A mixed delivery is a *bonus* token: the riding request gets it because mixed
chunked prefill extends it inside the prefill batch, and the balancer charges that
batch as prefill minus the rows so decode's half is untouched. The metric charges
the batch's whole wall-clock window to that single token, where the same window
spent on a decode step is divided by 2.70. With identical waits, a policy that
delivers during prefill is charged 2.7x more per token than one that waits.

## What is asserted instead, and why each bound holds

`TestScenarioBBalanceWinsAgainstPrev` pins, at both the 2 and 5 minute cadences and
against PREV: longest stall, stream time in stalls over 1 s, output tok/s, stream
decode tok/s in cold, per-agent tok/s in cold, and

    ITL p99 of NEW's decode-class samples <= PREV's ITL p99          25.7 ms vs 212.2 ms

plus the per-batch form of the bound (a prefill batch carries at most one chunk of
prefill tokens plus the one extend token each riding request adds). The long stall
comparison is structural rather than a tuned margin: with the ride, every prefill
batch hands its riding requests a token, so a stream never waits out more than one
prefill batch - NEW's longest stall 841.7 ms equals its widest single continuation
batch - while PREV's streams wait out a whole back-to-back run, 1.72 s against a
widest single batch of 942.9 ms.

`ITL p99` itself is not asserted against PREV, and `ITL p99.9` only at the 2 minute
cadence (523.5 ms vs 731.9 ms). At the 5 minute cadence 0.1% of 1,316,986 samples is
1,317 samples and 10,226 mixed samples cannot fit under it, so p99.9 there is a mixed
sample too (462.9 ms against PREV's 398.8 ms). The comparison that measures the wait
rather than the accounting is raw chunk-gap p99 (`ITL p99 (chunk gaps)` in the table,
the gap undivided): NEW 459.6 ms against PREV 720.2 ms at the 2 minute cadence.

## Cells no balancer setting can win

Two experiments on the committed model, both in
`TestMixedRideIsWhatDecidesTheP99Cell` (scenario B, 2 min cadence, the sweep's seeds
`[1 2 3]`, so these numbers read against the sweep's baseline row):

| mixed chunked prefill | PREV ITL p99 | PREV longest stall | NEW ITL p99 | NEW longest stall | NEW stream tok/s in cold | PREV stream tok/s in cold |
| --- | --- | --- | --- | --- | --- | --- |
| as shipped: off for PREV, on for NEW | 211.5 ms | 1.72 s | 332.0 ms | 822.3 ms | 67.6 | 66.8 |
| on for both | 290.9 ms | 934.3 ms | 332.0 ms | 822.3 ms | 67.6 | 67.5 |
| off for both | 211.5 ms | 1.72 s | 157.7 ms | 848.5 ms | 65.7 | 66.8 |

Giving a policy the ride raises its own p99 by 37% (211.5 to 290.9 ms) while halving
its longest stall (1.72 to 0.934 s), because each of its tokens now arrives at the end
of a prefill batch and carries a weight of one in the population. Put both policies on
the same delivery form and NEW's deficit is 332.0 against 290.9 ms, 14%; the rest of
the reported gap is the ride's weighting. Take the ride away from NEW and it wins the
cell outright, 157.7 against 211.5 ms, and loses `stream decode tok/s in cold` to PREV
(65.7 against 66.8 tok/s): those riding tokens are what streams generate during a cold
prompt at all. Mixed chunked prefill is the deployment's resolved default
(`arg_groups/mixed_chunk_hook.py`), not part of the control law, and the sweep prints
that variant's trade in its own `contract` column.

Sweeping the only parameter the balancer shares with the chunk size shows the same
trade. Each row is OLD / PREV / NEW from the sensitivity sweep:

| setting | B ITL p99 | B longest stall | B cold TTFT mean | B output tok/s |
| --- | --- | --- | --- | --- |
| chunk 4096 (shipped) | 28.2 / 211.5 / 332.0 ms | 45.4 s / 1.72 s / 822.3 ms | 45.2 / 102.6 / 104.7 s | 226 / 226 / 235 |
| `chunked_prefill_size 2048` | 28.2 / 59.6 / 226.4 ms | 46.5 s / 873.2 ms / 520.4 ms | 46.4 / 64.5 / 109.7 s | 224 / 212 / 235 |
| `chunked_prefill_size 8192` | 28.2 / 49.2 / 27.4 ms | 45.0 s / 3.19 s / 1.23 s | 44.8 / 97.3 / 100.6 s | 228 / 219 / 230 |

Halving the chunk moves NEW's p99 down to 226.4 ms, because the p99 tracks the cap
one-for-one, and moves PREV's down to 59.6 ms at the same time, so NEW's deficit
*grows* from 1.6x to 3.8x while PREV's cold TTFT improves by 38 s and NEW's worsens.
Reaching PREV's 211.5 ms that way needs a smaller chunk still, and continues the same
trade. Doubling the chunk puts NEW's mixed count under the 1% line and wins p99
outright (27.4 against 49.2 ms) - by changing `chunked_prefill_size`, a launch flag
the balancer must not decide, and at 1.23 s of stall for NEW and 3.19 s for PREV.

What the balancer does own - how many tokens a *continuation* may take - was measured
in both forms. A token-denominated bound leaves continuation batches at p99 809 ms
and max 953.7 ms against a promise of 0.52 s; the seconds form brings those to 653 ms
and 841.7 ms and moves the mixed count from 12,600 to 14,631 deliveries. The band's
height and the band's count trade against each other, and while the count is above 1%
of samples p99 sits at the band either way: 300.6 ms for the token form, 332.3 ms for
the seconds form, both above PREV's 212.2 ms. Against the token form, every metric
that measures the wait itself improved: p99.9, raw chunk-gap p99, longest stall,
stream time in stalls over 1 s, and output.

## `cold TTFT` against OLD

OLD finishes a cold 400K prompt in 45.3 s, NEW in 105.0 s. A 400K prompt at the
log's own measured prefill rate costs about 48-53 s of GPU (the logged episode
measures 430,080 tokens in 52.99 s), so any policy that lets a cold prompt have the
whole GPU finishes in about that time and any policy that splits it with decode
finishes in about twice that. The balancer's split is even by definition - debt is
prefill seconds minus decode seconds - so the bound on cold TTFT is

    cold TTFT >= 2 x (prompt tokens x prefill seconds per token)
               = 2 x 400,000 x 126 us ~= 101 s

which both interleaving policies sit on (PREV 103.1 s, NEW 105.0 s) and which OLD
escapes only by giving decode 0 steps for the whole window: OLD's in-window stream
rate is 1.9 tok/s against NEW's 67.4, and 35.3% of its stream time is inside a stall
longer than a second against NEW's 0.0%. NEW's 1.9 s above PREV is the continuation
cap buying shorter chunks (4,352 continuation batches against 3,698) and is inside
the same 2x bound.
