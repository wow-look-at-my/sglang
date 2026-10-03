# Inter-token tail under mixed chunked prefill

What each policy's inter-token latency can and cannot be compared on. The three-policy tables report `ITL p99` and `ITL p99.9` as contract metrics. NEW loses at least one of them to OLD and to PREV in most scenarios. This document names the measured quantities behind those cells and does the arithmetic. As a result, the assertions in `scenario_contract_test.go` are the ones the model supports and the unwinnable cells are not quietly dropped.

Everything below is scenario B (closed-loop agent streams at 100K-250K context plus one cold 400K prompt on a fixed cadence), seeds `[1 2 3 4 5]`, window 900 s, cost model calibrated from the embedded production log (`internal/trace/live_log.txt`), chunk 4096, page 64. Drafts accepted to 2.76 tokens per step on average.

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
| continuation batch cost, seconds cap (NEW, shipped) | 4,352 batches; tokens p50 3,606 p99 4,101; seconds p50 413.3 ms p99 652.7 ms max 841.7 ms | `WritePrefillCostBreakdown` |
| continuation batch cost, uncapped (PREV) | 3,698 batches; tokens p50 4,096; seconds p50 463.0 ms p99 825.8 ms max 942.9 ms | same |

## `ITL p99` is a per-class statement, not a per-policy one

The p99 cut of 1,048,428 samples names 10,485 of them. Mixed deliveries carry 14,631 samples and 10,477 of those sit above the cut. As a result, the reported p99 is drawn from the mixed band, not from decode steps:

    mixed share          = 14,631 / 1,048,428 = 1.40% > 1%
    reported p99         = 332.3 ms
    decode band          = p50 7.6 ms, p99 25.7 ms, p99.9 38.4 ms
    mixed band           = p50 390.6 ms, p99 624.0 ms, max 841.7 ms
    332.3 ms > 38.4 ms   => the p99 sample is not a decode sample

At the 5 minute cadence the same policy has 10,226 mixed samples out of 1,316,986:

    mixed share          = 0.78% < 1%
    reported p99         = 27.7 ms, decode band p99 = 27.1 ms

so there the p99 is a decode sample and NEW beats PREV's 39.1 ms. One count crossing one threshold flips the metric. The schedule did not change shape.

A mixed delivery is a *bonus* token: the riding request gets it because mixed chunked prefill extends it inside the prefill batch. The balancer charges that batch as prefill minus the rows so decode's half is untouched. The metric charges the batch's whole wall-clock window to that single token, where the same window spent on a decode step is divided by 2.70. With identical waits, a policy that delivers during prefill is charged 2.7x more per token than one that waits.

## What is asserted instead, and why each bound holds

`TestScenarioBBalanceWinsAgainstPrev` pins, at both the 2 and cadences and against PREV. Longest stall, stream time in stalls over 1 s, output tok/s, stream decode tok/s in cold, per-agent tok/s in cold, and

    ITL p99 of NEW's decode-class samples <= PREV's ITL p99          25.7 ms vs 212.2 ms

plus the per-batch form of the bound (a prefill batch carries at most one chunk of prefill tokens plus the extend token each riding request adds). The long stall comparison is structural rather than a tuned margin. With the ride, every prefill batch hands its riding requests a token. As a result, a stream never waits out more than one prefill batch - NEW's longest stall 841.7 ms equals its widest single continuation batch. This is while PREV's streams wait out a whole back-to-back run, 1.72 s against a widest single batch of 942.9 ms.

`ITL p99` itself is not asserted against PREV, and `ITL p99.9` only at the 2 minute cadence (523.5 ms vs 731.9 ms). At the 5 minute cadence 0.1% of 1,316,986 samples is 1,317 samples and 10,226 mixed samples cannot fit under it, so p99.9 there is a mixed sample too (462.9 ms against PREV's 398.8 ms). The comparison that measures the wait rather than the accounting is raw chunk-gap p99 (`ITL p99 (chunk gaps)` in the table, the gap undivided): NEW 459.6 ms against PREV 720.2 ms at the 2 minute cadence.

## Scenario A: what the 406 samples above the p99 cut are

`boundITLPercentile` argues a lost p99 or p99.9 cell only after it can say which class of delivery the named percentile falls. This is in, because the population is not a list of waits: `spread` in `metrics.go` divides a delivery's gap by the tokens that delivery carried. It files the result once per token. As a result, a mixed batch hands each riding request one token and files that batch's whole wall clock as a single sample. This is while a decode step files the same wait 2.7 times over at a third of the size. The precondition is therefore a count, not a share threshold, and the run prints it.

A, seed 7, the logged episode. NEW files 40,517 per-token samples, so a p99.0 names 406 of them:

| quantity | value | source |
| --- | --- | --- |
| samples in NEW's metric population | 40,517 | the bound's printed premise |
| slots a p99.0 names | 406 | 1% of the population |
| samples that ride a mixed batch, whole population | 361 | printed premise |
| decode-class samples that waited out a pass they were not a row on | 53 | printed premise |
| inside the tail: mixed 359, waited 47 | 406, which is the whole cut | printed premise |
| of the 53 waiting samples, first gap / mid-stream | 53 / 0 | printed premise |
| seconds the tail samples waited, total | 158.3 s | printed premise |
| share of that wait inside no launched pass | 0.00% | `newGPUTimeline` |
| prefill passes in NEW's widest gap | 1 | printed premise |
| NEW's reported p99 / longest launched pass | 77.6 ms / 748.2 ms | the metric, the batch trace |
| NEW's p99 within its decode class | 23.8 ms | the population with the pass-cost samples removed |
| OLD's reported p99 | 26.6 ms | the metric |

Mixed samples alone (361) do not fill the cut (406), so the *named* sample is not a mixed ride. It is the largest non-mixed one, at 77.6 ms. Together the classes fill it exactly, which is the precondition the bound checks before it argues anything. A cell where neither class reaches the cut is a plain decode sample no forward pass explains. The bound fails it outright rather than excusing it.

What prices the remaining 47 is one prefill pass, not two and not idle hardware. The widest gap spans one pass, 0.00% of the waited seconds fall outside a launched pass. All 47 are a stream's first gap - the wait between the token its own prefill sampled and the first token after it. The ceiling is what one such pass can cost at the worst context this run reached, built from the same terms. The scheduler caps a continuation with:

    chunk at this run's deepest mid-context   897.0 ms
    riding rows, 2 at the marginal rate          0.4 ms
    the largest host-tier copy any pass paid   282.7 ms
    calibrated step at the largest decode batch  23.9 ms
    ceiling                                    1.20 s  >  NEW's 77.6 ms and > 748.2 + 23.9 ms

The pass is a continuation of another request's chunked prompt, already under the seconds cap `prefill_token_budget` imposes. As a result, no further bound on chunk size is available to the scheduler here. Nor is the alternative delivery form better for the stream. A first token that rode the pass will carry exactly one token and file the pass's whole seconds as its sample. This is moving the gap from a divided wait to the mixed band instead of out of it.

The improvement claim in this cell is the last a couple of rows. Take the pass-cost samples out and NEW's own decode steps reach 23.8 ms at their 99th percentile against OLD's reported 26.6 ms. As a result, NEW is not the policy with the slower steps. The cell is lost on the weighting above. The bound fails the build the day a control law makes NEW's own steps the slower ones.

At p99.9 in the same scenario the cut is samples and 41 of them are mixed rides. As a result, the named sample is a ride under the longest forward pass (479.3 ms below 748.2 ms), and NEW's decode class reaches 89.7 ms there against OLD's 196.7 ms.

OLD's own numbers are why this cell is not evidence that it serves streams better. In the same run it delivers 0.0 tok/s of stream decode inside its cold windows and has a longest stall of 76.1 s against NEW's 748.2 ms. A stall does reach OLD's percentile population - the token that ends it files one sample of that size - but one sample. This is in 40,517 cannot rise to a rank that names 406. As a result, OLD's p99 is drawn from the waits between ordinary decode steps and never sees the stall. Where the stall share is printed (`TestScenarioBBalanceWinsAgainstPrev`, scenario B), OLD spends 35.14% of stream time inside a stall longer than a second and NEW 0.00%.

## Cells no balancer setting can win

Experiments on the committed model, both printed by `TestMixedRideIsWhatDecidesTheP99Cell` (scenario B, 2 min cadence, the sweep's seeds `[1 2 3]`, so these numbers read against the sweep's baseline row):

| mixed chunked prefill | PREV ITL p99 | PREV longest stall | NEW ITL p99 | NEW longest stall | NEW stream tok/s in cold | PREV stream tok/s in cold |
| --- | --- | --- | --- | --- | --- | --- |
| as shipped: off for PREV, on for NEW | 211.5 ms | 1.72 s | 332.0 ms | 822.3 ms | 66.5 | 65.7 |
| on for both | 290.9 ms | 934.3 ms | 332.0 ms | 822.3 ms | 66.5 | 66.7 |
| off for both | 211.5 ms | 1.72 s | 157.7 ms | 848.5 ms | 64.7 | 65.7 |

Giving a policy the ride raises its own p99 by 37% (211.5 to 290.9 ms) while halving its longest stall (1.72 to 0.934 s). This is because each of its tokens now arrives at the end of a prefill batch and carries a weight of one in the population. Put both policies on the same delivery form and NEW's deficit is 332.0 against 290.9 ms, 14%. The rest of the reported gap is the ride's weighting. Take the ride away from NEW and it wins the cell outright, 157.7 against 211.5 ms, and loses `stream decode tok/s in cold` to PREV (64.7 against 65.7 tok/s). Those riding tokens are what streams generate during a cold prompt at all. Mixed chunked prefill is the deployment's resolved default (`arg_groups/mixed_chunk_hook.py`), not part of the control law. The sweep prints that variant's trade in its own `contract` column.

Sweeping the only parameter the balancer shares with the chunk size shows the same trade. Each row is OLD / PREV / NEW from the sensitivity sweep:

| setting | B ITL p99 | B longest stall | B cold TTFT mean | B output tok/s |
| --- | --- | --- | --- | --- |
| chunk 4096 (shipped) | 28.2 / 211.5 / 332.0 ms | 45.4 s / 1.72 s / 822.3 ms | 45.2 / 102.6 / 104.7 s | 226 / 226 / 235 |
| `chunked_prefill_size 2048` | 28.2 / 59.6 / 226.4 ms | 46.5 s / 873.2 ms / 520.4 ms | 46.4 / 64.5 / 109.7 s | 224 / 212 / 235 |
| `chunked_prefill_size 8192` | 28.2 / 49.2 / 27.4 ms | 45.0 s / 3.19 s / 1.23 s | 44.8 / 97.3 / 100.6 s | 228 / 219 / 230 |

Halving the chunk moves NEW's p99 down to 226.4 ms, because the p99 tracks the cap one-for-one. The chunk moves PREV's down to 59.6 ms at the same time. As a result, NEW's deficit *grows* from 1.6x to 3.8x while PREV's cold TTFT improves by 38 s and NEW's worsens. Reaching PREV's 211.5 ms that way needs a smaller chunk still, and continues the same trade. Doubling the chunk puts NEW's mixed count under the 1% line. Wins p99 outright (27.4 against 49.2 ms) - by changing `chunked_prefill_size`, a launch flag the balancer must not decide, and at 1.23 s of stall for NEW. This is 3.19 s for PREV.

What the balancer does own - how many tokens a *continuation* may take - shows up as the band's height against the band's count. Uncapped (PREV) the continuation batches cost p50 463.0 ms, p99 825.8 ms and a max of 942.9 ms at 4,096 tokens each, mean 120 us per token. Capped in seconds (NEW) they cost p50 413.3 ms, p99 652.7 ms and a max of 841.7 ms at a median of 3,606 tokens, mean 126 us per token. There are 4,352 of them against PREV's 3,698 - 18% more passes to be ridden. The cap pays for itself in the wait metrics (longest stall 841.7 ms against PREV's 1.72 s, raw gap p99 459.6 against 720.2 ms). The cap costs p99, because while the riding count is above 1% of samples the reported percentile sits in the band whatever the band's height is.

## The same count decides the other cells the tables mark

Delivery-class shares for NEW, summed over each scenario's own seeds, as `TestDeliveryClassShareTable` prints them. PREV has no mixed class at all - mixed chunked prefill is off at its commit - so every one of its samples is a decode sample. Latencies are ms except where a unit is written.

| scenario | mixed share | p99 NEW / PREV | p99.9 NEW / PREV | raw gap p99 NEW / PREV | longest stall NEW / PREV |
| --- | --- | --- | --- | --- | --- |
| A: logged episode | 0.89% (361 of 40,517) | 77.6 / 161.5 | 479.3 / 646.7 | 422.0 / 541.7 | 748.2 ms / 1.76 s |
| C: 2 req/s, max 16 | 2.46% (28,850 of 1,170,763) | 139.5 / 94.5 | 223.4 / 244.0 | 192.5 / 238.0 | 322.7 / 611.7 |
| C: 0.5 req/s, max 16 | 0.33% (986 of 303,313) | 18.3 / 19.8 | 156.7 / 90.4 | 32.0 / 32.0 | 279.9 / 312.8 |
| D: one cold 400K | 0.68% (4,544 of 669,290) | 26.6 / 30.4 | 395.7 / 339.8 | 134.6 / 95.5 | 665.3 / 1.54 s |
| D: one cold 100K | 0.49% (3,581 of 728,029) | 26.0 / 27.1 | 145.5 / 147.3 | 63.3 / 72.0 | 363.2 / 699.7 |
| thrash host 4x, 600 s | 0.64% (3,679 of 573,269) | 28.4 / 84.3 | 427.8 / 287.5 | 263.4 / 284.4 | 831.0 / 899.8 |

The share says which band a percentile is drawn from - above 1% for p99, above 0.1% for p99.9. The band's height then says whether it beats PREV:

* The p99 cells NEW loses to PREV are exactly the ones whose share is above 1%: the 1 minute cadence (366.3 against 251.6 ms at 1.60%). This is B at a couple of minutes (332.3 against 212.2 ms at 1.40%), C at 2, 3 and 5 req/s (139.5 against 94.5, 167.7 against 148.7 and 165.8 against 148.8 ms, at 2.46%, 4.75% and 4.82%) and C at 5 req/s with `max_running_requests 6` (92.3 against 80.6 ms at 1.72%). Everywhere the share printed below 1% NEW wins p99 outright (77.6 against 161.5 in A, 26.0 against 27.0 at D 25K, 26.0 against 27.1 at D 100K, 26.6 against 30.4 at D 400K, 28.4 against 84.3 under thrash, 27.7 against 39.1 at the 5 minute cadence).
* Above 0.1% the p99.9 is a mixed sample, so NEW's number comes out of its band. The cell is won only where PREV's tail is longer than that: 479.3 against 646.7 in A, 523.5 against 731.9 at B a couple of minutes. This is 534.5 against 804.3 at the 1 minute cadence, 223.4/250.3/247.2 against 244.0/388.5/389.3 at C 2/3/5 req/s, 200.2 against 267.4 at C 5 req/s with `max_running_requests 6`, 145.5 against 147.3 at D 100K. This is every thrash episode below the 4x host tier (251.3-434.7 against 755.9-839.4). It is lost where PREV's tail is short: 90.4, 148.7. 147.2 at the C rows at 0.5 and 1 req/s, 398.8 at the 5 minute cadence, 90.1 to 453.1 at the D rows other than 100K. This is 287.5 and 304.6 at the 4x tier.
* In every cell the bound argues, NEW's own pure-decode steps are far tighter than the percentile. It is being compared to - `itlPassBand`'s decode-class cut, the same rank with the pass-cost samples removed, is 23.8 ms against OLD's 26.6. This is in A, 24.8 against PREV's 212.2 at B a couple of minutes, 24.2 against 251.6 at the 1 minute cadence, and 36.4/37.9/37.9/22.8 against 94.5/148.7/148.8/80.6 at C 2/3/5 req/s. This is C 5 with `max_running_requests 6`. At p99.9 the same cut is 89.7 ms in A against OLD's 196.7 and 27.1 to 27.4 ms across the D rows against their 81.0 to 85.1 ms. The undivided raw gap p99 favours NEW in four of the tabulated rows, ties at C 0.5 req/s and loses at D 400K.

Within the sensitivity sweep the same crossing decides the p99 column, which the sweep prints. A cheaper decode step moves OLD's reported p99 down into the decode band - 23.8, 22.7, 22.0 and 27.1 ms for `decode D0 x0.7`, `decode DCtx 0`, `decode DBS 0` and `MTP accept 3.5`, against the baseline's 28.2 ms. This is while NEW stays in its own mixed band (290.9, 307.9, 302.2 and 291.0 ms against the baseline's 332.0) and PREV lands between the two (174.5, 190.5, 186.0 and 189.1 against 211.5). Those rows hold a longest stall within 8% of the baseline's 822.3 ms (830.9, 774.3, 760.4 and 800.4 ms). `MTP accept 3.5` is the only decode-side variant whose `contract` column also names `ITL p99.9 vs PREV`. The sweep's own `mixed chunked prefill off` row is what `TestMixedRideIsWhatDecidesTheP99Cell` measures: NEW's p99 drops to 157.7 ms against PREV's 211.5, and `stream decode tok/s in cold vs PREV` joins its losses.

Other failure classes are not the tail mechanism at all:

* `output tok/s vs OLD` in A (132.1 against 139.7) is the half-share trade, not a regression: OLD gives the cold prompt the whole GPU. OLD is done with it in 70.0 s, NEW splits it. OLD is done in 128.8 s, and the tokens the streams did not generate during that window are the difference. NEW beats PREV there (132.1 against 125.3). The `output tok/s` carets on C 0.5 req/s, C 1 req/s. Thrash at 600 s are ties at the printed precision - C 0.5 req/s measures NEW 168.5950 against PREV 168.6128, a 0.011% shortfall - and `Better` flags any non-zero difference.
* `stream decode tok/s in cold` in D at 100K is the other class: NEW 64.2 against PREV's 66.1 tok/s. Its bound prices the ride rather than excusing it - NEW's 18,198 tokens over 283.9 stream-seconds against PREV's 17,844 over 269.9. As a result, the 14.0 extra seconds of stream life returned 25.2 tok/s against the 2.8 tok/s floor a riding row costs. This is with 453 of NEW's tokens arriving as prefill rows - and `docs/derivation-stream-rate-under-mixed-chunk.md` derives that floor. `per-agent tok/s in cold` in the same cell is a win (44.5 against PREV's 43.7). As a result, the shortfall is the longer cold window the streams are alive through, not slower steps.

## `cold TTFT` against OLD

OLD finishes a cold 400K prompt in 45.3 s, NEW in 105.0 s. A 400K prompt at the log's own measured prefill rate costs about 48-53 s of GPU (the logged episode measures 430,080 tokens in 52.99 s), so any policy that lets a cold prompt have. The whole GPU finishes in about that time and any policy that splits it with decode finishes in about twice that. The balancer's split is even by definition - debt is prefill seconds minus decode seconds - so the bound on cold TTFT is

    cold TTFT >= 2 x (prompt tokens x prefill seconds per token)
               = 2 x 400,000 x 126 us ~= 101 s

which both interleaving policies sit on (PREV 103.1 s, NEW 105.0 s) and which OLD escapes only by giving decode multiple steps for the whole window. OLD's in-window stream rate is 0.7 tok/s against NEW's 66.1. 35.14% of its stream time is inside a stall longer than a second against NEW's 0.00%. NEW's 1.9 s above PREV is the continuation cap buying shorter chunks (4,352 continuation batches against 3,698) and is inside the same 2x bound.
