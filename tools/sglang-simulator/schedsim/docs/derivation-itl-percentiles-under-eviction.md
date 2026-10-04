# Inter-token tail under eviction

Covers `boundEvictionTail`, the bound `contract_test.go` reads for an `ITL p99` or `ITL p99.9` cell in a `thrash-*` scenario. The ten-conversation episode has no cold prompt at all. As a result, the cold-window derivations have nothing to price here: the table's `stream decode tok/s in cold` and `cold TTFT mean` rows print `-` for every policy. `boundName` routes these metrics to this argument instead of the mixed-chunk one.

## What the episode measures

Conversations at 150K-430K contexts against a device pool of 1,406,118 tokens (derived from the log's `#full token / full token usage`, `internal/trace/calib.go`), `max_running` 10, seeds `[1 2 3]`. The working set is roughly twice the pool, so a turn's prefix has usually been evicted by the time the next turn arrives. It hits only the 11,584-token shared system prompt. Whether the evicted prefix comes back from the host tier or has to be recomputed is what the host tier decides. That is the difference the tables show:

| thrash episode | recomputes OLD / PREV / NEW | longest stall OLD / PREV / NEW | stream time in stalls > 1 s | output tok/s OLD / PREV / NEW |
| --- | --- | --- | --- | --- |
| host 0x, 600 s | 68 / 38 / 10 | 84.7 s / 1.53 s / 554.7 ms | 95.8% / 10.7% / 0.0% | 13.8 / 119.6 / 171.0 |
| host 1.5x, 600 s | 64 / 30 / 4 | 49.1 s / 1.77 s / 576.0 ms | 81.1% / 17.7% / 0.0% | 61.0 / 186.3 / 210.7 |
| host 0x, 1800 s | 193 / 101 / 43 | 84.7 s / 1.73 s / 583.8 ms | 96.0% / 12.2% / 0.0% | 13.9 / 107.9 / 151.6 |
| host 1.5x, 1800 s | 183 / 71 / 12 | 87.1 s / 2.00 s / 664.4 ms | 85.3% / 19.0% / 0.0% | 45.8 / 165.5 / 191.5 |
| host 4x, 600 s | 0 / 0 / 0 | 932.5 ms / 899.8 ms / 831.0 ms | 0.0% / 0.0% / 0.0% | 317.6 / 315.4 / 317.5 |

At 0x and 1.5x the pool is over-subscribed and OLD spends most of the run refilling it. 68 to multiple whole prefixes rebuilt, 84.7 s of longest stall, and a lower count.8% of all stream time inside a single gap longer than a second. NEW wins every latency cell in those episodes, so no bound is consulted. At 4x the host tier is large enough to hold the working set, recomputes fall to zero for all policies. The episode stops being about eviction -- which is where the cells this document argues live.

## The two cells, and the class they are drawn from

`ITL p99.9` at the 4x tier, against both opponents, at both run lengths. Cells, each with the numbers the test prints:

| cell | NEW p99.9 | opponent p99.9 | samples of NEW's above the opponent's | mixed deliveries in the population | NEW's longest forward pass |
| --- | --- | --- | --- | --- | --- |
| host 4x, 600 s vs OLD | 427.8 ms | 302.1 ms | 1,644 of 573,269 (0.287%) | 3,679 | 831.0 ms |
| host 4x, 600 s vs PREV | 427.8 ms | 287.5 ms | 1,825 of 573,269 (0.318%) | 3,679 | 831.0 ms |
| host 4x, 1800 s vs OLD | 453.7 ms | 307.6 ms | 4,601 of 1,544,298 (0.298%) | 8,180 | 1.0 s |
| host 4x, 1800 s vs PREV | 453.7 ms | 304.6 ms | 4,660 of 1,544,298 (0.302%) | 8,180 | 1.0 s |

A p99.9 names 0.1% of the population: samples at 600 s, 1,544 at 1800 s. The mixed deliveries are 0.64% of NEW's 573,269 samples at 600 s and 0.53% of its 1,544,298 at 1800 s -- 3,679 and 8,180 samples, since a mixed delivery files exactly one sample (one token per riding request, because speculative decoding hands out one token inside a prefill batch). So the mixed class is 6.4x and 5.3x the size of the cut. The reported percentile is a prefill batch's wall clock rather than a decode step's:

    3,679 mixed samples  >  574 samples a p99.9 names      (600 s)
    8,180 mixed samples  >  1,544 samples a p99.9 names    (1800 s)

That is why the same policy wins `ITL p99` outright in these scenarios -- 28.4 ms and 28.0 ms against 84.9 ms and 82.7 ms from OLD. This is 84.3 ms and 81.7 ms from PREV. That loses p99.9. At the p99 rank NEW is still inside its decode band, and one decade further into the tail. The rank is inside the band of tokens delivered during a prefill batch.

## Why a batch's cost is the right ceiling, and what the bound checks

The percentile is only comparable between multiple policies if it measures the same wait, so `boundITLTailUnderEviction` refuses the cell unless conditions hold. This is each on numbers the run prints:

1. NEW's percentile must not outlast the longest prefill batch it launched: 427.8 ms below 831.0 ms at 600 s, 453.7 ms below 1.0 s at 1800 s. Above that the tail is not a batch's stall and nothing here explains it.
2. The share of NEW's samples above the opponent's percentile must be at least the tail the metric names (0.1% for p99.9): measured 0.287% and 0.318% at 600 s. This is 0.298% and 0.302% at 1800 s. Below that floor the percentiles are not separated by this population at all -- they will be the same point in a different policy's ordering.
3. That count must fit inside the mixed rows the run served: 1,644 against 3,679 rows and 4,601 against 8,180. If more samples sat above the cut than there were riding tokens, some of the tail will not be the ride.
4. NEW's longest stall must not be the worse of the two: 831.0 ms against OLD's 932.5 ms and PREV's 899.8 ms at 600 s. This is 1.01 s against 1.19 s and 1.09 s at 1800 s.
5. NEW's median inter-token gap must not exceed the opponent's: 6.8 ms against 7.0 ms and 6.9 ms at 600 s, and 6.6 ms against 6.6 ms at 1800 s. A policy whose whole distribution sat higher will not be excused by a tail argument.
6. Both policies must have run the same deployment: the GPU busy share is 98.4% for all three at 600 s and 99.5% at 1800 s. The decode share is 77.8-78.1% at 600 s and 77.1-77.3% at 1800 s. `TestDeliveryClassShareTable` prints the busy and decode shares, the medians in condition 5 and the longest stalls in condition 4 for both tiers.

Conditions 1 and 3 together are the claim: the tail samples are tokens handed out inside a prefill batch. No such token can wait longer than the batch that carried it.

What the cells cost, stated as what the bound does not claim. 427.8 ms is one prefill batch's wall clock, charged to a single token, and the token exists at all. This is because the request rode a batch that was prefilled for someone else. Turning mixed chunked prefill off will move NEW's p99.9 back under the opponent's, and will also take away every token the streams generate. This is during those batches. The same trade the mixed-chunk derivation measures directly, where removing the ride drops NEW's p99 to 157.7 ms against PREV's 211.5 ms and loses `stream decode tok/s in cold` instead.

## Cells this derivation covers and refuses

Covers multiple cells: `ITL p99.9` in thrash at the 4x host tier over 600 s and 1800 s, against OLD and against PREV. Margins are +41.6% and +48.8% at 600 s, +47.5% and +49.0% at 1800 s.

Refuses:

* Every `ITL p99` cell under eviction. NEW wins all six: 25.0-42.5 ms against PREV's 81.7-258.4 ms and against OLD's 82.7 ms to 9.2 s, so there is nothing to argue.
* The `output tok/s` caret on thrash at 4x over 600 s (317.5 against OLD's 317.6). That is +0.02% at the precision the contract prints, inside the tie band it judges. A derivation will be claiming a loss the contract does not see.
* `longest stall` at the 4x tier. NEW's is the shortest of the three (831.0 ms). As a result, it is a win, not an exception.
* Any claim that the 4x tier is where the mechanism matters. At that tier the host cache holds the working set and nothing is recomputed by anyone. The episode is a busy short-chat mix with long prompts, and the tail belongs to the same delivery classes the mixed-chunk document describes. What this document adds is only that the cold-prompt accounts cannot speak for it, because there is no cold prompt.
