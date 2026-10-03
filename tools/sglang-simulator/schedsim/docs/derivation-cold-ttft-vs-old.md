# Cold TTFT against the policy that withholds decode

Covers `boundWholeGPU`, the bound `contract_test.go` reads for a `cold TTFT mean` cell lost to OLD. Every number below is a value the committed run prints or a constant it charges with: `go run ./cmd/schedsim` for the calibration. The three-policy tables, `go test ./internal/sim/ -run TestContract -v` for the per-cell accounts.

## What the bound asserts

For the same set of cold prompts, priced under both policies:

    gap      = NEW's wait  - OLD's wait
    explained = NEW's other prefill + decode rows + overhead + decode + idle
              + (NEW's own prefill - OLD's own prefill)

The cell is argued when `gap <= coldTTFTSlack * explained`, given preconditions. The policies charged the same prompt the same own prefill work, and OLD spent at most `oldDecodeShare` of the same windows decoding. Both sides are sums over paired windows, not means, so the arithmetic is over total seconds.

## Where each quantity comes from

| name | measured by | not |
| --- | --- | --- |
| window | `ColdWindow.FirstTok - Arrival`, arrival to the token the prefill batch itself sampled (`engine.firstToken`) | a queue wait, an inference queue, or a wall clock outside the run |
| own prefill | `AccountWindow` in `account.go`: the window's slice of each batch, weighted by `it.Extend x SecondsPerToken(prefix + Extend/2)` for the item whose request is this window's prompt | a prompt-length times a constant rate; the rate rises with context |
| other prefill | the same weighting over every *other* request's extend items in those batches | |
| decode rows | `SecondsPerToken(r.Context())` summed over the requests a mixed batch rode | a separate decode cost, which is why `PiggybackCredit` can subtract it from the batch's prefill charge |
| overhead | `b.ReloadSeconds + Cost.PerBatchSeconds()` split across the batch's parts in the batch's own proportions | prefill compute |
| decode | seconds of every pure decode batch inside the window | |
| idle | `Window - Total`, the residual, floored at zero | a measurement; it is what the five charged buckets did not fill |
| per-pass base | `Cost.PerBatchSeconds()` = `cal.Prefill.Base` = 10.0 ms | an assumed overhead |
| token rate | `cal.Prefill`: 68.403 us/token, + 4.72e-11/token/ctx + 6.05e-16/token/ctx^2, fitted on 104 chunks with 56 held out (fit 1.714%, holdout 1.399%) | a flat rate: `ChunkSeconds(0)`, one full chunk at an empty prefix, is 10.0 ms + 4096 x 68.5 us = 290.6 ms, and that is what scenario A's single fresh 4096-token batch measures, 290.6 ms; the same chunk deeper into a 430K context costs more per token, which is why a continuation is capped in seconds and not in tokens |

The pairing rule matters for every number below: `coldAccounts` walks NEW's windows. It looks up the window with the same tag in the same seed's run, so a prompt either policy failed to serve is excluded and counted, never priced from a first token that was never recorded. The buckets and idle are a partition of the window by construction. The content of the account is the split. The content of this bound is that NEW's later first token is other people's work, not more compute of its own.

## Precondition: the same prompt is the same computation

`ownTolerance` is 5%, and the excess it tolerates is not dropped - it is added to `explained`, so an own-work difference is carried as a real term of NEW's wait rather than waved away. Measured excess over OLD, per cell:

| scenario | own NEW (s) | own OLD (s) | excess (s) | excess as % of own | own tokens | own passes NEW / OLD |
| --- | --- | --- | --- | --- | --- | --- |
| A | 124.397253 | 124.397023 | 0.000230 | 0.00019 | 1,134,560 both | 351 / 280 |
| B 1 min | 3038.691454 | 3038.687603 | 0.003851 | 0.00013 | 27,600,000 both | 8,182 / 6,762 |
| B 2 min | 1541.365488 | 1541.363277 | 0.002211 | 0.00014 | 14,000,000 both | 4,385 / 3,430 |
| B 5 min | 660.585313 | 660.584262 | 0.001051 | 0.00016 | 6,000,000 both | 1,921 / 1,470 |
| D 25K | 8.639799 | 8.639798 | 0.000001 | 0.00002 | 125,000 both | 36 / 35 |
| D 100K | 36.389563 | 36.389560 | 0.000003 | 0.00001 | 500,000 both | 132 / 125 |
| D 200K | 81.187426 | 81.187369 | 0.000057 | 0.00007 | 1,000,000 both | 272 / 245 |
| D 400K | 220.195172 | 220.194754 | 0.000418 | 0.00019 | 2,000,000 both | 655 / 490 |
| D 500K | 326.498498 | 326.497819 | 0.000679 | 0.00021 | 2,500,000 both | 900 / 615 |

Own tokens are identical in every cell, so neither policy re-cut a prompt into a different total. Own *seconds* differ by the split. This is because a pass's seconds are shared out among its parts and a mixed pass gives its own request a smaller share of the same clock. The pass counts differ. This is where the whole excess comes from - the cede lets other requests into the chunk - and the bound prices that difference rather than assuming it away.

Where 5% comes from: the only way identical tokens can cost different seconds is the per-pass base and the reload copy. The base is `cal.Prefill.Base` = 10.0 ms. A pass charges its own request `k x ownCost` with `k = secs / (total + overhead)`, so re-cutting a fixed token count can move own work by at most the overhead share of each pass it runs - no more than the whole base of every own pass, on top of the tokens:

    A:      351 x 10.0 ms = 3.51 s  on 124.397 s of own work   = 2.82%
    B 1min: 8182 x 10.0 ms = 81.82 s on 3038.691 s             = 2.69%
    B 2min: 4385 x 10.0 ms = 43.85 s on 1541.365 s             = 2.85%
    B 5min: 1921 x 10.0 ms = 19.21 s on 660.585 s              = 2.91%
    D 25K:  36 x 10.0 ms = 0.36 s    on 8.640 s                = 4.17%
    D 100K: 132 x 10.0 ms = 1.32 s   on 36.390 s               = 3.63%
    D 200K: 272 x 10.0 ms = 2.72 s   on 81.187 s               = 3.35%
    D 400K: 655 x 10.0 ms = 6.55 s   on 220.195 s              = 2.97%
    D 500K: 900 x 10.0 ms = 9.00 s   on 326.498 s              = 2.76%

The largest is 4.17%, at D 25K, the scenario with the fewest tokens per pass. `ownTolerance` is 5%. As a result, the tolerance is that ceiling plus a margin. A cell can only fail this precondition if a policy computed more of the prompt than the same tokens amount to. What the cells measure is orders of magnitude inside it: the worst measured excess is 0.00021% of own work against a 5% allowance.

## Precondition: OLD spent the window on prefill only

`oldDecodeShare` is `ownTolerance`, not a second tuned number. The bound grants that at most 5% of OLD's cold-window seconds are anything other than prefill. OLD's decode seconds are exactly such a portion. As a result, a second tolerance will let the bound keep a term it does not account for. Measured:

| scenario | OLD decode (s) | OLD window (s) | OLD share | OLD's worst single window | NEW share |
| --- | --- | --- | --- | --- | --- |
| A | 0.000000 | 210.071732 | 0.0000% | 0.0000% | 38.80% |
| B 1 min | 1.858908 | 3117.393859 | 0.0596% | 0.1066% | 40.79% |
| B 2 min | 7.375508 | 1586.207961 | 0.4650% | 5.1776% | 50.34% |
| B 5 min | 4.051801 | 681.208623 | 0.5948% | 4.4497% | 50.36% |
| D 25K | 0.162849 | 9.513118 | 1.7118% | 2.3153% | 50.10% |
| D 100K | 0.162849 | 38.562030 | 0.4223% | 0.5629% | 50.44% |
| D 200K | 0.162849 | 84.053398 | 0.1937% | 0.2541% | 50.11% |
| D 400K | 0.162849 | 225.749416 | 0.0721% | 0.0946% | 49.78% |
| D 500K | 0.571843 | 333.873165 | 0.1713% | 0.6742% | 50.08% |

The aggregate share is what the test compares against 5%. The worst single window is reported in the failure message and is not gated. This is why B 2 min and B 5 min can show a 5.18% and 4.45% window without failing. The asymmetry the derivation rests on is the last column against the fourth: NEW spends 39-50% of the same windows decoding, OLD 0.00-1.71%. OLD's faster first token is the whole-GPU effect. The table is what distinguishes it from a policy that was also sharing.

## The account, cell by cell

| scenario | paired windows | NEW wait (s) | OLD wait (s) | gap (s) | other NEW | rows NEW | overhead NEW | decode NEW | idle NEW | explained (s) | explained / gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 3 | 386.336392 | 210.071732 | 176.264660 | 100.961886 | 0.064156 | 11.029618 | 149.883479 | 0.000000 | 261.939369 | 1.486 |
| B 1 min | 69 | 21490.949521 | 3117.393859 | 18373.555662 | 8804.513573 | 6.659178 | 875.003779 | 8766.081537 | 0.000000 | 18452.261918 | 1.004 |
| B 2 min | 35 | 3674.015726 | 1586.207961 | 2087.807766 | 131.933688 | 1.449789 | 149.944352 | 1849.322409 | 0.000000 | 2132.652449 | 1.021 |
| B 5 min | 15 | 1571.441287 | 681.208623 | 890.232664 | 55.381516 | 0.637510 | 63.524103 | 791.312845 | 0.000000 | 910.857025 | 1.023 |
| D 25K | 5 | 19.562175 | 9.513118 | 10.049057 | 0.748713 | 0.011132 | 0.361060 | 9.801471 | 0.000000 | 10.922377 | 1.087 |
| D 100K | 5 | 81.861354 | 38.562030 | 43.299324 | 2.812315 | 0.043633 | 1.321059 | 41.294784 | 0.000000 | 45.471793 | 1.050 |
| D 200K | 5 | 180.213419 | 84.053398 | 96.160022 | 5.904728 | 0.089094 | 2.721058 | 90.311113 | 0.000000 | 99.026050 | 1.030 |
| D 400K | 5 | 488.586241 | 225.749416 | 262.836825 | 16.762323 | 0.204808 | 8.213854 | 243.210085 | 0.000000 | 268.391488 | 1.021 |
| D 500K | 5 | 750.384094 | 333.873165 | 416.510928 | 26.085539 | 0.272665 | 21.726547 | 375.800845 | 0.000000 | 423.886275 | 1.018 |

`explained / gap` is at least 1 in every cell, so the 2% in `coldTTFTSlack` is never what makes a cell pass. The tightest is B at the 1 minute cadence at 1.004. The slack is there. This is because all quantities are sums of floats over hundreds of batches, not counts, and a part per million of float error on a 78 s residual will otherwise reject a cell that closed. The same account identity is what `coldTTFTIdentitySlack` (1e-6) polices on the NEW-versus-PREV bound. There a mis-attributed forward pass is the thing it is looking for. One pass of the log's 10.0 ms base against A's 6.748 s gap is 1.5e-3, fifteen hundred times the tolerance, so any bookkeeping error that can matter fails the identity rather than passing it.

Cells' accounts carry terms worth naming. In A, OLD also spent 81.314145 s on *other* requests' prefill inside its cold windows - the follow-up that queued 1.3 s into C1's prefill and the cold prompts behind it - so OLD's 210.07 s of windows is not 210 s of C1. The bound claims only that those windows were prefill and not decode. The decode column is where the policies differ.

At the 1 minute cadence the account closes per prompt (paired windows):

    own        44.04 s  both policies, the same 400,000-token prompt
    other      127.60 s NEW,  0.11 s OLD
    overhead    12.68 s NEW,   1.00 s OLD
    decode     127.04 s NEW,   0.03 s OLD
    rows          0.10 s NEW,   0.00 s OLD
    -------    -------
    wait       311.46 s NEW,  45.18 s OLD

Each column is per prompt: the scenario's own work is 44.04 s and OLD's whole wait is 45.18 s. As a result, OLD's cold prompt ran with the GPU to itself plus 1.1 s of other people's work. NEW's wait is that same 44.04 s plus 267.4 s it chose to spend elsewhere: 127.6 s of queued prefill that the 1-minute cadence stacked up, 127.0 s of decode for the five resident streams, and 12.7 s of per-pass base. Reload copy across the 118.6 forward passes per prompt (8,182 passes against OLD's 6,762 over the windows). The gap the bound has to explain is 266.29 s per prompt and the buckets leave 1.14 s per prompt of surplus, the tightest in the suite.

## Windows this bound will not price

`AccountWindow` classifies a window it cannot price instead of pricing it from nothing. `Unserved` is a run that ended before the prompt reached a first token, `Orphan` is a window no request in the run opened, `Empty` is a first token that is not after the arrival. The bound excludes them and says how many:

    NEW cold windows: 69/70 priced, excluded: seed 4 13/14 windows priced, 1 unserved

at B 1 min, which is one window in seventy.

The tag 13 prompt on seed 4 arrived at 840.000 s and never sampled a first token. Its prefix stopped at 399,841 of 400,000 tokens and it delivered 0 of its output tokens. A window opens with `FirstTok` at -1 (`newColdWindow` in `engine.go`, asserted by `cold_window_test.go`). This is because 0.0 is a legal timestamp and cannot also mark the unserved case, so `Done()` is false for this prompt. It is counted as `Unserved` rather than served at t = 0. The metric and the bound then agree: the table prints `cold prompts served/arrived 69/70` and a `cold TTFT mean` of 311.5 s. This is the paired-window mean this document's table computes (21490.949521 / 69 = 311.4630) with no term subtracted from it.

## Cells this derivation covers and refuses

Covers, all losses of `cold TTFT mean` against OLD where a cold prompt exists: A, B at 1, 2 and a few minutes, D at 25K, 100K, 200K, 400K and 500K. Refuses:

* C at every rate and `max_running`, and every thrash episode: no cold prompt arrives, metric 4 has no value, and `Judge` returns undefined before any bound is reached. `boundName` excludes the thrash family a second way, by key prefix.
* Any cell NEW ties or wins. `TieRelative` is 0.5%, and a cell inside it is a tie decided by which requests fell in the window. The loss list above is the whole set of cold-TTFT cells beyond it against OLD.
* The claim that splitting the prompt is *worth* the wait. This bound only shows the wait is other requests' work: OLD spends 98.3-100.0% of its cold windows on extend passes (its decode share is 0.00-1.71% and idle is 0.000000 s in every cell). NEW's gap over it is carried entirely by the other-prefill, rows, overhead and decode buckets, which together run from 1.004 to 1.486 times the gap. Whether that trade is right is the balancer's decision, not this document's.
* A per-window statement. Every figure is a sum over paired windows. Nothing here says each window closed, only that the account over the surviving windows does.
