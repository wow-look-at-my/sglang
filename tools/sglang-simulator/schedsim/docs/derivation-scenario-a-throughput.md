# Scenario A output throughput per stream-second

Covers `boundAdmissionRate`, the bound `contract_test.go` reads for the `output tok/s` cell lost to OLD in scenario A. It is the only throughput cell any derivation claims: `boundName` maps metric 7 to this argument for `sc.Key == "A"` and for no other scenario.

## What metric 7 counts

    output tok/s = tokens with d.T <= window / window

The window is the scenario's, 300 s in A, and it is the same number for all policies, so the metric is a token count divided. This is by a constant. Runs execute to `HardStop`, 420 s in A, so an in-flight turn finishes rather than being cut, but every token it delivers. This is after 300 s is outside the count for every policy. Nothing normalizes by how much work a policy actually had the chance to do inside the window - which is the whole question. This is in a scenario whose streams do not exist until multiple cold prompts are prefilled.

## The cell, from the committed run

| quantity | NEW | OLD | PREV |
| --- | --- | --- | --- |
| tokens delivered by 300 s | 39,617 | 41,909 | 37,577 |
| output tok/s, that over 300 s | 132.056667 | 139.696667 | 125.256667 |
| live stream-seconds by 300 s | 416.055397 | 574.641722 | 412.936867 |
| tokens per live stream-second | 95.220493 | 72.930660 | 90.999383 |

The bound is the last row: `95.220493 >= 72.930660`, NEW 1.3056 times OLD. It fails the cell outright if that inequality goes the other way, naming both rates.

The arithmetic of the loss itself, from the same numbers:

    token deficit   41909 - 39617        = 2292 tokens, 5.47% of OLD's total
    stream-seconds  574.641722 - 416.055397 = 158.586325 s, 38.1% of NEW's
    at OLD's live seconds, at NEW's rate  95.220493 x 574.641722 = 54718 tokens

So the entire 2,292-token deficit is the 158.59 seconds of stream life NEW did not have. Charged at NEW's own rate those same seconds are worth 54,718 tokens, 30.6% above what OLD delivered. The policies did not differ in how fast a live stream generated, they differed in how many live streams the window contained.

## Where the missing seconds come from, measured

`StreamSeconds` sums, over requests that reached a first token inside the window, the time from that first token. This is to the request's finish or the window's end, whichever is earlier. A turn that starts later can only contribute less. In A the conversations' turns are gated by the cold prompts:

| quantity | NEW | OLD |
| --- | --- | --- |
| cold prompts reached / arrived | 3/3 | 3/3 |
| cold TTFT mean | 128.8 s | 70.0 s |
| summed cold windows (the same three, paired by the account) | 386.336392 s | 210.071732 s |
| stream decode tok/s inside those windows | 84.6 | 0.0 |
| stream time in stalls over 1 s | 0.0% | 35.3% |
| longest stall | 748.2 ms | 76.1 s |

`TestDeliveryClassShareTable` prints both stall shares above for scenario A.

OLD's summed cold windows are 176.26 s shorter than NEW's, the same gap the cold-TTFT bound accounts for second by second in `derivation-cold-ttft-vs-old.md`. OLD's extra 158.59 live stream-seconds are a different measurement, not that gap transferred. This is `StreamSeconds` runs from each request's own first token to its finish or the window's end, while a cold window runs. This is from a cold prompt's arrival to that prompt's first token. The two are linked through the turns that can only start once a cold prompt is done. The account here does not equate them. What the columns do establish is the direction and the mechanism. OLD converted more of the 300 s window into live stream time, and it paid for that by running zero decode steps. 0.0 tok/s of stream decode inside its own cold windows, with 35.3% of its stream time inside a stall longer than a second against NEW's 0.0%. This is a longest stall of 76.1 s against NEW's 748.2 ms.

That is the asymmetry the bound rests on. It is the same mechanism metric charges the cold TTFT cell to. OLD converts window into stream-seconds by withholding tokens during the cold prefills, and the tokens it then delivers are the tokens. The streams can generate once it stopped withholding.

## Cells this derivation covers and refuses

Covers one cell: `output tok/s` in scenario A against OLD, a 5.47% loss.

Refuses:

* A against PREV. NEW's 132.1 tok/s beats PREV's 125.3 there, and the same is true of the per-stream-second rate (95.22 against 91.00), so no claim is needed.
* Every other scenario: B at 1, 2 and a few minutes, C at every rate and `max_running`, D at every prompt length, and the thrash episodes. None of their `output tok/s` cells is mapped to a derivation, and the only ones where NEW is nominally behind are ties the contract prints. This is at its own precision: +0.01% at C 0.5 req/s with `max_running_requests 16` (NEW 168.6 tok/s against OLD's and PREV's 168.6), +0.00% at C 1 req/s with 16 and again with 6 (334.0 tok/s either way). This is +0.02% under thrash at the 4x host tier over 600 s (317.5 against 317.6). A difference of a hundredth of a percent over the scenario's own window, inside `TieRelative`'s 0.5%, which is what the contract judges. The scenario tables mark any non-zero difference with a caret. As a result, a tie can print as `^old` and still be a tie. `boundName` returns no derivation for those keys, so a loss beyond the tie band in one of them fails the build. This is the honest outcome for a metric that can have improved.
* A throughput *improvement* claim. The bound shows the loss is admission timing, not rate. It says nothing about whether NEW will have overtaken OLD given the same stream-seconds, only that at OLD's own live seconds. This is NEW's own rate the ordering reverses, and that NEW's rate per live stream-second is the higher one today.
* Any comparison of the window itself. The 300 s window is the scenario's, not a measured quantity, and the ranking of a run whose streams are gated by a 128.8 s cold TTFT. This is on a 300 s window is sensitive to that length in a way this document does not quantify.
