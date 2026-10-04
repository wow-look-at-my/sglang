# Stream decode rate under mixed chunked prefill

Covers `boundStreamRate`, the bound `contract_test.go` reads for a `stream decode tok/s in cold` cell lost to PREV. It is the only one of the contract metrics whose denominator the policy itself creates, which is why the cell needs an argument. This is at all rather than a number.

## What metric 1 measures, and why it can fall while service improves

    stream decode tok/s in cold = agent tokens delivered inside cold windows
                                  / stream-seconds inside those windows

The numerator is a count. The denominator is `streamSeconds` in `metrics.go`: for each agent request, the overlap of its life past its first token with the merged cold windows. A stream-second exists only because some policy admitted the turn that produced it and got it to a first token. As a result, a policy can lower this ratio by keeping streams alive longer, not only by generating fewer tokens. The scenario tables print the mean of each seed's own ratio. The bound works on the sums over seeds, which are a different average and are what the arithmetic below uses.

The bound therefore admits a cell only if NEW delivered at least as many agent tokens in the cold windows. The bound then asks what NEW's extra seconds returned. The quantity that prices a second of stream time served as a prefill batch's decode row is one token per pass. As a result, this is the ceiling on such a second is one over the longest pass NEW ran. `rowTokenFloor` scans NEW's prefill batches that carry rows and finish inside a cold window and returns `1 / longest`. If the extra seconds returned less than that, they held streams that generated nothing and the bound says so instead of excusing the cell.

## The one cell this covers: D, one cold prompt of 100K

Sums over the scenario's seeds, from the run's bound evidence:

| quantity | NEW | PREV |
| --- | --- | --- |
| agent tokens in the cold windows | 18,198 | 17,844 |
| stream-seconds in the cold windows | 283.941131 | 269.896917 |
| ratio of those two (the table prints 64.2 and 66.1 as per-seed means) | 64.090750 | 66.114130 |

The shortfall is 2.87%, and the bound's arithmetic on those numbers:

    extra tokens   18198 - 17844      =    354 tokens
    extra seconds  283.941131 - 269.896917 = 14.044214 s
    marginal rate  354 / 14.044214    =  25.206 tokens per stream-second
    a riding row's floor 1 / 0.363234 =   2.753 tokens per stream-second

So NEW's fourteen extra seconds of stream life returned 25.2 tokens per second, several times what a stream can earn at the slowest delivery form. The deployment has - one token at the end of the widest prefill batch it rode, 0.363234 s. 453 of NEW's tokens in these windows arrived as prefill rows, one per pass, which is the channel the floor prices. The extra seconds were service, not life support.

The other conditions the bound reads, same run:

| check | NEW | PREV |
| --- | --- | --- |
| per-agent tok/s in cold (`sum / (window span x conversations)`, a denominator no policy controls) | 44.450435 | 43.730055 |
| longest stall | 363.2 ms | 699.7 ms |

Both point the same way. Spread over every conversation rather than over live streams, NEW is 1.6% ahead, and no stream in NEW waited out a longer gap than a stream. This is in PREV did. The 0.363234 s in the floor is the same batch as NEW's 363.2 ms longest stall, which is the ride's own structural claim. This is a stream cannot wait out more than one prefill batch. This is because every batch hands it a token.

## Cells this derivation covers and refuses

Covers one cell: `stream decode tok/s in cold` against PREV at D with a single 100K cold prompt, the only cold-window stream-rate loss in the suite.

Refuses:

* A and B at 1, 2 and a few minutes and D at 25K, 200K, 400K and 500K, where NEW's rate is the higher one. A is 84.6 against 77.1. This is B 2 min 66.1 against 65.8. The narrowest, D 200K and B 2 min, are 0.51% and 0.52% wins, outside the 0.5% tie band, and they are not argued here. This is because nothing needs arguing in a win.
* C at every rate and `max_running`, and every thrash episode: no cold window opens, so metric 1 has no value, `Judge` reports undefined. A rate over an empty denominator is not a thing to argue.
* The same metric against OLD in any scenario. OLD is prefill-priority and its in-window stream rate is 0.0-3.6 tok/s, so NEW wins those cells outright. `boundName` does not claim them for this derivation.
* A cell where NEW delivered fewer agent tokens than PREV, or where the marginal rate fell under a riding row's floor. Both are coded as failures of the bound, on purpose: the first is a defect rather than a trade, and the second means. NEW was holding streams that generated nothing. Neither is excused by a better per-agent rate.
* Any statement about tok/s *outside* a cold prefill. The windows are where the contention is, and this metric measures nothing else.
