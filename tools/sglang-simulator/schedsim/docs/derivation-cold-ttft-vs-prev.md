# Cold TTFT against the same balancer

Covers `boundOwnWork`, the bound `contract_test.go` reads for a `cold TTFT mean`
cell lost to PREV. PREV is the balancer at f15db9ee6a and NEW is the balancer on
this branch: same control law family, same cede against the same prompt, so there
is no whole-GPU asymmetry to invoke as there is against OLD (see
`derivation-cold-ttft-vs-old.md`). What can be claimed here is narrower, and it is
arithmetic.

## The identity, which is the whole content

`AccountWindow` fills five buckets by charging each batch's in-window seconds in
the batch's own proportions and sets `Idle = Window - Total`. Both policies'
accounts are therefore partitions of the same kind of interval, and subtracting
them is exact:

    gap = (ownNEW - ownPREV)
        + (otherNEW - otherPREV) + (rowsNEW - rowsPREV)
        + (overheadNEW - overheadPREV) + (decodeNEW - decodePREV)
        + (idleNEW - idlePREV)

The test computes `carried`, the four middle differences, `ownExcess` and
`idleExcess`, and requires `|gap - (ownExcess + carried + idleExcess)| <=
coldTTFTIdentitySlack x gap`. Because the identity holds by construction, this
check is not measuring a model, it is measuring the bookkeeping: it fails if a
second stops being attributed to exactly one bucket. `coldTTFTIdentitySlack` is 1e-6
because the two sides are sums of the same floats in a different order, and a part
per million of a multi-thousand-second gap is orders of magnitude above the double
round-off in such a sum and far below the smallest real term the account can move:
one forward pass of the log's own base is 10.0 ms, which against A's 6.748 s gap is
1.5e-3 - fifteen hundred times the tolerance. A single mis-attributed pass therefore
fails the identity instead of sliding under it.

Given a closed identity the other two checks reduce to one statement, which is worth
saying plainly rather than leaving implicit. `carried >= gap - ownTolerance * ownPREV`
is, after substituting the identity, `ownExcess + idleExcess <= ownTolerance * ownPREV`.
So the bound does not measure how much of the extra wait was other requests' work - the
identity already assigned every second to a bucket - it measures whether the wait grew
because NEW computed more of the prompt than PREV did, or left the GPU idle where PREV
did not. Neither is allowed, and the two named checks below are what enforce that.

## The measured cells

| scenario | paired windows | NEW wait (s) | PREV wait (s) | gap (s) | own excess (s) | other diff (s) | rows diff (s) | overhead diff (s) | decode diff (s) | idle diff (s) | carried (s) | residue (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 3 | 386.336392 | 379.588140 | 6.748252 | 0.000185 | +2.369064 | +0.064156 | +2.366226 | +1.948621 | 0.000000 | 6.748067 | 0.000000000 |
| B 2 min | 35 | 3674.015726 | 3608.840077 | 65.175650 | 0.000987 | +8.779108 | +1.449789 | +14.841296 | +40.104469 | 0.000000 | 65.174662 | 0.000000000 |
| B 5 min | 15 | 1571.441287 | 1547.343652 | 24.097635 | 0.000541 | +3.595574 | +0.637510 | +6.240807 | +13.623203 | 0.000000 | 24.097094 | 0.000000000 |
| D 500K | 5 | 750.384094 | 735.492702 | 14.891392 | 0.000418 | +2.383950 | +0.272665 | +3.994211 | +8.240147 | 0.000000 | 14.890973 | 0.000000000 |

The residue column is the identity's error, printed to nine decimals: zero in every
cell, against a tolerance of 6.7e-6 s in A and 1.5e-5 s at D 500K.

Same prompt, same computation. The second named check refuses a cell whose gap is a
different amount of compute, and its precondition is that the prompt's own prefill
tokens are equal:

| scenario | own tokens NEW | own tokens PREV | own seconds NEW | own seconds PREV | excess as % of own | own passes NEW / PREV |
| --- | --- | --- | --- | --- | --- | --- |
| A | 1,134,560 | 1,134,560 | 124.397253 | 124.397068 | 0.00015 | 351 / 289 |
| B 2 min | 14,000,000 | 14,000,000 | 1541.365488 | 1541.364501 | 0.00006 | 4,385 / 3,732 |
| B 5 min | 6,000,000 | 6,000,000 | 660.585313 | 660.584772 | 0.00008 | 1,921 / 1,598 |
| D 500K | 2,500,000 | 2,500,000 | 326.498498 | 326.498080 | 0.00013 | 900 / 676 |

Token counts match exactly, so the only remaining question is how the same tokens were
cut into passes. NEW runs 62 more own passes than PREV in A, 653 more at B 2 min, 323
more at B 5 min and 224 more at D 500K - the fair cede and the seconds-form chunk cap
both shorten a continuation - and `ownTolerance` (5%) is sized so that the whole
per-pass base of every one of those passes could be charged to own work without
tripping it: 653 x 10.0 ms = 6.53 s against 1541.4 s of own work is the largest such
ratio in the set, 0.42%, and the others are 0.50%, 0.29% and 0.69%. In the
account the difference does not even land in that bucket, because `AccountWindow`
splits a pass's base into `Overhead` rather than into the request whose chunk it was;
the measured excess is 0.00006-0.00015% of own work. The overhead the extra passes
cost does show up where it belongs, 2.37 s in A and 3.99 s at D 500K, and those seconds
are counted as work NEW bought for other requests, which is the cell's excuse.

Nothing is idled. `prevIdleSlack` is 0.2% of PREV's paired window seconds, and NEW's
idle excess is 0.000000 s in all four cells - the GPU is never left between batches
inside a cold prompt - so the tolerance is never consulted. Where 0.2% comes from: it
and `TieRelative` are stated against the same quantity per window, because a cold
window *is* the interval metric 4 reports. A cell only reaches this bound at 0.5% of
the wait; an idling allowance of 0.2% of that same interval is strictly smaller, so
the bound is never the reason a tie got excused. Measured against the cells it has to
clear, it is 11.2% of A's gap (0.759 s against 6.748 s), 11.1% at B 2 min (7.218 s
against 65.175 s), 12.8% at B 5 min (3.095 s against 24.098 s) and 9.9% at D 500K
(1.471 s against 14.891 s) - close enough to the gap that a real idle would be caught,
far enough that round-off in the residual bucket is not.

## Cells this derivation covers and refuses

Covers the four cells where NEW's cold TTFT mean is worse than PREV's by more than
`TieRelative`: A (+1.78%), B 2 min (+1.81%), B 5 min (+1.56%), D 500K (+2.02%).

Refuses:

* B 1 min, D 25K, D 100K, D 200K and D 400K. B 1 min and D 200K and D 400K are cells
  NEW wins against PREV (-573.036365 s, -1.014346 s, -3.125518 s of paired wait); D 25K
  (+0.008200 s, 0.04%) and D 100K (+0.260784 s, 0.32%) are inside the 0.5% tie band and
  the test never reaches a bound for them. Both of those ties would pass this bound if
  they moved out of the band - their accounts close and their idle excess is zero - and
  they are not claimed here because no cell is argued that the tables do not mark.
* C at every rate and `max_running`, and the thrash episodes: no cold prompt arrives,
  metric 4 has no value, and `boundName` maps these keys to no derivation at all. The
  thrash family is excluded by key prefix even though its accounts would be empty rather
  than unfavourable.
* The claim that the gap is small. It is 1.56-2.02% and it is real: NEW's extra own
  passes and the follow-up prefill they let in beside the chunk cost its prompt between
  2.24 s and 65.18 s of paired wait. What this derivation supports is only that the cost
  is other requests' work and not extra compute or wasted GPU.
* A single-window reading. D 500K's five windows share a workload and the account is a
  sum over them; nothing here prices one prompt's wait alone.
