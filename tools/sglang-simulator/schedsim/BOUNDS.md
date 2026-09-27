# What decoding during a cold prompt costs

OLD runs every chunk of a cold prompt back to back and decodes nothing until the prompt is done. That gives OLD the best cold TTFT and a low ITL p99, and it pays with a stall the length of the whole prompt (tens of seconds). NEW keeps the running streams decoding while the cold prompt prefills. This page derives what any schedule that does that must pay against OLD, from the simulator's own numbers, and shows where NEW sits.

`go run ./cmd/schedsim -only bounds` prints the table below. `internal/scenario/bounds.go` computes it, and `TestColdPromptBound` asserts it.

## Cold TTFT

Let `P` be the time the cold prompt's own chunk batches take when they run back to back. `coldPrefill` prices each chunk with the calibrated cost model: the batch overhead plus the extend cost of that chunk at its offset. No schedule gets the prompt out in less than `P`.

Let `f` be the share of the GPU that goes to decode while a cold prompt waits. The simulator records it as `WindowDecode / WindowSeconds` over every interval in which a cold prompt is queued or prefilling. If the GPU is never idle in those windows, the prompt gets at most `1 - f` of the GPU, so:

    TTFT >= P / (1 - f)

The bound is tight only when the GPU never idles in the cold windows. The test checks that NEW's idle share there is under 0.5%. It is 0.00% in every row. So NEW's cold TTFT is the price of the decode share it gives, plus queueing behind other prefill work (other cold prompts, follow-up turns and recomputes).

`f` is a choice. The balancer gives decode an equal share of GPU time while both classes have work, so `f` is near 0.45 when streams are running. It falls when few streams run (A: 0.11, follow-ups 2.5-4K: 0.04) and when the chunk is small (chunk 2048: 0.17).

## ITL tail

A stream that decodes during a cold prompt must stop for every prefill batch it does not ride in. Let `S` be NEW's longest stall. A schedule whose stalls are no longer than `S` cuts each cold prompt into at least `ceil(P / S)` stalls, and each running stream sees every one. Over the run:

    tail share = streams x sum(ceil(P / S)) / gaps

Here `streams` is the mean number of requests decoding in the cold windows and `gaps` is the count of inter-token gaps. A cold prompt that already has its first token counts as a stream: its gaps are in `gaps`. It sits through the next cold prompt like any other request. If the tail share is at least `1 - q`, then at least that share of gaps contains a stall. A stall contains at least one chunk, so the ITL q-quantile is at least the cheapest chunk. When the tail share is below `1 - q`, the derivation does not force the quantile up. The test then demands NEW be no worse than OLD.

A quantile is a property of one run, and the mean of per-seed quantiles is not the quantile of anything. So the test derives the tail bound per seed, from that seed's NEW run, and compares it with that seed's OLD run. It checks a seed only when the seed-mean quantile of NEW is worse than OLD's and that seed's NEW is worse than its OLD.

## Numbers

Cells are OLD / NEW, each the mean of 16 workload seeds. Device pool 1397211 tokens, host tier 2794422 tokens.

| scenario | prefill alone s | NEW decode share | TTFT bound s | TTFT OLD / NEW s | NEW idle | cheapest chunk ms | tail share | ITL p99 OLD / NEW ms | ITL p99.9 OLD / NEW ms |
|---|---|---|---|---|---|---|---|---|---|
| A (logged episode) | 42.4 | 0.109 | 47.6 | 68.6 / 78.2 | 0.00% | 63 | 0.66% | 30 / 599 | 76076 / 644 |
| B-1 (cold every 1 min) | 45.0 | 0.467 | 84.5 | 43.6 / 144.6 | 0.00% | 291 | 1.68% | 29 / 686 | 3122 / 1319 |
| B-2 (cold every 2 min) | 45.0 | 0.456 | 82.7 | 45.2 / 87.6 | 0.00% | 291 | 0.77% | 30 / 605 | 184 / 1281 |
| B-5 (cold every 5 min) | 45.0 | 0.458 | 83.1 | 45.2 / 88.3 | 0.00% | 291 | 0.28% | 31 / 28 | 191 / 1066 |
| D-25K | 1.8 | 0.446 | 3.2 | 1.9 / 3.4 | 0.00% | 40 | 0.07% | 22 / 22 | 170 / 194 |
| D-100K | 7.5 | 0.440 | 13.4 | 7.7 / 14.1 | 0.00% | 144 | 0.18% | 24 / 22 | 173 / 512 |
| D-200K | 16.7 | 0.448 | 30.3 | 16.8 / 31.9 | 0.00% | 291 | 0.24% | 26 / 23 | 176 / 700 |
| D-400K | 45.0 | 0.450 | 81.8 | 45.2 / 86.5 | 0.00% | 291 | 0.23% | 28 / 25 | 178 / 990 |
| D-500K | 66.5 | 0.454 | 121.9 | 66.8 / 128.7 | 0.00% | 80 | 0.22% | 29 / 26 | 177 / 1198 |
| B-2 prefill x0.75 | 34.0 | 0.458 | 62.8 | 34.1 / 65.7 | 0.00% | 220 | 0.72% | 30 / 431 | 148 / 978 |
| B-2 prefill x1.33 | 59.6 | 0.459 | 110.2 | 59.7 / 114.0 | 0.00% | 383 | 0.83% | 29 / 811 | 235 / 1653 |
| B-2 batch overhead 0 | 44.0 | 0.453 | 80.6 | 44.2 / 85.2 | 0.00% | 281 | 0.74% | 30 / 587 | 175 / 1261 |
| B-2 batch overhead 40ms | 48.0 | 0.456 | 88.1 | 48.1 / 93.0 | 0.00% | 321 | 0.80% | 30 / 663 | 213 / 1337 |
| B-2 decode base x0.7 | 45.0 | 0.423 | 78.1 | 45.2 / 83.0 | 0.00% | 291 | 0.58% | 26 / 193 | 177 / 1209 |
| B-2 decode per-req x3 | 45.0 | 0.465 | 84.1 | 45.2 / 88.9 | 0.00% | 291 | 0.91% | 40 / 658 | 192 / 1325 |
| B-2 decode per-ctx x0 | 45.0 | 0.439 | 80.3 | 45.2 / 85.4 | 0.00% | 291 | 0.60% | 18 / 290 | 181 / 1223 |
| B-2 decode per-ctx x3 | 45.0 | 0.480 | 86.5 | 45.2 / 90.7 | 0.00% | 291 | 1.12% | 53 / 746 | 205 / 1351 |
| B-2 accept 2.2 | 45.0 | 0.472 | 85.3 | 45.2 / 89.8 | 0.00% | 291 | 0.78% | 30 / 625 | 181 / 1306 |
| B-2 accept 3.5 | 45.0 | 0.427 | 78.6 | 45.2 / 83.6 | 0.00% | 291 | 0.74% | 30 / 538 | 191 / 1250 |
| B-2 overlap off | 45.0 | 0.456 | 82.8 | 45.1 / 87.9 | 0.00% | 291 | 1.50% | 30 / 448 | 186 / 695 |
| B-2 chunk 2048 | 46.0 | 0.171 | 55.5 | 46.1 / 55.8 | 0.00% | 128 | 0.54% | 30 / 75 | 187 / 393 |
| B-2 chunk 8192 | 44.5 | 0.439 | 79.3 | 44.7 / 83.6 | 0.00% | 572 | 0.38% | 30 / 30 | 187 / 2088 |
| B-2 shortest-prefill-first | 45.0 | 0.454 | 82.4 | 45.4 / 87.3 | 0.00% | 291 | 0.77% | 30 / 611 | 206 / 1284 |
| B-2 no host tier | 45.0 | 0.456 | 82.7 | 45.2 / 87.6 | 0.00% | 291 | 0.77% | 30 / 605 | 184 / 1281 |
| B-2 host 1.5x | 45.0 | 0.456 | 82.7 | 45.2 / 87.6 | 0.00% | 291 | 0.77% | 30 / 605 | 184 / 1281 |
| B-2 reload x10 | 45.0 | 0.456 | 82.7 | 45.2 / 87.6 | 0.00% | 291 | 0.77% | 30 / 605 | 184 / 1281 |
| B-2 mixed chunk off | 45.0 | 0.456 | 82.7 | 45.2 / 87.6 | 0.00% | 291 | 0.76% | 30 / 635 | 184 / 1309 |
| B-2 faster agents | 45.0 | 0.499 | 89.9 | 45.2 / 97.0 | 0.00% | 291 | 0.89% | 75 / 688 | 218 / 1288 |
| B-2 10 agents | 45.0 | 0.503 | 90.6 | 45.2 / 99.4 | 0.00% | 291 | 1.03% | 116 / 745 | 227 / 1255 |
| B-2 follow-ups 2.5-4K | 45.0 | 0.040 | 46.9 | 45.2 / 47.0 | 0.00% | 291 | 0.17% | 31 / 31 | 436 / 457 |
| B-2 outputs 1-3K | 45.0 | 0.503 | 90.5 | 45.1 / 92.3 | 0.00% | 291 | 0.91% | 30 / 702 | 165 / 1358 |

## What this shows

- **TTFT.** NEW idles 0.00% of every cold window, and its cold TTFT is 2 to 8% above `P / (1 - f)` in single-cold rows. A and B-1 sit further above, because there cold prompts queue behind each other. OLD's lower TTFT is the price of a stall as long as the whole prompt.
- **ITL p99.9.** The tail share exceeds 0.1% in every row but D-25K. As a result, the p99.9 of any schedule with NEW's stall length is at least the cheapest chunk, which NEW meets. OLD's p99.9 is lower where its long stall is a smaller share of gaps than NEW's many short ones.
- **ITL p99 is not proven.** The tail share reaches 1% only in the rows `B-1`, `B-2 decode per-ctx x3`, `B-2 10 agents` and `B-2 overlap off`. In B-2 and most sensitivity rows it is 0.6 to 0.9%. There a schedule with NEW's longest stall can, in principle, put stalls in under 1% of gaps by bursting longer and less often. With overlap, a NEW burst is a pair of chunks, and its longest stall is set by rarer events (a mixed batch of retracted requests, a follow-up that rides in the chunk). So NEW's ITL p99 in those rows is above the bound, and `TestColdPromptBound` fails there. A burst that runs until it reaches the stall length already paid may close the gap. It is not implemented or measured.
