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

## Throughput of a drained run

A run that goes on until every request finishes (A) has a throughput of its output tokens over its makespan. Before the last cold prompt's first token, the GPU runs every cold prompt's chunks, `sum(P)`. It also runs the decode time `D` spent while a cold prompt waits (`WindowDecode`). After that token, the prompt still decodes its other `n - 1` output tokens. A step gives at most `accept` tokens and costs at least a one-request step at the prompt's length, `step(1, prompt)`. So a schedule that decodes for `D` while the cold prompts wait has:

    makespan >= sum(P) + D + ceil((n - 1) / accept) x step(1, prompt)

Its throughput is at most the output tokens over that makespan. The test takes the smallest decode tail over the cold prompts. Per seed, the test checks that NEW is under this bound. Where NEW is under OLD, it also checks that the bound is under OLD. OLD's makespan does not contain `D`, and no schedule that decodes for `D` during the cold prompts gets it back. A run that stops at a horizon has no makespan. This bound does not apply to it.

## Numbers

OLD / NEW cells are the mean over the workload seeds in `Seeds`. Bound columns are the mean of the per-seed bounds, and NEW idle is the largest over the seeds. The forced columns count the seeds whose tail share holds that quantile at or above the cheapest chunk. Device pool 1397211 tokens, host tier 2794422 tokens.

| scenario | prefill alone s | NEW decode share | TTFT bound s | TTFT OLD / NEW s | NEW idle | cheapest chunk ms | tail share | p99 forced | ITL p99 OLD / NEW ms | p99.9 forced | ITL p99.9 OLD / NEW ms | throughput bound / OLD / NEW tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A (logged episode) | 42.4 | 0.109 | 47.6 | 68.6 / 78.2 | 0.00% | 63 | 2.88% | 16/16 | 30 / 599 | 16/16 | 76076 / 644 | 24.2 / 26.1 / 24.1 |
| B-1 (cold every 1 min) | 45.0 | 0.467 | 84.6 | 43.6 / 144.6 | 0.00% | 291 | 1.83% | 16/16 | 29 / 686 | 16/16 | 3122 / 1319 | - |
| B-2 (cold every 2 min) | 45.0 | 0.455 | 82.7 | 45.2 / 87.6 | 0.00% | 291 | 0.77% | 0/16 | 30 / 605 | 16/16 | 184 / 1281 | - |
| B-5 (cold every 5 min) | 45.0 | 0.458 | 83.1 | 45.2 / 88.3 | 0.00% | 291 | 0.28% | 0/16 | 31 / 28 | 16/16 | 191 / 1066 | - |
| D-25K | 1.8 | 0.426 | 3.2 | 1.9 / 3.4 | 0.00% | 40 | 0.07% | 0/16 | 22 / 22 | 3/16 | 170 / 194 | - |
| D-100K | 7.5 | 0.432 | 13.4 | 7.7 / 14.1 | 0.00% | 144 | 0.18% | 0/16 | 24 / 22 | 16/16 | 173 / 512 | - |
| D-200K | 16.7 | 0.445 | 30.3 | 16.8 / 31.9 | 0.00% | 291 | 0.24% | 0/16 | 26 / 23 | 16/16 | 176 / 700 | - |
| D-400K | 45.0 | 0.449 | 81.8 | 45.2 / 86.5 | 0.00% | 291 | 0.23% | 0/16 | 28 / 25 | 16/16 | 178 / 990 | - |
| D-500K | 66.5 | 0.453 | 121.8 | 66.8 / 128.7 | 0.00% | 80 | 0.22% | 0/16 | 29 / 26 | 16/16 | 177 / 1198 | - |
| B-2 prefill x0.75 | 34.0 | 0.458 | 62.8 | 34.1 / 65.7 | 0.00% | 220 | 0.72% | 0/16 | 30 / 431 | 16/16 | 148 / 978 | - |
| B-2 prefill x1.33 | 59.6 | 0.459 | 110.2 | 59.7 / 114.0 | 0.00% | 383 | 0.87% | 0/16 | 29 / 811 | 16/16 | 235 / 1653 | - |
| B-2 batch overhead 0 | 44.0 | 0.453 | 80.5 | 44.2 / 85.2 | 0.00% | 281 | 0.75% | 0/16 | 30 / 587 | 16/16 | 175 / 1261 | - |
| B-2 batch overhead 40ms | 48.0 | 0.456 | 88.1 | 48.1 / 93.0 | 0.00% | 321 | 0.79% | 0/16 | 30 / 663 | 16/16 | 213 / 1337 | - |
| B-2 decode base x0.7 | 45.0 | 0.423 | 78.1 | 45.2 / 83.0 | 0.00% | 291 | 0.58% | 0/16 | 26 / 193 | 16/16 | 177 / 1209 | - |
| B-2 decode per-req x3 | 45.0 | 0.465 | 84.1 | 45.2 / 88.9 | 0.00% | 291 | 0.91% | 0/16 | 40 / 658 | 16/16 | 192 / 1325 | - |
| B-2 decode per-ctx x0 | 45.0 | 0.439 | 80.3 | 45.2 / 85.4 | 0.00% | 291 | 0.60% | 0/16 | 18 / 290 | 16/16 | 181 / 1223 | - |
| B-2 decode per-ctx x3 | 45.0 | 0.479 | 86.5 | 45.2 / 90.7 | 0.00% | 291 | 1.12% | 16/16 | 53 / 746 | 16/16 | 205 / 1351 | - |
| B-2 accept 2.2 | 45.0 | 0.472 | 85.3 | 45.2 / 89.8 | 0.00% | 291 | 0.78% | 0/16 | 30 / 625 | 16/16 | 181 / 1306 | - |
| B-2 accept 3.5 | 45.0 | 0.427 | 78.6 | 45.2 / 83.6 | 0.00% | 291 | 0.74% | 0/16 | 30 / 538 | 16/16 | 191 / 1250 | - |
| B-2 overlap off | 45.0 | 0.456 | 82.8 | 45.1 / 87.9 | 0.00% | 291 | 1.50% | 16/16 | 30 / 448 | 16/16 | 186 / 695 | - |
| B-2 chunk 2048 | 46.0 | 0.169 | 55.5 | 46.1 / 55.8 | 0.00% | 128 | 0.54% | 0/16 | 30 / 75 | 16/16 | 187 / 393 | - |
| B-2 chunk 8192 | 44.5 | 0.439 | 79.3 | 44.7 / 83.6 | 0.00% | 572 | 0.38% | 0/16 | 30 / 30 | 16/16 | 187 / 2088 | - |
| B-2 shortest-prefill-first | 45.0 | 0.454 | 82.4 | 45.4 / 87.3 | 0.00% | 291 | 0.77% | 0/16 | 30 / 611 | 16/16 | 206 / 1284 | - |
| B-2 no host tier | 45.0 | 0.455 | 82.7 | 45.2 / 87.6 | 0.00% | 291 | 0.77% | 0/16 | 30 / 605 | 16/16 | 184 / 1281 | - |
| B-2 host 1.5x | 45.0 | 0.455 | 82.7 | 45.2 / 87.6 | 0.00% | 291 | 0.77% | 0/16 | 30 / 605 | 16/16 | 184 / 1281 | - |
| B-2 reload x10 | 45.0 | 0.455 | 82.7 | 45.2 / 87.6 | 0.00% | 291 | 0.77% | 0/16 | 30 / 605 | 16/16 | 184 / 1281 | - |
| B-2 mixed chunk off | 45.0 | 0.456 | 82.7 | 45.2 / 87.6 | 0.00% | 291 | 0.76% | 0/16 | 30 / 635 | 16/16 | 184 / 1309 | - |
| B-2 faster agents | 45.0 | 0.499 | 89.9 | 45.2 / 97.0 | 0.00% | 291 | 0.89% | 0/16 | 75 / 688 | 16/16 | 218 / 1288 | - |
| B-2 10 agents | 45.0 | 0.503 | 90.6 | 45.2 / 99.4 | 0.00% | 291 | 1.04% | 16/16 | 116 / 745 | 16/16 | 227 / 1255 | - |
| B-2 follow-ups 2.5-4K | 45.0 | 0.040 | 46.9 | 45.2 / 47.0 | 0.00% | 291 | 0.17% | 0/16 | 31 / 31 | 13/16 | 436 / 457 | - |
| B-2 outputs 1-3K | 45.0 | 0.503 | 90.5 | 45.1 / 92.3 | 0.00% | 291 | 0.92% | 0/16 | 30 / 702 | 16/16 | 165 / 1358 | - |

## What this shows

- **TTFT.** NEW idles 0.00% of every cold window, and its cold TTFT is at most 8% above `P / (1 - f)` in single-cold rows. A and B-1 sit further above, because there cold prompts queue behind each other. OLD's lower TTFT is the price of a stall as long as the whole prompt.
- **Throughput (A).** A drains. NEW's throughput is at its bound. The bound is under OLD's: OLD's makespan has no decode time while a cold prompt waits.
- **ITL p99.9.** The tail share exceeds 0.1% in every seed of every row but D-25K. As a result, the p99.9 of any schedule with NEW's stall length is at least the cheapest chunk, which NEW meets. OLD's p99.9 is lower where its long stall is a smaller share of gaps than NEW's many short ones. In D-25K the prompt is only a few chunks long. Its tail share is 0.03 to 0.19% per seed. The test fails in the seeds where NEW is above OLD and the share is under 0.1%.
- **ITL p99 is not proven in B-2 and most of its sensitivity rows.** Once the cold prompts that are already decoding count as streams, the tail share reaches 1% in A, B-1, `B-2 decode per-ctx x3`, `B-2 10 agents` and `B-2 overlap off`. In the other B-2 rows it is 0.5 to 0.9% on average over the seeds, and `TestColdPromptBound` fails there.

NEW's longest stall in those rows is the final pair of chunks of the 400K prompt (1.48 s). Every stall before it is a cheaper pair. So NEW has about twice the stalls of the bound's schedule. The simulator measures the schedules below, which close part of that gap. Both lose to PREV:

| change to NEW | B-2 ITL p99 ms | B-2 ITL p99.9 ms | B-2 longest stall s | other effect |
|---|---|---|---|---|
| none (OLD / PREV / NEW) | 30 / 635 / 605 | 184 / 1309 / 1281 | 45.49 / 1.52 / 1.48 | |
| every burst as long as the prompt's last two chunks, from the cost model | 168 | 1453 | 1.56 | D-400K p99.9 1276 ms against PREV's 1011 |
| bursts packed up to the longest burst paid so far | 405 | 1392 | 1.51 | the limit creeps: Thrash 1.5x 1800s longest stall 1.91 s against PREV's 1.34 |

Even the first schedule does not reach OLD's p99. Follow-up turns also stall every stream, and OLD's own gaps above its p99 outside NEW's cold windows are another 0.15 to 0.25% of gaps. The bound does not count them, because which streams run during a turn depends on the schedule. So in these rows a lower p99 costs p99.9 and longest stall against PREV. The cold prompts alone do not prove that OLD's p99 is out of reach.
