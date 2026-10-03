// Intern-S2-Mobius per-cell benchmark numbers, keyed by the same `match`
// tuple as intern-s2-mobius.jsx cells. All H200 numbers measured in this work
// on 2xH200 (TP=2, sglang main @ e0828ee3 + head — model landed in main, so
// lmsysorg/sglang:dev is equivalent now).
export const benchmarks = [
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "main @ e0828ee3",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 177.91, tpot_ms: 3.13, tokens_per_sec_per_gpu: 1283.2 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1139.88, tpot_ms: 6.84, tokens_per_sec_per_gpu: 9014.6 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 4440.38, tpot_ms: 12.17, tokens_per_sec_per_gpu: 13016.5 },
    ],
    accuracy: { gsm8k_pct: 96.66, gpqa_pct: 79.23 },
  },

  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "main @ e0828ee3",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1166.47, tpot_ms: 14.26, tokens_per_sec_per_gpu: 4679.2 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 4112.65, tpot_ms: 22.91, tokens_per_sec_per_gpu: 10697.7 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 16290.63, tpot_ms: 50.31, tokens_per_sec_per_gpu: 17393.4 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 182448.22, tpot_ms: 123.82, tokens_per_sec_per_gpu: 11875.5 },
    ],
    accuracy: { gsm8k_pct: 96.82 },
  },

  // ==== B200 recipes are inferred from the H200 ones — benchmarks pending. ====
  { match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
];
