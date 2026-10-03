// One entry per cell `match` tuple (same keys as the config cells).

export const benchmarks = [
  {
    match: { hw: "rtx5090", variant: "default", quant: "bf16", nodes: "single" },
    sglang_version: "dev @ 30705c004c",
    latencyPercentile: "Mean",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 34, tpot_ms: 4.0, tokens_per_sec_per_gpu: 496 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 128 },
        ttft_ms: null, tpot_ms: 11.8, tokens_per_sec_per_gpu: 19280 },
    ],
  },
  {
    match: { hw: "dgx-spark", variant: "default", quant: "bf16", nodes: "single" },
    sglang_version: "dev @ 30705c004c",
    latencyPercentile: "Mean",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 85, tpot_ms: 27.7, tokens_per_sec_per_gpu: 72 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 64 },
        ttft_ms: 1373, tpot_ms: 42.9, tokens_per_sec_per_gpu: 2892 },
    ],
  },
  // Pending — no numbers taken on these platforms yet.
  { match: { hw: "h200",    variant: "default", quant: "bf16", nodes: "single" } },
  { match: { hw: "rtx6000", variant: "default", quant: "bf16", nodes: "single" } },
];
