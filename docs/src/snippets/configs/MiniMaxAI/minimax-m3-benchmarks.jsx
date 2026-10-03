// MiniMax-M3 per-cell benchmark numbers, keyed by the same `match` tuple as
// minimax-m3.jsx cells. See _deployment.jsx for the speed/accuracy schema.
export const benchmarks = [
  {
    // B200 re-measured at tp8 on minimax-m3-upstream.
    match: { hw: "b200", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #27944",
    speed: [
      // bench_serving --flush-cache, MSA path, tp8; warm steady-state (3-run, identical).
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 64, num_prompts: 128 },
        ttft_ms: 1580, tpot_ms: 24.1, tokens_per_sec_per_gpu: 2385 },
    ],
    accuracy: { gpqa_pct: 89.1, gsm8k_pct: 96.5, mmmu_pro_pct: 72.7 },
  },
  {
    // Hopper H200: bf16 build (MXFP8 is Blackwell-only) at tp8, built-in Triton sparse path (MSA is Blackwell-only).
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #27944",
    speed: [
      // bench_serving --flush-cache, bf16 Triton path; warm steady-state (3-run, cold-start run-1 excluded).
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 64, num_prompts: 128 },
        ttft_ms: 1054, tpot_ms: 70.8, tokens_per_sec_per_gpu: 1044 },
    ],
    accuracy: { gsm8k_pct: 97.0 },
  },
  {
    match: { hw: "b300", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #27944",
    speed: [
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 64 },
        ttft_ms: null, tpot_ms: 32.8, tokens_per_sec_per_gpu: 3285 },
    ],
    accuracy: { gsm8k_pct: null },
  },
  // GB200: inferred-supported, not directly benchmarked.
  { match: { hw: "gb200", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" } },
  {
    match: { hw: "gb300", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #27944",
    speed: [
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 64 },
        ttft_ms: 4746, tpot_ms: 39.3, tokens_per_sec_per_gpu: 2493 },
      { workload: { dataset: "random", isl: 8192, osl: 256, max_concurrency: 24 },
        ttft_ms: 3324, tpot_ms: 32.9, tokens_per_sec_per_gpu: 4323 },
    ],
    accuracy: { gsm8k_pct: null },
  },
  // MI355X (gfx950): native MXFP8. No TTFT/TPOT reported for this run.
  {
    match: { hw: "mi355x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #27944",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 64, num_prompts: 640 },
        ttft_ms: null, tpot_ms: null, tokens_per_sec_per_gpu: 420 },
    ],
    accuracy: { gsm8k_pct: null },
  },
  // MI350X (gfx950): inferred-supported from MI355X, not separately benchmarked.
  { match: { hw: "mi350x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" } },
  {
    match: { hw: "mi300x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #27944",
    accuracy: { gsm8k_pct: null },
  },
  // MI325X (gfx942): inferred-supported from MI300X, not separately benchmarked.
  { match: { hw: "mi325x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" } },
];
