// GLM-5.2 per-cell benchmark numbers, keyed by the same `match` tuple as glm-5.2.jsx cells.
// See _deployment.jsx for the speed/accuracy schema.
// Numbers pending: each entry is a bare `match` stub (renders "pending") until measured
// end-to-end on the corresponding hardware, then filled with sglang_version + speed/accuracy.
export const benchmarks = [
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "v0.5.14 @ 49e384ce",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 668, tpot_ms: 5.05, tokens_per_sec_per_gpu: 197 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 6148, tpot_ms: 16.11, tokens_per_sec_per_gpu: 813 },
    ],
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "v0.5.14 @ 49e384ce",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 7473, tpot_ms: 23.49, tokens_per_sec_per_gpu: 2343 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 80562, tpot_ms: 28.08, tokens_per_sec_per_gpu: 2391 },
    ],
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "v0.5.14 @ 49e384ce",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 553480, tpot_ms: 61.71, tokens_per_sec_per_gpu: 1656 },
    ],
  },
  {
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "main @ 09ca4fc",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 757, tpot_ms: 3.22, tokens_per_sec_per_gpu: 288 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 3188, tpot_ms: 9.12, tokens_per_sec_per_gpu: 1476 },
    ],
  },
  {
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "main @ 09ca4fc",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 5742, tpot_ms: 17.65, tokens_per_sec_per_gpu: 3078 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 18744, tpot_ms: 32.61, tokens_per_sec_per_gpu: 5022 },
    ],
  },
  {
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "main @ 09ca4fc",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 177620, tpot_ms: 47.99, tokens_per_sec_per_gpu: 4059 },
    ],
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "main @ 09ca4fc",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 374, tpot_ms: 4.55, tokens_per_sec_per_gpu: 459 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 3719, tpot_ms: 11.5, tokens_per_sec_per_gpu: 2376 },
    ],
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "main @ 09ca4fc",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 7429, tpot_ms: 25.21, tokens_per_sec_per_gpu: 4437 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 27488, tpot_ms: 48.43, tokens_per_sec_per_gpu: 6804 },
    ],
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "main @ 09ca4fc",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 231101, tpot_ms: 86.01, tokens_per_sec_per_gpu: 6039 },
    ],
  },
  // ---- B300 + FP8 ---- (8-GPU single node, TP8; serve recipe in glm-5.2.jsx; benchmark pending re-measurement)
  { match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
  // ---- B300 + BF16 ---- (unquantized zai-org/GLM-5.2, TP8.
  { match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
  // ---- BF16 multi-node (inferred) ---- benchmarks pending
  { match: { hw: "h200",  variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "multi-2" } },
  { match: { hw: "h200",  variant: "default", quant: "bf16", strategy: "balanced",        nodes: "multi-2" } },
  { match: { hw: "h200",  variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" } },
  { match: { hw: "b200",  variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "multi-2" } },
  { match: { hw: "b200",  variant: "default", quant: "bf16", strategy: "balanced",        nodes: "multi-2" } },
  { match: { hw: "b200",  variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "multi-2" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced",        nodes: "multi-2" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" } },
  {
    match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 295, tpot_ms: 1.85, tokens_per_sec_per_gpu: 527 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2491, tpot_ms: 5.43, tokens_per_sec_per_gpu: 2289 },
    ],
  },
  {
    match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 5837, tpot_ms: 12.70, tokens_per_sec_per_gpu: 3770 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 16736, tpot_ms: 30.00, tokens_per_sec_per_gpu: 5343 },
    ],
  },
  {
    match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 130174, tpot_ms: 67.12, tokens_per_sec_per_gpu: 5305 },
    ],
  },
  {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    accuracy: { aime25_pct: 89.58 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 196, tpot_ms: 1.86, tokens_per_sec_per_gpu: 459 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 274, tpot_ms: 6.95, tokens_per_sec_per_gpu: 2016 },
    ],
  },
  {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
    accuracy: { aime25_pct: 89.58 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 680, tpot_ms: 48.9, tokens_per_sec_per_gpu: 1377 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 3010, tpot_ms: 149, tokens_per_sec_per_gpu: 1845 },
    ],
  },
  {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    accuracy: { aime25_pct: 89.58 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 6370, tpot_ms: 280, tokens_per_sec_per_gpu: 3870 },
    ],
  },
  // ---- MI355X + FP8 ---- gfx950, TP8, DSA tilelang, NO MTP (disabled on
  // AMD). Measured on
  // lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260618, flush-cache
  // every run. No spec-decoding, so not directly comparable to the NVIDIA
  // low-latency cells (EAGLE MTP).
  {
    match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 634, tpot_ms: 13.56, tokens_per_sec_per_gpu: 81 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 5411, tpot_ms: 23.60, tokens_per_sec_per_gpu: 621 },
    ],
  },
  {
    match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 19526, tpot_ms: 46.50, tokens_per_sec_per_gpu: 1098 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 117866, tpot_ms: 56.12, tokens_per_sec_per_gpu: 1044 },
    ],
  },
  {
    match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 432058, tpot_ms: 106.44, tokens_per_sec_per_gpu: 1269 },
    ],
  },
];
