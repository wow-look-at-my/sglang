// Laguna-M.1 benchmarks — one entry per cell `match` (same keys as
// laguna-m1.jsx cells).
//
// All numbers below are REAL measured values; cells without measurements are
// bare `{ match }` pending stubs (the card renders "pending"). NO
// fabricated/dummy numbers.

export const benchmarks = [
  // ===== H200 — BF16 / FP8 =====
  {
    // ✅ REAL — 8xH200, BF16, tp8.
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    verified: true,
    sglang_version: "main @ 3f668733 (#28400 + #28604)",
    speed: [
      { workload: { dataset: "random", isl: 4096, osl: 1024, max_concurrency: 1 },
        ttft_ms: 81.9, tpot_ms: 8.91, tokens_per_sec_per_gpu: 69 },
      { workload: { dataset: "random", isl: 4096, osl: 1024, max_concurrency: 128 },
        ttft_ms: 200.1, tpot_ms: 52.1, tokens_per_sec_per_gpu: 1415 },
    ],
    accuracy: { gsm8k_pct: 93.02 },
  },
  {
    // ✅ REAL — 8xH200, FP8, tp8. (Hopper: no --fp8-gemm-backend flag needed.)
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    verified: true,
    sglang_version: "main @ 3f668733 (#28400 + #28604 + g_proj FP8 fix #28649)",
    accuracy: { gsm8k_pct: 93.25 },
  },

  // ===== B200 (8-GPU HGX) — BF16 / FP8 / NVFP4 =====
  {
    // ✅ REAL — 8xB200, BF16, tp8.
    match: { hw: "b200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    verified: true,
    sglang_version: "PR #28400 + #28604",
    speed: [
      { workload: { dataset: "random", isl: 4096, osl: 1024, max_concurrency: 1 },
        ttft_ms: 108, tpot_ms: 9.0, tokens_per_sec_per_gpu: 68 },
      { workload: { dataset: "random", isl: 4096, osl: 1024, max_concurrency: 128 },
        ttft_ms: 170, tpot_ms: 43.3, tokens_per_sec_per_gpu: 1655 },
    ],
    accuracy: { gsm8k_pct: 91.88 },
  },
  {
    // ✅ REAL — 8xB200, FP8, tp8, with --fp8-gemm-backend triton.
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    verified: true,
    sglang_version: "main + #28649 + --fp8-gemm-backend triton (DeepGEMM UE8M0 workaround; fix = PR #28662)",
    accuracy: { gsm8k_pct: 93.78 },
  },
  {
    // ✅ REAL — 8xB200, NVFP4, tp8.
    match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
    verified: true,
    sglang_version: "PR #28400 + #28604",
    accuracy: { gsm8k_pct: 89.38 },
  },

  // ===== B300 / GB200 / GB300 — BF16 / FP8 / NVFP4, UNVERIFIED → bare "pending" stubs (no fabricated numbers).
  { match: { hw: "b300",  variant: "default", quant: "bf16",  strategy: "balanced", nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "fp8",   strategy: "balanced", nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" } },
  { match: { hw: "gb200", variant: "default", quant: "bf16",  strategy: "balanced", nodes: "single" } },
  { match: { hw: "gb200", variant: "default", quant: "fp8",   strategy: "balanced", nodes: "single" } },
  { match: { hw: "gb200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16",  strategy: "balanced", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "fp8",   strategy: "balanced", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" } },
];
