// Laguna-XS-2.1 benchmarks — one entry per cell `match` (same keys as
// laguna-xs21.jsx cells).
//
// All numbers below are REAL measured values; cells without measurements are
// bare `{ match }` pending stubs (the card renders "pending"). NO
// fabricated/dummy numbers.

//
// Thinking ENABLED by serving with a copy of the model's chat template
// whose enable_thinking default is flipped to true — sgl-eval's
// --thinking sets the generic 'thinking' key, which Laguna's template
// ignores (see Configuration Tips: Thinking).
export const benchmarks = [
  // ===== H200 (8-GPU HGX; bf16 tp fp8/int4 tp8+ep8) — ✅ REAL, full GSM8K =====
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 76.12, aime25_pct: 63.96 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 75.97, aime25_pct: 65.00 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 73.54, aime25_pct: 64.79 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 74.53, aime25_pct: 62.50 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 67.02, aime25_pct: 63.33 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 66.57, aime25_pct: 64.17 },
  },

  // ===== B300 (8-GPU HGX; bf16/nvfp4 tp fp8/int4 tp8+ep8) — REAL, full GSM8K =====
  // (accuracy measured as 2x(4xGB300) tp8/MNNVL — same GPU + shard math as one B300 node)
  {
    // REAL — BF16 dense, tp8, backend auto->trtllm_mha.
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main; run @ main 0543246184)",
    accuracy: { gsm8k_pct: 75.59, aime25_pct: 65.21 },
  },
  {
    // REAL — BF16 + DFlash (matched bf16 draft), tp8, trtllm_mha.
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main; run @ main 0543246184)",
    accuracy: { gsm8k_pct: 75.36, aime25_pct: 65.62 },
  },
  {
    // REAL — FP8 dense, tp8+ep8+SGLANG_SHARED_EXPERT_TP1=1 (plain tp8 impossible: block-FP8 scale granularity).
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main; run @ main 0543246184)",
    accuracy: { gsm8k_pct: 71.19, aime25_pct: 61.67 },
  },
  {
    // REAL — FP8 + DFlash (matched fp8-calibrated draft), tp8+ep8+flag, trtllm_mha.
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main; run @ main 0543246184)",
    accuracy: { gsm8k_pct: 71.87, aime25_pct: 62.50 },
  },
  {
    // REAL — NVFP4 dense, tp8 — NO escape needed (group_size=16 shards 8-way cleanly).
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main; run @ main 0543246184)",
    accuracy: { gsm8k_pct: 78.01, aime25_pct: 57.92 },
  },
  {
    // REAL — NVFP4 + DFlash (matched nvfp4-calibrated draft), tp8, trtllm_mha.
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main; run @ main 0543246184)",
    accuracy: { gsm8k_pct: 77.79, aime25_pct: 60.21 },
  },
  {
    // REAL — INT4 dense (mixed 4/8-bit MoE), tp8+ep8 (plain tp8 impossible: Marlin gs=128 'scales is not contiguous', same signature as H200).
    match: { hw: "b300", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main; run @ main 0543246184)",
    accuracy: { gsm8k_pct: 67.25, aime25_pct: 63.54 },
  },
  {
    // REAL — INT4 + DFlash (matched int4-calibrated draft), tp8+ep8, trtllm_mha.
    match: { hw: "b300", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main; run @ main 0543246184)",
    accuracy: { gsm8k_pct: 66.72, aime25_pct: 62.92 },
  },

  {
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 75.66, aime25_pct: 62.50 },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 76.19, aime25_pct: 65.83 },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 71.87, aime25_pct: 63.12 },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 72.02, aime25_pct: 63.12 },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 78.39, aime25_pct: 60.00 },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 74.53, aime25_pct: 60.00 },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 66.79, aime25_pct: 64.79 },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 67.02, aime25_pct: 61.04 },
  },
];
