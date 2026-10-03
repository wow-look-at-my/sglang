export const benchmarks = [


  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 23742, tpot_ms: 256.8, tokens_per_sec_per_gpu: 5264 },
    ],
    accuracy: { gsm8k_pct: 93.18, aime25_pct: null },
  },
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 102.3, tpot_ms: 3.48, tokens_per_sec_per_gpu: 305 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 97.7, tpot_ms: 6.45, tokens_per_sec_per_gpu: 2094 },
    ],
    accuracy: { gsm8k_pct: 93.33, aime25_pct: null },
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 16199, tpot_ms: 293.2, tokens_per_sec_per_gpu: 5175 },
    ],
    accuracy: { gsm8k_pct: 94.24, aime25_pct: 0.654 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 97.7, tpot_ms: 4.43, tokens_per_sec_per_gpu: 242 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 99.4, tpot_ms: 7.16, tokens_per_sec_per_gpu: 1956 },
    ],
    accuracy: { gsm8k_pct: 94.47, aime25_pct: null },
  },
  {
    match: { hw: "h200", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 19886, tpot_ms: 318.5, tokens_per_sec_per_gpu: 5055 },
    ],
    accuracy: { gsm8k_pct: 95.00, aime25_pct: 0.690 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 113.1, tpot_ms: 3.83, tokens_per_sec_per_gpu: 276 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 94.1, tpot_ms: 6.40, tokens_per_sec_per_gpu: 2125 },
    ],
    accuracy: { gsm8k_pct: 94.24, aime25_pct: null },
  },


  {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 93.71, aime25_pct: null },
  },
  {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 93.63, aime25_pct: null },
  },
  {
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.39, aime25_pct: 66.88 },
  },
  {
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.69, aime25_pct: null },
  },
  {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.54, aime25_pct: 66.46 },
  },
  {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 95.30, aime25_pct: null },
  },
  {
    match: { hw: "b300", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.62, aime25_pct: 68.33 },
  },
  {
    match: { hw: "b300", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.69, aime25_pct: null },
  },


  {
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 93.33, aime25_pct: null },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 114, tpot_ms: 4.2, tokens_per_sec_per_gpu: 562 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 141, tpot_ms: 8.4, tokens_per_sec_per_gpu: 3413 },
    ],
    accuracy: { gsm8k_pct: 93.33, aime25_pct: null },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 10062, tpot_ms: 216, tokens_per_sec_per_gpu: 9536 },
    ],
    accuracy: { gsm8k_pct: 94.31, aime25_pct: null },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 109, tpot_ms: 6.1, tokens_per_sec_per_gpu: 369 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 135, tpot_ms: 10.7, tokens_per_sec_per_gpu: 2578 },
    ],
    accuracy: { gsm8k_pct: 94.54, aime25_pct: null },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 7186, tpot_ms: 211, tokens_per_sec_per_gpu: 9658 },
    ],
    accuracy: { gsm8k_pct: 94.47, aime25_pct: null },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 115, tpot_ms: 11.2, tokens_per_sec_per_gpu: 204 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 125, tpot_ms: 18.8, tokens_per_sec_per_gpu: 1549 },
    ],
    accuracy: { gsm8k_pct: 94.77, aime25_pct: null },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 10428, tpot_ms: 216, tokens_per_sec_per_gpu: 9184 },
    ],
    accuracy: { gsm8k_pct: 94.69, aime25_pct: null },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 134, tpot_ms: 5.4, tokens_per_sec_per_gpu: 410 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 127, tpot_ms: 10.1, tokens_per_sec_per_gpu: 2880 },
    ],
    accuracy: { gsm8k_pct: 95.00, aime25_pct: null },
  },
];
