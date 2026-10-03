// Hy3 per-cell benchmark numbers, keyed by the same `match` tuple as hy3.jsx cells. See
// _deployment.jsx for the speed/accuracy schema. FP8 cells not yet verified.
export const benchmarks = [
  { match: { hw: "h200",  variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" }, gsm8k_pct: 95.75 },
  { match: { hw: "h200",  variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" }, gsm8k_pct: 95.83 },
  { match: { hw: "b200",  variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b200",  variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "gb200", variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb200", variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "h200",  variant: "default",  quant: "fp8",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "h200",  variant: "default",  quant: "fp8",  strategy: "balanced",        nodes: "single" } },
  { match: { hw: "b200",  variant: "default",  quant: "fp8",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b200",  variant: "default",  quant: "fp8",  strategy: "balanced",        nodes: "single" } },
  { match: { hw: "b300",  variant: "default",  quant: "fp8",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b300",  variant: "default",  quant: "fp8",  strategy: "balanced",        nodes: "single" } },
  { match: { hw: "gb300", variant: "default",  quant: "fp8",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb300", variant: "default",  quant: "fp8",  strategy: "balanced",        nodes: "single" } },
  { match: { hw: "gb200", variant: "default",  quant: "fp8",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb200", variant: "default",  quant: "fp8",  strategy: "balanced",        nodes: "single" } },
];
