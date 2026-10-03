// Laguna-M.1 (poolside) — config-driven cookbook page. Consumed by the
// shared _deployment.jsx + _playground.jsx engines (no model code there).
//
// Build: M.1 needs SGLang (softplus per-element output gating, MERGED) AND
// (global-attention SWA fix — M.1 is sliding_window=0 / all-global; without
// it M.1 crashes ~1s into any concurrent batch with AssertionError: ...
// swa_lock_ref=0). Both are merged on main (verified on a 3f668733 build).
// The shipped recipe carries NO SWA workaround flag, but the pinned build
// MUST contain BOTH; the #28400-merge wheel 0.5.14.dev20260618+g343aeeef39 is
// #28400-ONLY and crashes under load.

export const config = {
  modelName: "Laguna-M.1",

  supportedHardware: ["h200", "b200", "b300", "gb200", "gb300"],

  variants: [
    { id: "default", label: "Default" },
  ],

  quantizations: [
    { id: "bf16",  label: "BF16"  },
    { id: "fp8",   label: "FP8"   },
    { id: "nvfp4", label: "NVFP4" },
  ],

  // Single balanced operating point (maintainer decision — no LL/HT split).
  strategies: [
    { id: "balanced", label: "Balanced" },
  ],

  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "default|bf16":  "poolside/Laguna-M.1",
    "default|fp8":   "poolside/Laguna-M.1-FP8",
    "default|nvfp4": "poolside/Laguna-M.1-NVFP4",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",         default: "0.0.0.0"         },
    PORT:      { target: "command", label: "Bind port",         default: "30000"           },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl",    label: "Server host",       default: "localhost"       },
    CURL_PORT: { target: "curl",    label: "Server port",       default: "30000"           },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,

  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}}`,
    // GSM8K is the required accuracy sanity on every verified cell
    // (cookbook_guide §3), via sgl-eval.
    accuracy: {
      gsm8k_pct:
`# pip install sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --num-threads 128`,
    },
    numPromptsByConc: { 1: 8, 16: 32, 64: 128, 128: 256, 256: 512, 1024: 2048, 4096: 4096 },
  },

  // Hardware-independent accuracy default (null = no variant-wide default; real numbers are per-cell
  // in laguna-m1-benchmarks.jsx).
  defaultAccuracy: {
    default: { gsm8k_pct: null },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  dockerImages: {
    h200:  "lmsysorg/sglang:latest",
    b200:  "lmsysorg/sglang:latest",
    b300:  "lmsysorg/sglang:latest",
    gb200: "lmsysorg/sglang:latest",
    gb300: "lmsysorg/sglang:latest",
  },

  github: {
    cookbookModel: "poolside/Laguna-M.1",
  },

  playgroundFeatures: {

    // M.1 is global-attention (no SWA); expose TP + DP-Attention here.
    attention: {
      knobs: [
        { id: "tp",     label: "TP",           values: [null, 1, 2, 4, 8] },
        { id: "dpAttn", label: "DP-Attention", values: [null, false, 1, 2, 4, 8],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    // 256-expert top-16 MoE — EP degree only.
    moe: {
      ep: { label: "EP", values: [null, 1, 2, 4, 8] },
    },

    // Reasoning + tool-call parsers (poolside_v1, same family as Laguna-XS.2). ALSO baked into
    // every Deploy cell below (the maintainer's baseline carries them).
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser poolside_v1" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser poolside_v1" },
      ],
    },

    // HiCache (hierarchical KV cache).
    hicache: {
      writePolicies: [
        { id: "auto",          label: "Auto" },
        { id: "write_through", label: "Write-through" },
        { id: "write_back",    label: "Write-back" },
      ],
    },

    // Prefill-Decode disaggregation (§3.3). M.1 is standard-KV (global
    // attention, no sparse index buffer), so it disaggregates with the
    // --disaggregation-* flags — no model-specific backend pinning.
    pdDisagg: {
      modes: [
        { id: "off",     label: "Off" },
        { id: "prefill", label: "Prefill role" },
        { id: "decode",  label: "Decode role" },
      ],
      transferBackends: [
        // mooncake (recommended): honors --disaggregation-ib-device, no transfer cold-start.
        // The NCCL/MNNVL env is only needed on NVLink-multinode Grace-Blackwell (GB200/GB300).
        { id: "mooncake", label: "Mooncake",
          env: [
            "NCCL_MNNVL_ENABLE=1",
            "NCCL_CUMEM_ENABLE=1",
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True",
            "MC_FORCE_MNNVL=1",
          ],
          envWhen: { hw: ["gb200", "gb300"] } },
        // NiXL ignores --disaggregation-ib-device; its UCX backend needs the
        // NIC pinned via UCX_NET_DEVICES or every KV transfer hangs.
        { id: "nixl",     label: "NiXL",
          env: ["UCX_NET_DEVICES=mlx5_0:1"],
          envWhen: { hw: ["h200", "b200", "b300"] } },
      ],
      ibDevices: [{ id: "auto", label: "Auto" }, "mlx5_0", "mlx5_7"],
      router: {
        port: 8000,
        command:
`python3 -m sglang_router.launch_router \\
  --pd-disaggregation \\
  --prefill http://<prefill-host>:{{PREFILL_PORT}} \\
  --decode http://<decode-host>:{{DECODE_PORT}} \\
  --policy round_robin \\
  --host 0.0.0.0 --port {{ROUTER_PORT}}`,
      },
    },
  },

  // One Balanced cell per valid (hw × quant): H200×{BF16,FP8}; each Blackwell×{BF16,FP8,NVFP4}. Baseline
  // recipe (parsers poolside_v1 + --trust-remote-code) on every cell. Absent verified = yellow/unverified
  // badge.
  cells: [
    // ===== NVIDIA Hopper (H200) — BF16 / FP8 (NVFP4 is Blackwell-only) =====
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ===== NVIDIA Blackwell B200 (8-GPU HGX) — BF16 / FP8 / NVFP4 =====
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--fp8-gemm-backend triton",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ===== NVIDIA Blackwell B300 (8-GPU HGX) — BF16 / FP8 / NVFP4 (UNVERIFIED) =====
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--fp8-gemm-backend triton",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 8",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ===== NVIDIA Grace-Blackwell GB200 (4-GPU single node) — BF16 / FP8 / NVFP4 (UNVERIFIED) =====
    {
      match: { hw: "gb200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--fp8-gemm-backend triton",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // ===== NVIDIA Grace-Blackwell GB300 (4-GPU single node) — BF16 / FP8 / NVFP4 (UNVERIFIED) =====
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--fp8-gemm-backend triton",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser poolside_v1",
        "--tool-call-parser poolside_v1",
        "--tp 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
