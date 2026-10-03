// Hy4-Preview cookbook config. Consumed by _deployment.jsx + _playground.jsx;
// see _deployment.jsx header for the field contract.
//
// Sizing (drives the TP/nodes choices below): 770B total / 49B active MoE.
// MXFP8 ≈ 760GB → TP4 on B300/GB300 (288GB), TP8 on B200; the MXFP8 kernel
// path requires SM100+, and H200 (SM90) was tested and cannot serve it — H200
// gets BF16 cells only.

export const config = {
  modelName: "Hy4-Preview",

  supportedHardware: ["h200", "b200", "b300", "gb300"],

  variants: [
    { id: "default", label: "Default" },
  ],
  quantizations: [
    { id: "bf16",  label: "BF16"  },
    { id: "mxfp8", label: "MXFP8" },
  ],
  strategies: [
    { id: "low-latency",     label: "Low-Latency"     },
    { id: "high-throughput", label: "High-Throughput" },
  ],
  nodesOptions: [
    { id: "single",  label: "Single Node" },
    { id: "multi-2", label: "Multi-Nodes" },
  ],

  modelNames: {
    "default|bf16":  "tencent/Hy4-preview",
    "default|mxfp8": "tencent/Hy4-preview-FP8",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",        default: "0.0.0.0"        },
    PORT:      { target: "command", label: "Bind port",        default: "30000"          },
    NODE0_IP:  { target: "command", label: "Head node IP",     default: "<node0-ip>"     },
    NODE_RANK: { target: "command", label: "This node rank",   default: "<node-rank>"    },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl",    label: "Server host",      default: "localhost"      },
    CURL_PORT: { target: "curl",    label: "Server port",      default: "30000"          },
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
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --flush-cache`,
    // GSM8K harness — keep these exact settings for comparability across runs.
    accuracy: {
      gsm8k_pct:
`# To install sgl-eval: pip install sgl-eval
sgl-eval run gsm8k \\
  --base-url http://{{CURL_HOST}}:{{CURL_PORT}}/v1 \\
  --model {{MODEL_NAME}} \\
  --num-examples 1319 --num-threads 64 \\
  --max-tokens 4096 --temperature 0 --top-p 0.95 --seed 0`,
    },
    numPromptsByConc: { 1: 32, 16: 32, 64: 128, 256: 512, 1024: 2048 },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  multiNodeHints: {
    // GB300 NVL/MNNVL fabric may need manual NIC configuration; NVSHMEM lines
    // only matter if you switch the MoE a2a backend to DeepEP in the Playground.
    gb300: [
      "The following env vars may be needed depending on your cluster:",
      "  GLOO_SOCKET_IFNAME=<your-nic>",
      "  NVSHMEM_ENABLE_NIC_PE_MAPPING=1",
      "  NVSHMEM_HCA_LIST=<your-hca-list>",
    ],
  },

  dockerImages: {
    // The hy4-preview image bundles the HYV4 model code, the suffix-aware `hunyuan` parsers, and the NEXTN MTP runtime.
    h200:  "lmsysorg/sglang:hy4-preview",
    b200:  "lmsysorg/sglang:hy4-preview",
    b300:  "lmsysorg/sglang:hy4-preview",
    gb300: "lmsysorg/sglang:hy4-preview",
  },

  github: {
    cookbookModel: "tencent/Hy4-preview",
  },

  playgroundFeatures: {

    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [
          null,
          { value: 4,
            disable: [
              { when: { quant: ["bf16"] },
                reason: "TP=4 cannot hold the ~1.5TB BF16 weights (~380GB/rank)." },
              { when: { hw: ["h200", "b200"] },
                reason: "TP=4 MXFP8 needs ~190GB/rank — requires 288GB GPUs (B300/GB300)." },
            ] },
          { value: 8,
            disable: [
              { when: { hw: ["h200", "b200"], quant: ["bf16"] },
                reason: "TP=8 BF16 needs ~190GB/rank — exceeds H200/B200 VRAM; use TP=16 across 2 nodes." },
              { when: { hw: ["gb300"], nodes: ["single"] },
                reason: "GB300 hosts carry 4 GPUs — TP=8 needs Multi-Nodes (2×4)." },
            ] },
          { value: 16,
            disable: [
              { when: { nodes: ["single"] },
                reason: "TP=16 requires 16 ranks — switch the Deploy panel's Nodes to Multi-Nodes first." },
              { when: { hw: ["gb300"] },
                reason: "GB300 hosts carry 4 GPUs — 2 nodes provide only 8 ranks." },
            ] },
        ]},
        { id: "dpAttn", label: "DP-Attention",
          values: [
            null,
            false,
            4,
            { value: 8,
              disable: [
                { when: { effTp: [4] },
                  reason: "DP-Attention=8 needs TP ≥ 8 (TP must be divisible by the DP degree) — raise TP in this card first." },
                { when: { hw: ["gb300"], nodes: ["single"] },
                  reason: "GB300 hosts carry 4 GPUs — 8 attention ranks need Multi-Nodes (2×4)." },
              ] },
            { value: 16,
              disable: [
                { when: { effTp: [4, 8] },
                  reason: "DP-Attention=16 needs TP=16 — raise TP in this card first." },
                { when: { nodes: ["single"] },
                  reason: "DP-Attention=16 requires 16 ranks — switch the Deploy panel's Nodes to Multi-Nodes first." },
                { when: { hw: ["gb300"] },
                  reason: "GB300 hosts carry 4 GPUs — 2 nodes provide only 8 ranks." },
              ] },
          ],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    moe: {
      backend: {
        options: [
          { id: null,     label: "Inherited" },
          { id: "deepep", label: "DeepEP (EP = TP)", flags: ["--moe-a2a-backend deepep"] },
        ],
      },
    },

    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser auto" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser auto" },
      ],
    },

    // No NGRAM option — untested against the DSA sparse-attention backend.
    speculative: {
      options: [
        { id: "current",  label: "Inherited from base" },
        { id: "off",      label: "Off (greedy)" },
        { id: "nextn-34", label: "MTP / NextN 3-1-4",
          flags: ["--speculative-algorithm NEXTN", "--speculative-num-steps 3",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4"] },
      ],
    },

    pdDisagg: {
      modes: [
        { id: "off",     label: "Off" },
        { id: "prefill", label: "Prefill role" },
        { id: "decode",  label: "Decode role" },
      ],
      transferBackends: [
        { id: "mooncake", label: "Mooncake",
          env: [
            "NCCL_MNNVL_ENABLE=1",
            "NCCL_CUMEM_ENABLE=1",
            "SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True",
            "MC_FORCE_MNNVL=1",
          ],
          envWhen: { hw: ["gb300"] } },
        { id: "nixl",     label: "NiXL" },
      ],
      // `auto` is a sentinel (emits no --disaggregation-ib-device flag).
      ibDevices: [{ id: "auto", label: "Auto" }, "mlx5_0", "mlx5_7"],
      // Router fronting the prefill + decode roles; substitute <prefill-host>/<decode-host>.
      router: {
        port: 8000,
        command:
`python3 -m sglang_router.launch_router \\
  --pd-disaggregation \\
  --prefill http://<prefill-host>:{{PREFILL_PORT}} \\
  --decode http://<decode-host>:{{DECODE_PORT}} \\
  --host 0.0.0.0 --port {{ROUTER_PORT}} \\
  --disable-circuit-breaker \\
  --health-check-interval-secs 999999`,
      },
    },

    hicache: {
      backends: [
        { id: null,       label: "Auto" },
        { id: "file",     label: "File" },
        { id: "mooncake", label: "Mooncake" },
        { id: "hf3fs",    label: "HF3FS" },
        { id: "nixl",     label: "NiXL" },
      ],
      writePolicies: [
        { id: "auto",                    label: "Auto" },
        { id: "write_through",           label: "Write-through" },
        { id: "write_back",              label: "Write-back" },
        { id: "write_through_selective", label: "Write-through (selective)" },
      ],
    },

    hisparse: {
      requiredFlags: ["--disable-radix-cache"],
      config: { top_k: 2048, device_buffer_size: 6144 },
      hostRatios: [
        { id: 5,  label: "5 (~1TB host)" },
        { id: 10, label: "10 (~2TB host)" },
      ],
      defaultHostRatio: 10,
    },
  },

  cells: [
    //
    // The runtime derives the rest from the checkpoint and model defaults:
    // quantization comes from the ModelOpt hf_quant_config, the DSA
    // sparse-attention backend is auto-selected for HYV4, and decode
    // CUDA-graph capture is on by default.

    // ====================================================================
    // B300 (288GB) × MXFP8 — TP4 single node (~190GB weights/rank).
    // ====================================================================
    {
      match: { hw: "b300", variant: "default", quant: "mxfp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend deep_gemm",
        "--fp8-gemm-backend deep_gemm",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "mxfp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend deep_gemm",
        "--fp8-gemm-backend deep_gemm",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B300 (288GB) × BF16 — TP8 single node (~190GB weights/rank on an
    // 8-GPU node; same per-rank footprint as MXFP8 TP4).
    // ====================================================================
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // H200 (141GB) × BF16 — TP16 across 2x8 (~95GB weights/rank; the
    // ~25GB/rank KV pool ≈ 260K tokens — size --context-length to it).
    // No H200 MXFP8 cells: tested — SM90 cannot serve the MXFP8 checkpoint
    // (the kernel path requires SM100+).
    // ====================================================================
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "multi-2" },
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" },
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B200 (192GB) × MXFP8 — TP8 single node (~95GB weights/rank).
    // ====================================================================
    {
      match: { hw: "b200", variant: "default", quant: "mxfp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend deep_gemm",
        "--fp8-gemm-backend deep_gemm",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "mxfp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--moe-runner-backend deep_gemm",
        "--fp8-gemm-backend deep_gemm",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // B200 (192GB) × BF16 — TP16 across 2x8 (~95GB weights/rank; 8x192GB
    // cannot hold the ~1.5TB weights single-node).
    // ====================================================================
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "multi-2" },
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" },
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 16",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // GB300 (288GB, 4-GPU hosts, sm_103 + aarch64) × MXFP8 — TP4 single
    // node, same per-rank footprint as B300.
    // ====================================================================
    {
      match: { hw: "gb300", variant: "default", quant: "mxfp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend deep_gemm",
        "--fp8-gemm-backend deep_gemm",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "mxfp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--moe-runner-backend deep_gemm",
        "--fp8-gemm-backend deep_gemm",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ====================================================================
    // GB300 (288GB, 4-GPU hosts) × BF16 — TP8 across 2x4 nodes (~190GB
    // weights/rank; a single 4-GPU host cannot hold the ~1.5TB weights).
    // NCCL MNNVL env follows the GB-platform multi-node convention.
    // ====================================================================
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "multi-2" },
      verificationStatus: "in-progress",
      env: ["NCCL_MNNVL_ENABLE=1", "NCCL_CUMEM_ENABLE=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" },
      verificationStatus: "in-progress",
      env: ["NCCL_MNNVL_ENABLE=1", "NCCL_CUMEM_ENABLE=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--reasoning-parser auto",
        "--tool-call-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
