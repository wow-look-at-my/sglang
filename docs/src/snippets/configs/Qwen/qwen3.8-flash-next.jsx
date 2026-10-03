// Single `export const config` literal — no spreads/calls/IIFE (Mintlify
// re-evals at hydration). Cells are denormalized: no
// `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals —
// engine injects them.
//
// Qwen3.8-Flash-Next: 176B total params (51B of that is the N-gram embedding
// table) with 6B active per token. Multimodal (text + image in, text out).

export const config = {
  modelName: "Qwen3.8-Flash-Next",

  supportedHardware: ["h200", "b200", "b300", "gb300", "rtx6000", "dgx-spark", "mi350x", "mi355x"],

  hardware: [
    { id: "rtx6000", label: "RTX PRO 6000", vram: "96GB", vendor: "blackwell" },
  ],

  variants: [
    { id: "default", label: "Default" },
  ],
  // Checkpoint precisions.
  quantizations: [
    { id: "bf16",       label: "BF16"         },
    { id: "fp8",        label: "FP8"          },
    { id: "nvfp4",      label: "NVFP4 (RDXA)" },
    { id: "nvfp4-nvda", label: "NVFP4 (NVDA)" },
  ],
  strategies: [
    { id: "low-latency",     label: "Low Latency"     },
    { id: "balanced",        label: "Balanced"        },
    { id: "high-throughput", label: "High Throughput" },
  ],
  // `multi-N` id carries the node count for `--nnodes N`; only the DGX Spark
  // NVFP4 cells use it.
  nodesOptions: [
    { id: "single",  label: "Single Node" },
    { id: "multi-2", label: "Multi-Node"  },
  ],

  // Orthogonal knobs — layered onto the matched cell, never part of the cell
  // key (see overlayDims in _deployment.jsx).
  overlayDims: [
    {
      id: "pleOffload",
      title: "PLE Offload",
      // Offloads the 51B N-gram embedding table to CPU pinned memory and prefetches it on a side CUDA stream.
      showWhen: (sel) => !["mi350x", "mi355x"].includes(sel.hw),
      default: "auto",
      options: [
        { id: "auto", label: "Auto",
          disabled: (sel) => sel.hw === "dgx-spark" || sel.hw === "rtx6000",
          disableReason: (sel) => sel.hw === "rtx6000"
            ? "RTX PRO 6000 (96 GB) only fits this checkpoint with the 47.7 GiB FP8 N-gram table in pinned host RAM; the verified cells pass --ple-offload-embedding explicitly, so On is the only pick."
            : "DGX Spark is unified memory: PLE offload to RAM frees nothing (the pinned table shares the 128 GB pool with the weights). The verified settings are Off for the two-node cells and On (NVMe file) for a single Spark.",
          hints: ["PLE Offload: auto-enabled for BF16 on CUDA, off otherwise"] },
        { id: "on",   label: "On",
          disabled: (sel) => sel.hw === "dgx-spark",
          disableReason: "DGX Spark is unified memory: PLE offload to RAM frees nothing (the pinned table shares the 128 GB pool with the weights). The verified settings are Off for the two-node cells and On (NVMe file) for a single Spark.",
          flags: ["--ple-offload-embedding"] },
        { id: "off",  label: "Off",
          disabled: (sel) => sel.hw === "rtx6000" || (sel.hw === "dgx-spark" && sel.nodes === "single"),
          disableReason: (sel) => sel.hw === "dgx-spark"
            ? "A single DGX Spark cannot hold the 126 GiB checkpoint in its 128 GB of unified memory; the verified single-Spark cells keep the 47.7 GiB FP8 N-gram table in a file on the local NVMe (On (NVMe file))."
            : "RTX PRO 6000 (96 GB) cannot hold the 47.7 GiB FP8 N-gram table alongside the other 78 GiB of the checkpoint; the table must be offloaded to pinned host RAM (On).",
          flags: ["--no-ple-offload-embedding"] },
        // Requires the device attribute
        // cudaDevAttrPageableMemoryAccessUsesHostPageTables, which GB10 has; hidden on
        // other hardware. Verified only single-node — the 2-node cells keep the table
        // GPU-resident instead.
        { id: "file", label: "On (NVMe file)",
          showWhen: (sel) => sel.hw === "dgx-spark",
          disabled: (sel) => sel.nodes !== "single",
          disableReason: "The file-backed table is verified for the single-Spark cells; the 2-node cells shard the table across both GPUs instead (Off).",
          flags: ["--ple-offload-embedding", "--ple-offload-backend file"],
          hints: [
            "PLE table -> sparse 47.7 GiB file under $SGLANG_CACHE_DIR/ple/<model> (put it on local NVMe; --ple-offload-dir relocates it).",
            "Delete the previous table file before each boot until the rewrite is fixed upstream: rewriting a populated file runs at ~17 MB/s (~55 min), a fresh sparse file at GB/s (~8 min).",
          ] },
      ],
    },
  ],

  modelNames: {
    "default|bf16":  "Qwen/Qwen3.8-Flash-Next",
    // Separate repos, not revisions of the BF16 one.
    "default|fp8":   "Qwen/Qwen3.8-Flash-Next-FP8",
    "default|nvfp4": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
    "default|nvfp4-nvda": "nvidia/Qwen3.8-Flash-Next-NVFP4",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",         default: "0.0.0.0"         },
    PORT:      { target: "command", label: "Bind port",         default: "30000"           },
    NODE0_IP:  { target: "command", label: "Head node IP",      default: "<node0-ip>"      },
    NODE_RANK: { target: "command", label: "This node rank",    default: "<node-rank>"     },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl",    label: "Server host",       default: "localhost"       },
    CURL_PORT: { target: "curl",    label: "Server port",       default: "30000"           },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,

  // The "⚡ Reproduce" modal's benchmark command. --random-range-ratio pins ISL
  // exactly rather than drawing a range, so runs stay comparable.
  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang-oai \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} --random-range-ratio 1 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --request-rate inf \\
  --flush-cache`,
    numPromptsByConc: { 1: 8, 16: 32, 64: 128, 256: 512, 1024: 2048, 4096: 4096 },
  },

  accuracyLabels: [
    ["gsm8k_pct",    "GSM8K",    "%"],
    ["aime26_pct",   "AIME26",   "%"],
    ["mmmu_pro_pct", "MMMU-Pro", "%"],
  ],

  // Launch images — this is a day-0 model with no release cut, so both tags are
  // purpose-built rather than a version. The ROCm build targets CDNA4 (gfx950)
  // and is not interchangeable with the CUDA one.
  // Prepended as `# ...` comments above multi-node commands.
  multiNodeHints: {
    "dgx-spark": [
      "Run the same command on both Sparks: rank 1 first, then rank 0 (node 0 = --dist-init-addr host).",
      "Point the rendezvous and NCCL at the ConnectX-7 link, not the management NIC:",
      "  NCCL_SOCKET_IFNAME=<200GbE-nic>  GLOO_SOCKET_IFNAME=<200GbE-nic>",
      "Cross-node decode CUDA graphs verified with the NCCL these images load (2.29.7 in dev-qwen38-next-local, 2.30.7 in qwen38flashnext);",
      "confirm with the startup log line 'sglang is using nccl=='.",
    ],
  },

  dockerImages: {
    h200:   "lmsysorg/sglang:qwen38flashnext",
    // DGX Spark and RTX PRO multiple recipes need the qwen4-main-squashed build.
    "dgx-spark": "lmsysorg/sglang:dev-qwen38-next-local",
    rtx6000: "lmsysorg/sglang:dev-qwen38-next-local",
    b200:   "lmsysorg/sglang:qwen38flashnext",
    "b200|nvfp4-nvda": "lmsysorg/sglang:latest",
    b300:   "lmsysorg/sglang:qwen38flashnext",
    "b300|nvfp4-nvda": "lmsysorg/sglang:latest",
    gb300:  "lmsysorg/sglang:qwen38flashnext",
    "gb300|nvfp4-nvda": "lmsysorg/sglang:latest",
    mi350x: "lmsysorg/sglang-rocm:qwen38flashnext",
    mi355x: "lmsysorg/sglang-rocm:qwen38flashnext",
  },

  github: {
    cookbookModel: "Qwen/Qwen3.8-Flash-Next",
  },

  playgroundFeatures: {

    // ----- Card: "Attention Parallelism" ----- TP only.
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 1, 2, 4, 8] },
      ],
    },

    // ----- Card: "MoE Parallelism" ----- EP degree only.
    moe: {
      ep: { label: "EP", values: [null, 1, 2, 4, 8] },
    },

    // ----- Card: "Parsers" ----- Both chips emit `auto`.
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser auto" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser auto" },
      ],
    },

    // ----- Card: "Speculative Decoding" ----- The in-checkpoint MTP head,
    // trained with multiple steps.
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "mtp",     label: "NEXTN / MTP",
          flags: ["--speculative-algorithm NEXTN", "--speculative-num-steps 3",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4"] },
      ],
    },
  },

  // Every cell below is a verified recipe. Ordering: the first cell seeds the
  // Deploy panel's default selection.
  //
  // Within a quantization the NVIDIA cells are identical across
  // H200/B200/B300/GB300, and `--linear-attn-{prefill,decode}-backend flashinfer` is
  // pinned explicitly rather than left to the GDN default, which differs by GPU
  // generation (Triton on SM90, and the flashinfer decode default is gated on
  // `--mamba-ssm-dtype bfloat16`). Pinning both makes one recipe portable.
  cells: [
    // Without an explicit --max-running-requests a speculative run takes
    // the speculative hook's default rather than a memory-derived ceiling.
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--linear-attn-verify-backend triton",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--max-running-requests 96",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // High throughput: speculation off, EP4 across the same ranks.
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--max-running-requests 96",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--max-running-requests 96",
        "--reasoning-parser auto",
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
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--max-running-requests 96",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // Unlike the BF16/NVFP4 low-latency cells these keep EP4 and pin no
    // --max-running-requests, so the panel's speculative hint applies.
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--linear-attn-verify-backend triton",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--ep 4",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ==== NVFP4 (Blackwell only) ==== Single-GPU: the FP4 weights fit one card,
    // so these run TP1 rather than the TP4 the BF16/FP8 cells use. That leaves
    // one rank, so there is no EP to spend and both tiers differ only by the MTP
    // head — `ep_size * moe_dp_size <= tp_size` would reject an --ep here.
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    {
      match: { hw: "b200", variant: "default", quant: "nvfp4-nvda", strategy: "low-latency", nodes: "single" },
      verified: false,
      warn: "Requires SGLang v0.5.20 or later for this NVIDIA ModelOpt MIXED_PRECISION export. Quantization and MoE backends are selected automatically from the checkpoint.",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    {
      match: { hw: "b200", variant: "default", quant: "nvfp4-nvda", strategy: "high-throughput", nodes: "single" },
      verified: false,
      warn: "Requires SGLang v0.5.20 or later for this NVIDIA ModelOpt MIXED_PRECISION export. Quantization and MoE backends are selected automatically from the checkpoint.",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4-nvda", strategy: "low-latency", nodes: "single" },
      verified: false,
      warn: "Requires SGLang v0.5.20 or later for this NVIDIA ModelOpt MIXED_PRECISION export. Quantization and MoE backends are selected automatically from the checkpoint.",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", variant: "default", quant: "nvfp4-nvda", strategy: "high-throughput", nodes: "single" },
      verified: false,
      warn: "Requires SGLang v0.5.20 or later for this NVIDIA ModelOpt MIXED_PRECISION export. Quantization and MoE backends are selected automatically from the checkpoint.",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4-nvda", strategy: "low-latency", nodes: "single" },
      verified: false,
      warn: "Requires SGLang v0.5.20 or later for this NVIDIA ModelOpt MIXED_PRECISION export. Quantization and MoE backends are selected automatically from the checkpoint.",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4-nvda", strategy: "high-throughput", nodes: "single" },
      verified: false,
      warn: "Requires SGLang v0.5.20 or later for this NVIDIA ModelOpt MIXED_PRECISION export. Quantization and MoE backends are selected automatically from the checkpoint.",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--linear-attn-prefill-backend flashinfer",
        "--linear-attn-decode-backend flashinfer",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser auto",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "multi-2" },
      verified: true,
      warn: "2x DGX Spark only (GB10 pair, TP=2 over ConnectX-7); in Docker mode use the lmsysorg/sglang:dev-qwen38-next-local image, the qwen4-main-squashed build the Spark rows are generated for. Memory headroom at --mem-fraction-static 0.85 is ~8-12 GiB per node; keep a host memory watchdog for long-context runs. See [DGX Spark notes](#spark-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--quantization modelopt_fp4",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--page-size 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--max-running-requests 24",
        "--max-mamba-cache-size 120",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "multi-2" },
      verified: true,
      warn: "2x DGX Spark only (GB10 pair, TP=2 over ConnectX-7); in Docker mode use the lmsysorg/sglang:dev-qwen38-next-local image, the qwen4-main-squashed build the Spark rows are generated for. At 96 concurrent requests the KV pool is ~1.07M tokens (~11k per request when full); for long-context workloads drop --max-running-requests and --max-mamba-cache-size, and SGLang sizes the state pool from --context-length and gives the rest to the KV pool. See [DGX Spark notes](#spark-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--quantization modelopt_fp4",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--page-size 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--max-running-requests 96",
        "--max-mamba-cache-size 384",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    {
      match: { hw: "rtx6000", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      warn: "Single RTX PRO 6000 (96 GB). Use the lmsysorg/sglang:dev-qwen38-next-local image, the build this cell is verified on. The FP8 N-gram table lives in pinned host RAM: keep >= 64 GB of host memory free and run Docker with --ulimit memlock=-1. The KV pool is ~78k tokens (~4.9k per request at 16 concurrent); for long-context workloads drop --max-running-requests and --max-mamba-cache-size, and SGLang sizes the state pool from --context-length and gives the rest to the KV pool. See [RTX PRO 6000 notes](#rtx6000-note).",
      env: ["PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--quantization modelopt_fp4",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--moe-runner-backend flashinfer_cutlass",
        "--page-size 64",
        "--mamba-track-interval 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--max-running-requests 16",
        "--max-mamba-cache-size 48",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.96",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "rtx6000", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      warn: "Single RTX PRO 6000 (96 GB). Use the lmsysorg/sglang:dev-qwen38-next-local image, the build this cell is verified on. The FP8 N-gram table lives in pinned host RAM: keep >= 64 GB of host memory free and run Docker with --ulimit memlock=-1. At 64 concurrent requests the KV pool is ~98k tokens (~1.5k per request when full); for long-context workloads drop --max-running-requests and --max-mamba-cache-size, and SGLang sizes the state pool from --context-length and gives the rest to the KV pool. See [RTX PRO 6000 notes](#rtx6000-note).",
      env: ["PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--quantization modelopt_fp4",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--moe-runner-backend flashinfer_cutlass",
        "--page-size 64",
        "--mamba-track-interval 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--max-running-requests 64",
        "--max-mamba-cache-size 192",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.93",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // Full-set GSM8K in the benchmarks config.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
      verified: true,
      warn: "Single DGX Spark (GB10, 128 GB unified). The N-gram table is a 47.7 GiB sparse file on the local NVMe (PLE Offload = On (NVMe file)); keep ~50 GB free there and mount that directory into the container. Boot writes the whole table each time: delete the previous file first (a populated file rewrites at ~17 MB/s). Concurrency is memory-bound at 8 with MTP. See [DGX Spark notes](#spark-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--quantization modelopt_fp4",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--page-size 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--max-running-requests 8",
        "--max-mamba-cache-size 40",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
      verified: true,
      warn: "Single DGX Spark (GB10, 128 GB unified). The N-gram table is a 47.7 GiB sparse file on the local NVMe (PLE Offload = On (NVMe file)); keep ~50 GB free there and mount that directory into the container. Boot writes the whole table each time: delete the previous file first (a populated file rewrites at ~17 MB/s). At 24 concurrent requests the KV pool is ~286k tokens; for long-context workloads drop --max-running-requests and --max-mamba-cache-size, and SGLang sizes the state pool from --context-length and gives the rest to the KV pool. See [DGX Spark notes](#spark-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--quantization modelopt_fp4",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--page-size 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--max-running-requests 24",
        "--max-mamba-cache-size 96",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ==== NVFP4 (NVDA) on 2x DGX Spark — nvidia/Qwen3.8-Flash-Next-NVFP4
    // ==== ModelOpt MIXED_PRECISION export: NVFP4 routed experts, FP8 N-gram
    // table, FP8_BLOCK_SCALES (128-wide) MTP experts.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4-nvda", strategy: "low-latency", nodes: "multi-2" },
      verified: true,
      warn: "2x DGX Spark only (GB10 pair, TP=2 over ConnectX-7). Verified on the qwen4-main-squashed branch (the Python install path above). In Docker mode use the lmsysorg/sglang:dev-qwen38-next-local image (the qwen4-main-squashed build); the qwen38flashnext image predates the MIXED_PRECISION loader ([sgl-project/sglang#38121](https://github.com/sgl-project/sglang/pull/38121)) and cannot load this export. The MTP draft is read from the RadixArk export (same head, BF16) because this export's fp8 block-scaled MTP experts cannot be split across two ranks. See [DGX Spark notes](#spark-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--moe-runner-backend flashinfer_cutlass",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--page-size 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--speculative-draft-model-path RadixArk/Qwen3.8-Flash-Next-NVFP4",
        "--speculative-draft-model-quantization modelopt_fp4",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--max-running-requests 24",
        "--max-mamba-cache-size 120",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4-nvda", strategy: "high-throughput", nodes: "multi-2" },
      verified: true,
      warn: "2x DGX Spark only (GB10 pair, TP=2 over ConnectX-7). Verified on the qwen4-main-squashed branch (the Python install path above). In Docker mode use the lmsysorg/sglang:dev-qwen38-next-local image (the qwen4-main-squashed build); the qwen38flashnext image predates the MIXED_PRECISION loader ([sgl-project/sglang#38121](https://github.com/sgl-project/sglang/pull/38121)) and cannot load this export. At 96 concurrent requests the KV pool is ~1.1M tokens; for long-context workloads drop --max-running-requests and --max-mamba-cache-size, and SGLang sizes the state pool from --context-length and gives the rest to the KV pool. See [DGX Spark notes](#spark-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 2",
        "--moe-runner-backend flashinfer_cutlass",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--page-size 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--max-running-requests 96",
        "--max-mamba-cache-size 384",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // At TP=1 the in-checkpoint MTP head loads directly: its fp8
    // block-scaled experts need no sharding, so the RadixArk draft used by
    // the 2-node cell is not needed. Full-set GSM8K in the benchmarks
    // config. The smaller fp8 draft leaves a 174k-token KV pool with MTP (vs
    // 93k for the RDXA cell) and 300k without.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4-nvda", strategy: "low-latency", nodes: "single" },
      verified: true,
      warn: "Single DGX Spark (GB10, 128 GB unified). Use the lmsysorg/sglang:dev-qwen38-next-local image: this ModelOpt MIXED_PRECISION export needs the loader from [sgl-project/sglang#38121](https://github.com/sgl-project/sglang/pull/38121), which the qwen38flashnext image does not have. The N-gram table is a 47.7 GiB sparse file on the local NVMe (PLE Offload = On (NVMe file)); keep ~50 GB free there and mount that directory into the container. Boot writes the whole table each time: delete the previous file first (a populated file rewrites at ~17 MB/s). Concurrency is memory-bound at 8 with MTP. See [DGX Spark notes](#spark-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--moe-runner-backend flashinfer_cutlass",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--page-size 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--max-running-requests 8",
        "--max-mamba-cache-size 40",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4-nvda", strategy: "high-throughput", nodes: "single" },
      verified: true,
      warn: "Single DGX Spark (GB10, 128 GB unified). Use the lmsysorg/sglang:dev-qwen38-next-local image: this ModelOpt MIXED_PRECISION export needs the loader from [sgl-project/sglang#38121](https://github.com/sgl-project/sglang/pull/38121), which the qwen38flashnext image does not have. The N-gram table is a 47.7 GiB sparse file on the local NVMe (PLE Offload = On (NVMe file)); keep ~50 GB free there and mount that directory into the container. Boot writes the whole table each time: delete the previous file first (a populated file rewrites at ~17 MB/s). At 24 concurrent requests the KV pool is ~300k tokens; for long-context workloads drop --max-running-requests and --max-mamba-cache-size, and SGLang sizes the state pool from --context-length and gives the rest to the KV pool. See [DGX Spark notes](#spark-note).",
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--moe-runner-backend flashinfer_cutlass",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--page-size 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--max-running-requests 24",
        "--max-mamba-cache-size 96",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // Same shape, pools, flags and headroom as the RDXA cells above, with
    // differences:
    //   - no `--quantization` (the checkpoint resolves to modelopt_mixed);
    //   - low latency keeps the in-checkpoint MTP head. At TP=1 its fp8
    //     block-scaled experts need no sharding, and #38121 runs them on triton
    //     under the flashinfer_cutlass pin. The RadixArk BF16 draft
    //     (--speculative-draft-model-path) measured the same on this card
    //     (accept 3.33 vs 3.31, TPOT 18.5 vs 19.1 ms at 16), so the
    //     single-checkpoint command stays.
    {
      match: { hw: "rtx6000", variant: "default", quant: "nvfp4-nvda", strategy: "low-latency", nodes: "single" },
      verified: true,
      warn: "Single RTX PRO 6000 (96 GB). Use the lmsysorg/sglang:dev-qwen38-next-local image: this ModelOpt MIXED_PRECISION export needs the loader from [sgl-project/sglang#38121](https://github.com/sgl-project/sglang/pull/38121), which the qwen38flashnext image does not have. The FP8 N-gram table lives in pinned host RAM: keep >= 64 GB of host memory free and run Docker with --ulimit memlock=-1. The KV pool is ~170k tokens (~10k per request at 16 concurrent). See [RTX PRO 6000 notes](#rtx6000-note).",
      env: ["PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--moe-runner-backend flashinfer_cutlass",
        "--page-size 64",
        "--mamba-track-interval 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--speculative-algorithm NEXTN",
        "--speculative-num-steps 3",
        "--speculative-eagle-topk 1",
        "--speculative-num-draft-tokens 4",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--max-running-requests 16",
        "--max-mamba-cache-size 48",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.96",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "rtx6000", variant: "default", quant: "nvfp4-nvda", strategy: "high-throughput", nodes: "single" },
      verified: true,
      warn: "Single RTX PRO 6000 (96 GB). Use the lmsysorg/sglang:dev-qwen38-next-local image: this ModelOpt MIXED_PRECISION export needs the loader from [sgl-project/sglang#38121](https://github.com/sgl-project/sglang/pull/38121), which the qwen38flashnext image does not have. The FP8 N-gram table lives in pinned host RAM: keep >= 64 GB of host memory free and run Docker with --ulimit memlock=-1. At 64 concurrent requests the KV pool is ~98k tokens (~1.5k per request when full); for long-context workloads drop --max-running-requests and --max-mamba-cache-size, and SGLang sizes the state pool from --context-length and gives the rest to the KV pool. See [RTX PRO 6000 notes](#rtx6000-note).",
      env: ["PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp 1",
        "--fp4-gemm-backend flashinfer_cutlass",
        "--moe-runner-backend flashinfer_cutlass",
        "--page-size 64",
        "--mamba-track-interval 64",
        "--chunked-prefill-size 4096",
        "--context-length 262144",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--max-running-requests 64",
        "--max-mamba-cache-size 192",
        "--mamba-ssm-dtype bfloat16",
        "--reasoning-parser qwen3",
        "--mem-fraction-static 0.93",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ==== AMD CDNA4 (MI350X / MI355X) ==== One recipe, identical for BF16 and FP8
    // and for both cards (same gfx950, same 288GB, same ROCm image) — hence
    // `balanced` on all cells. This is its own shape rather than a port of the
    // NVIDIA one: TP8, the aiter attention backend with `--page-size 32`, and a
    // 16384-token prefill chunk. `--kv-cache-dtype auto` is stated rather than
    // left off so the checkpoint's own declaration is visibly what decides KV
    // precision.
    {
      match: { hw: "mi350x", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--attention-backend aiter",
        "--page-size 32",
        "--kv-cache-dtype auto",
        "--chunked-prefill-size 16384",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.9",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--trust-remote-code",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi350x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--attention-backend aiter",
        "--page-size 32",
        "--kv-cache-dtype auto",
        "--chunked-prefill-size 16384",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.9",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--trust-remote-code",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--attention-backend aiter",
        "--page-size 32",
        "--kv-cache-dtype auto",
        "--chunked-prefill-size 16384",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.9",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--trust-remote-code",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--attention-backend aiter",
        "--page-size 32",
        "--kv-cache-dtype auto",
        "--chunked-prefill-size 16384",
        "--watchdog-timeout 1200",
        "--mem-fraction-static 0.9",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--trust-remote-code",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
