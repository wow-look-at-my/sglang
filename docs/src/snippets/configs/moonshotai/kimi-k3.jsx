// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// Served today from the public sgl-project/sglang kimi-k3 branch.

export const config = {
  modelName: "Kimi-K3",

  supportedHardware: ["b300", "gb300", "b200", "gb200", "h200", "h100", "mi350x", "mi355x", "a3", "a5"],

  // ---- Cell introspection (config-internal; the engines ignore these keys)
  // ---- The overlay callbacks below are handed only the SELECTION.
  cellFor(s) {
    return (config.cells || []).find(
      (c) => c.match.hw === s.hw &&
             c.match.pdMode === s.pdMode &&
             c.match.strategy === s.strategy);
  },
  // Flags are authored as "--name value" / "--name=value" strings, matching how
  // _deployment.jsx splits them for stripPrefixes.
  flagOf(cell, name) {
    return ((cell || {}).flags || []).find((f) => f.split(/[\s=]/)[0] === name);
  },
  sizeOf(cell, name) {
    const f = config.flagOf(cell, name);
    return f ? Number(f.split(/[\s=]/)[1]) || 1 : 1;
  },
  // Both NPU recipes (A3 and A5) each ship exactly one operating point —
  // Unified PD, the Balanced strategy, DSPARK, no HiCache.
  isNpuHw(s) {
    return s.hw === "a3" || s.hw === "a5";
  },
  // Does this recipe shard the TP-replicated MLA KV?
  hasDcp(s) {
    return !!config.flagOf(config.cellFor(s), "--dcp-size");
  },
  // A pp_size == 1 speculative algorithm cannot run a pipelined recipe.
  isPipelined(s) {
    return config.sizeOf(config.cellFor(s), "--pp-size") > 1;
  },
  specCollapses(s) {
    return config.isPipelined(s) && !!(config.cellFor(s) || {}).specCollapsePp;
  },
  // The playground's chip `disable` is declarative only (no predicates), so the
  // DCP-carrying recipes are enumerated — but generated from the cells here
  // rather than typed out per platform, so it is the same single source of truth
  // as hasDcp() above. One rule per (hardware, PD role): which strategies carry
  // DCP varies along both axes — B200 reaches DCP only at High-Throughput when
  // unified, but carries it on Balanced too in the decode role — and keying on
  // hw+strategy alone (as the hand-written rules did) cannot express that without
  // over-matching one role or under-matching the other.
  get dcpStorageDisableRules() {
    const groups = new Map();
    for (const c of (config.cells || [])) {
      if (!config.flagOf(c, "--dcp-size")) continue;
      const k = `${c.match.hw}|${c.match.pdMode}`;
      if (!groups.has(k)) groups.set(k, new Set());
      groups.get(k).add(c.match.strategy);
    }
    return [...groups].map(([k, strategies]) => {
      const [hw, pdMode] = k.split("|");
      return {
        when: { hw: [hw], pdMode: [pdMode], strategy: [...strategies],
                hicache: ["l2"], spec: ["none", "dspark"] },
        reason: "This recipe runs DCP, and a storage backend (L3) under DCP is rejected at startup. Switch HiCache to L3 in the Deploy panel (that drops DCP), or stay on L1+L2.",
      };
    });
  },
  // Companion to the hand-written TP/DP-Attention rules below, which enumerate
  // the platforms whose recipes are too small for a 16-rank knob. Those cover the
  // flat recipes; this covers the other reason a knob cannot widen — the recipe
  // spends its ranks on pipeline stages instead. Only cells that never collapse
  // the pipeline are listed (a specCollapsePp cell becomes flat under DSPARK and
  // is handled by its own spec-keyed rule), so no `spec` key is needed here.
  get pipelinedKnobDisableRules() {
    const groups = new Map();
    for (const c of (config.cells || [])) {
      if (config.sizeOf(c, "--pp-size") <= 1 || c.specCollapsePp) continue;
      const k = `${c.match.hw}|${c.match.pdMode}`;
      if (!groups.has(k)) groups.set(k, new Set());
      groups.get(k).add(c.match.strategy);
    }
    return [...groups].map(([k, strategies]) => {
      const [hw, pdMode] = k.split("|");
      return {
        when: { hw: [hw], pdMode: [pdMode], strategy: [...strategies] },
        reason: "This recipe spends its ranks on pipeline stages (--pp-size > 1 with a small --tp-size), so widening the attention parallelism would need more GPUs than the deployment has. Pick a flat recipe to change TP or DP-Attention.",
      };
    });
  },
  // Fold the pipeline back into the other axes at constant world size:
  // TP8 × PP2 -> TP16, and DCP8/EP8 scale with it -> DCP16/EP16. Derived from
  // the cell rather than written out, so it stays right if a cell is re-shaped.
  specCollapsedFlags(s) {
    const cell = config.cellFor(s);
    const pp = config.sizeOf(cell, "--pp-size");
    const out = [`--tp-size ${config.sizeOf(cell, "--tp-size") * pp}`];
    // L3 rejects DCP at startup, so under L3 the DCP half is dropped, not scaled.
    if (config.flagOf(cell, "--dcp-size") && s.hicache !== "l3") {
      out.push(`--dcp-size ${config.sizeOf(cell, "--dcp-size") * pp}`);
    }
    if (config.flagOf(cell, "--ep-size")) {
      out.push(`--ep-size ${config.sizeOf(cell, "--ep-size") * pp}`);
    }
    return out;
  },

  // Single checkpoint and a single shipped quantization (MXFP4), so neither is a
  // reader-facing axis. Node count is fixed by the hardware recipe (B200 2x8,
  // H100 4x8, B300 1x8, H200 2x8 — 4x8 on Unified High-Throughput, GB200 4x4,
  // GB300 2x4, MI350X/MI355X 1x8), so it rides on the cell rather than on a
  // selector.
  matchDims: [
    {
      id: "pdMode",
      title: "PD Mode",
      options: [
        { id: "unified", label: "Unified"  },
        {
          id: "prefill",
          label: "Prefill",
          disabled: (s) => config.isNpuHw(s),
          disableReason: (s) => (config.isNpuHw(s) ? "Only Unified PD is supported on this recipe." : ""),
        },
        {
          id: "decode",
          label: "Decode",
          disabled: (s) => config.isNpuHw(s),
          disableReason: (s) => (config.isNpuHw(s) ? "Only Unified PD is supported on this recipe." : ""),
        },
      ],
    },
    {
      // Prefill nodes are sized by context length.
      id: "strategy",
      title: "Strategy",
      options: [
        {
          id: "low-latency",
          label: "Low-Latency",
          showWhen: (s) => s.pdMode !== "prefill",
          disabled: (s) => config.isNpuHw(s),
          disableReason: (s) => (config.isNpuHw(s) ? "Only the Balanced operating point is supported on this recipe." : ""),
        },
        { id: "balanced",        label: "Balanced",        showWhen: (s) => s.pdMode !== "prefill" },
        {
          id: "high-throughput",
          label: "High-Throughput",
          showWhen: (s) => s.pdMode !== "prefill",
          disabled: (s) => config.isNpuHw(s),
          disableReason: (s) => (config.isNpuHw(s) ? "Only the Balanced operating point is supported on this recipe." : ""),
        },
        { id: "default",         label: "Default",         showWhen: (s) => s.pdMode === "prefill" },
        { id: "long-context",    label: "Long-Context",    showWhen: (s) => s.pdMode === "prefill" },
      ],
    },
  ],

  // Orthogonal to the cell grid: the picked option layers flags onto whichever
  // cell is showing, so turning speculation on does not triple the cell count.
  overlayDims: [
    {
      // Checkpoint choice, orthogonal to the cell grid: MXFP4 is the shipping default.
      id: "quant",
      title: "Quantization",
      default: "mxfp4",
      options: [
        { id: "mxfp4", label: "MXFP4", subtitle: "Moonshot AI checkpoint",
          // The 950PR/DT recipe serves this checkpoint; the A3 Series recipe serves the W4A8 build under it instead.
          disabled: (s) => s.hw === "a3",
          disableReason: (s) => (s.hw === "a3" ? "The A3 Series recipe serves the sgl-npu Modelslim (W4A8) checkpoint." : ""),
        },
        {
          // A3 Series only (ModelSlim W4A8 checkpoint).
          id: "modelslim",
          label: "Modelslim (W4A8)",
          subtitle: "ModelScope NPU checkpoint",
          showWhen: (s) => s.hw === "a3",
        },
        {
          id: "nvfp4",
          label: "NVFP4",
          subtitle: "NVIDIA checkpoint",
          // The NVFP4 MoE kernels (FlashInfer TRT-LLM) are Blackwell-only.
          disabled: (s) => !["b200", "gb200", "b300", "gb300"].includes(s.hw),
          disableReason:
            "The nvidia/Kimi-K3-NVFP4 checkpoint needs Blackwell: its routed experts run on FlashInfer TRT-LLM NVFP4 kernels (SiTU), which do not exist for Hopper or AMD.",
          // B200's Balanced/High-Throughput cells pin flashinfer_mxfp4.
          stripPrefixes: ["--moe-runner-backend"],
          flags: ["--moe-runner-backend flashinfer_trtllm"],
          hints: [
            "Use docker image lmsysorg/sglang:dev-dev-kimi-k3-nvfp4 (CUDA 13).",
          ],
        },
      ],
    },
    {
      id: "mmTransport",
      title: "VLM Transport",
      default: "auto",
      showWhen: (s) => s.pdMode !== "decode" && !config.isNpuHw(s),
      options: [
        {
          id: "auto",
          label: "Auto (topology-aware)",
          hints: (s) => {
            if (s.pdMode !== "unified") {
              return [
                "VLM transport: Auto -> CPU for PD; KV/KDA transfer is separate.",
              ];
            }
            if (s.hw === "b300") {
              return [
                "VLM transport: Auto -> CUDA IPC (up to 1 GiB HBM; CPU fallback when full).",
              ];
            }
            if (["gb200", "gb300"].includes(s.hw)) {
              return [
                "VLM transport: Auto -> CUDA VMM with IMEX, otherwise CPU (up to 1 GiB HBM).",
              ];
            }
            return ["VLM transport: Auto -> CPU on this topology."];
          },
        },
        {
          id: "cpu",
          label: "CPU (save HBM)",
          flags: ["--mm-feature-transport cpu"],
          hints: ["VLM transport: CPU; no GPU feature pool."],
        },
      ],
    },
    {
      id: "spec",
      title: "Spec Decode",
      default: "dspark",
      options: [
        { id: "none", label: "Non-Spec",
          env: (s) => (["mi350x", "mi355x"].includes(s.hw) ? ["SGLANG_MLA_DECODE_TUNE=1"] : []),
          disabled: (s) => config.isNpuHw(s),
          disableReason: (s) => (config.isNpuHw(s) ? "Only DSPARK is supported on this recipe." : ""),
        },
        {
          id: "dspark",
          label: "DSPARK",
          // DSPARK requires pp_size == 1, so a pipelined recipe is either re-laid flat (cells opting in with `specCollapsePp`).
          disabled: (s) => config.isPipelined(s) && !config.specCollapses(s),
          disableReason:
            "This recipe is pipelined (--pp-size > 1) and DSPARK requires pp_size == 1. Pick a recipe that runs a single pipeline stage, or run this one NOSPEC.",
          // Where a cell does opt in, DSPARK rewrites its parallelism instead
          // of layering on top.
          stripPrefixes: (s) =>
            config.specCollapses(s)
              ? ["--tp-size", "--pp-size", "--dcp-size", "--ep-size"]
              : [],
          flags: (s) => [
            ...(config.specCollapses(s) ? config.specCollapsedFlags(s) : []),
            "--speculative-algorithm DSPARK",
            "--speculative-draft-model-path RadixArk/Kimi-K3-DSpark",
            "--speculative-dspark-block-size 7",
            ...(config.isNpuHw(s)
              ? [
                  "--speculative-draft-attention-backend ascend",
                  "--speculative-eagle-topk 1",
                  "--speculative-draft-model-quantization unquant",
                ]
              : []),
            // ReplaySSM moves the per-draft intermediate SSM states onto a
            // fixed ring.
            ...(!config.isNpuHw(s) && s.pdMode !== "prefill"
              ? ["--enable-linear-replayssm-spec"]
              : []),
          ],
        },
        {
          id: "dflash",
          label: "DFLASH",
          // DFLASH doesn't support pipeline parallelism yet and rejects DP
          // attention off NPU, so pipelined and DP-attention recipes are
          // unavailable. The NPU recipes ship DSPARK only.
          disabled: (s) =>
            config.isNpuHw(s) ||
            ["h100", "h200", "mi350x", "mi355x"].includes(s.hw) ||
            config.isPipelined(s) ||
            !!config.flagOf(config.cellFor(s), "--enable-dp-attention"),
          disableReason: (s) =>
            config.isNpuHw(s)
              ? "Only DSPARK is supported on this recipe."
              : ["h100", "h200", "mi350x", "mi355x"].includes(s.hw)
                ? "K3 DFLASH has not been validated on this hardware yet; use DSPARK."
                : "DFLASH doesn't support pipeline parallelism or DP attention yet. Pick a recipe that runs a single pipeline stage without DP attention, or use DSPARK.",
          flags: (s) => [
            "--speculative-algorithm DFLASH",
            "--speculative-draft-model-path modal-labs/Kimi-K3-DFlash",
            "--speculative-dflash-block-size 8",
            // Same ReplaySSM rule as DSPARK: the PD prefill role rejects the flag.
            ...(s.pdMode !== "prefill" ? ["--enable-linear-replayssm-spec"] : []),
          ],
        },
      ],
    },
    {
      // Recipes transcribed from the measured HiCache rounds.
      id: "hicache",
      title: "HiCache",
      default: "off",
      options: [
        { id: "off", label: "Off" },
        {
          id: "l2",
          label: "L1+L2 (host)",
          disabled: (s) => config.isNpuHw(s),
          disableReason: (s) => (config.isNpuHw(s)
            ? "NPU HiCache does not support mamba cache (K3's KDA state is one)."
            : ""),
          flags: [
            "--enable-hierarchical-cache",
          ],
          stripPrefixes: (s) =>
            !["none", "dspark"].includes(s.spec) && config.hasDcp(s)
              ? ["--dcp-size", "--dcp-comm-backend"]
              : [],
          hints: (s) =>
            !["none", "dspark"].includes(s.spec) && config.hasDcp(s)
              ? [
                  "HiCache under DCP only accepts DSPARK speculative decoding, so this",
                  "recipe drops DCP and serves the MLA KV TP-replicated (any PP/EP in the",
                  "cell stays). Per-request context is far shorter than the DCP version —",
                  "DCP is what buys KV capacity. Run the cell NOSPEC or DSPARK to keep DCP.",
                ]
              : [],
        },
        {
          id: "l3",
          label: "+ L3 (Mooncake)",
          disabled: (s) => config.isNpuHw(s),
          disableReason: (s) => (config.isNpuHw(s)
            ? "NPU HiCache does not support mamba cache (K3's KDA state is one)."
            : ""),
          flags: [
            "--enable-hierarchical-cache",
            "--hicache-storage-backend mooncake",
          ],
          env: ["SGLANG_HICACHE_MOONCAKE_CONFIG_PATH={{MOONCAKE_CONFIG}}"],
          // L3 under DCP is rejected at startup (the rank-0 replicated-MLA
          // backup and the storage keys are not dcp_rank-aware).
          stripPrefixes: (s) =>
            config.hasDcp(s) ? ["--dcp-size", "--dcp-comm-backend"] : [],
          hints: (s) =>
            [
              "L3 also needs a mooncake_master process on rank 0 and the config file",
              "above present on every rank — the launch command alone is not enough.",
              ...(config.hasDcp(s)
                ? [
                    "L3 storage keys are not dcp_rank-aware yet, so this recipe drops DCP",
                    "and serves the MLA KV TP-replicated (any PP/EP in the cell stays).",
                    "Concurrency lands on a similar target, but",
                    "per-request context is far shorter than the DCP version — DCP is",
                    "what buys KV capacity.",
                  ]
                : []),
            ],
        },
      ],
    },
  ],

  modelNames: {
    default: "moonshotai/Kimi-K3",
    nvfp4: "nvidia/Kimi-K3-NVFP4",
    a3: "sgl-npu/Kimi-K3-W4A8",
    // The 950PR/DT recipe serves the official Moonshot checkpoint (MXFP4) from ModelScope.
    a5: "moonshotai/Kimi-K3",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",        default: "0.0.0.0"        },
    PORT:      { target: "command", label: "Bind port",        default: "30000"          },
    NODE0_IP:  { target: "command", label: "Head node IP",     default: "<node0-ip>"     },
    NODE_RANK: { target: "command", label: "This node rank",   default: "<node-rank>"    },
    LOCAL_IP:  { target: "command", label: "This node IP",     default: "<this-node-ip>" },
    NETWORK_IFACE: { target: "command", label: "Cross-node NIC", default: "<your-nic>"   },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    MOONCAKE_CONFIG: { target: "command", label: "Mooncake config path", default: "<mooncake.json>" },
    CURL_HOST: { target: "curl",    label: "Server host",      default: "localhost"      },
    CURL_PORT: { target: "curl",    label: "Server port",      default: "30000"          },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"Hello"}] }'`,

  // Reproduce command for the benchmark card's "⚡ Reproduce" modal. The
  // sweep is 8k-in / 1k-out random, num_prompts = 5 × concurrency, cache-cold.
  benchmarkCommands: {
    speed:
`python3 -m sglang.bench_serving \\
  --backend sglang \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --dataset-name {{DATASET}} \\
  --random-input-len {{ISL}} --random-output-len {{OSL}} --random-range-ratio 1.0 \\
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{MAX_CONCURRENCY}} \\
  --warmup-requests 64 --flush-cache`,
    numPromptsByConc: { 1: 16, 16: 80, 64: 320, 256: 1280, 1024: 5120 },
  },

  dockerImages: {
    h100:   "lmsysorg/sglang:kimi-k3",
    h200:   "lmsysorg/sglang:kimi-k3",
    b300:   "lmsysorg/sglang:kimi-k3",
    gb300:  "lmsysorg/sglang:kimi-k3",
    b200:   "lmsysorg/sglang:kimi-k3",
    gb200:  "lmsysorg/sglang:kimi-k3",
    mi350x: "lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260916",
    mi355x: "lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260916",
    "b300|nvfp4":  "lmsysorg/sglang:dev-dev-kimi-k3-nvfp4",
    "gb300|nvfp4": "lmsysorg/sglang:dev-dev-kimi-k3-nvfp4",
    "b200|nvfp4":  "lmsysorg/sglang:dev-dev-kimi-k3-nvfp4",
    "gb200|nvfp4": "lmsysorg/sglang:dev-dev-kimi-k3-nvfp4",
    a3:     "quay.io/ascend/sglang:main-cann9.0.0-a3",
    a5:     "swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-950-B070",
  },
  // Pre-selects the issue template's `model` field on "Submit verified cell".
  github: {
    cookbookModel: "moonshotai/kimi-k3",
  },

  playgroundFeatures: {

    // ----- Card: "Attention Parallelism" ----- DP-Attention is a combined knob: value = DP
    // degree AND toggles `--enable-dp-attention`. No CP knob: K3 uses decode context parallel
    // (`--dcp-size`), a different lever from prefill `--attn-cp-size`.
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [
          null,
          {
            value: 8,
            get disable() { return [
              {
                when: { hw: ["a3"] },
                reason: "Only TP64 is supported on this recipe.",
              },
              {
                when: { hw: ["a5"] },
                reason: "Only TP32 is supported on this recipe.",
              },
            ]; },
          },
          {
            value: 16,
            get disable() { return [
              {
                when: { hw: ["a3"] },
                reason: "Only TP64 is supported on this recipe.",
              },
              {
                when: { hw: ["a5"] },
                reason: "Only TP32 is supported on this recipe.",
              },
              {
                when: { hw: ["b300", "gb300"] },
                reason: "TP=16 needs 16 ranks; the B300 and GB300 recipes have 8 ranks.",
              },
              {
                when: { hw: ["b200"], pdMode: ["unified"], spec: ["none"] },
                reason: "With Spec Decode off, the B200 Unified recipes already use all 16 GPUs as TP8 × PP2, so changing TP to 16 would require 32 ranks. Switch Spec Decode to DSPARK — it drops the pipeline and re-lays the same GPUs as flat TP16.",
              },
              ...config.pipelinedKnobDisableRules,
            ]; },
          },
          {
            // 950PR/DT Series only: ranks (nodes × cards, one rank per card); hidden on the other recipes.
            value: 32,
            hide: { hw: ["b300", "gb300", "b200", "gb200", "h200", "h100", "mi350x", "mi355x", "a3"] },
          },
          {
            // A3 Series only: ranks (nodes × cards × dies); hidden on the other recipes.
            value: 64,
            hide: { hw: ["b300", "gb300", "b200", "gb200", "h200", "h100", "mi350x", "mi355x", "a5"] },
          },
        ]},
        { id: "dpAttn", label: "DP-Attention",
          values: [
            null,
            {
              value: 1,
              hide: { hw: ["b300", "gb300", "b200", "gb200", "h200", "h100", "mi350x", "mi355x", "a3"] },
            },
            {
              value: false,
              get disable() { return [
                {
                  when: { hw: ["a3"] },
                  reason: "Only DP-Attention=4 is supported on this recipe.",
                },
                {
                  when: { hw: ["a5"] },
                  reason: "Only DP-Attention=1 is supported on this recipe.",
                },
              ]; },
            },
            {
              value: 2,
              get disable() { return [
                {
                  when: { hw: ["a3"] },
                  reason: "Only DP-Attention=4 is supported on this recipe.",
                },
                {
                  when: { hw: ["a5"] },
                  reason: "Only DP-Attention=1 is supported on this recipe.",
                },
              ]; },
            },
            {
              value: 4,
              get disable() { return [
                {
                  when: { hw: ["a5"] },
                  reason: "Only DP-Attention=1 is supported on this recipe.",
                },
              ]; },
            },
            {
              value: 8,
              get disable() { return [
                {
                  when: { hw: ["a3"] },
                  reason: "Only DP-Attention=4 is supported on this recipe.",
                },
                {
                  when: { hw: ["a5"] },
                  reason: "Only DP-Attention=1 is supported on this recipe.",
                },
                {
                  when: { hw: ["b300", "gb300"] },
                  reason: "On an 8-rank deployment (B300 1×8, GB300 2×4) dp=8 leaves attn_tp=1, so each rank holds the full unsharded MLA KV and OOMs — prefer dp=2/attn_tp=4.",
                },
                {
                  when: { hw: ["b200"], pdMode: ["unified"], spec: ["none"] },
                  reason: "With Spec Decode off, the B200 Unified recipes run TP8 within each PP2 stage, so dp=8 leaves attn_tp=1 and each rank holds the full unsharded MLA KV — prefer dp=2/attn_tp=4, or switch Spec Decode to DSPARK for the flat TP16 shape.",
                },
                ...config.pipelinedKnobDisableRules,
              ]; },
            },
            {
              value: 16,
              get disable() { return [
                {
                  when: { hw: ["a3"] },
                  reason: "Only DP-Attention=4 is supported on this recipe.",
                },
                {
                  when: { hw: ["a5"] },
                  reason: "Only DP-Attention=1 is supported on this recipe.",
                },
                {
                  when: { hw: ["b300", "gb300"] },
                  reason: "DP-Attention=16 needs 16 TP ranks; the B300 and GB300 recipes have 8.",
                },
                {
                  when: { hw: ["b200"], pdMode: ["unified"], spec: ["none"] },
                  reason: "With Spec Decode off, the B200 Unified recipes use TP8 within each PP2 stage, so DP-Attention cannot exceed 8. Switch Spec Decode to DSPARK for the flat TP16 shape.",
                },
                ...config.pipelinedKnobDisableRules,
              ]; },
            },
          ],
          labels: { "auto": "Auto", "false": "Off" } },
      ],
    },

    // Marlin (W4A16) is the accuracy runner; the MXFP4 / a2a runners (FlashInfer
    // MXFP4, DeepEP, MegaMoE) are throughput levers.
    moe: {
      backend: {
        options: [
          { id: null,               label: "Inherited" },
          { id: "deepep",           label: "DeepEP",            flags: ["--moe-a2a-backend deepep"] },
          // Blackwell-only kernel-fusion path; selecting it reveals the Quantization sub-select.
          { id: "megamoe",          label: "MegaMoE",           flags: ["--moe-a2a-backend megamoe"],
            requiresHw: ["b200", "b300", "gb200", "gb300"] },
          // Blackwell-only: runs FlashInfer's official trtllm-gen SiTU kernels.
          { id: "flashinfer_mxfp4", label: "FlashInfer (MXFP4)", flags: ["--moe-runner-backend flashinfer_mxfp4"],
            requiresHw: ["b200", "b300", "gb200", "gb300"],
            disable: [{ when: { hw: ["a3", "a5"] },
              reason: "FlashInfer is a CUDA kernel and is not supported on NPU." }] },
          { id: "marlin",           label: "Marlin (W4A16)",    flags: ["--moe-runner-backend marlin"],
            disable: [{ when: { hw: ["a3", "a5"] },
              reason: "Marlin is a CUDA kernel and is not supported on NPU." }] },
        ],
      },
      // MegaMoE quantization sub-select — shown only when backend === "megamoe".
      megamoeQuant: {
        stripEnv: ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"],
        options: [
          { id: "w4a8", label: "W4A8",
            env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320"] },
          { id: "w4a4", label: "W4A4",
            flags: ["--enable-w4a4-mxfp4-megamoe"],
            env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320"] },
        ],
      },
      ep: {
        showWhen: (b) => !config.isNpuHw(b),
        label: "EP",
        values: [
          null, 1, 2, 4, 8,
          {
            value: 16,
            get disable() { return [
              {
                when: { hw: ["b300", "gb300"] },
                reason: "EP=16 needs 16 TP ranks; the B300 and GB300 recipes have 8.",
              },
              {
                when: { hw: ["b200"], pdMode: ["unified"], spec: ["none"] },
                reason: "With Spec Decode off, the B200 Unified recipes use TP8 within each PP2 stage, so EP cannot exceed 8. Switch Spec Decode to DSPARK for the flat TP16 shape.",
              },
              ...config.pipelinedKnobDisableRules,
            ]; },
          },
        ],
      },
    },

    // ----- Card: "Parsers" -----
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser kimi_k3" },
        // Tool calling is not yet supported on the NPU recipes, so the item
        // hides on a3 and a5.
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser kimi_k3",
          hide: { hw: ["a3", "a5"] } },
      ],
    },

    pdDisagg: {
      showWhen: (b) => b.pdMode === "prefill" || b.pdMode === "decode",
      transferBackends: [
        { id: "nixl", label: "NiXL" },
        { id: "mooncake", label: "Mooncake" },
      ],
      // `auto` is a sentinel (emits no --disaggregation-ib-device flag).
      ibDevices: [{ id: "auto", label: "Auto" }, "mlx5_0"],
      router: {
        port: 8000,
        // Ports come from the engine's PD_PORTS, the same source the role commands rewrite `--port` from, so the router always targets the ports.
        command:
`python3 -m sglang_router.launch_router \\
  --pd-disaggregation \\
  --prefill http://<prefill-host>:{{PREFILL_PORT}} 8998 \\
  --decode http://<decode-host>:{{DECODE_PORT}} \\
  --host 0.0.0.0 --port {{ROUTER_PORT}} \\
  --disable-circuit-breaker \\
  --health-check-interval-secs 999999`,
      },
    },

    // ----- Card: "HiCache" ----- (K3 hybrid L1/L2/L3, incl. KDA state)
    hicache: {
      showWhen: (b) => b.hicache !== undefined && b.hicache !== "off",
      // Picking a storage backend here IS L3, and L3 under DCP is rejected at
      // startup. The Deploy panel's own L3 option drops DCP first; reaching L3
      // through this card would not, so gate it on the DCP recipes unless Deploy
      // already switched to L3 (hicache "off"/"l2" means DCP is still standing).
      backends: [
        { id: null,       label: "Auto" },
        { id: "file",     label: "File",
          get disable() { return config.dcpStorageDisableRules; } },
        { id: "mooncake", label: "Mooncake",
          get disable() { return config.dcpStorageDisableRules; } },
        { id: "hf3fs",    label: "HF3FS",
          get disable() { return config.dcpStorageDisableRules; } },
        { id: "nixl",     label: "NiXL",
          get disable() { return config.dcpStorageDisableRules; } },
      ],
      writePolicies: [
        { id: "auto",                    label: "Auto" },
        { id: "write_through",           label: "Write-through" },
        { id: "write_back",              label: "Write-back" },
        { id: "write_through_selective", label: "Write-through (selective)" },
      ],
    },

    // ----- Axis: Flag Selects (K3 hybrid dual-pool knobs) -----
    // KDA state pool vs full-KV pool levers (measured dual-pool analysis). The
    // flagless option is the accuracy-safe default; the others are capacity/long-ctx
    // levers whose accuracy A/B is workload-gated — Playground opt-ins, not cells.
    flagSelects: [
      {
        // Deployment picks this from the strategy; the row is the override.
        id: "proposedDraftTokens", title: "Proposed Draft Tokens",
        showWhen: (b) => b.spec === "dspark" && !config.isNpuHw(b),
        control: "slider",
        stripPrefixes: [
          "--speculative-dspark-block-size",
          "--speculative-dflash-block-size",
          "--speculative-num-steps",
        ],
        options: [
          { id: "1", label: "1", flags: ["--speculative-dspark-block-size 1"] },
          { id: "2", label: "2", flags: ["--speculative-dspark-block-size 2"] },
          { id: "3", label: "3", flags: ["--speculative-dspark-block-size 3"] },
          { id: "4", label: "4", flags: ["--speculative-dspark-block-size 4"] },
          { id: "5", label: "5", flags: ["--speculative-dspark-block-size 5"] },
          { id: "6", label: "6", flags: ["--speculative-dspark-block-size 6"] },
          { id: "7", label: "7", flags: ["--speculative-dspark-block-size 7"] },
        ],
      },
      {
        // ReplaySSM moves the per-draft intermediate SSM states onto a fixed ring.
        id: "replaySsm", title: "ReplaySSM (spec)",
        showWhen: (b) => b.spec === "dspark" && !config.isNpuHw(b),
        stripPrefixes: ["--enable-linear-replayssm-spec"],
        options: [
          { id: "off", label: "Off" },
          {
            id: "on", label: "On",
            // The ring is spec-verify-only scratch and a prefill server never runs verify.
            disable: { pdMode: ["prefill"] },
            disableReason:
              "A PD prefill server never runs speculative verify, so --enable-linear-replayssm-spec is rejected at startup.",
            flags: ["--enable-linear-replayssm-spec"],
          },
        ],
      },
      {
        // Compact prunes the verify layout to the SPS budget.
        id: "raggedVerify", title: "Ragged Verify Mode (spec)",
        // The NPU recipes pin static.
        showWhen: (b) => b.spec === "dspark" && !config.isNpuHw(b),
        stripEnv: ["SGLANG_RAGGED_VERIFY_MODE"],
        options: [
          { id: "static",  label: "Auto (static)" },
          { id: "compact", label: "Compact (requires SPS table)", env: ["SGLANG_RAGGED_VERIFY_MODE=compact"] },
        ],
      },
      {
        id: "kvCacheDtype", title: "KV Cache Precision",
        showWhen: (b) => !config.isNpuHw(b),
        stripPrefixes: ["--kv-cache-dtype"],
        options: [
          { id: "auto", label: "Auto (BF16)" },
          { id: "fp8",  label: "FP8 (E4M3) — halves KV memory", flags: ["--kv-cache-dtype fp8_e4m3"] },
        ],
      },
      {
        id: "mambaSsmDtype", title: "KDA State Precision",
        showWhen: (b) => !config.isNpuHw(b),
        stripPrefixes: ["--mamba-ssm-dtype"],
        options: [
          { id: "auto", label: "Auto (FP32)" },
          { id: "bf16", label: "BFloat16 — halves state memory", flags: ["--mamba-ssm-dtype bfloat16"] },
          // The dtype --enable-mamba-cache-stochastic-rounding requires; no serving round has tried it, unlike bf16.
          { id: "fp16", label: "Float16 — stochastic-rounding capable", flags: ["--mamba-ssm-dtype float16"] },
        ],
      },
      {
        // Whole-model prefix cache (the radix tree spans MLA KV + KDA state).
        id: "prefixCache", title: "Prefix Cache",
        showWhen: (b) => !config.isNpuHw(b),
        stripPrefixes: ["--disable-radix-cache"],
        options: [
          { id: "on",  label: "On" },
          { id: "off", label: "Off", flags: ["--disable-radix-cache"] },
        ],
      },
      {
        // How KDA state buffers for radix reuse; no strategy exists with the prefix cache off.
        id: "mambaRadix", title: "KDA Radix Cache Strategy",
        showWhen: (b, v, d) => (!config.isNpuHw(b))
          && ((((v && v.prefixCache) ?? (d && d.prefixCache)) !== "off")),
        stripPrefixes: ["--mamba-radix-cache-strategy"],
        options: [
          { id: "auto",  label: "Auto (extra_buffer)" },
          { id: "lazy",  label: "extra_buffer_lazy", flags: ["--mamba-radix-cache-strategy extra_buffer_lazy"] },
          { id: "nobuf", label: "no_buffer",         flags: ["--mamba-radix-cache-strategy no_buffer"] },
        ],
      },
      {
        // Experimental env toggle: SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK skips the decode-time mamba lock.
        id: "mambaSlotSaving", title: "KDA Slot Saving (experimental)",
        showWhen: (b) => !config.isNpuHw(b),
        stripEnv: ["SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK"],
        options: [
          { id: "off", label: "Off" },
          { id: "on",  label: "On", env: ["SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1"] },
        ],
      },
      {
        // AITER SiTU v2 activation-quant mode (ROCm only).
        id: "situActMode", title: "SiTU MoE Activation Quant (AMD)",
        showWhen: (b) => ["mi350x", "mi355x"].includes(b.hw),
        stripEnv: ["AITER_SITUV2_A8W4", "AITER_SITUV2_A4W4"],
        options: [
          { id: "a8w4", label: "A8W4 (default)", env: ["AITER_SITUV2_A8W4=1"] },
          { id: "a4w4", label: "A4W4 — correct, ~1% slower", env: ["AITER_SITUV2_A4W4=1"] },
        ],
      },
      {
        // Opt-in fused gfx950 KDA decode boundary (f_b + conv + recurrence + gated RMSNorm).
        id: "kdaFusedDecode", title: "Fused KDA Decode (AMD gfx950)",
        showWhen: (b) => ["mi350x", "mi355x"].includes(b.hw),
        stripEnv: ["SGLANG_K3_KDA_FUSED_BACKEND"],
        options: [
          { id: "off",   label: "Off" },
          { id: "aiter", label: "On (AITER fused boundary)", env: ["SGLANG_K3_KDA_FUSED_BACKEND=aiter"] },
        ],
      },
      {
        // Only meaningful with EP a2a on (MoE card or a large-scale preset).
        id: "eplb", title: "Expert Rebalancing (EPLB)",
        showWhen: (b) => !config.isNpuHw(b),
        stripPrefixes: ["--enable-eplb"],
        options: [
          { id: "off", label: "Off" },
          { id: "on",  label: "On (requires EP a2a)", flags: ["--enable-eplb"] },
        ],
      },
      {
        // Prefill has no graph by default; BCG captures it as a breakable graph.
        id: "prefillGraph", title: "Prefill CUDA Graph",
        showWhen: (b) => !config.isNpuHw(b),
        stripPrefixes: ["--cuda-graph-backend-prefill"],
        options: [
          { id: "auto", label: "Auto (off)" },
          { id: "bcg",  label: "Breakable (BCG)", flags: ["--cuda-graph-backend-prefill breakable"] },
        ],
      },
      {
        id: "pdDecodeRadix", title: "PD Decode Radix Cache",
        showWhen: (b) => b.pdMode === "decode",
        stripPrefixes: ["--disaggregation-decode-enable-radix-cache"],
        options: [
          { id: "off", label: "Off (chunk cache)" },
          { id: "on",  label: "On", flags: ["--disaggregation-decode-enable-radix-cache"] },
        ],
      },
      {
        // Both rows below compose: Cluster Size picks N, Large-Scale Preset resolves it into the full parallelism shape.
        id: "lsGpus", title: "Cluster Size (large-scale)",
        showWhen: (b) => (!config.isNpuHw(b)) && (b.pdMode === undefined || b.pdMode === "unified"),
        // Default follows the base cell's own GPU count (tp8 lanes -> 8,
        // tp16 lanes -> 16), so a preset starts from "same hardware, new shape".
        default: (b) =>
          ({ b300: "8", gb300: "8", b200: "16", gb200: "16",
             h200: "16", h100: "32", mi350x: "8", mi355x: "8" })[(b || {}).hw] || "32",
        stripPrefixes: [],
        options: [
          { id: "8",  label: "8 GPUs" },
          { id: "16", label: "16 GPUs" },
          { id: "32", label: "32 GPUs" },
          { id: "64", label: "64 GPUs" },
        ],
      },
      {
        id: "lsPreset", title: "Large-Scale Preset",
        showWhen: (b) => (!config.isNpuHw(b)) && (b.pdMode === undefined || b.pdMode === "unified"),
        stripPrefixes: [
          "--tp-size", "--tp", "--tensor-parallel-size",
          "--ep-size", "--ep", "--expert-parallel-size",
          "--enable-dp-attention", "--dp-size", "--enable-dp-lm-head",
          "--dcp-size", "--dcp-comm-backend",
          "--pp-size", "--pipeline-parallel-size",
          "--moe-a2a-backend", "--moe-runner-backend",
          "--kv-cache-dtype", "--mamba-ssm-dtype", "--mamba-radix-cache-strategy",
          "--mem-fraction-static", "--disable-radix-cache", "--enable-symm-mem",
        ],
        stripEnv: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK"],
        options: [
          { id: "off", label: "Off", flags: () => null },
          {
            id: "serving", label: "Peak Throughput",
            disable: { hw: ["h100", "h200", "mi350x", "mi355x"] },
            disableReason: "The large-scale presets ride the MegaMoE a2a lane (SM100/SM103) — Blackwell only.",
            env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=20480"],
            flags: (v, b) => {
              const n = Number(v.lsGpus) || 32;
              const dp = n / 8;
              const gpusPerNode = ["gb200", "gb300"].includes(b.hw) ? 4 : 8;
              const nnodes = n / gpusPerNode;
              return [
                `--tp-size ${n}`, `--ep-size ${n}`,
                ...(dp > 1 ? ["--enable-dp-attention", `--dp-size ${dp}`, "--enable-dp-lm-head"] : []),
                ...(nnodes > 1 ? [`--nnodes ${nnodes}`, "--node-rank {{NODE_RANK}}", "--dist-init-addr {{NODE0_IP}}:20000"] : []),
                "--moe-a2a-backend megamoe", "--moe-runner-backend deep_gemm",
                "--kv-cache-dtype fp8_e4m3", "--mamba-ssm-dtype bfloat16",
                "--mamba-radix-cache-strategy extra_buffer_lazy",
                "--mem-fraction-static 0.92",
              ];
            },
          },
          {
            id: "capacity", label: "Peak Capacity (+DCP8)",
            // The only playground option that re-adds DCP, so it is also the
            // only one that can resurrect the DCP + L3 combination the Deploy
            // panel stripped out.
            disable: [
              {
                when: { hw: ["h100", "h200", "mi350x", "mi355x"] },
                reason: "The large-scale presets ride the MegaMoE a2a lane (SM100/SM103) — Blackwell only.",
              },
              {
                when: { hicache: ["l3"] },
                reason: "L3 storage keys are not dcp_rank-aware, so DCP and L3 cannot run together. Use Peak Throughput, or switch HiCache to L1+L2.",
              },
            ],
            env: ["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=20480"],
            flags: (v, b) => {
              const n = Number(v.lsGpus) || 32;
              const dp = n / 8;
              const gpusPerNode = ["gb200", "gb300"].includes(b.hw) ? 4 : 8;
              const nnodes = n / gpusPerNode;
              return [
                `--tp-size ${n}`, `--ep-size ${n}`,
                ...(dp > 1 ? ["--enable-dp-attention", `--dp-size ${dp}`, "--enable-dp-lm-head"] : []),
                "--dcp-size 8",
                ...(nnodes > 1 ? [`--nnodes ${nnodes}`, "--node-rank {{NODE_RANK}}", "--dist-init-addr {{NODE0_IP}}:20000"] : []),
                "--moe-a2a-backend megamoe", "--moe-runner-backend deep_gemm",
                "--kv-cache-dtype fp8_e4m3", "--mamba-ssm-dtype bfloat16",
                "--mamba-radix-cache-strategy extra_buffer_lazy",
                "--mem-fraction-static 0.92",
              ];
            },
          },
        ],
      },
    ],
  },

  // Verification marks: every cell carries "Final Verification In Progress" —
  // the recipe runs, but its serving round on the final weights and current
  // code is still open. Flip a cell to `verified: true` (and drop its
  // `verificationStatus`) once that round lands.
  cells: [
    {
      match: { hw: "b300", pdMode: "unified", strategy: "low-latency" },
      nnodes: 1,
      verified: true,
      env: [],
      // No --enable-symm-mem: it makes the fused all-reduce auto-probe skip.
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", pdMode: "unified", strategy: "balanced" },
      nnodes: 1,
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", pdMode: "unified", strategy: "high-throughput" },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
      redirect: true,
      warn: "High-Throughput is the large-scale lane: pick a Cluster Size and a Large-Scale Preset in the [Playground](#playground) to compose the DP x EP command on top of this hardware's Balanced recipe.",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--disable-custom-all-reduce",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // NOSPEC shape.
      match: { hw: "b200", pdMode: "unified", strategy: "low-latency" },
      nnodes: 2,
      // Under a pp_size == 1 speculative algorithm, re-lay this recipe flat at constant world size instead.
      specCollapsePp: true,
      verified: false,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--pp-size 2",
        "--mem-fraction-static 0.85",
        "--disable-flashinfer-autotune",
        "--watchdog-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // PP2 × DCPEP8: DCP8 deduplicates the TP-replicated MLA KV within each pipeline stage.
      match: { hw: "b200", pdMode: "unified", strategy: "balanced" },
      nnodes: 2,
      // Under a pp_size == 1 speculative algorithm, re-lay this recipe flat at constant world size instead.
      specCollapsePp: true,
      verified: false,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--pp-size 2",
        "--dcp-size 8",
        "--ep-size 8",
        // Both backends are pinned to the validated recipe even though automatic resolution selects them on Blackwell.
        "--moe-runner-backend flashinfer_mxfp4",
        "--decode-attention-backend cutedsl_mla",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--disable-flashinfer-autotune",
        "--watchdog-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Balanced baseline; High-Throughput routes to the large-scale presets.
      match: { hw: "b200", pdMode: "unified", strategy: "high-throughput" },
      nnodes: 2,
      // Under a pp_size == 1 speculative algorithm, re-lay this recipe flat at constant world size instead.
      specCollapsePp: true,
      verified: false,
      verificationStatus: "in-progress",
      redirect: true,
      warn: "High-Throughput is the large-scale lane: pick a Cluster Size and a Large-Scale Preset in the [Playground](#playground) to compose the DP x EP command on top of this hardware's Balanced recipe.",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--pp-size 2",
        "--dcp-size 8",
        "--ep-size 8",
        "--moe-runner-backend flashinfer_mxfp4",
        "--decode-attention-backend cutedsl_mla",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 8192",
        "--disable-flashinfer-autotune",
        "--watchdog-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // MI350X and MI355X use the same single-node TP8/DCP8 ROCm/AITER profile.
      match: { hw: "mi350x", pdMode: "unified", strategy: "balanced" },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_USE_AITER=1",
        "SGLANG_AITER_K3_OPT=1",
        "AITER_FLYDSL_FORCE=1",
        "AITER_SITUV2_A8W4=1",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp-size 8",
        "--dcp-size 8",
        "--dcp-comm-backend a2a",
        "--prefill-attention-backend aiter",
        "--decode-attention-backend aiter",
        "--kv-cache-dtype fp8_e4m3",
        "--dtype bfloat16",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 256",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Same ROCm/AITER profile as MI350X.
      match: { hw: "mi355x", pdMode: "unified", strategy: "balanced" },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_USE_AITER=1",
        "SGLANG_AITER_K3_OPT=1",
        "AITER_FLYDSL_FORCE=1",
        "AITER_SITUV2_A8W4=1",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp-size 8",
        "--dcp-size 8",
        "--dcp-comm-backend a2a",
        "--prefill-attention-backend aiter",
        "--decode-attention-backend aiter",
        "--kv-cache-dtype fp8_e4m3",
        "--dtype bfloat16",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 256",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Latency operating point.
      match: { hw: "h100", pdMode: "unified", strategy: "low-latency" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "NCCL_CUMEM_ENABLE=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_K3_ATTN_RES_MODE=jit",
        "SGLANG_MOE_FUSED_GATE_RADIX=1",
        "SGLANG_HOST_IP={{LOCAL_IP}}",
        "NCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 32",
        "--ep-size 32",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--mem-fraction-static 0.85",
        "--dist-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Accuracy-preserving default.
      match: { hw: "h100", pdMode: "unified", strategy: "balanced" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "NCCL_CUMEM_ENABLE=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_K3_ATTN_RES_MODE=jit",
        "SGLANG_MOE_FUSED_GATE_RADIX=1",
        "SGLANG_HOST_IP={{LOCAL_IP}}",
        "NCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 32",
        "--ep-size 32",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--mem-fraction-static 0.85",
        "--dist-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", pdMode: "unified", strategy: "high-throughput" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "NCCL_CUMEM_ENABLE=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_K3_ATTN_RES_MODE=jit",
        "SGLANG_MOE_FUSED_GATE_RADIX=1",
        "SGLANG_HOST_IP={{LOCAL_IP}}",
        "NCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 32",
        "--ep-size 32",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--mem-fraction-static 0.85",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--dist-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", pdMode: "unified", strategy: "low-latency" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--ep-size 16",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", pdMode: "unified", strategy: "balanced" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--ep-size 16",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // The H200 cell that widens past a single pair of nodes.
      match: { hw: "h200", pdMode: "unified", strategy: "high-throughput" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_K3_ATTN_RES_MODE=jit",
        "SGLANG_MOE_FUSED_GATE_RADIX=1",
        "SGLANG_HOST_IP={{LOCAL_IP}}",
        "NCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 32",
        "--ep-size 32",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--enable-symm-mem",
        "--mem-fraction-static 0.90",
        "--mamba-radix-cache-strategy extra_buffer_lazy",
        "--dist-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", pdMode: "unified", strategy: "low-latency" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", pdMode: "unified", strategy: "balanced" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", pdMode: "unified", strategy: "high-throughput" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      redirect: true,
      warn: "High-Throughput is the large-scale lane: pick a Cluster Size and a Large-Scale Preset in the [Playground](#playground) to compose the DP x EP command on top of this hardware's Balanced recipe.",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", pdMode: "unified", strategy: "low-latency" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Same balanced operating point as GB300; TP/DCP span all ranks.
      match: { hw: "gb200", pdMode: "unified", strategy: "balanced" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dcp-size 16",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Same high-throughput operating point as GB300; TP/DCP span all ranks.
      match: { hw: "gb200", pdMode: "unified", strategy: "high-throughput" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      redirect: true,
      warn: "High-Throughput is the large-scale lane: pick a Cluster Size and a Large-Scale Preset in the [Playground](#playground) to compose the DP x EP command on top of this hardware's Balanced recipe.",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dcp-size 16",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ----- Prefill role: chunked. The prefill role keeps radix caching, so
    // the Unified 5-slots-per-request state cost still holds (pool split
    // rides the calculator-driven ratio). Deep PP is one pipeline stage per
    // GPU, which turns the parallelism comm from something you wait for
    // into something the next microbatch hides.
    {
      match: { hw: "b300", pdMode: "prefill", strategy: "default" },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--disable-custom-all-reduce",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend nixl",
        // Must match the positional bootstrap port the router passes after --prefill.
        "--disaggregation-bootstrap-port 8998",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // PP8xTP1: no TP collective left to accelerate.
      match: { hw: "b300", pdMode: "prefill", strategy: "long-context" },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--pp-size 8",
        "--disable-custom-all-reduce",
        "--mem-fraction-static 0.90",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--disable-flashinfer-autotune",
        "--weight-loader-prefetch-checkpoints",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend nixl",
        "--disaggregation-bootstrap-port 8998",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", pdMode: "prefill", strategy: "default" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend nixl",
        "--disaggregation-bootstrap-port 8998",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", pdMode: "prefill", strategy: "long-context" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--pp-size 8",
        "--mem-fraction-static 0.90",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--disable-flashinfer-autotune",
        "--weight-loader-prefetch-checkpoints",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend nixl",
        "--disaggregation-bootstrap-port 8998",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    {
      // PP16 x TP1 spans all ranks.
      match: { hw: "gb200", pdMode: "prefill", strategy: "default" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--pp-size 16",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--disable-flashinfer-autotune",
        "--weight-loader-prefetch-checkpoints",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend nixl",
        "--disaggregation-bootstrap-port 8998",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", pdMode: "prefill", strategy: "long-context" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--pp-size 16",
        "--mem-fraction-static 0.90",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--disable-flashinfer-autotune",
        "--weight-loader-prefetch-checkpoints",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend nixl",
        "--disaggregation-bootstrap-port 8998",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    {
      match: { hw: "b200", pdMode: "prefill", strategy: "default" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--pp-size 16",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--disable-flashinfer-autotune",
        "--weight-loader-prefetch-checkpoints",
        "--watchdog-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        // Explicit multithread_load is also what keeps it on: prefetch otherwise forces the single-threaded loader.
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend nixl",
        "--disaggregation-bootstrap-port 8998",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", pdMode: "prefill", strategy: "long-context" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 1",
        "--pp-size 16",
        "--mem-fraction-static 0.90",
        "--chunked-prefill-size 16384",
        "--max-prefill-tokens 16384",
        "--disable-flashinfer-autotune",
        "--weight-loader-prefetch-checkpoints",
        "--watchdog-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--disaggregation-mode prefill",
        "--disaggregation-transfer-backend nixl",
        "--disaggregation-bootstrap-port 8998",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // ----- Decode role: the unified cell for the same hw and strategy, plus
    // the PD role and transport flags, and re-sized KDA state.
    //
    // Comparisons hold; absolutes would be higher behind the PP16 x TP1
    // prefill cell above.
    {
      match: { hw: "b300", pdMode: "decode", strategy: "balanced" },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--disable-custom-all-reduce",
        "--mem-fraction-static 0.85",
        "--disaggregation-decode-extra-slots 16",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", pdMode: "decode", strategy: "low-latency" },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--disable-custom-all-reduce",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--disaggregation-decode-extra-slots 16",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b300", pdMode: "decode", strategy: "high-throughput" },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--disable-custom-all-reduce",
        "--mem-fraction-static 0.92",
        "--disaggregation-decode-extra-slots 16",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", pdMode: "decode", strategy: "low-latency" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--mem-fraction-static 0.85",
        "--disaggregation-decode-extra-slots 16",
        "--disable-flashinfer-autotune",
        "--watchdog-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", pdMode: "decode", strategy: "balanced" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dcp-size 16",
        "--mem-fraction-static 0.85",
        "--disaggregation-decode-extra-slots 16",
        "--disable-flashinfer-autotune",
        "--watchdog-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", pdMode: "decode", strategy: "high-throughput" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dcp-size 16",
        "--mem-fraction-static 0.92",
        "--disaggregation-decode-extra-slots 16",
        "--disable-flashinfer-autotune",
        "--watchdog-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi350x", pdMode: "decode", strategy: "balanced" },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_USE_AITER=1",
        "SGLANG_AITER_K3_OPT=1",
        "AITER_FLYDSL_FORCE=1",
        "AITER_SITUV2_A8W4=1",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp-size 8",
        "--attention-backend triton",
        "--kv-cache-dtype fp8_e4m3",
        "--dtype bfloat16",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 256",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mi355x", pdMode: "decode", strategy: "balanced" },
      nnodes: 1,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_USE_AITER=1",
        "SGLANG_AITER_K3_OPT=1",
        "AITER_FLYDSL_FORCE=1",
        "AITER_SITUV2_A8W4=1",
      ],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--tp-size 8",
        "--attention-backend triton",
        "--kv-cache-dtype fp8_e4m3",
        "--dtype bfloat16",
        "--mem-fraction-static 0.85",
        "--cuda-graph-max-bs-decode 256",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", pdMode: "decode", strategy: "low-latency" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "NCCL_CUMEM_ENABLE=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_K3_ATTN_RES_MODE=jit",
        "SGLANG_MOE_FUSED_GATE_RADIX=1",
        "SGLANG_HOST_IP={{LOCAL_IP}}",
        "NCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 32",
        "--ep-size 32",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--mem-fraction-static 0.85",
        "--dist-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", pdMode: "decode", strategy: "balanced" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "NCCL_CUMEM_ENABLE=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_K3_ATTN_RES_MODE=jit",
        "SGLANG_MOE_FUSED_GATE_RADIX=1",
        "SGLANG_HOST_IP={{LOCAL_IP}}",
        "NCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 32",
        "--ep-size 32",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--mem-fraction-static 0.85",
        "--dist-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", pdMode: "decode", strategy: "high-throughput" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "NCCL_CUMEM_ENABLE=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
        "SGLANG_K3_ATTN_RES_MODE=jit",
        "SGLANG_MOE_FUSED_GATE_RADIX=1",
        "SGLANG_HOST_IP={{LOCAL_IP}}",
        "NCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 32",
        "--ep-size 32",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--mem-fraction-static 0.85",
        "--dist-timeout 3600",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", pdMode: "decode", strategy: "low-latency" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--ep-size 16",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", pdMode: "decode", strategy: "balanced" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--ep-size 16",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--enable-symm-mem",
        "--mem-fraction-static 0.85",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", pdMode: "decode", strategy: "high-throughput" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "NCCL_MNNVL_ENABLE=1",
        "NCCL_CUMEM_ENABLE=1",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--ep-size 16",
        "--moe-runner-backend marlin",
        "--decode-attention-backend flashmla",
        "--enable-symm-mem",
        "--mem-fraction-static 0.90",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", pdMode: "decode", strategy: "low-latency" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--mem-fraction-static 0.85",
        "--disaggregation-decode-extra-slots 16",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", pdMode: "decode", strategy: "balanced" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--mem-fraction-static 0.85",
        "--disaggregation-decode-extra-slots 16",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", pdMode: "decode", strategy: "high-throughput" },
      nnodes: 2,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--dcp-size 8",
        "--mem-fraction-static 0.92",
        "--disaggregation-decode-extra-slots 16",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", pdMode: "decode", strategy: "low-latency" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      warn: "SGLang requires `decode pp_size == prefill pp_size or 1`, so this cell must be paired with a PP2 x TP8 prefill (`--tp-size 8 --pp-size 2`) rather than the PP16 x TP1 Prefill recipe. That prefill delivers 2919 prefill tok/s/GPU against PP16's 4550, which is the trade for the decode-side latency.",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 8",
        "--pp-size 2",
        "--mem-fraction-static 0.85",
        "--disaggregation-decode-extra-slots 16",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", pdMode: "decode", strategy: "balanced" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dcp-size 16",
        "--ep-size 16",
        "--mem-fraction-static 0.85",
        "--disaggregation-decode-extra-slots 16",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb200", pdMode: "decode", strategy: "high-throughput" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp-size 16",
        "--dcp-size 8",
        "--dp-size 2",
        "--enable-dp-attention",
        "--ep-size 16",
        "--mem-fraction-static 0.85",
        "--disaggregation-decode-extra-slots 16",
        "--reasoning-parser kimi_k3",
        "--tool-call-parser kimi_k3",
        "--disaggregation-mode decode",
        "--disaggregation-transfer-backend nixl",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "a3", pdMode: "unified", strategy: "balanced" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_USE_MODELSCOPE=1",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "HCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "SGLANG_SET_CPU_AFFINITY=1",
        "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1",
        "STREAMS_PER_DEVICE=32",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT=1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128",
        "HCCL_BUFFSIZE=200",
        "HCCL_OP_EXPANSION_MODE=AIV",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND=64",
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512",
        "DEEPEP_HCCL_BUFFSIZE=1800",
        "SGLANG_K3_SHARED_EXPERTS_ATTN_TP=1",
        "SGLANG_K3_DENSE_MLP_ATTN_TP=1",
        "SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE=1",
        "SGLANG_RAGGED_VERIFY_MODE=static",
        "SGLANG_DSPARK_FOLDED_PROPOSAL=0",
        "SGLANG_DSPARK_FOLDED_SAMPLING=0",
        "SGLANG_DSPARK_STACKED_CTX_KV=0",
        "SGLANG_DSPARK_EMBED_IN_GRAPH=0",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tokenizer-path {{MODEL_NAME}}",
        "--attention-backend ascend",
        "--device npu",
        "--quantization modelslim",
        "--dtype bfloat16",
        "--tp-size 64",
        "--enable-dp-attention",
        "--dp-size 4",
        "--enable-dp-lm-head",
        "--mem-fraction-static 0.78",
        "--chunked-prefill-size 16384",
        "--cuda-graph-bs-decode 2 4 8 16",
        "--max-running-requests 64",
        "--max-mamba-cache-size 64",
        "--moe-a2a-backend deepep",
        "--deepep-mode auto",
        "--reasoning-parser kimi_k3",
        "--watchdog-timeout 9000",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Ascend 950PR/DT Series: nodes × cards, one rank per card (TP32).
      match: { hw: "a5", pdMode: "unified", strategy: "balanced" },
      nnodes: 4,
      verified: false,
      verificationStatus: "in-progress",
      env: [
        "SGLANG_USE_MODELSCOPE=1",
        "GLOO_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "HCCL_SOCKET_IFNAME={{NETWORK_IFACE}}",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "SGLANG_NPU_USE_FIAS_V2_BSND=True",
        "SGLANG_NPU_FINE_GRAINED_MOE_DUAL_STREAM=True",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1",
        "SGLANG_RAGGED_VERIFY_MODE=static",
        "SGLANG_DSPARK_FOLDED_PROPOSAL=0",
        "SGLANG_DSPARK_FOLDED_SAMPLING=0",
        "SGLANG_DSPARK_STACKED_CTX_KV=0",
        "SGLANG_DSPARK_EMBED_IN_GRAPH=0",
        "STREAMS_PER_DEVICE=32",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT=1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128",
        "HCCL_BUFFSIZE=2000",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND=64",
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512",
        "HCCL_OP_EXPANSION_MODE=AIV",
      ],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tokenizer-path {{MODEL_NAME}}",
        "--attention-backend ascend",
        "--device npu",
        "--dtype bfloat16",
        "--tp-size 32",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--mem-fraction-static 0.9",
        "--chunked-prefill-size 8192",
        "--cuda-graph-bs-decode 32",
        "--max-running-requests 32",
        "--enable-shared-experts-attn-tp",
        "--enable-dense-mlp-attn-tp",
        "--shared-experts-tp-size 4",
        "--reasoning-parser kimi_k3",
        "--moe-a2a-backend deepep",
        "--deepep-mode auto",
        "--linear-attn-verify-backend triton",
        "--disable-radix-cache",
        "--disable-custom-all-reduce",
        "--watchdog-timeout 9000",
        "--model-loader-extra-config '{\"enable_multithread_load\": true}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],

  // Cross-node fabric env (substitute the NIC used by every rank).
  multiNodeHints: {
    b200: [
      // One hint list per hw, shared by every cell, so it has to name the shape per role rather than assume the Unified one.
      "Unified with Spec Decode off runs TP8 within a node and PP2 across the two (+DCP8/EP8 on Balanced and High-Throughput); with DSPARK (pp_size == 1) it runs TP16 across both nodes instead.",
      "Prefill is TP1 × PP16 — one pipeline stage per GPU, non-speculative only. Decode is flat TP16 (+DCP16 on Balanced and High-Throughput).",
      "Multi-node K3 needs the cross-node NIC pinned on BOTH ranks:",
      "  GLOO_SOCKET_IFNAME=<your-nic>   # bootstrap interface",
      "  NCCL_SOCKET_IFNAME=<your-nic>   # force NCCL off kube-ipvs0",
      "  SGLANG_HOST_IP=<this-node-ip>",
      "  NCCL_IB_HCA=<hca0,hca1,...>     # RDMA fabrics only",
    ],
    gb200: [
      "Allocate all four nodes within a single NVL72 domain.",
      "MNNVL is off by default — set on every rank:",
      "  NCCL_MNNVL_ENABLE=1",
      "  NCCL_CUMEM_ENABLE=1",
      "Point the JIT caches at GB200-only paths; GB200 is SM100 and GB300 is SM103,",
      "so the two architectures need separate caches:",
      "  TORCH_EXTENSIONS_DIR / TRITON_CACHE_DIR / TVM_FFI_CACHE_DIR",
    ],
    h100: [
      "Set This node IP separately on each node; use the same cross-node NIC name on all four nodes.",
    ],
    h200: [
      "Low-Latency and Balanced run TP16/EP16 across 2 nodes; Unified High-Throughput widens to TP32/EP32 across 4.",
      "Multi-node K3 needs the cross-node NIC pinned on EVERY node:",
      "  GLOO_SOCKET_IFNAME=<your-nic>   # e.g. bond0",
      "  NCCL_SOCKET_IFNAME=<your-nic>   # force NCCL off kube-ipvs0",
      "  SGLANG_HOST_IP=<this-node-ip>",
    ],
    a3: [
      "Run the same command on all four nodes with --node-rank 0/1/2/3.",
      "NPU collectives use HCCL. Pin the cross-node NIC on EVERY node:",
      "  GLOO_SOCKET_IFNAME=<your-nic>   # bootstrap interface",
      "  HCCL_SOCKET_IFNAME=<your-nic>   # HCCL transport interface",
      "  SGLANG_HOST_IP=<this-node-ip>   # this node's IP on that NIC",
      "If running outside the official image, source set_env.sh on every node first.",
    ],
    a5: [
      "Run the same command on all four nodes with --node-rank 0/1/2/3.",
      "NPU collectives use HCCL. Pin the cross-node NIC on EVERY node:",
      "  GLOO_SOCKET_IFNAME=<your-nic>   # bootstrap interface",
      "  HCCL_SOCKET_IFNAME=<your-nic>   # HCCL transport interface",
      "  SGLANG_HOST_IP=<this-node-ip>   # this node's IP on that NIC",
      "If running outside the official image, source both set_env.sh scripts on every",
      "node first: /usr/local/Ascend/ascend-toolkit/set_env.sh and",
      "/usr/local/Ascend/nnal/atb/set_env.sh.",
    ],
  },
};
