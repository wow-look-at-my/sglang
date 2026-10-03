// Single `export const config` literal — no spreads/calls/IIFE (Mintlify re-evals at hydration).
// Cells are denormalized: no `--nnodes`/`--node-rank`/`--dist-init-addr`/`--host`/`--port` literals — engine injects them.
//
// Single-GPU on every supported card, hence one node and no parallelism flags
// in any cell.
//
// Verification per cell is documented above cells[]. A hardware x
// quantization combination with no launch recipe has no cell, and the
// engine greys it out.

export const config = {
  modelName: "Qwen3.8-27B",

  supportedHardware: ["h200", "rtx6000", "rtx5090", "dgx-spark", "gb300"],

  hardware: [
    { id: "rtx6000", label: "RTX PRO 6000", vram: "96GB", vendor: "blackwell" },
    { id: "rtx5090", label: "RTX 5090", vram: "32GB", vendor: "blackwell" },
  ],

  // Every cell pins `--kv-cache-dtype fp8_e4m3` at the maintainers' direction
  // (sign-off recorded in the PR description). Both RadixArk NVFP4 exports: a
  // no-op made visible (their `kv_cache_quant_algo: FP8` already resolved
  // `auto` to fp8_e4m3). BF16/FP8 and the NVIDIA NVFP4 export: a real
  // quality/capacity trade — halves kv_bytes_per_token, and those checkpoints
  // declare no KV scheme at all, so `auto` would leave the pool in bf16.
  matchDims: [
    { id: "variant", title: "Model Variant", options: [
      { id: "default", label: "Default" },
    ] },
    { id: "quant", title: "Quantization", options: [
      { id: "bf16",  label: "BF16"  },
      { id: "fp8",   label: "FP8"   },
      // NVFP4 exports ship separately, differing only in the lm_head: one keeps it dense bf16.
      { id: "nvfp4-bf16-head", label: "NVFP4-BF16-Head" },
      { id: "nvfp4-fp4-head",  label: "NVFP4-FP4-Head"  },
      // NVIDIA's own ModelOpt export of the same W4A4 body: identical quantized-layer map (layers, FP8 attention projections + NVFP4 MLPs).
      { id: "nvfp4-nvidia",    label: "NVFP4-NVIDIA"    },
    ] },
    { id: "nodes", title: "Nodes", options: [
      { id: "single", label: "Single Node" },
    ] },
  ],

  overlayDims: [
    {
      id: "spec",
      title: "Speculative Decoding",
      default: "none",
      options: [
        { id: "none", label: "None" },
        {
          id: "eagle", label: "EAGLE",
          disabled: (sel) => sel.hw === "rtx5090" && !String(sel.quant).startsWith("nvfp4"),
          disableReason:
            "On the 32GB RTX 5090 the MTP head only fits on top of the NVFP4 weights",
          stripPrefixes: (sel) =>
            sel.hw === "rtx5090"
              ? ["--mem-fraction-static", "--mamba-full-memory-ratio",
                 "--max-total-tokens"]
              : [],
          flags: (sel) => [
            "--speculative-algorithm EAGLE",
            "--speculative-num-steps 3",
            "--speculative-eagle-topk 1",
            "--speculative-num-draft-tokens 4",
            // ReplaySSM spec-verify only on SM120/SM121, where it was
            // exercised.
            ...(["rtx5090", "rtx6000", "dgx-spark"].includes(sel.hw)
              ? ["--enable-linear-replayssm-spec"]
              : []),
            ...(sel.hw === "rtx5090"
              ? [sel.ssmDtype === "float32"
                  ? "--mem-fraction-static 0.94"
                  : "--mem-fraction-static 0.93"]
              : []),
            // The dense-lm_head export is the case where replayssm's tiny
            // state pool still is not enough: its head costs ~3.2GB more at
            // runtime, and at fp32 the default split leaves the pool short of
            // its slots.
            ...(sel.hw === "rtx5090" &&
                sel.quant === "nvfp4-bf16-head" &&
                sel.ssmDtype === "float32"
              ? ["--max-total-tokens 16384",
                 sel.tier === "low-latency"
                   ? "--mamba-full-memory-ratio 2.67"
                   : "--mamba-full-memory-ratio 2.14"]
              : []),
          ],
        },
        {
          id: "dspark", label: "DSPARK",
          disabled: (sel) => sel.hw === "rtx5090" && !String(sel.quant).startsWith("nvfp4"),
          disableReason:
            "On the 32GB RTX 5090 the DSpark draft model only fits on top of the NVFP4 weights",
          stripPrefixes: (sel) =>
            sel.hw === "rtx5090"
              ? ["--mem-fraction-static", "--mamba-full-memory-ratio",
                 "--chunked-prefill-size", "--max-total-tokens"]
              : [],
          flags: (sel) => [
            "--speculative-algorithm DSPARK",
            "--speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark",
            "--speculative-draft-attention-backend flashinfer",
            ...(sel.hw === "rtx5090"
              ? [
                  "--max-total-tokens 16384",
                  ...(sel.quant === "nvfp4-bf16-head"
                    ? ["--mem-fraction-static 0.92",
                       "--chunked-prefill-size 512",
                       sel.tier === "low-latency"
                         ? "--mamba-full-memory-ratio 3.38"
                         : "--mamba-full-memory-ratio 2.71"]
                    : sel.ssmDtype === "float32"
                      ? ["--mem-fraction-static 0.91",
                         "--chunked-prefill-size 1024",
                         sel.tier === "low-latency"
                           ? "--mamba-full-memory-ratio 6.94"
                           : "--mamba-full-memory-ratio 5.56"]
                      : ["--mem-fraction-static 0.88",
                         sel.tier === "low-latency"
                           ? "--mamba-full-memory-ratio 3.38"
                           : "--mamba-full-memory-ratio 3.12"]),
                ]
              : []),
          ],
        },
        {
          id: "dflash", label: "DFLASH2",
          // Trained block-diffusion draft, a separate checkpoint.
          disabled: (sel) => sel.hw === "rtx5090" && !String(sel.quant).startsWith("nvfp4"),
          disableReason:
            "On the 32GB RTX 5090 the DFlash2 draft model only fits on top of the NVFP4 weights",
          // fp32 needs the balanced ratio overridden, so that family is
          // stripped as well and re-emitted below.
          stripPrefixes: (sel) =>
            sel.hw === "rtx5090"
              ? sel.ssmDtype === "float32"
                ? ["--mem-fraction-static", "--mamba-full-memory-ratio"]
                : ["--mem-fraction-static"]
              : [],
          flags: (sel) => [
            "--speculative-algorithm DFLASH",
            "--speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2",
            "--speculative-num-draft-tokens 8",
            ...(sel.hw === "rtx5090"
              ? sel.ssmDtype === "float32"
                // FP4-head export, High-Throughput only (the SSM dtype row
                // greys out the Low-Latency tier).
                ? ["--mem-fraction-static 0.895",
                   "--mamba-full-memory-ratio 10"]
                : ["--mem-fraction-static 0.91",
                   "--chunked-prefill-size 1024"]
              : []),
            // The dense-lm_head export carries ~3.2GB more weight at runtime,
            // which is the difference between serving at the pins above and
            // needing the pools pinned outright. Measured on v0.5.19.
            ...(sel.hw === "rtx5090" && sel.quant === "nvfp4-bf16-head"
              ? ["--max-total-tokens 16384",
                 sel.tier === "low-latency"
                   ? "--mamba-full-memory-ratio 3.38"
                   : "--mamba-full-memory-ratio 3.12"]
              : []),
          ],
        },
      ],
    },
    {
      // Serving tier via the GDN radix-cache strategy, the knob that sets S (state slots per running request): extra_buffer S=5 (latency tier).
      id: "tier",
      title: "Serving Strategy",
      default: "low-latency",
      // Owns the flag outright: strip whatever a cell pinned, then re-emit.
      stripPrefixes: ["--mamba-radix-cache-strategy"],
      options: [
        { id: "low-latency", label: "Low-Latency",
          flags: ["--mamba-radix-cache-strategy extra_buffer"] },
        { id: "high-throughput", label: "High-Throughput",
          flags: ["--mamba-radix-cache-strategy extra_buffer_lazy"] },
      ],
    },
    {
      id: "ssmDtype",
      title: "Mamba SSM Dtype",
      default: "float32",
      options: [
        {
          id: "float32", label: "float32",
          disabled: (sel) =>
            sel.hw === "rtx5090" &&
            (sel.quant === "nvfp4-bf16-head"
              ? sel.spec === "dflash" || sel.spec === "dspark"
              : sel.spec === "dflash" && sel.tier === "low-latency"),
          disableReason:
            "On the 32GB RTX 5090 this combination has no fp32 GDN state pool that " +
            "also leaves room for prefill graph capture — use bfloat16",
          flags: ["--mamba-ssm-dtype float32"],
        },
        {
          id: "bfloat16", label: "bfloat16",
          disabled: (sel) => sel.hw === "rtx5090" && !String(sel.quant).startsWith("nvfp4"),
          disableReason:
            "On the 32GB RTX 5090 the bf16 GDN state pool is only a live choice for NVFP4; " +
            "the BF16 and FP8 checkpoints have no serviceable cell on this card",
          flags: ["--mamba-ssm-dtype bfloat16"],
        },
      ],
    },
  ],
  modelNames: {
    "default|bf16":  "Qwen/Qwen3.8-27B",
    "default|fp8":   "Qwen/Qwen3.8-27B-FP8",
    "default|nvfp4-bf16-head": "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead",
    "default|nvfp4-fp4-head":  "RadixArk/Qwen3.8-27B-NVFP4",
    "default|nvfp4-nvidia":    "nvidia/Qwen3.8-27B-NVFP4",
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

  // The source page's measurement protocol, kept reproducible:
  // --random-range-ratio multiple pins ISL exactly; --flush-cache
  // measures cache-cold (bench_serving's `random` prompts are deterministic).
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
    accuracy: {
      gsm8k_pct:
`python3 -m sglang.test.run_eval \\
  --host {{CURL_HOST}} --port {{CURL_PORT}} \\
  --model {{MODEL_NAME}} \\
  --eval-name gsm8k \\
  --num-examples 1319`,
    },
  },

  accuracyLabels: [
    ["gsm8k_pct", "GSM8K", "%"],
  ],

  dockerImages: {
    h200:    "lmsysorg/sglang:latest",
    rtx6000: "lmsysorg/sglang:latest",
    rtx5090: "lmsysorg/sglang:latest",
    "dgx-spark": "lmsysorg/sglang:latest",
    gb300:   "lmsysorg/sglang:latest",
  },

  github: {
    cookbookModel: "Qwen/Qwen3.8-27B",
  },

  playgroundFeatures: {

    // No "Attention Parallelism" card: the page is single-GPU throughout, and
    // a TP knob would desync the ratio calculator (TP1-only geometry).

    // ----- Card: "Parsers" ----- Same parser pair the Qwen3.8 flagship page
    // ships, and baked into every cell: this model is used through agent
    // harnesses.
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser qwen3" },
        { id: "toolCall",  label: "Tool Call Parser", flag: "--tool-call-parser qwen3_coder" },
      ],
    },

    // ----- Card: "Speculative Decoding" -----
    // The in-checkpoint 1-layer MTP head. The source page wrote the preset as
    // `--speculative-algorithm NEXTN`; NEXTN is an alias of EAGLE, so it is
    // normalized here (the Playground strips/derives by the first token, and an
    // alias would survive toggles and double up). DSpark is the trained draft
    // model, a separate checkpoint.
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off (greedy)" },
        { id: "mtp",     label: "EAGLE / MTP",
          flags: ["--speculative-algorithm EAGLE", "--speculative-num-steps 3",
                  "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 4"] },
        { id: "dspark",  label: "DSpark",
          // Same flags as the Deploy panel's DSPARK option, so both paths
          // compose identical commands.
          flags: ["--speculative-algorithm DSPARK",
                  "--speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark",
                  "--speculative-draft-attention-backend flashinfer"] },
        { id: "dflash",  label: "DFlash2",
          // Same trio as the Deploy panel's DFLASH2 option.
          flags: ["--speculative-algorithm DFLASH",
                  "--speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2",
                  "--speculative-num-draft-tokens 8"] },
      ],
    },

    // ----- Card: single-selects over one flag family each -----
    flagSelects: [
      {
        // trtllm_mha is SM100-only, so every cell bakes flashinfer (SM90 and SM120 alike).
        id: "attnBackend", title: "Attention Backend",
        stripPrefixes: ["--attention-backend"],
        options: [
          { id: "flashinfer", label: "FlashInfer (default)",
            flags: ["--attention-backend flashinfer"] },
          { id: "fa3", label: "FlashAttention-3 (SM90 only)",
            flags: ["--attention-backend fa3"] },
          { id: "triton", label: "Triton — MTP fallback on older FlashInfer",
            flags: ["--attention-backend triton"] },
        ],
      },
      {
        id: "kvCacheDtype", title: "KV Cache Precision",
        stripPrefixes: ["--kv-cache-dtype"],
        options: [
          { id: "auto", label: "Auto (checkpoint-declared)" },
          { id: "fp8",  label: "FP8 (E4M3) — halves KV memory", flags: ["--kv-cache-dtype fp8_e4m3"] },
          { id: "bf16", label: "BFloat16", flags: ["--kv-cache-dtype bfloat16"] },
        ],
      },
      {
        id: "mambaSsmDtype", title: "GDN State Precision",
        stripPrefixes: ["--mamba-ssm-dtype"],
        options: [
          { id: "auto", label: "Auto (FP32)" },
          { id: "bf16", label: "BFloat16 — halves state memory", flags: ["--mamba-ssm-dtype bfloat16"] },
        ],
      },
      {
        // Whole-model prefix cache.
        id: "prefixCache", title: "Prefix Cache",
        stripPrefixes: ["--disable-radix-cache"],
        options: [
          { id: "on",  label: "On" },
          { id: "off", label: "Off (S=1)", flags: ["--disable-radix-cache"] },
        ],
      },
      {
        id: "mambaRadix", title: "GDN Radix Cache Strategy",
        showWhen: (b, v, d) => (((v && v.prefixCache) ?? (d && d.prefixCache)) !== "off"),
        stripPrefixes: ["--mamba-radix-cache-strategy"],
        options: [
          { id: "auto",  label: "Auto (extra_buffer, S=5)" },
          { id: "lazy",  label: "extra_buffer_lazy (S=4)", flags: ["--mamba-radix-cache-strategy extra_buffer_lazy"] },
          { id: "nobuf", label: "no_buffer (S=3)",         flags: ["--mamba-radix-cache-strategy no_buffer"] },
        ],
      },
    ],
  },

  // DGX Spark was measured across its whole overlay envelope too, but to a
  // weaker standard (boot-and-serve only — see the cell block comment below).
  //
  // Cells carry NO --mamba-full-memory-ratio: the ratio depends on workload,
  // S, D and kv_bytes_per_token, so the page's calculator computes it live
  // and the engine pins it into the rendered command — which it only does
  // while the cell stays ratio-free (_deployment.jsx `cellWithRatio`).
  cells: [
    {
      // H200 141GB, FP8 blockwise (~28.5GB of weights).
      match: { hw: "h200", variant: "default", quant: "fp8", nodes: "single" },
      verified: true,
      // DFLASH2 has not been exercised on this platform; every other overlay
      // pick keeps this cell's original validation.
      verificationStatus: (sel) =>
        sel.spec === "dflash" ? "in-progress" : "verified",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.85",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 32768",
        "--max-prefill-tokens 32768",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // H200, BF16 reference checkpoint (~54GB of weights).
      match: { hw: "h200", variant: "default", quant: "bf16", nodes: "single" },
      verified: true,
      // DFLASH2 has not been exercised on this platform; every other overlay
      // pick keeps this cell's original validation.
      verificationStatus: (sel) =>
        sel.spec === "dflash" ? "in-progress" : "verified",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.85",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 32768",
        "--max-prefill-tokens 32768",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // The page's headline recipe: NVFP4 W4A4 on the 96GB workstation card, ~16.5GB of weights, fp8 KV auto-enabled.
      match: { hw: "rtx6000", variant: "default", quant: "nvfp4-bf16-head", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.85",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Same recipe as the BF16-head cell above: the FP4 head is smaller.
      match: { hw: "rtx6000", variant: "default", quant: "nvfp4-fp4-head", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.85",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // NVIDIA's ModelOpt export of the same W4A4 body and FP4 lm_head as the RadixArk FP4-head checkpoint above.
      match: { hw: "rtx6000", variant: "default", quant: "nvfp4-nvidia", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.85",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // FP8 blockwise, ~28.5GB of weights — comfortable on 96GB.
      match: { hw: "rtx6000", variant: "default", quant: "fp8", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.85",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // BF16, the reference checkpoint.
      match: { hw: "rtx6000", variant: "default", quant: "bf16", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.85",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "rtx5090", variant: "default", quant: "nvfp4-bf16-head", nodes: "single" },
      verified: true,
      // Rendered with the cell so nobody ships the bs=1 pins into a multi-user deployment unaware.
      warn:
        "This recipe serves ONE request at a time: --max-running-requests 1 " +
        "and --cuda-graph-max-bs-decode 1 pin it to the validated single-stream " +
        "envelope. To handle more concurrent requests, raise both flags " +
        "together and re-derive --mamba-full-memory-ratio (and mem-fraction) " +
        "with the [Mamba ratio calculator](#mamba-ratio-calculator) — on this " +
        "32GB card the GDN state pool, not KV, is what runs out first.",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.9",
        "--attention-backend flashinfer",
        "--max-running-requests 1",
        "--cuda-graph-max-bs-decode 1",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Same recipe as the BF16-head cell above: the FP4 head is smaller.
      match: { hw: "rtx5090", variant: "default", quant: "nvfp4-fp4-head", nodes: "single" },
      verified: true,
      // Rendered with the cell so nobody ships the bs=1 pins into a multi-user deployment unaware.
      warn:
        "This recipe serves ONE request at a time: --max-running-requests 1 " +
        "and --cuda-graph-max-bs-decode 1 pin it to the validated single-stream " +
        "envelope. To handle more concurrent requests, raise both flags " +
        "together and re-derive --mamba-full-memory-ratio (and mem-fraction) " +
        "with the [Mamba ratio calculator](#mamba-ratio-calculator) — on this " +
        "32GB card the GDN state pool, not KV, is what runs out first.",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.9",
        "--attention-backend flashinfer",
        "--max-running-requests 1",
        "--cuda-graph-max-bs-decode 1",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // NVIDIA's ModelOpt export: same body, same FP4 lm_head, same 21.9GB of weights as the RadixArk FP4-head checkpoint, so the 32GB fit.
      match: { hw: "rtx5090", variant: "default", quant: "nvfp4-nvidia", nodes: "single" },
      verified: true,
      // Rendered with the cell so nobody ships the bs=1 pins into a multi-user deployment unaware.
      warn:
        "This recipe serves ONE request at a time: --max-running-requests 1 " +
        "and --cuda-graph-max-bs-decode 1 pin it to the validated single-stream " +
        "envelope. To handle more concurrent requests, raise both flags " +
        "together and re-derive --mamba-full-memory-ratio (and mem-fraction) " +
        "with the [Mamba ratio calculator](#mamba-ratio-calculator) — on this " +
        "32GB card the GDN state pool, not KV, is what runs out first.",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.9",
        "--attention-backend flashinfer",
        "--max-running-requests 1",
        "--cuda-graph-max-bs-decode 1",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // DGX Spark (GB10, SM121): single node, 128GB coherent unified memory
    // shared with the CPU — every checkpoint fits, so all quants get a cell.
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4-bf16-head", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.80",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Same recipe as the BF16-head cell above: the FP4 head is smaller.
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4-fp4-head", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.80",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // NVIDIA's ModelOpt export of the same W4A4 body as the RadixArk FP4-head checkpoint, on that cell's recipe.
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4-nvidia", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.80",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "dgx-spark", variant: "default", quant: "fp8", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.80",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "dgx-spark", variant: "default", quant: "bf16", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.80",
        "--attention-backend flashinfer",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // GB300 (SM103), single 288GB GPU. Base and plain-MTP arms measured on
    // lmsysorg/sglang:dev @ c4271c3fe (attention resolves to engine default
    // — no pin, so cell and measurement see the same kernel). Verified
    // envelope: spec none|eagle at engine-default tier/state dtype.
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4-bf16-head", nodes: "single" },
      verified: true,
      // DFLASH2 has not been exercised on this platform; every other overlay
      // pick keeps this cell's original validation.
      verificationStatus: (sel) =>
        sel.spec === "dflash" ? "in-progress" : "verified",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      // Same recipe as the BF16-head cell above: the FP4 head is smaller.
      match: { hw: "gb300", variant: "default", quant: "nvfp4-fp4-head", nodes: "single" },
      verified: true,
      // DFLASH2 has not been exercised on this platform; every other overlay
      // pick keeps this cell's original validation.
      verificationStatus: (sel) =>
        sel.spec === "dflash" ? "in-progress" : "verified",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "nvfp4-nvidia", nodes: "single" },
      verified: true,
      // DFLASH2 has not been exercised on this platform; every other overlay
      // pick keeps this cell's original validation.
      verificationStatus: (sel) =>
        sel.spec === "dflash" ? "in-progress" : "verified",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "fp8", nodes: "single" },
      verified: true,
      // DFLASH2 has not been exercised on this platform; every other overlay
      // pick keeps this cell's original validation.
      verificationStatus: (sel) =>
        sel.spec === "dflash" ? "in-progress" : "verified",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "gb300", variant: "default", quant: "bf16", nodes: "single" },
      verified: true,
      // DFLASH2 has not been exercised on this platform; every other overlay
      // pick keeps this cell's original validation.
      verificationStatus: (sel) =>
        sel.spec === "dflash" ? "in-progress" : "verified",
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.85",
        "--chunked-prefill-size 2048",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
