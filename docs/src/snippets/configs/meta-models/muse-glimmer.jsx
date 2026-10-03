export const config = {
  modelName: "Muse Glimmer",

  supportedHardware: ["b200", "h200", "rtx5090", "rtx6000", "dgx-spark", "mac"],

  hardware: [
    { id: "rtx5090",   label: "RTX 5090",     vram: "32GB",  vendor: "blackwell" },
    { id: "rtx6000",   label: "RTX PRO 6000", vram: "96GB",  vendor: "blackwell" },
    // dgx-spark is NOT listed here -- it's already in the shared HARDWARE_CATALOG under blackwell (with its multi-node docker flags).
    { id: "mac",       label: "Apple Silicon", vram: "48GB+", vendor: "apple" },
  ],

  variants: [{ id: "default", label: "Default" }],

  quantizations: [
    { id: "bf16",         label: "BF16" },
    { id: "gguf",         label: "GGUF Q4_K_M" },
    { id: "nvfp4",        label: "NVFP4" },
    // MLX artifacts, all Apple-Silicon-only. gs128 is the one with a measured GSM8K / CIMemories round (see the cookbook §3.4 table).
    { id: "mlx-q4",       label: "MLX Q4" },
    { id: "mlx-q4km",     label: "MLX Q4_K_M (gs128)" },
    { id: "mlx-q4k-dyn",  label: "MLX Q4_K (dynamic)" },
  ],

  // No Docker path on Apple Silicon — the MLX backend runs native-only.
  runModes: (s) => (s.hw === "mac" ? ["python"] : ["python", "docker"]),

  strategies: [
    { id: "standard", label: "Standard" },
    { id: "dflash",   label: "DFlash" },
  ],

  nodesOptions: [{ id: "single", label: "Single Node" }],

  overlayDims: [
    {
      id: "modality",
      title: "Modality",
      default: "text",
      // GGUF and every MLX artifact are text-only; no modality choice there.
      showWhen: (s) => s.quant !== "gguf" && !(s.quant || "").startsWith("mlx-"),
      options: [
        // NVFP4 ships with no vision weights despite config.json declaring vision_config.
        { id: "mm", label: "Image + text", showWhen: (s) => s.quant === "bf16" },
        {
          id: "text",
          label: "Text only",
          flags: ["--language-model-only"],
        },
      ],
    },
  ],

  modelNames: {
    "default|bf16": "meta-models/Muse-Glimmer-30B",
    "default|gguf": "meta-models/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
    "default|nvfp4": "RadixArk/Muse-Glimmer-NVFP4",
    "default|mlx-q4": "RadixArk/Muse-Glimmer-q4-MLX",
    "default|mlx-q4km": "RadixArk/Muse-Glimmer-q4km-gs128-MLX",
    "default|mlx-q4k-dyn": "RadixArk/Muse-Glimmer-q4k-dynamic-MLX",
  },

  placeholders: {
    HOST_IP:    { target: "command", label: "Bind host",         default: "0.0.0.0" },
    PORT:       { target: "command", label: "Bind port",         default: "30000" },
    HF_TOKEN:   { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    DRAFT_PATH: { target: "command", label: "DFlash draft checkpoint", default: "meta-models/Muse-Glimmer-30B-assistant" },
    QUANT_PATH: { target: "command", label: "NVFP4 checkpoint", default: "RadixArk/Muse-Glimmer-NVFP4" },
    CURL_HOST:  { target: "curl",    label: "Server host",       default: "localhost" },
    CURL_PORT:  { target: "curl",    label: "Server port",       default: "30000" },
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
    numPromptsByConc: { 1: 8, 16: 32, 64: 128, 256: 512, 1024: 2048, 4096: 4096 },
  },

  // No `accuracy` template and no `accuracyLabels`: no benchmarks entry carries
  // accuracy data, so the Reproduce modal renders speed only. Re-add both
  // together (matching keys) when an eval round lands.

  dockerImages: {
    b200:    "lmsysorg/sglang:dev-muse-glimmer",
    h200:    "lmsysorg/sglang:dev-muse-glimmer",
    rtx5090: "lmsysorg/sglang:dev-muse-glimmer",
    rtx6000: "lmsysorg/sglang:dev-muse-glimmer",
    // GB10 needs an aarch64 manifest under this tag to resolve natively
    "dgx-spark": "lmsysorg/sglang:dev-muse-glimmer",
  },

  github: {
    cookbookModel: "meta-models/Muse-Glimmer-30B",
  },

  playgroundFeatures: {
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 1, 2] },
      ],
    },

    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off",     label: "Off" },
        { id: "dflash",  label: "DFlash",
          flags: ["--speculative-algorithm DFLASH",
                  "--speculative-draft-model-path {{DRAFT_PATH}}"] },
      ],
    },
  },

  cells: [
    {
      match: { hw: "rtx5090", variant: "default", quant: "gguf", strategy: "standard", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "rtx5090", variant: "default", quant: "gguf", strategy: "dflash", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path {{DRAFT_PATH}}",
        "--speculative-draft-load-format auto",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    {
      match: { hw: "rtx5090", variant: "default", quant: "nvfp4", strategy: "standard", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{QUANT_PATH}}",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    // Margin is tight though -- ~1.1GB free after CUDA graph capture --
    // so this is the ceiling, not a value with headroom to spare.
    {
      match: { hw: "rtx5090", variant: "default", quant: "nvfp4", strategy: "dflash", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{QUANT_PATH}}",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path {{DRAFT_PATH}}",
        "--speculative-draft-model-quantization fp8",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.9",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    {
      match: { hw: "rtx6000", variant: "default", quant: "bf16", strategy: "standard", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "rtx6000", variant: "default", quant: "bf16", strategy: "dflash", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path {{DRAFT_PATH}}",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "rtx6000", variant: "default", quant: "nvfp4", strategy: "standard", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{QUANT_PATH}}",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "rtx6000", variant: "default", quant: "nvfp4", strategy: "dflash", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{QUANT_PATH}}",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path {{DRAFT_PATH}}",
        "--speculative-draft-model-quantization fp8",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    // GB10: 128GB unified CPU/GPU memory -- the mem-fraction values in the
    // dgx-spark cells below are deliberately lower than the discrete-GPU
    // cells (the OS and client share the same pool; 0.85-style fractions
    // OOM the box during load).
    {
      match: { hw: "dgx-spark", variant: "default", quant: "bf16", strategy: "standard", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--mem-fraction-static 0.75",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "dgx-spark", variant: "default", quant: "bf16", strategy: "dflash", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path {{DRAFT_PATH}}",
        "--mem-fraction-static 0.65",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "standard", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{QUANT_PATH}}",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.40",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "dflash", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{QUANT_PATH}}",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path {{DRAFT_PATH}}",
        "--speculative-draft-model-quantization fp8",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.38",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "standard", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "dflash", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path {{DRAFT_PATH}}",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    {
      match: { hw: "mac", variant: "default", quant: "mlx-q4", strategy: "standard", nodes: "single" },
      verified: false,
      env: ["SGLANG_USE_MLX=1", "SGLANG_MLX_CACHE_LIMIT_GB=8"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--disable-radix-cache",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mac", variant: "default", quant: "mlx-q4km", strategy: "standard", nodes: "single" },
      verified: true,
      env: ["SGLANG_USE_MLX=1", "SGLANG_MLX_CACHE_LIMIT_GB=8"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--disable-radix-cache",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "mac", variant: "default", quant: "mlx-q4k-dyn", strategy: "standard", nodes: "single" },
      verified: false,
      env: ["SGLANG_USE_MLX=1", "SGLANG_MLX_CACHE_LIMIT_GB=8"],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--trust-remote-code",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--disable-radix-cache",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },

    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "standard", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "dflash", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{MODEL_NAME}}",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path {{DRAFT_PATH}}",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "standard", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{QUANT_PATH}}",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "dflash", nodes: "single" },
      verified: true,
      env: [],
      flags: [
        "--model-path {{QUANT_PATH}}",
        "--reasoning-parser muse",
        "--tool-call-parser muse",
        "--speculative-algorithm DFLASH",
        "--speculative-draft-model-path {{DRAFT_PATH}}",
        "--speculative-draft-model-quantization fp8",
        "--kv-cache-dtype fp8_e4m3",
        "--mem-fraction-static 0.85",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
