"""Low-level primitives used by the CUDA graph backends.

Subpackages:
  - breakable_cuda_graph: BreakableCUDAGraph + capture context,
    eager_on_graph decorator, is_in_breakable_cuda_graph flag.
  - piecewise_cuda_graph: shared piecewise context manager
    (set_tc_piecewise_forward_context, is_in_tc_piecewise_cuda_graph).

Backends in cuda_graph_backend/ import from here. Runners do not.
"""

import torch

_OPEN_ISSUE_MSG = (
    "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
)
# Lowercased; cuBLAS reports a failed workspace allocation as NOT_INITIALIZED.
_OOM_MARKERS = (
    "out of memory",
    "cublas_status_alloc_failed",
    "cublas_status_not_initialized",
)
# Lowercased; the CUDA error text of the custom all-reduce kernels names their source file.
_CUSTOM_AR_IPC_MARKERS = ("custom_all_reduce", "custom all-reduce", "cudaipc")


def cuda_graph_capture_failed_msg(error: BaseException) -> str:
    """Remedy hint for a decode-style CUDA graph capture failure, matched to its cause."""
    text = str(error).lower()
    if isinstance(error, torch.OutOfMemoryError) or any(
        marker in text for marker in _OOM_MARKERS
    ):
        return (
            "Possible solutions:\n"
            "1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
            "2. set --cuda-graph-max-bs-decode to a smaller value (e.g., 16)\n"
            "3. disable decode CUDA graph by --cuda-graph-backend-decode=disabled. "
            "(Not recommended. Huge performance loss)\n" + _OPEN_ISSUE_MSG
        )
    if any(marker in text for marker in _CUSTOM_AR_IPC_MARKERS):
        return (
            "The failure came from custom all-reduce sharing buffers between GPUs "
            "through CUDA IPC, not from a GPU memory shortage.\n" + _OPEN_ISSUE_MSG
        )
    return (
        "This is not an out-of-memory error, so lowering --mem-fraction-static or "
        "--cuda-graph-max-bs-decode will not help.\n" + _OPEN_ISSUE_MSG
    )


PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG = (
    "Fail when using backend: {backend} for prefill runner.\n"
    "Possible suggestions:\n"
    "{suggestions}"
    "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
)
