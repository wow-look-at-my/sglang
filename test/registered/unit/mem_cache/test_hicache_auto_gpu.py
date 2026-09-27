"""Automatic HiCache pinning against the real CUDA host-registration API."""

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache import hicache_auto as auto
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.common import (
    _cuda_host_register,
    _cuda_host_unregister,
    probe_host_registration,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context, get_memory
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

PAGE_SIZE = 16


def _is_registered(buffer: torch.Tensor) -> bool:
    """A second registration of a pinned range fails; of an unpinned one, succeeds."""
    cudart = torch.cuda.cudart()
    size = buffer.numel() * buffer.element_size()
    rc = int(cudart.cudaHostRegister(buffer.data_ptr(), size, 0))
    if rc == 0:
        cudart.cudaHostUnregister(buffer.data_ptr())
        return False
    return True


class TestAutoHiCachePinningGpu(CustomTestCase):
    def setUp(self):
        if not (torch.cuda.is_available() and is_cuda()):
            self.skipTest("CUDA is required for host registration.")
        override = get_context().override_server_args(
            enable_hierarchical_cache=True, hicache_mem_layout="layer_first"
        )
        override.install()
        self.addCleanup(override.restore)

    def test_probe_pins_and_releases(self):
        self.assertIsNone(probe_host_registration())

    def test_aborted_attempt_unpins_and_the_retry_keeps_its_pool(self):
        """A peer's failure on the first attempt unpins this rank's host pool;
        the half-size retry commits a pool that stays pinned."""
        device_pool = MHATokenToKVPool(
            size=4096,
            page_size=PAGE_SIZE,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=64,
            layer_num=2,
            device="cuda",
            enable_memory_saver=False,
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Mock(get_kvcache=Mock(return_value=device_pool)),
            page_size=PAGE_SIZE,
        )
        built = []

        def attach(tree_cache, gate):
            def build():
                host = MHATokenToKVPoolHost(
                    device_pool,
                    get_memory().hicache_ratio,
                    0,
                    PAGE_SIZE,
                    "layer_first",
                    pin_memory=True,
                    device="cpu",
                )
                built.append((get_memory().hicache_ratio, host))
                return host

            gate.run(build)

        gate_calls = []

        def all_reduce(value, op):
            # The first commit vote reports one failed peer; sizing is unanimous.
            if op == torch.distributed.ReduceOp.SUM:
                gate_calls.append(value)
                return value + (1 if len(gate_calls) == 1 else 0)
            return value

        with patch.object(auto, "_all_reduce", side_effect=all_reduce):
            _, enabled = auto.build_tree_cache_with_auto_hicache(
                create_tree_cache=lambda: Mock(spec=UnifiedRadixCache),
                attach_hicache=attach,
                params=params,
                draft_plan=None,
            )

        self.assertTrue(enabled)
        self.assertEqual(len(built), 2)
        (first_ratio, first), (second_ratio, second) = built
        self.assertEqual(second_ratio, first_ratio / 2)
        self.assertFalse(_is_registered(first.kv_buffer))
        self.assertTrue(_is_registered(second.kv_buffer))
        second.destroy()

    def test_rollback_tolerates_a_buffer_its_pool_already_released(self):
        buffer = torch.empty(1 << 20, dtype=torch.uint8)
        _cuda_host_register(buffer)
        _cuda_host_unregister(buffer)
        _cuda_host_unregister(buffer)
        self.assertFalse(_is_registered(buffer))


if __name__ == "__main__":
    unittest.main()
