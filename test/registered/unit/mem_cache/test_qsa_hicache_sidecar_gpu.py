"""QSA compressed-K HiCache round trip through the real transfer kernels."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.l2_transfer import L2TransferEngine
from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.runtime_context import get_context
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

PAGE_SIZE = 64
RATIO = 4
ROWS_PER_PAGE = PAGE_SIZE // RATIO
LAYOUTS = (
    ("kernel", "layer_first"),
    ("kernel", "page_first"),
    ("direct", "layer_first"),
    ("direct", "page_first_direct"),
)


def _qsa_pool(full_attention_layer_ids, num_pages=8):
    return QSATokenToKVPool(
        size=PAGE_SIZE * num_pages,
        dtype=torch.bfloat16,
        page_size=PAGE_SIZE,
        head_num=2,
        head_dim=64,
        full_attention_layer_ids=list(full_attention_layer_ids),
        device="cuda",
        mamba_pool=None,
        qsa_index_kv_heads=1,
        qsa_index_head_dim=128,
        qsa_compress_ratio=RATIO,
        qsa_token_topk=64,
        num_request_slots=4,
    )


def _token_indices(pages, device):
    return torch.cat(
        [torch.arange(p * PAGE_SIZE, (p + 1) * PAGE_SIZE) for p in pages]
    ).to(device=device, dtype=torch.int64)


class _AnchorHost:
    """Anchor stand-in; the sidecar only borrows the anchor's slot space."""

    device = "cpu"
    can_use_write_back_jit = False
    size_per_token = 0

    def __init__(self, layout, num_host_pages=8):
        self.layout = layout
        self.page_size = PAGE_SIZE
        self.page_num = num_host_pages
        self.size = self.logical_size = num_host_pages * PAGE_SIZE

    def prepare_transfer_indices(self, host_indices, device_indices, io_backend):
        return host_indices, device_indices

    def backup_from_device_all_layer_physical(self, *args, **kwargs):
        pass

    def load_to_device_per_layer_physical(self, *args, **kwargs):
        pass


class TestQsaCompressedHiCacheGpu(CustomTestCase):
    def setUp(self):
        if not (torch.cuda.is_available() and is_cuda()):
            self.skipTest("CUDA is required for the HiCache transfer kernels.")

    def _run(self, io_backend, layout):
        override = get_context().override_server_args(
            hicache_mem_layout=layout, hicache_io_backend=io_backend
        )
        override.install()
        self.addCleanup(override.restore)

        target = _qsa_pool(full_attention_layer_ids=(3, 7))
        draft = _qsa_pool(full_attention_layer_ids=(0,))
        for seed, pool in enumerate((target, draft)):
            generator = torch.Generator(device="cuda").manual_seed(seed)
            for buf in pool.qsa_compressed_k_buffer_pool:
                buf.copy_(torch.randn(buf.shape, generator=generator, device="cuda"))

        params = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                mamba_allocator=SimpleNamespace(alloc=None, free=None)
            ),
            mtp_draft_device_pools=(draft,),
            page_size=PAGE_SIZE,
            token_to_kv_pool_allocator=None,
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
        )
        with (
            patch.object(
                assembler, "build_kv_host_pool", return_value=_AnchorHost(layout)
            ),
            patch.object(assembler, "MambaPoolHost", return_value=_AnchorHost(layout)),
            patch.object(assembler, "HybridCacheController"),
            host_memory_budget_scope(1 << 40),
        ):
            group, _ = assembler.build_hybrid_mamba_stack(
                params=params,
                kv_pool=target.full_kv_pool,
                mamba_pool=object(),
                full_layer_mapping=assembler._stage_local_layer_mapping(
                    target.full_attention_layer_id_mapping, target.start_layer
                ),
                mamba_layer_mapping={0: 0},
                load_cache_event=None,
                storage_backend=None,
                use_mla=False,
                qsa_pool=target,
            )
        host_pool = group.get_pool(PoolName.QSA_COMPRESSED_K)
        self.addCleanup(host_pool.destroy)

        src_pages, host_pages, dst_pages = [1, 5], [2, 6], [7, 4]
        expected = {
            id(pool): [
                [
                    buf[p * ROWS_PER_PAGE : (p + 1) * ROWS_PER_PAGE].clone()
                    for p in src_pages
                ]
                for buf in pool.qsa_compressed_k_buffer_pool
            ]
            for pool in (target, draft)
        }
        host_device = "cuda" if io_backend == "kernel" else "cpu"

        def sidecar_transfers(device_pages):
            return group.resolve_host_transfers(
                [
                    PoolTransfer(
                        name=PoolName.QSA_COMPRESSED_K,
                        indices_from_pool=PoolName.KV,
                    )
                ],
                primary_device_indices=_token_indices(device_pages, host_device),
                primary_host_indices=_token_indices(host_pages, host_device),
            )

        controller = SimpleNamespace(mem_pool_host=group, transfer_layer_id_max=8)
        controller._l2_transfers = partial(
            HybridCacheController._l2_transfers, controller
        )
        empty = torch.empty(0, dtype=torch.int64, device=host_device)
        engine = L2TransferEngine(io_backend)
        engine.submit_device_to_host(
            controller._l2_transfers(empty, empty, sidecar_transfers(src_pages))
        )
        torch.cuda.synchronize()
        for pool in (target, draft):
            for buf in pool.qsa_compressed_k_buffer_pool:
                buf.zero_()
        engine.submit_host_to_device(
            HybridCacheController._l2_load_transfers(
                controller, empty, empty, sidecar_transfers(dst_pages)
            ),
            transfer_layer_id_max=controller.transfer_layer_id_max,
        )
        torch.cuda.synchronize()

        for pool in (target, draft):
            for layer, buf in enumerate(pool.qsa_compressed_k_buffer_pool):
                for i, page in enumerate(dst_pages):
                    full_slots = torch.arange(
                        page * PAGE_SIZE, (page + 1) * PAGE_SIZE, device="cuda"
                    )
                    self.assertTrue(
                        torch.equal(
                            buf[full_slots[::RATIO] // RATIO],
                            expected[id(pool)][layer][i],
                        ),
                        f"{io_backend}/{layout} "
                        f"pool={'draft' if pool is draft else 'target'} "
                        f"layer={layer} page={page}",
                    )

    def test_round_trip_into_different_device_pages(self):
        """Compressed keys backed up from one set of device pages must read
        back, at new_full_slot // ratio, after loading into other pages."""
        for io_backend, layout in LAYOUTS:
            with self.subTest(io_backend=io_backend, layout=layout):
                self._run(io_backend, layout)


if __name__ == "__main__":
    unittest.main()
