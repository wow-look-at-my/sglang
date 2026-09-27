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


class TestUnifiedTargetHostTransfersGpu(CustomTestCase):
    """The unified pool's host paths through the real transfer kernels."""

    def setUp(self):
        if not (torch.cuda.is_available() and is_cuda()):
            self.skipTest("CUDA is required for the HiCache transfer kernels.")

    def _kv_pool(self, layer_num, seed):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        pool = MHATokenToKVPool(
            size=8 * PAGE_SIZE,
            page_size=PAGE_SIZE,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=64,
            layer_num=layer_num,
            device="cuda",
            enable_memory_saver=False,
        )
        generator = torch.Generator(device="cuda").manual_seed(seed)
        for buf in (*pool.k_buffer, *pool.v_buffer):
            buf.copy_(torch.randn(buf.shape, generator=generator, device="cuda"))
        return pool

    def test_drafts_back_up_with_their_own_ids_on_every_layout(self):
        """Behind a translating target, a packed MTP draft layer outside the
        target's page envelopes must be backed up from the untranslated ids,
        one inside them (sharing the target's translate) from the resolved
        ids, and target layers from resolved ones."""
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

        for io_backend, layout, draft_shares_translate in (
            ("kernel", "layer_first", False),
            ("kernel", "page_first", False),
            ("direct", "layer_first", False),
            ("kernel", "layer_first", True),
            ("kernel", "page_first", True),
            ("direct", "layer_first", True),
        ):
            with self.subTest(
                io_backend=io_backend,
                layout=layout,
                draft_shares_translate=draft_shares_translate,
            ):
                target, draft = self._kv_pool(2, seed=0), self._kv_pool(1, seed=1)
                target.host_transfer_translate = lambda ids: (
                    (ids + 3 * PAGE_SIZE) % (8 * PAGE_SIZE)
                )
                if draft_shares_translate:
                    draft.host_transfer_translate = target.host_transfer_translate
                with host_memory_budget_scope(1 << 40):
                    host = MHATokenToKVPoolHost(
                        target,
                        2.0,
                        0,
                        PAGE_SIZE,
                        layout,
                        mtp_draft_device_pools=(draft,),
                    )
                self.addCleanup(host.destroy)
                controller = SimpleNamespace(
                    mem_pool_host=SimpleNamespace(
                        anchor_entry=SimpleNamespace(
                            host_pool=host, device_pool=target, layer_mapper=None
                        ),
                        entry_map={},
                    )
                )
                controller._l2_transfers = partial(
                    HybridCacheController._l2_transfers, controller
                )
                index_device = "cuda" if io_backend == "kernel" else "cpu"
                device_ids = _token_indices([1, 6], index_device)
                host_ids = _token_indices([2, 5], index_device)
                L2TransferEngine(io_backend).submit_device_to_host(
                    HybridCacheController._l2_write_transfers(
                        controller, host_ids, device_ids
                    )
                )
                torch.cuda.synchronize()
                resolved = target.host_transfer_translate(device_ids.cuda())
                host_rows = host_ids.cpu()
                for layer in range(2):
                    torch.testing.assert_close(
                        host.k_data_refs[layer][host_rows],
                        target.k_buffer[layer][resolved].cpu(),
                    )
                draft_ids = resolved if draft_shares_translate else device_ids.cuda()
                self.assertEqual(host.packs_draft_backup, draft_shares_translate)
                torch.testing.assert_close(
                    host.k_data_refs[2][host_rows], draft.k_buffer[0][draft_ids].cpu()
                )
                torch.testing.assert_close(
                    host.v_data_refs[2][host_rows], draft.v_buffer[0][draft_ids].cpu()
                )

    def test_ple_side_states_round_trip_through_the_kernels(self):
        """The PLE short-conv window (hundreds of KiB per slot) and the N-gram
        context (16 B per slot) ride the state slot through host memory, both
        as their own tensors and inside a unified pool's slot envelopes."""
        for enveloped in (False, True):
            with self.subTest(enveloped=enveloped):
                self._ple_round_trip(enveloped=enveloped)

    def _ple_round_trip(self, *, enveloped: bool):
        from sglang.srt.mem_cache.layout.page_major import build_slot_sibling_views
        from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
        from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost

        slots = 8
        conv_view = context_view = None
        if enveloped:
            # A state envelope of 4 KiB ahead of the side states, as in the pool.
            state_bytes, side_bytes = 4096, 2 * 8 + 640 * 9 * 2
            entry = state_bytes + side_bytes + 8
            raw = torch.zeros((slots + 1) * entry, dtype=torch.uint8, device="cuda")
            context_view, conv_view = build_slot_sibling_views(
                raw,
                layouts=(((2,), torch.int64), ((1, 640, 9), torch.bfloat16)),
                entry_bytes=entry,
                first_offset_bytes=state_bytes,
                max_slots=slots + 1,
            )
            conv_view = conv_view.transpose(0, 1)
        short_conv = ShortConvPool(
            size=slots,
            state_shape=(640, 9),
            layer_ids=[1],
            dtype=torch.bfloat16,
            device="cuda",
            conv_state=conv_view,
        )
        ngram = NGramPool(
            size=slots,
            context_len=2,
            eos_token_id=7,
            device="cuda",
            context=context_view,
        )
        short_conv.conv_state.copy_(torch.randn(short_conv.conv_state.shape))
        ngram.context.copy_(
            torch.arange(ngram.context.numel(), device="cuda").view_as(ngram.context)
        )
        views = short_conv.slot_major_views() + ngram.slot_major_views()
        device_pool = SimpleNamespace(
            num_mamba_layers=1,
            size=slots,
            host_capacity_tokens=None,
            device="cuda",
            mamba_cache=SimpleNamespace(
                conv=[torch.zeros(1, slots + 1, 64, 3, device="cuda")],
                temporal=torch.zeros(1, slots + 1, 2, 16, 16, device="cuda"),
            ),
            slot_sibling_views=lambda: views,
        )
        for io_backend, layout in (
            ("kernel", "page_first"),
            ("direct", "page_first_direct"),
        ):
            with self.subTest(io_backend=io_backend, layout=layout):
                with host_memory_budget_scope(1 << 40):
                    host = MambaPoolHost(device_pool, 2.0, 0, layout=layout)
                self.addCleanup(host.destroy)
                src = torch.tensor([1, 2], device="cuda")
                dst = torch.tensor([5, 6], device="cuda")
                host_slots = torch.tensor([3, 4], device="cuda")
                if io_backend == "direct":
                    host_slots = host_slots.cpu()
                conv_before = short_conv.conv_state[:, src].clone()
                context_before = ngram.context[src].clone()
                host.backup_from_device_all_layer(
                    device_pool, host_slots, src, io_backend
                )
                torch.cuda.synchronize()
                host.load_to_device_per_layer(
                    device_pool, host_slots, dst, 0, io_backend
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(short_conv.conv_state[:, dst], conv_before)
                torch.testing.assert_close(ngram.context[dst], context_before)


if __name__ == "__main__":
    unittest.main()
