"""CPU tests for the QSA compressed-K HiCache sidecar."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache import hicache_auto_size as sizing
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.l2_transfer import L2TransferEngine
from sglang.srt.mem_cache.pool_host import dsa as dsa_host
from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS
from sglang.srt.mem_cache.pool_host.qsa import (
    QSACompressedKPoolHost,
    qsa_compressed_bytes,
)
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

PAGE_SIZE = 64
RATIO = 4
ROWS_PER_PAGE = PAGE_SIZE // RATIO


def _qsa_pool(full_attention_layer_ids=(3, 7), num_pages=6, start_layer=None):
    return QSATokenToKVPool(
        size=PAGE_SIZE * num_pages,
        dtype=torch.bfloat16,
        page_size=PAGE_SIZE,
        head_num=1,
        head_dim=8,
        full_attention_layer_ids=list(full_attention_layer_ids),
        device="cpu",
        mamba_pool=None,
        qsa_index_kv_heads=2,
        qsa_index_head_dim=16,
        qsa_compress_ratio=RATIO,
        qsa_token_topk=8,
        num_request_slots=4,
        start_layer=start_layer,
    )


def _token_indices(pages):
    return torch.cat(
        [torch.arange(p * PAGE_SIZE, (p + 1) * PAGE_SIZE) for p in pages]
    ).to(torch.int64)


def _reference_transfer_kv_direct(
    src_layers, dst_layers, src_indices, dst_indices, page_size
):
    # sgl_kernel.kvcacheio.transfer_kv_direct: row copy per layer pair.
    assert page_size == 1
    for src, dst in zip(src_layers, dst_layers, strict=True):
        dst[dst_indices.long()] = src[src_indices.long()]


def _unpinned_host_alloc(dims, dtype, device, pin_memory, allocator, **_):
    # CPU-only torch cannot register host memory with CUDA.
    return torch.zeros(dims, dtype=dtype, device=device)


def _fill_compressed(pool, seed):
    generator = torch.Generator().manual_seed(seed)
    for buffer in pool.qsa_compressed_k_buffer_pool:
        buffer.copy_(torch.randn(buffer.shape, generator=generator))


class _StubHostPool:
    """Anchor/Mamba stand-in: the QSA sidecar only borrows the anchor's slots."""

    layout = "layer_first"
    device = "cpu"
    can_use_write_back_jit = False
    size_per_token = 0

    def __init__(self, num_host_pages=8):
        self.page_size = PAGE_SIZE
        self.page_num = num_host_pages
        self.size = self.logical_size = num_host_pages * PAGE_SIZE

    def prepare_transfer_indices(self, host_indices, device_indices, io_backend):
        return host_indices, device_indices

    def backup_from_device_all_layer_physical(self, *args, **kwargs):
        pass

    def load_to_device_per_layer_physical(self, *args, **kwargs):
        pass


class TestQsaCompressedHostRoundTrip(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            hicache_mem_layout="layer_first", hicache_io_backend="direct"
        )
        override.install()
        self.addCleanup(override.restore)

    def _build_group(self, target, drafts):
        params = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                mamba_allocator=SimpleNamespace(alloc=None, free=None)
            ),
            mtp_draft_device_pools=drafts,
            page_size=PAGE_SIZE,
            token_to_kv_pool_allocator=None,
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
        )
        with (
            patch.object(assembler, "build_kv_host_pool", return_value=_StubHostPool()),
            patch.object(assembler, "MambaPoolHost", return_value=_StubHostPool()),
            patch.object(assembler, "HybridCacheController"),
            patch.dict(ALLOC_MEMORY_FUNCS, {"cpu": _unpinned_host_alloc}),
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
        return group

    @staticmethod
    def _sidecar_transfers(group, host_pages, device_pages):
        transfer = PoolTransfer(
            name=PoolName.QSA_COMPRESSED_K, indices_from_pool=PoolName.KV
        )
        return group.resolve_host_transfers(
            [transfer],
            primary_device_indices=_token_indices(device_pages),
            primary_host_indices=_token_indices(host_pages),
        )

    def test_load_back_into_new_pages_reads_the_backed_up_groups(self):
        """A page loaded into a different device page must expose, at
        compressed slot new_full_slot // ratio, the keys that were at
        old_full_slot // ratio when it was backed up -- for every target
        layer and for the packed MTP draft."""
        target = _qsa_pool(full_attention_layer_ids=(3, 7))
        draft = _qsa_pool(full_attention_layer_ids=(0,))
        _fill_compressed(target, seed=0)
        _fill_compressed(draft, seed=1)
        group = self._build_group(target, drafts=(draft,))
        src_pages, host_pages, dst_pages = [1, 4], [5, 2], [3, 2]
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
        # Stage-local transfer IDs span the full-attention layers 3 and 7.
        controller = SimpleNamespace(mem_pool_host=group, transfer_layer_id_max=8)
        controller._l2_transfers = partial(
            HybridCacheController._l2_transfers, controller
        )
        no_anchor_indices = torch.empty(0, dtype=torch.int64)
        engine = L2TransferEngine("direct")

        with patch.object(
            dsa_host, "transfer_kv_direct", _reference_transfer_kv_direct, create=True
        ):
            engine.submit_device_to_host(
                controller._l2_transfers(
                    no_anchor_indices,
                    no_anchor_indices,
                    self._sidecar_transfers(group, host_pages, src_pages),
                )
            )
            for pool in (target, draft):
                for buf in pool.qsa_compressed_k_buffer_pool:
                    buf.zero_()
            engine.submit_host_to_device(
                HybridCacheController._l2_load_transfers(
                    controller,
                    no_anchor_indices,
                    no_anchor_indices,
                    self._sidecar_transfers(group, host_pages, dst_pages),
                ),
                transfer_layer_id_max=controller.transfer_layer_id_max,
            )

        for pool in (target, draft):
            for layer, buf in enumerate(pool.qsa_compressed_k_buffer_pool):
                for i, page in enumerate(dst_pages):
                    full_slots = torch.arange(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
                    self.assertTrue(
                        torch.equal(
                            buf[full_slots[::RATIO] // RATIO],
                            expected[id(pool)][layer][i],
                        ),
                        f"pool={'draft' if pool is draft else 'target'} "
                        f"layer={layer} page={page}",
                    )

    def test_draft_geometry_must_match_target(self):
        with self.assertRaisesRegex(ValueError, "one compressed-K layer"):
            with host_memory_budget_scope(1 << 40):
                QSACompressedKPoolHost(
                    _qsa_pool(),
                    _StubHostPool(),
                    "layer_first",
                    mtp_draft_device_pools=(
                        _qsa_pool(full_attention_layer_ids=(0, 1)),
                    ),
                    pin_memory=False,
                )


class TestQsaCompressedLayerWait(CustomTestCase):
    def test_reading_compressed_keys_waits_for_that_layers_load(self):
        """The indexer reads compressed keys before attention reads KV, so the
        read itself must wait for the layer's HiCache load-back."""
        pool = _qsa_pool(full_attention_layer_ids=(13, 17), start_layer=10)
        waited = []
        pool.register_layer_transfer_counter(SimpleNamespace(wait_until=waited.append))
        pool.get_qsa_compressed_k_buffer(17)
        self.assertEqual(waited, [7])


class TestQsaStrategySelection(CustomTestCase):
    def test_qsa_pool_selects_the_sidecar_strategy(self):
        pool = _qsa_pool()
        strategy = assembler._select_strategy(
            pool, {ComponentType.FULL, ComponentType.MAMBA}
        )
        self.assertIsInstance(strategy, assembler._QsaMambaStrategy)
        self.assertIs(strategy._qsa_pool(pool), pool)

    def test_mixed_qsa_and_plain_packed_drafts_are_rejected(self):
        with self.assertRaises(NotImplementedError):
            assembler._qsa_draft_device_pools((_qsa_pool(), object()))


class TestQsaHostSizing(CustomTestCase):
    def test_auto_size_counts_compressed_keys(self):
        pool = _qsa_pool()
        kv_bytes = sum(pool.full_kv_pool.get_kv_size_bytes())
        self.assertGreater(qsa_compressed_bytes(pool), 0)
        self.assertEqual(
            sizing._pool_bytes(pool), kv_bytes + qsa_compressed_bytes(pool)
        )


if __name__ == "__main__":
    unittest.main()
