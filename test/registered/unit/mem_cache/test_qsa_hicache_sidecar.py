"""CPU tests for HiCache on QSA models: the compressed-K sidecar, the packed
MTP draft and the PLE side states."""

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
from sglang.srt.runtime_context import get_context, get_parallel
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


def _unified_qsa_pools():
    """Target and MTP draft QSA pools over one unified buffer, like `_qsa_pool`."""
    from sglang.srt.layers.attention.qsa.config import QSAProfile
    from sglang.srt.mem_cache.unified_memory_pool import (
        build_unified_qsa_draft_pool,
        init_unified_mamba_pools,
    )

    with get_parallel().override(attn_dcp_size=1):
        bundle = init_unified_mamba_pools(
            device="cpu",
            kv_cache_dtype=torch.bfloat16,
            head_num=1,
            head_dim=8,
            page_size=PAGE_SIZE,
            start_layer=0,
            end_layer=8,
            is_draft_worker=False,
            use_mla_backend=False,
            mamba_layer_ids=[0],
            full_attention_layer_ids=[3, 7],
            mamba2_cache_params=SimpleNamespace(
                shape=SimpleNamespace(conv=[(3, 8)], temporal=(4, 16, 16)),
                dtype=SimpleNamespace(conv=torch.bfloat16, temporal=torch.float32),
                layers=[0],
            ),
            model_context_len=4 * PAGE_SIZE,
            extra_max_context_len=4,
            max_total_num_tokens=8 * PAGE_SIZE,
            max_mamba_cache_size=2,
            max_num_reqs=4,
            enable_memory_saver=False,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=None,
            disable_overlap_schedule=True,
            need_sort=False,
            qsa_profile=QSAProfile(
                n_heads=4, kv_heads=2, head_dim=16, budget=8, compress_ratio=RATIO
            ),
            draft_layer_num=1,
        )
        draft = build_unified_qsa_draft_pool(
            allocator=bundle.token_to_kv_pool_allocator,
            target_pool=bundle.token_to_kv_pool,
            full_attention_layer_ids=[0],
            mamba_pool=bundle.req_to_token_pool.mamba_pool,
            start_layer=8,
            num_request_slots=bundle.req_to_token_pool.req_to_token.shape[0],
        )
    return bundle.token_to_kv_pool, draft, bundle.token_to_kv_pool_allocator


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


def _reference_transfer_kv_direct_paged(
    src_layers, dst_layers, src_indices, dst_indices, page_size
):
    # Page-granular callers pass whole pages of token indices.
    _reference_transfer_kv_direct(src_layers, dst_layers, src_indices, dst_indices, 1)


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

    def test_unified_pool_round_trip_follows_the_page_table(self):
        """On the unified pool the compressed keys live in the full pages'
        envelopes: a backup by virtual ids and a load into other virtual pages
        must land where `qsa_index_slots` finds them, for target and draft."""
        target, draft, allocator = _unified_qsa_pools()
        _fill_compressed(target, seed=0)
        _fill_compressed(draft, seed=1)
        group = self._build_group(target, drafts=(draft,))
        locs = allocator.alloc(4 * PAGE_SIZE)
        pages = (locs[::PAGE_SIZE] // PAGE_SIZE).tolist()
        src_pages, host_pages, dst_pages = pages[:2], [5, 2], pages[2:]

        def row_ids(pool, page):
            slots = torch.arange(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
            return pool.qsa_index_slots(slots[::RATIO]) // RATIO

        def rows(pool, buf, page):
            return buf[row_ids(pool, page)]

        expected = {
            id(pool): [
                [rows(pool, buf, p).clone() for p in src_pages]
                for buf in pool.qsa_compressed_k_buffer_pool
            ]
            for pool in (target, draft)
        }
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
                    for page in dst_pages:
                        buf[row_ids(pool, page)] = 0
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
                    torch.testing.assert_close(
                        rows(pool, buf, page), expected[id(pool)][layer][i]
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


def _reference_all_layer_direct_lf_pf(
    src_ptrs, dst_ptrs, src_indices, dst_indices, page_size
):
    # One page-first host buffer [slot, layer, 1, ...]; src is one tensor per layer.
    assert page_size == 1 and len(dst_ptrs) == 1
    for layer, src in enumerate(src_ptrs):
        dst_ptrs[0][dst_indices.long(), layer, 0] = src[src_indices.long()]


def _reference_per_layer_direct_pf_lf(
    src_ptrs, dst_ptrs, src_indices, dst_indices, layer_id, page_size
):
    assert page_size == 1
    for src, dst in zip(src_ptrs, dst_ptrs, strict=True):
        dst[dst_indices.long()] = src[src_indices.long(), layer_id, 0]


class TestUnifiedTargetDraftBackup(CustomTestCase):
    """A unified target resolves its backup indices to kernel-facing ids. A
    draft outside its page envelopes is indexed by the untranslated ids; one
    inside them shares the target's translate and is packed with it."""

    def test_each_draft_is_backed_up_with_its_own_ids(self):
        target, draft, host, device_ids, host_ids = self._backup(
            draft_shares_translate=False
        )
        self.assertFalse(host.packs_draft_backup)
        torch.testing.assert_close(
            host.k_buffer[2][host_ids], draft.k_buffer[0][device_ids]
        )
        torch.testing.assert_close(
            host.v_buffer[2][host_ids], draft.v_buffer[0][device_ids]
        )

    def test_envelope_draft_is_packed_with_the_target(self):
        target, draft, host, device_ids, host_ids = self._backup(
            draft_shares_translate=True
        )
        self.assertTrue(host.packs_draft_backup)
        resolved = target.host_transfer_translate(device_ids)
        torch.testing.assert_close(
            host.k_buffer[2][host_ids], draft.k_buffer[0][resolved]
        )
        torch.testing.assert_close(
            host.v_buffer[2][host_ids], draft.v_buffer[0][resolved]
        )

    def _backup(self, *, draft_shares_translate: bool):
        from sglang.srt.mem_cache import l2_transfer
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.mem_cache.pool_host import mha as mha_host

        def kv_pool(layer_num, seed):
            pool = MHATokenToKVPool(
                size=6 * PAGE_SIZE,
                page_size=PAGE_SIZE,
                dtype=torch.bfloat16,
                head_num=1,
                head_dim=8,
                layer_num=layer_num,
                device="cpu",
                enable_memory_saver=False,
            )
            generator = torch.Generator().manual_seed(seed)
            for buf in (*pool.k_buffer, *pool.v_buffer):
                buf.copy_(torch.randn(buf.shape, generator=generator))
            return pool

        target, draft = kv_pool(2, seed=0), kv_pool(1, seed=1)
        # The unified target's views are addressed by a permutation of the ids.
        target.host_transfer_translate = lambda ids: (
            (ids + 2 * PAGE_SIZE) % (6 * PAGE_SIZE)
        )
        if draft_shares_translate:
            draft.host_transfer_translate = target.host_transfer_translate
        with (
            patch.dict(ALLOC_MEMORY_FUNCS, {"cpu": _unpinned_host_alloc}),
            host_memory_budget_scope(1 << 40),
        ):
            host = mha_host.MHATokenToKVPoolHost(
                target,
                2.0,
                0,
                PAGE_SIZE,
                "layer_first",
                pin_memory=False,
                mtp_draft_device_pools=(draft,),
            )
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
        device_ids, host_ids = _token_indices([1]), _token_indices([3])
        with (
            patch.object(
                mha_host,
                "transfer_kv_direct",
                # Token indices of whole pages: the page copy is a row copy.
                _reference_transfer_kv_direct_paged,
                create=True,
            ),
            patch.object(l2_transfer.device_module, "Stream", create=True),
        ):
            L2TransferEngine("direct").submit_device_to_host(
                HybridCacheController._l2_write_transfers(
                    controller, host_ids, device_ids
                )
            )
        resolved = target.host_transfer_translate(device_ids)
        for layer in range(2):
            torch.testing.assert_close(
                host.k_buffer[layer][host_ids], target.k_buffer[layer][resolved]
            )
        return target, draft, host, device_ids, host_ids


class TestPleSideStatesOnHost(CustomTestCase):
    """PLE side states (short-conv window, N-gram context) belong to a state slot;
    a prefix loaded back from host must carry them, not zeroed rows."""

    def test_backup_and_load_carry_the_side_states(self):
        from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
        from sglang.srt.mem_cache.pool_host import mamba as mamba_host

        slots = 8
        short_conv = ShortConvPool(
            size=slots,
            state_shape=(6, 3),
            layer_ids=[1],
            dtype=torch.bfloat16,
            device="cpu",
        )
        ngram = NGramPool(size=slots, context_len=2, eos_token_id=7, device="cpu")
        short_conv.conv_state.copy_(torch.randn(short_conv.conv_state.shape))
        ngram.context.copy_(torch.arange(ngram.context.numel()).view_as(ngram.context))
        views = short_conv.slot_major_views() + ngram.slot_major_views()
        device_pool = SimpleNamespace(
            num_mamba_layers=1,
            size=slots,
            host_capacity_tokens=None,
            device="cpu",
            mamba_cache=SimpleNamespace(
                conv=[torch.zeros(1, slots + 1, 3, 4)],
                temporal=torch.zeros(1, slots + 1, 2, 4, 4),
            ),
            slot_sibling_views=lambda: views,
        )
        with (
            patch.dict(ALLOC_MEMORY_FUNCS, {"cpu": _unpinned_host_alloc}),
            host_memory_budget_scope(1 << 40),
        ):
            host = mamba_host.MambaPoolHost(
                device_pool, 2.0, 0, pin_memory=False, layout="page_first_direct"
            )
        src, host_slots, dst = (
            torch.tensor([1, 2]),
            torch.tensor([3, 4]),
            torch.tensor([5, 6]),
        )
        conv_before = short_conv.conv_state[:, src].clone()
        context_before = ngram.context[src].clone()
        with (
            patch.object(
                mamba_host,
                "transfer_kv_direct",
                _reference_transfer_kv_direct,
                create=True,
            ),
            patch.object(
                mamba_host,
                "transfer_kv_all_layer_direct_lf_pf",
                _reference_all_layer_direct_lf_pf,
                create=True,
            ),
            patch.object(
                mamba_host,
                "transfer_kv_per_layer_direct_pf_lf",
                _reference_per_layer_direct_pf_lf,
                create=True,
            ),
        ):
            host.backup_from_device_all_layer(device_pool, host_slots, src, "direct")
            host.load_to_device_per_layer(device_pool, host_slots, dst, 0, "direct")
        torch.testing.assert_close(short_conv.conv_state[:, dst], conv_before)
        torch.testing.assert_close(ngram.context[dst], context_before)

    def test_ngram_read_waits_for_the_first_state_layer_load(self):
        """The N-gram context is read at embedding time, before any layer runs,
        so it must wait for the load that brings the side states back."""
        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
        from sglang.srt.mem_cache.ple_state_pool import NGramPool

        pool = object.__new__(HybridReqToTokenPool)
        pool.ngram_pool = NGramPool(size=4, context_len=2, eos_token_id=7, device="cpu")
        pool.mamba_map = {12: 0, 13: 1, 15: 2}
        pool.start_layer = 10
        waited = []
        pool.layer_transfer_counter = SimpleNamespace(wait_until=waited.append)
        pool.get_ngram_context(torch.tensor([1]))
        self.assertEqual(waited, [2])


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
