"""The unified memory pool with QSA's pools: compressed index keys and an MTP
draft in each full page's envelope, PLE side states in each state slot's.

Every byte the static pools spend per token or per state slot lives in the
envelope it belongs to, so the shared buffer is the static footprint rearranged:
it holds at least the static KV whatever the state occupancy, and compaction
moves a page's keys and draft rows, and a slot's side states, with it.
"""

import types
import unittest

import torch

from sglang.srt.layers.attention.qsa.config import QSAProfile
from sglang.srt.layers.attention.qsa.metadata import build_pending_ring_slots
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.unified_memory_pool import init_unified_mamba_pools
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_PAGE = 8
_RATIO = 4
_STATE_SLOTS = 4
_BUDGET = 1 << 20
_PROFILE = QSAProfile(n_heads=4, kv_heads=1, head_dim=8, budget=16, compress_ratio=4)
_PLE = dict(
    short_conv_layer_ids=[0],
    short_conv_state_shape=(6, 3),
    ngram_context_len=2,
    ngram_eos_token_id=7,
)
# Per token and layer: K + V rows of 1 head x 8 dims in bf16, and one 8-dim bf16
# index key per ratio tokens.
_KV_PER_LAYER = 2 * 8 * 2
_INDEX_PER_LAYER = 8 * 2 // _RATIO


def _cache_params():
    return types.SimpleNamespace(
        shape=types.SimpleNamespace(conv=[(3, 8)], temporal=(4, 16, 16)),
        dtype=types.SimpleNamespace(conv=torch.bfloat16, temporal=torch.float32),
        layers=[0],
    )


def _static_cell(draft_layers: int) -> int:
    """What the static configurator charges per token (target + draft)."""
    return (1 + draft_layers) * (_KV_PER_LAYER + _INDEX_PER_LAYER)


def _static_kv_tokens(cell: int) -> int:
    return _BUDGET // cell // _PAGE * _PAGE


def _build(**over):
    kw = dict(
        device="cpu",
        kv_cache_dtype=torch.bfloat16,
        head_num=1,
        head_dim=8,
        page_size=_PAGE,
        start_layer=0,
        end_layer=2,
        is_draft_worker=False,
        use_mla_backend=False,
        mamba_layer_ids=[0],
        full_attention_layer_ids=[1],
        mamba2_cache_params=_cache_params(),
        model_context_len=64,
        extra_max_context_len=4,
        max_total_num_tokens=_static_kv_tokens(_static_cell(0)),
        max_mamba_cache_size=_STATE_SLOTS,
        max_num_reqs=4,
        enable_memory_saver=False,
        enable_mamba_extra_buffer=False,
        speculative_num_draft_tokens=None,
        disable_overlap_schedule=True,
        need_sort=False,
        qsa_profile=_PROFILE,
        ple_req_pool_kwargs=_PLE,
        unified_total_bytes=_BUDGET,
        token_cell_bytes=_static_cell(0),
    )
    kw.update(over)
    return init_unified_mamba_pools(**kw)


def _build_with_draft():
    bundle = _build(
        draft_layer_num=1,
        token_cell_bytes=_static_cell(1),
        max_total_num_tokens=_static_kv_tokens(_static_cell(1)),
    )
    # Imported here so the capacity tests also run against trees without it.
    from sglang.srt.mem_cache.unified_memory_pool import build_unified_qsa_draft_pool

    draft = build_unified_qsa_draft_pool(
        allocator=bundle.token_to_kv_pool_allocator,
        target_pool=bundle.token_to_kv_pool,
        full_attention_layer_ids=[0],
        mamba_pool=bundle.req_to_token_pool.mamba_pool,
        start_layer=2,
        num_request_slots=bundle.req_to_token_pool.req_to_token.shape[0],
    )
    return bundle, draft


def _kv_tokens_with_state_slots(bundle, state_slots: int) -> int:
    """Tokens the full side still hands out with ``state_slots`` states resident."""
    slots = bundle.req_to_token_pool.mamba_allocator
    allocator = bundle.token_to_kv_pool_allocator
    for _ in range(state_slots):
        assert slots.alloc(1) is not None, "state slot allocation failed"
    tokens = 0
    while allocator.alloc(_PAGE) is not None:
        tokens += _PAGE
    return tokens


class _UnifiedPoolCase(unittest.TestCase):
    def setUp(self):
        self.enterContext(get_parallel().override(attn_dcp_size=1))


class TestUnifiedKVCapacityVsStatic(_UnifiedPoolCase):
    def test_unified_kv_capacity_at_full_state_occupancy_is_at_least_static(self):
        """With every configured state slot resident the unified pool must still
        hold the static pool's KV tokens: per-token side rows (compressed keys)
        and per-slot side states (PLE) may not be reserved for KV or states
        that cannot coexist with them."""
        cell = _static_cell(0)
        for state_slots in (0, 1, _STATE_SLOTS):
            with self.subTest(state_slots=state_slots):
                tokens = _kv_tokens_with_state_slots(_build(), state_slots)
                self.assertGreaterEqual(tokens, _static_kv_tokens(cell))

    def test_draft_rows_do_not_cost_kv_capacity_at_full_state_occupancy(self):
        bundle, _ = _build_with_draft()
        tokens = _kv_tokens_with_state_slots(bundle, _STATE_SLOTS)
        self.assertGreaterEqual(tokens, _static_kv_tokens(_static_cell(1)))

    def test_buffer_is_the_static_footprint(self):
        """Budget + the static state pool (slot 0 included) + its padding page,
        and nothing allocated beside it for the compressed keys or PLE."""
        bundle = _build()
        shared = bundle.unified_memory_pool
        state_entry = shared.spec("mamba").entry_bytes()
        page_bytes = _PAGE * shared.spec("full").entry_bytes()
        static = _BUDGET + (_STATE_SLOTS + 1) * state_entry + page_bytes
        self.assertGreaterEqual(shared.total_bytes, static)
        self.assertLess(shared.total_bytes, static + 4096)
        raw = shared._raw
        raw_end = raw.data_ptr() + raw.numel()
        req_pool = bundle.req_to_token_pool
        for view in (
            *bundle.token_to_kv_pool.qsa_compressed_k_buffer_pool,
            req_pool.short_conv_pool.conv_state,
            req_pool.ngram_pool.context,
        ):
            self.assertTrue(raw.data_ptr() <= view.data_ptr() < raw_end)


class TestUnifiedQSAPoolCompaction(_UnifiedPoolCase):
    def test_compressed_keys_and_draft_rows_move_with_their_page(self):
        """Compaction relocates whole page envelopes; the compressed keys and
        the draft's K/V and keys must be found at the page's new home through
        the same translation the kernels use."""
        bundle, draft = _build_with_draft()
        target = bundle.token_to_kv_pool
        allocator = bundle.token_to_kv_pool_allocator
        first = allocator.alloc(2 * _PAGE)
        second = allocator.alloc(2 * _PAGE)

        def kernel_ids(locs):
            return allocator.translate_kv_loc_for_kernel(locs)

        def rows(pool, locs):
            return (
                pool.full_kv_pool.k_buffer[0][kernel_ids(locs)].clone(),
                pool.qsa_compressed_k_buffer_pool[0][
                    pool.qsa_index_slots(locs[::_RATIO]) // _RATIO
                ].clone(),
            )

        for pool in (target, draft):
            k_view = pool.full_kv_pool.k_buffer[0]
            k_view[kernel_ids(second)] = torch.randn(len(second), *k_view.shape[1:]).to(
                k_view.dtype
            )
            compressed = pool.qsa_compressed_k_buffer_pool[0]
            groups = pool.qsa_index_slots(second[::_RATIO]) // _RATIO
            compressed[groups] = torch.randn(len(groups), *compressed.shape[1:]).to(
                compressed.dtype
            )
        before = {id(pool): rows(pool, second) for pool in (target, draft)}
        moved_from = kernel_ids(second).clone()

        allocator.free(first)  # eager compaction fills the freed pages

        self.assertFalse(
            torch.equal(kernel_ids(second), moved_from),
            "no page moved; test is vacuous",
        )
        for pool in (target, draft):
            for got, want in zip(rows(pool, second), before[id(pool)]):
                torch.testing.assert_close(got, want)
        target_rows = rows(target, second)
        self.assertFalse(
            torch.equal(target_rows[0], rows(draft, second)[0]),
            "target and draft rows alias",
        )

    def test_ple_side_states_move_with_their_state_slot(self):
        bundle = _build()
        req_pool = bundle.req_to_token_pool
        slots = req_pool.mamba_allocator
        first, second = slots.alloc(1), slots.alloc(1)
        conv = req_pool.short_conv_pool.conv_state
        context = req_pool.ngram_pool.context
        phys = req_pool.translate_mamba_indices(second).long()
        conv_row = torch.randn(conv.shape[2:]).to(conv.dtype)
        conv[0, phys] = conv_row
        context[phys] = torch.tensor([3, 5])

        slots.free(first)  # eager compaction fills the freed slot

        moved = req_pool.translate_mamba_indices(second).long()
        self.assertFalse(torch.equal(moved, phys), "no slot moved; test is vacuous")
        torch.testing.assert_close(conv[0, moved].squeeze(0), conv_row)
        self.assertEqual(context[moved].flatten().tolist(), [3, 5])

    def test_ngram_context_starts_as_eos(self):
        bundle = _build()
        context = bundle.req_to_token_pool.ngram_pool.context
        self.assertTrue(bool((context == _PLE["ngram_eos_token_id"]).all()))


class TestUnifiedQSADraftTranslation(_UnifiedPoolCase):
    def test_draft_runner_translates_the_allocator_ids(self):
        """The draft pool is a view of the allocator's envelopes, so the draft
        runner's writes and reads must be translated like the target's."""
        bundle, draft = _build_with_draft()
        translators = [
            KVIndexTranslator(
                req_to_token=bundle.req_to_token_pool.req_to_token,
                token_to_kv_pool_allocator=bundle.token_to_kv_pool_allocator,
                token_to_kv_pool=pool,
                page_size=_PAGE,
                device="cpu",
            )
            for pool in (bundle.token_to_kv_pool, draft)
        ]
        self.assertTrue(all(t.is_translating for t in translators))
        self.assertEqual(
            translators[1].full_page_stride,
            bundle.token_to_kv_pool.full_kv_pool.kernel_page_stride,
        )


class TestUnifiedQSAPendingRing(_UnifiedPoolCase):
    def test_ring_covers_every_request_row_including_preallocated_ones(self):
        """The ring is addressed req_pool_idx * ratio + position % ratio; a PD
        decode pool preallocates rows past max_num_reqs, and those must land
        inside the ring too (row 0 is the inert dump)."""
        bundle = _build(decode_pre_alloc_size=3)
        kvcache = bundle.token_to_kv_pool
        rows = bundle.req_to_token_pool.req_to_token.shape[0]
        self.assertEqual(rows, 4 + 3 + 1)
        ring = kvcache.qsa_key_state_buffer_pool[0]
        slots = build_pending_ring_slots(
            token_to_batch_idx=torch.zeros(_RATIO, dtype=torch.int32),
            req_pool_indices=torch.tensor([rows - 1]),
            sequence_lengths=torch.tensor([_RATIO + 1]),
            logical_positions=torch.arange(_RATIO),
            compress_ratio=_RATIO,
            is_extend=False,
        )
        self.assertLess(int(slots.max()), ring.shape[0])
        self.assertGreaterEqual(int(slots.min()), _RATIO)


if __name__ == "__main__":
    unittest.main()
