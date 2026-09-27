"""QSA compression in mixed prefill + decode batches."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.qsa.kernel import average_pool_qsa_keys
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RATIO = 4
HEAD_DIM = 8


class _Pool:
    qsa_compress_ratio = RATIO

    def __init__(self, ring_slots, index_slots=None):
        self.ring = torch.zeros(ring_slots, 1, HEAD_DIM)
        self.ring_rope = torch.zeros(ring_slots, 3, dtype=torch.int64)
        self.compressed = {}
        # A unified pool's page move into the compressed rows' slot space.
        self._index_slots = index_slots

    def get_qsa_key_state_buffer(self, layer_id):
        return self.ring

    def get_qsa_rope_position_buffer(self, loc):
        return self.ring_rope[loc.long()]

    def set_qsa_compressed_k_buffer(self, layer_id, loc, value):
        for slot, row in zip(loc.tolist(), value):
            self.compressed[slot] = row.clone()

    def qsa_index_slots(self, full_slots):
        if self._index_slots is None:
            return full_slots
        return self._index_slots(full_slots)


class _Indexer:
    """The real compress path with identity normalization, recording RoPE starts."""

    layer_id = 0
    compress_ratio = RATIO
    rotary_emb = SimpleNamespace()
    update_key_state_and_compress = QSAIndexer.update_key_state_and_compress
    _rope_from_matrix = QSAIndexer._rope_from_matrix

    def __init__(self):
        self.rope_starts = None

    def _use_fused_compress(self, pool):
        return False

    def normalize_compressed_keys(self, pooled, block_positions):
        self.rope_starts = block_positions.tolist()
        return pooled


class TestQsaMixedChunk(unittest.TestCase):
    def test_decode_tail_completes_group_started_in_pending_ring(self):
        """A mixed batch whose decode tail completes a compress group must not
        trip the aligned-prefix assert, and must compress the group from its
        ring-held members plus the tail token, not from other rows' tokens.
        On the unified pool (compressed rows in each page's envelope, found
        through the live page table) the groups land where the page is now."""

        def moved_pages(slots):
            page = 2 * RATIO
            return (slots // page + 3) * (page * 5) + slots % page

        for index_slots in (None, moved_pages):
            with self.subTest(unified=index_slots is not None):
                self._decode_tail_completes_group(index_slots)

    def _decode_tail_completes_group(self, index_slots):
        # Row 0: fresh 8-token prefill (request slot 2). Row 1: decode tail at
        # position 7 of request slot 1, whose positions 4..6 are in the ring.
        prefill_req, tail_req = 2, 1
        pool = _Pool(ring_slots=4 * RATIO, index_slots=index_slots)
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend.token_to_kv_pool = pool
        lengths = torch.tensor([8, 8])
        extend_lens = torch.tensor([8, 1])
        slot_table = torch.arange(2 * 8, dtype=torch.int32).reshape(2, 8)
        slot_table[1] += 100
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.MIXED,
            extend_seq_lens=extend_lens,
            input_ids=torch.zeros(9, dtype=torch.long),
        )
        write_locs, group_ends, _, _, mixed = backend._qsa_build_write_plan(
            forward_batch=forward_batch,
            speculative_paged=False,
            token_slot_table=slot_table,
            sequence_lengths=lengths,
            row_req_pool_indices=torch.tensor([prefill_req, tail_req]),
        )
        member_locs, prior_ring_locs = mixed

        ring_keys = {pos: torch.randn(1, HEAD_DIM) for pos in (4, 5, 6)}
        for pos, key in ring_keys.items():
            pool.ring[tail_req * RATIO + pos % RATIO] = key
            pool.ring_rope[tail_req * RATIO + pos % RATIO] = pos
        token_k = torch.randn(9, 1, HEAD_DIM)
        forward_rope = torch.cat([torch.arange(8), torch.tensor([7])])
        metadata = SimpleNamespace(
            token_to_kv_pool=pool,
            compress_member_rows=torch.zeros(1),
            compress_member_locs=member_locs,
            is_cuda_graph=False,
            write_locs=write_locs,
            compress_group_positions=group_ends,
            extend_rope_matrix=torch.cat(
                [
                    pool.get_qsa_rope_position_buffer(prior_ring_locs),
                    forward_rope[:, None].expand(-1, 3),
                ]
            ),
        )
        indexer = _Indexer()
        indexer.update_key_state_and_compress(
            token_k,
            logical_positions=forward_rope,
            rope_positions=forward_rope,
            metadata=metadata,
            state_stored=True,
            prior_keys=pool.ring[prior_ring_locs],
        )

        def compressed_row(full_slot):
            return int(pool.qsa_index_slots(full_slot.long())) // RATIO

        tail_slot = compressed_row(slot_table[1, 4])
        expected_tail = average_pool_qsa_keys(
            torch.stack([ring_keys[4], ring_keys[5], ring_keys[6], token_k[8]])[None]
        )[0]
        torch.testing.assert_close(pool.compressed[tail_slot], expected_tail)
        for block in range(2):
            slot = compressed_row(slot_table[0, block * RATIO])
            expected = average_pool_qsa_keys(
                token_k[block * RATIO : (block + 1) * RATIO][None]
            )[0]
            torch.testing.assert_close(pool.compressed[slot], expected)
        # The tail group rotates by its first member's position, held in the ring.
        entry = write_locs.tolist().index(tail_slot)
        self.assertEqual(indexer.rope_starts[entry], 4)


class TestQsaUnifiedPoolReads(unittest.TestCase):
    def test_kv_reads_and_compressed_writes_translate(self):
        """On the unified pool req_to_token holds virtual slots: the sparse K/V
        read must go through the kernel-facing translation, and the compressed
        keys, which live in the same page envelopes, through the pool's
        index-slot translation."""
        from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention

        def to_kernel(ids):
            # A unified pool's views: page moved and scaled by the layer blocks.
            return (ids // RATIO + 3) * (RATIO * 4) + ids % RATIO

        def to_index_slots(ids):
            # The same page move, scaled by the envelope's index pages.
            return (ids // RATIO + 3) * (RATIO * 2) + ids % RATIO

        rows = to_kernel(torch.tensor(15)) + 1
        k_view, v_view = torch.randn(rows, 1, HEAD_DIM), torch.randn(rows, 1, HEAD_DIM)
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend.kv_index_translator = SimpleNamespace(
            is_translating=True,
            reads_are_translated=True,
            translate_full_attn_ids=to_kernel,
        )
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda layer_id: k_view,
            get_value_buffer=lambda layer_id: v_view,
            qsa_compress_ratio=RATIO,
            qsa_index_slots=to_index_slots,
        )
        virtual = torch.arange(16, dtype=torch.int32).reshape(1, 16)
        backend.forward_metadata = SimpleNamespace(
            token_to_batch_idx=torch.zeros(1, dtype=torch.int32),
            sequence_lengths=torch.tensor([8], dtype=torch.int32),
            token_slot_table=virtual,
        )
        layer = SimpleNamespace(
            tp_q_head_num=2, head_dim=HEAD_DIM, layer_id=0, scaling=1.0
        )
        q = torch.randn(1, 2, HEAD_DIM)
        topk = torch.tensor([[0, 5, 7, -1]], dtype=torch.int32)

        out = backend._forward_paged_attention(q, layer, None, topk)

        slots = torch.tensor([[0, 5, 7]])
        expected = qsa_sparse_attention(
            q,
            k_view,
            v_view,
            torch.cat([to_kernel(slots), -torch.ones(1, 1, dtype=torch.long)], 1),
            1.0,
        )
        torch.testing.assert_close(out, expected.reshape(1, -1))
        # Decode at length 8 completes the group of slots 4..7.
        write_locs, *_ = backend._qsa_build_write_plan(
            forward_batch=SimpleNamespace(
                forward_mode=SimpleNamespace(is_decode=lambda: True)
            ),
            speculative_paged=False,
            token_slot_table=virtual,
            sequence_lengths=torch.tensor([8], dtype=torch.int32),
            row_req_pool_indices=torch.tensor([1], dtype=torch.int32),
        )
        self.assertEqual(write_locs.tolist(), [int(to_index_slots(4)) // RATIO])

    def test_mixed_batch_reads_translate(self):
        """A mixed batch (a prefill row plus a decode tail) reads K/V through
        the extend path, which must translate virtual slots like decode does."""
        from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention

        def to_kernel(ids):
            return (ids // RATIO + 3) * (RATIO * 4) + ids % RATIO

        rows = to_kernel(torch.tensor(31)) + 1
        k_view, v_view = torch.randn(rows, 1, HEAD_DIM), torch.randn(rows, 1, HEAD_DIM)
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend.kv_index_translator = SimpleNamespace(
            is_translating=True,
            reads_are_translated=True,
            translate_full_attn_ids=to_kernel,
        )
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda layer_id: k_view,
            get_value_buffer=lambda layer_id: v_view,
        )
        virtual = torch.arange(32, dtype=torch.int32).reshape(2, 16)
        # Row 0 prefills 2 tokens; row 1 is a decode tail at length 9.
        backend.forward_metadata = SimpleNamespace(
            token_to_batch_idx=torch.tensor([0, 0, 1], dtype=torch.int32),
            sequence_lengths=torch.tensor([2, 9], dtype=torch.int32),
            token_slot_table=virtual,
        )
        layer = SimpleNamespace(
            tp_q_head_num=2, head_dim=HEAD_DIM, layer_id=0, scaling=1.0
        )
        q = torch.randn(3, 2, HEAD_DIM)
        topk = torch.tensor([[0, -1, -1], [0, 1, -1], [8, 3, 5]], dtype=torch.int32)

        out = backend.forward_extend(
            q,
            None,
            None,
            layer,
            SimpleNamespace(forward_mode=ForwardMode.MIXED),
            save_kv_cache=False,
            topk_indices=topk,
        )

        slots = torch.tensor([[0, -1, -1], [0, 1, -1], [24, 19, 21]])
        kernel_slots = torch.where(slots >= 0, to_kernel(slots.clamp(min=0)), -1)
        expected = qsa_sparse_attention(q, k_view, v_view, kernel_slots, 1.0)
        torch.testing.assert_close(out, expected.reshape(3, -1))


if __name__ == "__main__":
    unittest.main()
