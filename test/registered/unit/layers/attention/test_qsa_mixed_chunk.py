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

    def __init__(self, ring_slots):
        self.ring = torch.zeros(ring_slots, 1, HEAD_DIM)
        self.ring_rope = torch.zeros(ring_slots, 3, dtype=torch.int64)
        self.compressed = {}

    def get_qsa_key_state_buffer(self, layer_id):
        return self.ring

    def get_qsa_rope_position_buffer(self, loc):
        return self.ring_rope[loc.long()]

    def set_qsa_compressed_k_buffer(self, layer_id, loc, value):
        for slot, row in zip(loc.tolist(), value):
            self.compressed[slot] = row.clone()


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
        ring-held members plus the tail token, not from other rows' tokens."""
        # Row 0: fresh 8-token prefill (request slot 2). Row 1: decode tail at
        # position 7 of request slot 1, whose positions 4..6 are in the ring.
        prefill_req, tail_req = 2, 1
        pool = _Pool(ring_slots=4 * RATIO)
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

        tail_slot = int(slot_table[1, 4]) // RATIO
        expected_tail = average_pool_qsa_keys(
            torch.stack([ring_keys[4], ring_keys[5], ring_keys[6], token_k[8]])[None]
        )[0]
        torch.testing.assert_close(pool.compressed[tail_slot], expected_tail)
        for block in range(2):
            slot = int(slot_table[0, block * RATIO]) // RATIO
            expected = average_pool_qsa_keys(
                token_k[block * RATIO : (block + 1) * RATIO][None]
            )[0]
            torch.testing.assert_close(pool.compressed[slot], expected)
        # The tail group rotates by its first member's position, held in the ring.
        entry = write_locs.tolist().index(tail_slot)
        self.assertEqual(indexer.rope_starts[entry], 4)


if __name__ == "__main__":
    unittest.main()
