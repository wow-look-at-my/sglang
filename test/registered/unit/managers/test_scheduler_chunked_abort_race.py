"""Tests for deferred chunked-prefill aborts."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class _FakeReq:
    """Minimal stand-in for Req: only the fields the abort paths touch."""

    def __init__(self, rid: str):
        self.rid = rid
        # Mirrors Req.kv; the abort paths read only these two predicates.
        self.kv = SimpleNamespace(holds_kv=True, holds_mamba=False)
        self.to_finish = None
        self._finished = False
        self.return_logprob = False
        self.output_ids = []
        self.weight_version_events = []
        self.cache_request_handle = None
        self.time_stats = SimpleNamespace(trace_ctx=Mock())

    def finished(self):
        return self._finished


def _make_scheduler(pending_req, *, chunked_req, running_reqs) -> Scheduler:
    sched = Scheduler.__new__(Scheduler)
    sched.chunked_req = chunked_req
    sched._pending_chunked_abort_req = pending_req
    sched.waiting_queue = []
    sched.dllm_config = None
    sched.grammar_manager = Mock()
    sched.disaggregation_mode = None
    sched.enable_hicache_storage = False
    sched.mm_receiver = None
    sched.running_batch = SimpleNamespace(reqs=running_reqs)
    sched.last_batch = None
    return sched


class TestPendingChunkedAbortRace(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def test_req_left_chunked_slot_is_aborted(self):
        req = _FakeReq("zombie_rid")
        sched = _make_scheduler(req, chunked_req=None, running_reqs=[req])

        sched.process_pending_chunked_abort()

        self.assertIsNotNone(req.to_finish, "recorded abort was never applied")
        self.assertIsNone(sched._pending_chunked_abort_req)

    def test_finished_req_only_clears_marker(self):
        req = _FakeReq("done_rid")
        req._finished = True
        sched = _make_scheduler(req, chunked_req=None, running_reqs=[])

        sched.process_pending_chunked_abort()

        self.assertIsNone(req.to_finish)
        self.assertIsNone(sched._pending_chunked_abort_req)

    def test_pp_disagg_prefill_drops_aborted_chunked_req_before_next_chunk(self):
        """A PP prefill server must release an aborted in-progress chunked
        request before scheduling its next chunk, not prefill the whole prompt."""
        enter_scope(self, published_topology(pp_size=2))
        req = _FakeReq("chunked_rid")
        sched = _make_scheduler(req, chunked_req=req, running_reqs=[])
        sched.tree_cache = Mock()
        sched.ipc_channels = Mock()
        sched.init_pp_loop_state = Mock()
        sched.pp_loop_size = 1
        sched.pp_group = SimpleNamespace(is_last_rank=True)
        sched.running_mbs = [sched.running_batch]
        sched.last_mbs = [None]
        sched.ingest_requests = Mock(return_value=[])
        sched._pp_pd_get_bootstrapped_ids = Mock(return_value=[])
        sched._pp_pd_get_prefill_transferred_ids = Mock(return_value=[])
        sched._pp_commit_comm_work = Mock()
        sched._process_hicache_events = Mock()
        chunked_req_at_chunk_step = []
        sched.process_prefill_chunk = lambda **_: chunked_req_at_chunk_step.append(
            sched.chunked_req
        )
        # Stop the infinite loop at admission; no PP transport or GPU is needed.
        sched.get_new_batch_prefill = Mock(side_effect=StopIteration)

        def release(r, *_args, **_kwargs):
            r.kv.holds_kv = False

        with patch(
            "sglang.srt.managers.scheduler.release_kv_cache", side_effect=release
        ):
            with self.assertRaises(StopIteration):
                sched.event_loop_pp_disagg_prefill()

        self.assertEqual(chunked_req_at_chunk_step, [None])
        self.assertFalse(req.kv.holds_kv, "aborted chunked request kept its KV")
        self.assertIsNone(sched._pending_chunked_abort_req)
        (sent,) = sched.ipc_channels.send_to_tokenizer.send_output.call_args_list
        self.assertEqual(sent.args[0].rid, "chunked_rid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
