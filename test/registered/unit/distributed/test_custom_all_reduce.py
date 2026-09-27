"""Unit tests for the CUDA-graph path of the v1 CustomAllreduce -- CPU-only.

The custom all-reduce kernel extension and torch.distributed are replaced by
fakes that follow the C++ contract: a graph-captured all-reduce records the
pointer it reads, and exporting that pointer through cudaIpcGetMemHandle fails
unless it is one of the cudaMalloc'd IPC buffers.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.distributed.device_communicators import custom_all_reduce as car
from sglang.srt.distributed.device_communicators import custom_all_reduce_v2 as car_v2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_RANK = 0
_WORLD_SIZE = 2
_MAX_SIZE = 4096
# cudaMalloc'd IPC buffers handed out by create_shared_buffer (meta, then buffer).
_META_PTRS = [0x1000, 0x2000]
_BUFFER_PTRS = [0x3000, 0x4000]


class _FakeCustomArKernel:
    def __init__(self):
        self.ipc_exportable = set(_META_PTRS + _BUFFER_PTRS)
        self.graph_unreg_buffers = []
        self.registered_graph_buffers = 0

    def all_reduce(self, _fa, inp, out, reg_buffer, reg_buffer_sz_bytes):
        if torch.cuda.is_current_stream_capturing():
            self.graph_unreg_buffers.append(reg_buffer or inp.data_ptr())

    def get_graph_buffer_ipc_meta(self, _fa):
        for ptr in self.graph_unreg_buffers:
            if ptr not in self.ipc_exportable:
                raise RuntimeError(
                    "invalid argument\ncsrc/allreduce/custom_all_reduce.cuh:614"
                )
        num = len(self.graph_unreg_buffers)
        return b"h" * (64 * num), [0] * num

    def register_graph_buffers(self, _fa, handles, offsets):
        self.registered_graph_buffers += len(self.graph_unreg_buffers)
        self.graph_unreg_buffers.clear()


class _FakePeerGroup:
    """Rank 0's view of a 2-rank CPU group; the peer's payloads are scripted."""

    def __init__(self, peer_payloads):
        self.peer_payloads = list(peer_payloads)
        self.last_local = None

    def broadcast_object_list(self, object_list, src, group, device):
        if src == _RANK:
            self.last_local = list(object_list)
            return
        payload = self.peer_payloads.pop(0) if self.peer_payloads else None
        object_list[:] = list(self.last_local) if payload is None else payload


class TestCustomAllreduceGraphCapture(CustomTestCase):
    def _build(self, *, allocator_is_vmm, peer_payloads=()):
        kernel = _FakeCustomArKernel()
        peer = _FakePeerGroup(peer_payloads)
        shared_buffers = iter([list(_META_PTRS), list(_BUFFER_PTRS)])
        patches = [
            mock.patch.object(car, "_is_cuda", True),
            mock.patch.object(car, "_is_hip", False),
            mock.patch.object(car.ops, "IS_CUSTOM_AR_AVAILABLE", True, create=True),
            mock.patch.object(car.ops, "meta_size", lambda: 0, create=True),
            mock.patch.object(car.ops, "init_custom_ar", lambda *a: 1, create=True),
            mock.patch.object(car.ops, "register_buffer", lambda *a: None, create=True),
            mock.patch.object(car.ops, "dispose", lambda *a: None, create=True),
            mock.patch.object(car.ops, "all_reduce", kernel.all_reduce, create=True),
            mock.patch.object(
                car.ops,
                "get_graph_buffer_ipc_meta",
                kernel.get_graph_buffer_ipc_meta,
                create=True,
            ),
            mock.patch.object(
                car.ops,
                "register_graph_buffers",
                kernel.register_graph_buffers,
                create=True,
            ),
            mock.patch.object(car.dist, "get_rank", lambda group=None: _RANK),
            mock.patch.object(
                car.dist, "get_world_size", lambda group=None: _WORLD_SIZE
            ),
            mock.patch.object(
                car.dist,
                "get_process_group_ranks",
                lambda group: list(range(_WORLD_SIZE)),
            ),
            mock.patch.object(
                car.dist, "broadcast_object_list", peer.broadcast_object_list
            ),
            # A PCIe-only pair: admitted, without full NVLink.
            mock.patch.object(
                car, "can_use_custom_all_reduce_with_nvlink", lambda **kw: False
            ),
            mock.patch.object(
                car, "is_vmm_backed_allocator", lambda device: allocator_is_vmm
            ),
            mock.patch.object(
                car.CustomAllreduce,
                "create_shared_buffer",
                staticmethod(lambda size, group=None: next(shared_buffers)),
            ),
            mock.patch.object(car.CustomAllreduce, "close", lambda self: None),
            mock.patch.object(torch.cuda, "is_current_stream_capturing", lambda: True),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        ca = car.CustomAllreduce(
            group=object(), device=torch.device("cpu"), max_size=_MAX_SIZE
        )
        self.assertFalse(ca.disabled)
        return ca, kernel

    def _capture_one_all_reduce(self, ca):
        with ca.capture():
            self.assertIsNotNone(ca.custom_all_reduce(torch.zeros(64)))

    def test_graph_capture_on_vmm_allocator_registers_instead_of_crashing(self):
        """Under expandable_segments, graph-captured inputs are VMM memory that
        cudaIpcGetMemHandle rejects; capture must still register and keep custom AR."""
        ca, kernel = self._build(allocator_is_vmm=True)

        self._capture_one_all_reduce(ca)

        self.assertFalse(ca.disabled)
        self.assertEqual(kernel.registered_graph_buffers, 1)

    def test_peer_ipc_export_failure_disables_custom_ar_on_healthy_rank(self):
        """A rank whose own graph buffers export fine must still drop custom AR
        and report a recoverable error when a peer's export failed."""
        peer_failed = [None, None, "invalid argument"]
        ca, kernel = self._build(allocator_is_vmm=False, peer_payloads=[peer_failed])

        with self.assertRaises(car.CustomAllreduceGraphRegistrationError):
            self._capture_one_all_reduce(ca)

        self.assertTrue(ca.disabled)
        self.assertEqual(kernel.registered_graph_buffers, 0)
        self.assertIsNone(ca.custom_all_reduce(torch.zeros(64)))

    def test_peer_ipc_open_failure_disables_custom_ar_on_healthy_rank(self):
        """A failure while a peer opens the exchanged handles must also leave the
        whole group without custom AR, not just the failing rank."""
        ca, _ = self._build(
            allocator_is_vmm=False,
            peer_payloads=[None, ["cudaIpcOpenMemHandle: invalid argument"]],
        )

        with self.assertRaises(car.CustomAllreduceGraphRegistrationError):
            self._capture_one_all_reduce(ca)

        self.assertTrue(ca.disabled)
        self.assertIsNone(ca.custom_all_reduce(torch.zeros(64)))


class _FakeIpcManager:
    def batch_get_handles(self, ptrs):
        return [(b"h" * 64, 0) for _ in ptrs]

    def batch_open_handles(self, handles):
        return [0x9000 + i for i in range(len(handles))]


class _FailingVmmGraphInputManager:
    def register_graph_inputs(self):
        raise RuntimeError("cuMemMap: CUDA_ERROR_INVALID_VALUE")


class TestCustomAllReduceV2GraphRegistration(CustomTestCase):
    """Rank 0 of a 2-rank group; the peer's all_gather_object payloads are scripted."""

    def _build(self, *, vmm_inputs, peer_payloads):
        ar = car_v2.CustomAllReduceV2.__new__(car_v2.CustomAllReduceV2)
        ar.disabled = False
        ar.override_algo = None
        ar.group = object()
        ar.rank = _RANK
        ar.world_size = _WORLD_SIZE
        ar._graph_inputs = [(0x5000, 256)]
        ar._ipc_manager = _FakeIpcManager()
        ar._vmm_graph_input_manager = _FailingVmmGraphInputManager()
        peer_payloads = list(peer_payloads)

        def all_gather_object(gathered, local, group):
            gathered[_RANK] = local
            gathered[1 - _RANK] = peer_payloads.pop(0)

        for patch in (
            mock.patch.object(car_v2, "is_vmm_pointer", lambda ptr: vmm_inputs),
            mock.patch.object(car_v2.dist, "all_gather_object", all_gather_object),
            mock.patch.object(car_v2.CustomAllReduceV2, "close", lambda self: None),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        return ar

    def _assert_group_fell_back_to_nccl(self, ar):
        with self.assertRaises(car_v2.CustomAllreduceGraphRegistrationError):
            ar._register_graph_inputs()
        self.assertTrue(ar.disabled)
        self.assertFalse(ar.should_custom_ar(torch.zeros(64)))

    def test_peer_ipc_export_failure_disables_v2_on_healthy_rank(self):
        """A peer that failed to export its cudaIpc handles must not leave this
        rank opening garbage handles; the whole group drops v2 custom AR."""
        ar = self._build(
            vmm_inputs=False,
            peer_payloads=["invalid argument", "invalid argument"],
        )
        self._assert_group_fell_back_to_nccl(ar)

    def test_local_vmm_mapping_failure_is_recoverable(self):
        """A VMM peer-mapping failure must disable v2 custom AR on every rank
        and surface as the recoverable error, not crash startup."""
        ar = self._build(vmm_inputs=True, peer_payloads=[None])
        self._assert_group_fell_back_to_nccl(ar)


if __name__ == "__main__":
    unittest.main()
