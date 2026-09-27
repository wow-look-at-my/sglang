"""CPU coverage for the single-CG multi-layer EAGLE draft-extend capture loop.

The runner is built via ``__new__``; ``graph_capture`` (the context that
IPC-registers captured custom all-reduce buffers on exit) and the per-bucket
capture are faked so the loop runs without a GPU.
"""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

from sglang.srt.speculative import (
    multi_layer_eagle_draft_extend_cuda_graph_runner as mod,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_RunnerCls = mod.OneGraphMultiLayerEagleMultiStepDraftExtendCudaGraphRunner


class _FakeBackend:
    @contextmanager
    def capture_session(self, stream):
        yield


class _FakeGraphCapture:
    """Stands in for parallel_state.graph_capture and reports whether it is open."""

    def __init__(self):
        self.active = False

    @contextmanager
    def __call__(self, stream=None):
        self.active = True
        try:
            yield SimpleNamespace(stream=stream)
        finally:
            self.active = False


class TestSingleCgCapture(CustomTestCase):
    def test_buckets_are_captured_inside_graph_capture(self):
        """Captured custom all-reduces are IPC-registered only when capture runs
        inside graph_capture(); outside it they replay with unregistered buffers."""
        graph_capture = _FakeGraphCapture()
        backend = _FakeBackend()
        first = SimpleNamespace(capture_bs=[1, 2], backend=backend)
        runner = _RunnerCls.__new__(_RunnerCls)
        buckets_captured_inside_graph_capture = []

        def capture_one_graph(bs, runners):
            buckets_captured_inside_graph_capture.append(graph_capture.active)

        runner._capture_one_graph = capture_one_graph
        with (
            mock.patch.object(mod, "graph_capture", graph_capture),
            mock.patch.object(
                mod, "get_or_create_global_graph_capture_stream", lambda: "stream"
            ),
        ):
            runner._capture_all_graphs([first])

        self.assertEqual(buckets_captured_inside_graph_capture, [True, True])


if __name__ == "__main__":
    unittest.main()
