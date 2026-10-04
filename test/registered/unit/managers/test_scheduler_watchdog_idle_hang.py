import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components import invariant_checker

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _scheduler(*, waiting: list, running_empty: bool):
    return SimpleNamespace(
        is_initializing=False,
        cur_batch_for_debug=None,
        forward_ct=7,
        event_loop_ct=3,
        waiting_queue=waiting,
        running_batch=SimpleNamespace(is_empty=lambda: running_empty),
    )


def _fires(scheduler, *, timeout: float = 0.2, wait: float = 1.5) -> bool:
    fired = threading.Event()
    with patch.object(invariant_checker, "WatchdogRaw") as raw:
        invariant_checker.create_scheduler_watchdog(
            scheduler, watchdog_timeout=timeout, soft=True
        )
        kwargs = raw.call_args.kwargs

    from sglang.srt.utils import watchdog

    def on_error(*args, **kwargs):
        fired.set()

    with patch.object(watchdog, "pyspy_dump_schedulers"):
        with patch.object(watchdog.logger, "error", side_effect=on_error):
            raw = watchdog.WatchdogRaw(
                debug_name="Scheduler",
                get_counter=kwargs["get_counter"],
                is_active=kwargs["is_active"],
                watchdog_timeout=timeout,
                soft=True,
            )
            try:
                return fired.wait(wait)
            finally:
                # A soft watchdog re-arms after firing; a live one from an
                # earlier test would trip this test's logger patch.
                raw.stop()


class TestSchedulerWatchdogIdleHang(CustomTestCase):
    def test_stalled_loop_with_queued_request_fires(self):
        # No batch is in flight, yet a request waits and the loop stopped turning.
        self.assertTrue(_fires(_scheduler(waiting=["req"], running_empty=True)))

    def test_stalled_loop_with_running_request_fires(self):
        self.assertTrue(_fires(_scheduler(waiting=[], running_empty=False)))

    def test_idle_scheduler_with_no_work_does_not_fire(self):
        self.assertFalse(_fires(_scheduler(waiting=[], running_empty=True)))

    def test_turning_loop_does_not_fire(self):
        scheduler = _scheduler(waiting=["req"], running_empty=True)
        stop = threading.Event()

        def spin():
            while not stop.is_set():
                scheduler.event_loop_ct += 1
                stop.wait(0.01)

        threading.Thread(target=spin, daemon=True).start()
        try:
            self.assertFalse(_fires(scheduler))
        finally:
            stop.set()


if __name__ == "__main__":
    unittest.main()
