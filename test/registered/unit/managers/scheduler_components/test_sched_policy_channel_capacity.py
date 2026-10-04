"""A queued request sends its whole prompt, so the sched-policy channel is sized for one.

A channel sized below a full prompt rejects the queue-time notification with
MessageTooLarge, which the scheduler thread cannot catch: the process then dies
with SIGQUIT and the whole TP group restarts.
"""

import unittest

import goipc

from sglang.srt.managers.scheduler_components import sched_policy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPromptChannelCapacity(unittest.TestCase):
    def test_a_whole_prompt_fits_the_service_payload(self):
        for context_len in (4096, 131072, 262144, 1 << 20):
            capacity = sched_policy._prompt_channel_capacity(context_len)
            self.assertEqual(capacity & (capacity - 1), 0, "a ring takes a power of two")
            payload = (
                goipc.wire.max_message_size(capacity)
                - goipc.wire.SERVICE_SEQUENCE_SIZE
            )
            self.assertGreaterEqual(
                payload, sched_policy._TOKEN_BYTES * context_len, context_len
            )

    def test_a_short_context_keeps_the_smallest_ring(self):
        self.assertEqual(sched_policy._prompt_channel_capacity(0), goipc.MIN_CAPACITY)


if __name__ == "__main__":
    unittest.main()
