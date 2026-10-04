"""A queued request sends its whole prompt, so the sched-policy channel is sized for one.

A channel sized below a full prompt rejects the queue-time notification with
MessageTooLarge, which the scheduler thread cannot catch: the process then dies
with SIGQUIT and the whole TP group restarts.
"""

import unittest

import goipc

from sglang.srt.managers.scheduler_components import sched_policy
from sglang.srt.managers.scheduler_components import sched_policy_messages as msg
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# More tokens than a service channel of go-ipc's default size carries.
_PROMPT_TOKENS = 1 << 17


# The policy process answers Hello with HelloOk and every other request with Ack.
def _answer(session, type_, payload):
    if type_ == msg.Hello.TYPE_ID:
        return (
            msg.HelloOk.TYPE_ID,
            msg.HelloOk(protocol=sched_policy.PROTOCOL_VERSION).encode(),
        )
    return msg.Ack.TYPE_ID, msg.Ack().encode()


class TestPromptChannelCapacity(unittest.TestCase):
    def test_a_whole_prompt_fits_the_service_payload(self):
        for context_len in (4096, 131072, 262144, 1 << 20):
            capacity = sched_policy._prompt_channel_capacity(context_len)
            self.assertEqual(
                capacity & (capacity - 1), 0, "a ring takes a power of two"
            )
            payload = (
                goipc.wire.max_message_size(capacity) - goipc.wire.SERVICE_SEQUENCE_SIZE
            )
            self.assertGreaterEqual(
                payload, sched_policy._TOKEN_BYTES * context_len, context_len
            )

    def test_a_short_context_still_gives_a_valid_ring(self):
        capacity = sched_policy._prompt_channel_capacity(0)
        self.assertGreaterEqual(capacity, goipc.MIN_CAPACITY)
        self.assertEqual(capacity & (capacity - 1), 0)


class TestPromptChannelCarriesThePrompt(unittest.TestCase):
    """The shipped send path. A channel of go-ipc's default size rejects this prompt."""

    def _serve(self):
        name = sched_policy.new_name()
        service = goipc.service.serve(name, _answer)
        self.addCleanup(service.close)
        return name

    def test_the_policy_wrapper_sends_a_whole_prompt(self):
        name = self._serve()
        client = sched_policy.SchedPolicy(
            name=name, rank=0, world=1, timeout=5.0, context_len=_PROMPT_TOKENS
        )
        self.addCleanup(client.close)
        client.on_request_queued(rid="prompt", token_ids=range(_PROMPT_TOKENS))

    def test_the_policy_wrapper_sends_a_long_context_prompt(self):
        name = self._serve()
        context_len = 1 << 20
        client = sched_policy.SchedPolicy(
            name=name, rank=0, world=1, timeout=10.0, context_len=context_len
        )
        self.addCleanup(client.close)
        client.on_request_queued(rid="long", token_ids=range(context_len))

    def test_go_ipcs_default_channel_rejects_the_same_prompt(self):
        name = self._serve()
        client = goipc.service.connect(name, timeout=5.0, messages=msg.MESSAGES)
        self.addCleanup(client.close)
        request = msg.RequestQueued(
            now=0.0,
            rid="prompt",
            tokens=sched_policy._tokens(range(_PROMPT_TOKENS)),
        )
        with self.assertRaises(goipc.MessageTooLarge):
            client.call_typed(request)


if __name__ == "__main__":
    unittest.main()
