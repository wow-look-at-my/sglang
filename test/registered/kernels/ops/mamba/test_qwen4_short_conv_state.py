"""The fused PLE short-conv state step on slots spaced wider than one state.

Under the unified memory pool the short-conv windows live in each state slot's
envelope, so consecutive slots are an envelope apart; a kernel that assumed
packed slots would read and advance another slot's window.
"""

import unittest

import torch

from sglang.kernels.ops.mamba.qwen4_short_conv import (
    can_fuse_qwen4_short_conv_state,
    fused_qwen4_short_conv_state,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestQwen4ShortConvStateInSlotEnvelopes(CustomTestCase):
    def test_strided_slots_match_packed_slots(self):
        slots, channels, state_len, envelope_pad = 6, 160, 3, 37
        generator = torch.Generator(device="cuda").manual_seed(0)
        packed = torch.randn(
            (slots, channels, state_len),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        slot_elems = channels * state_len + envelope_pad
        envelope = torch.randn(
            slots * slot_elems, device="cuda", generator=generator
        ).to(torch.bfloat16)
        strided = torch.as_strided(
            envelope, (slots, channels, state_len), (slot_elems, state_len, 1)
        )
        strided.copy_(packed)
        pads = envelope.view(slots, slot_elems)[:, channels * state_len :].clone()
        indices = torch.tensor([4, 1, 0, 2], device="cuda", dtype=torch.long)
        x = torch.randn(
            (indices.numel(), channels),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )

        self.assertTrue(can_fuse_qwen4_short_conv_state(strided, indices, x))
        want = fused_qwen4_short_conv_state(packed, indices, x)
        got = fused_qwen4_short_conv_state(strided, indices, x)

        torch.testing.assert_close(got, want, rtol=0, atol=0)
        torch.testing.assert_close(strided, packed, rtol=0, atol=0)
        torch.testing.assert_close(
            envelope.view(slots, slot_elems)[:, channels * state_len :], pads
        )


if __name__ == "__main__":
    unittest.main()
