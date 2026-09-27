"""Qwen3.8-Flash-Next (QSA + GDN + NEXTN MTP) with the automatic HiCache tier.

The device pool is capped far below the prompt set, so the warm-up prefill
evicts most prefixes to host memory and the cache-hit generation loads them
back into different device pages. A load-back that restored the KV and GDN
state but not the QSA compressed keys scores the sparse indexer against
stale keys, which shows as a large prefill-cache-hit KL divergence.
"""

import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=900, stage="nightly", runner_config="4-gpu-b200")

QWEN38_FLASH_NEXT_NVFP4 = "RadixArk/Qwen3.8-Flash-Next-NVFP4"


class TestQwen38FlashNextAutoHiCacheKL(KLDivergenceMixin, DefaultServerBase):
    model = QWEN38_FLASH_NEXT_NVFP4
    # Not calibrated on hardware yet: MoE routing amplifies float noise (see
    # the kl-consistency-test skill), while stale compressed keys change which
    # KV blocks QSA attends to and read orders of magnitude above this.
    kl_div_thres = 0.02
    # The cookbook's B200 NVFP4 low-latency cell (TP1, NEXTN 3/1/4) with the
    # device pool capped so the 32 x ~3k-token prompts overflow it into host
    # memory. --enable-hierarchical-cache is deliberately left unset.
    other_args = [
        "--tp",
        "1",
        "--linear-attn-prefill-backend",
        "flashinfer",
        "--linear-attn-decode-backend",
        "flashinfer",
        "--mamba-ssm-dtype",
        "bfloat16",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--max-total-tokens",
        "32768",
    ]

    def test_hicache_resolves_on_without_the_flag(self):
        server_info = requests.get(self.base_url + "/server_info").json()
        self.assertTrue(server_info["enable_hierarchical_cache"])


if __name__ == "__main__":
    unittest.main()
