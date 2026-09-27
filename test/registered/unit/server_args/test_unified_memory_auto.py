"""An unset `--enable-unified-memory` resolves by itself for hybrid Mamba models.

The invariant: auto only turns the unified pool on for a configuration the
explicit-flag handlers (`handle_page_major_kv_layout`,
`handle_unified_memory_pool`) accept, and a configuration they would reject
resolves off instead of failing the boot.
"""

import unittest
from types import SimpleNamespace

import msgspec
from transformers import LlamaConfig

from sglang.srt.arg_groups.kv_cache_hook import (
    handle_page_major_kv_layout,
    handle_unified_memory_pool,
    resolve_unified_memory_default,
)
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.configs import Qwen3NextConfig
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

# QSA indexer fields as Qwen3.8-Flash-Next (Qwen4-Exp) declares them.
_QSA_FIELDS = dict(
    indexer_n_heads=32,
    indexer_kv_heads=1,
    indexer_head_dim=128,
    indexer_budget=2048,
    indexer_compress_ratio=4,
)


def _resolve(*, hf_config=None, has_asymmetric_kv=False, prefill_graph=None, **fields):
    """Resolve the flag on a GDN-hybrid stand-in, then run both handlers."""
    sa = ServerArgs(model_path="dummy")
    values = {
        "enable_unified_memory": None,
        "attention_backend": "triton",
        "prefill_attention_backend": None,
        "decode_attention_backend": None,
        "linear_attn_backend": "triton",
        "linear_attn_decode_backend": None,
        "linear_attn_prefill_backend": None,
        "mamba_backend": "triton",
        "disaggregation_mode": "null",
        "speculative_algorithm": None,
        "speculative_eagle_topk": None,
        "speculative_draft_model_path": None,
        "enable_hierarchical_cache": False,
        "enable_lmcache": False,
        "enable_two_batch_overlap": False,
        "enable_dp_attention": False,
        "pp_size": 1,
        "dcp_size": 1,
        "kv_cache_dtype": "auto",
        "cuda_graph_config": SimpleNamespace(
            prefill=SimpleNamespace(backend=prefill_graph or Backend.DISABLED),
            decode=SimpleNamespace(backend=Backend.FULL),
        ),
    }
    values.update(fields)
    for name, value in values.items():
        msgspec.Struct.__setattr__(sa, name, value)
    sa._model_config = SimpleNamespace(
        hf_config=hf_config if hf_config is not None else Qwen3NextConfig(),
        is_draft_model=False,
        linear_attn_registry_result=None,
        is_hybrid_swa=False,
        attention_arch=AttentionArch.MHA,
        has_asymmetric_kv=has_asymmetric_kv,
        head_dim=192 if has_asymmetric_kv else 128,
        v_head_dim=128,
        swa_head_dim=128,
        swa_v_head_dim=128,
    )
    with override_platform(is_cuda=True):
        resolve_unified_memory_default(sa)
        handle_page_major_kv_layout(sa)
        handle_unified_memory_pool(sa)
    return resolution_result(sa, "enable_unified_memory")


class TestUnifiedMemoryAuto(CustomTestCase):
    def test_supported_hybrid_model_turns_it_on(self):
        self.assertIs(_resolve(), True)

    def test_an_explicit_choice_is_kept(self):
        self.assertIs(_resolve(enable_unified_memory=False), False)

    def test_non_hybrid_model_stays_off(self):
        self.assertIs(_resolve(hf_config=LlamaConfig()), False)

    def test_qsa_model_with_its_chain_mtp_draft_turns_it_on(self):
        """The deployed QSA recipe (built-in MTP draft, linear chain) resolves on,
        with or without an explicit HiCache host tier: the two now coexist."""
        qsa = Qwen3NextConfig(**_QSA_FIELDS)
        mtp = dict(speculative_algorithm="EAGLE", speculative_eagle_topk=1)
        self.assertIs(_resolve(hf_config=qsa), True)
        self.assertIs(_resolve(hf_config=qsa, **mtp), True)
        self.assertIs(
            _resolve(hf_config=qsa, enable_hierarchical_cache=True, **mtp), True
        )
        self.assertIs(_resolve(enable_hierarchical_cache=True), True)

    def test_speculation_outside_the_audited_qsa_chain_stays_off(self):
        qsa = Qwen3NextConfig(**_QSA_FIELDS)
        cases = {
            # Tree verify is not audited for the unified pool.
            "QSA tree draft": dict(
                hf_config=qsa, speculative_algorithm="EAGLE", speculative_eagle_topk=4
            ),
            # A separate draft checkpoint is not the packed MTP pool.
            "QSA external draft": dict(
                hf_config=qsa,
                speculative_algorithm="EAGLE",
                speculative_eagle_topk=1,
                speculative_draft_model_path="some/draft",
            ),
        }
        for name, fields in cases.items():
            with self.subTest(name):
                self.assertIs(_resolve(**fields), False)

    def test_unsupported_configs_resolve_off_instead_of_failing(self):
        cases = {
            # Rejected by the explicit-flag validators.
            "two-batch overlap": dict(enable_two_batch_overlap=True),
            "asymmetric K/V rows": dict(has_asymmetric_kv=True),
            "unwired attention backend": dict(attention_backend="aiter"),
            "strided-state linear decode": dict(linear_attn_decode_backend="cutedsl"),
            # Would silently lose FULL prefill CUDA graphs.
            "FULL prefill graphs": dict(prefill_graph=Backend.FULL),
            "speculative decoding": dict(
                speculative_algorithm="EAGLE", speculative_eagle_topk=1
            ),
        }
        for name, fields in cases.items():
            with self.subTest(name):
                self.assertIs(_resolve(**fields), False)


if __name__ == "__main__":
    unittest.main()
