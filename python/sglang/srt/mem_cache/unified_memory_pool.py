# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""UnifiedKVPool -- one physical `uint8` byte buffer shared by N sub-pools.

Two END `MultiEndedAllocator`s grow inward from opposite ends; optional
"float" MIDDLE pools live between their frontiers (chain order
`[up end, floats..., down end]`). Eager- or lazy-compacting `free` keeps each
pool's byte range reclaimable. Layout is envelope-major (a slot's data for all
its layers in one contiguous byte envelope) so a freed slot vacates a region a
neighbor can grow into. Everything above the allocator stores virtual slot
IDs; the allocator owns the per-sub-pool virtual<->physical tables and
compaction only mutates those (no reference rewriting).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Mapping, NamedTuple, Optional, Tuple

import torch
from torch.profiler import record_function

from sglang.kernels.ops.kvcache.zero_pages import zero_pages
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.config import QSAProfile
from sglang.srt.mem_cache.layout.page_major import (
    build_mha_views,
    build_mla_views,
    build_page_major_mamba_views,
    build_row_block_views,
    build_slot_sibling_views,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MambaPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    unwrap_write_loc,
)
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


def _prod(iterable) -> int:
    out = 1
    for x in iterable:
        out *= int(x)
    return out


def _store_dtype_for(kv_cache_dtype: torch.dtype) -> torch.dtype:
    if kv_cache_dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        return torch.uint8
    return kv_cache_dtype


@dataclass(frozen=True, kw_only=True)
class SubPoolSpec(ABC):
    """Abstract per-slot layout of one sub-pool in a `UnifiedKVPool`.

    ``grow_direction`` places the sub-pool in the buffer's chain: the two
    ``"up"``/``"down"`` END pools own the buffer's two ends and grow inward;
    ``"float"`` middles live between the ends' frontiers (a float's position is
    carried entirely by its allocator's two watermarks — every view spans the
    whole buffer, so relocation never rebuilds views).
    """

    # Grow directions this subclass accepts. Scratch-class specs (e.g. the
    # spec-decode band) narrow this to ("float",).
    _allowed_grow_directions: ClassVar[Tuple[str, ...]] = ("up", "down", "float")

    name: str
    layer_num: int
    grow_direction: str  # "up" | "down" | "float"

    def __post_init__(self):
        assert self.grow_direction in self._allowed_grow_directions, (
            f"{type(self).__name__}.grow_direction must be one of "
            f"{self._allowed_grow_directions}; got {self.grow_direction!r}"
        )
        assert self.layer_num > 0, f"layer_num must be positive; got {self.layer_num}"

    @abstractmethod
    def entry_bytes(self) -> int:
        """Bytes for one slot across all `layer_num` layers."""
        raise NotImplementedError

    @abstractmethod
    def get_dtype(self) -> torch.dtype:
        """Storage dtype (informational). Multi-dtype subclasses return the dominant buffer's."""
        raise NotImplementedError

    def view_tail_pad_bytes(self, page_size: int) -> int:
        """Bytes this sub-pool's views reach PAST its last page envelope."""
        return 0

    def blocks_per_page(self) -> int:
        """Row-blocks one page holds in this sub-pool's kernel-facing id space.

        The page envelope is a uniform array of equally wide row-blocks, so a
        kernel-facing id is the physical page scaled by this count (see
        `MultiEndedAllocator.translate_kv_loc_for_kernel`). 1 means the kernel-facing ids are the physical ones.
        """
        return 1

    def kernel_page_stride(self, page_size: int) -> int:
        """Kernel-facing ids one physical page spans: ``kernel_id(t) =
        phys_page * stride + t % ps``."""
        return self.blocks_per_page() * page_size


@dataclass(frozen=True, kw_only=True)
class MHASubPoolSpec(SubPoolSpec):
    """Per-slot layout of one MHA-shaped sub-pool. `v_head_dim` defaults to `head_dim`.

    The page envelope may also carry an MTP draft's K/V layers after the
    target's, then compressed index-key rows (``ps // index_ratio`` per layer,
    target layers first), so compaction moves them with the page they index.
    """

    head_num: int
    head_dim: int
    store_dtype: torch.dtype
    v_head_dim: Optional[int] = None
    draft_layer_num: int = 0
    index_layer_num: int = 0
    index_head_num: int = 0
    index_head_dim: int = 0
    index_dtype: torch.dtype = torch.bfloat16
    index_ratio: int = 1

    def __post_init__(self):
        super().__post_init__()
        assert self.head_num > 0, f"head_num must be positive; got {self.head_num}"
        assert self.head_dim > 0, f"head_dim must be positive; got {self.head_dim}"
        if self.v_head_dim is None:
            object.__setattr__(self, "v_head_dim", self.head_dim)
        assert self.v_head_dim > 0, (
            f"v_head_dim must be positive; got {self.v_head_dim}"
        )
        assert self.draft_layer_num >= 0 and self.index_layer_num >= 0
        assert (
            self.index_row_bytes() * self.index_layer_num
        ) % self.index_ratio == 0, (
            f"index rows of {self.index_row_bytes()} B x {self.index_layer_num} "
            f"layers do not split into whole bytes per token at ratio "
            f"{self.index_ratio}"
        )

    def k_row_bytes(self) -> int:
        return self.head_num * self.head_dim * self.store_dtype.itemsize

    def v_row_bytes(self) -> int:
        return self.head_num * self.v_head_dim * self.store_dtype.itemsize

    def index_row_bytes(self) -> int:
        return self.index_head_num * self.index_head_dim * self.index_dtype.itemsize

    def envelope_layer_num(self) -> int:
        """K/V layers per page envelope: the target's, then the draft's."""
        return self.layer_num + self.draft_layer_num

    def kv_bytes_per_token(self, layer_num: int) -> int:
        return layer_num * (self.k_row_bytes() + self.v_row_bytes())

    def index_bytes_per_token(self, layer_num: int) -> int:
        return layer_num * self.index_row_bytes() // self.index_ratio

    def entry_bytes(self) -> int:
        return self.kv_bytes_per_token(
            self.envelope_layer_num()
        ) + self.index_bytes_per_token(self.index_layer_num)

    # Page-major byte math: within a page block K/V group per layer
    # [L0_K*ps | L0_V*ps | L1_K*ps | ...], then the index rows [I0*ps/r | ...];
    # at ps==1 this collapses to the per-slot envelope.

    def page_bytes(self, page_size: int) -> int:
        return page_size * self.entry_bytes()

    def view_tail_pad_bytes(self, page_size: int) -> int:
        return page_size * self.entry_bytes()

    def blocks_per_page(self) -> int:
        """Row-blocks per page in the kernel-facing id space (one K + one V per layer)."""
        assert self.index_layer_num == 0, (
            "an envelope with index rows is not a whole number of K/V blocks; "
            "use kernel_page_stride"
        )
        return 2 * self.envelope_layer_num()

    def kernel_page_stride(self, page_size: int) -> int:
        page_bytes = self.page_bytes(page_size)
        assert page_bytes % self.k_row_bytes() == 0, (
            f"page envelope of {page_bytes} B is not a whole number of "
            f"{self.k_row_bytes()} B K/V rows"
        )
        return page_bytes // self.k_row_bytes()

    def index_block_offsets_bytes(self, page_size: int) -> List[int]:
        """Byte offset of each index layer's rows inside a page envelope."""
        kv_bytes = page_size * self.kv_bytes_per_token(self.envelope_layer_num())
        rows_bytes = page_size // self.index_ratio * self.index_row_bytes()
        return [kv_bytes + j * rows_bytes for j in range(self.index_layer_num)]

    def index_page_multiplier(self, page_size: int) -> int:
        """Index pages one physical page spans in the index views' row space:
        a full page's index rows sit at ``phys_page * this * ps / ratio``."""
        assert page_size % self.index_ratio == 0
        page_index_bytes = page_size // self.index_ratio * self.index_row_bytes()
        page_bytes = self.page_bytes(page_size)
        assert page_bytes % page_index_bytes == 0, (
            f"page envelope of {page_bytes} B is not a whole number of "
            f"{page_index_bytes} B index pages"
        )
        return page_bytes // page_index_bytes

    def get_dtype(self) -> torch.dtype:
        return self.store_dtype


@dataclass(frozen=True, kw_only=True)
class MLASubPoolSpec(SubPoolSpec):
    """Per-slot layout of one MLA-shaped sub-pool.

    One latent row (``kv_lora_rank + qk_rope_head_dim``) per token per layer; V
    is a prefix slice of the same row, so there is no separate V region. Not a
    subclass of ``MHASubPoolSpec`` — the K+V byte math and the ``v_head_dim > 0``
    invariant there do not apply.
    """

    kv_lora_rank: int
    qk_rope_head_dim: int
    store_dtype: torch.dtype

    def __post_init__(self):
        super().__post_init__()
        assert self.kv_lora_rank > 0, (
            f"kv_lora_rank must be positive; got {self.kv_lora_rank}"
        )
        assert self.qk_rope_head_dim > 0, (
            f"qk_rope_head_dim must be positive; got {self.qk_rope_head_dim}"
        )

    @property
    def kv_cache_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    def entry_bytes(self) -> int:
        return self.layer_num * self.kv_cache_dim * self.store_dtype.itemsize

    def view_tail_pad_bytes(self, page_size: int) -> int:
        return page_size * self.entry_bytes()

    def blocks_per_page(self) -> int:
        """One latent row per layer, so L blocks per page (MHA has 2L: a K
        block and a V block per layer)."""
        return self.layer_num

    def get_dtype(self) -> torch.dtype:
        return self.store_dtype


@dataclass(frozen=True, kw_only=True)
class MambaSubPoolSpec(SubPoolSpec):
    """Per-slot layout of one Mamba-shaped sub-pool."""

    conv_state_shapes: Tuple[Tuple[int, ...], ...]  # one shape per conv tensor
    conv_dtype: torch.dtype
    temporal_state_shape: Tuple[int, ...]
    temporal_dtype: torch.dtype
    conv_slice_axis: int = 0
    # Per-slot side states packed after the temporal rows, as (shape, dtype),
    # so compaction moves them with their slot.
    sibling_layouts: Tuple[Tuple[Tuple[int, ...], torch.dtype], ...] = ()

    def __post_init__(self):
        super().__post_init__()
        assert len(self.conv_state_shapes) > 0, "conv_state_shapes must be non-empty"

    def conv_row_bytes(self, idx: int) -> int:
        return _prod(self.conv_state_shapes[idx]) * self.conv_dtype.itemsize

    def temporal_row_bytes(self) -> int:
        return _prod(self.temporal_state_shape) * self.temporal_dtype.itemsize

    def state_bytes(self) -> int:
        total = 0
        for i in range(len(self.conv_state_shapes)):
            total += self.layer_num * self.conv_row_bytes(i)
        total += self.layer_num * self.temporal_row_bytes()
        return total

    def sibling_offset_bytes(self) -> int:
        # 8-byte aligned so an int64 side state can follow any state dtype.
        return -(-self.state_bytes() // 8) * 8

    def entry_bytes(self) -> int:
        if not self.sibling_layouts:
            return self.state_bytes()
        siblings = sum(
            _prod(shape) * dtype.itemsize for shape, dtype in self.sibling_layouts
        )
        return -(-(self.sibling_offset_bytes() + siblings) // 8) * 8

    def get_dtype(self) -> torch.dtype:
        return self.conv_dtype  # representative state dtype; matches MambaPool.dtype


# ---------------------------------------------------------------------------
# UnifiedKVPool — the byte buffer + the per-sub-pool views
# ---------------------------------------------------------------------------


def unified_memory_supported_for_model(model_config, *, use_mla_backend: bool) -> bool:
    """Whether this model's KV geometry can back the unified memory pool."""
    return use_mla_backend or not model_config.has_asymmetric_kv


def _assert_kernel_id_bound(*, sub_pool_name: str, n_rows: int) -> None:
    """Check if kernel-facing ids can flow through int32 read-index buffers."""
    assert n_rows < 2**31, (
        f"sub-pool {sub_pool_name!r}: kernel-facing id space has {n_rows} rows, "
        f"exceeding the int32 bound (2^31) that read-index buffers assume. "
        "Reduce max_total_num_tokens or the layer count."
    )


def _reserved_floor_bytes(sub_pool_specs: List[SubPoolSpec], page_size: int) -> int:
    """Bytes at the bottom of the buffer reserved as the slot-0 padding sink.

    Slot-0 dummy writes for every sub-pool land here; each sub-pool's first
    allocatable slot is chosen so real data starts past it. For a PAGE-AWARE
    sub-pool the slot-0 write touches layer blocks spread across the whole
    page-0 envelope (page_size * entry_bytes), not just one slot envelope --
    but a mamba sub-pool is page_size=1, so its entry is charged ONCE. Charging
    a mamba entry per page would reserve page_size * ~100 MB of buffer that the
    sink never touches.

    Single source of truth: `UnifiedKVPool` reserves exactly this, and the
    factories' bs=1 feasibility floors charge exactly this.
    """
    return max(
        [max(s.entry_bytes() for s in sub_pool_specs)]
        + [
            page_size * s.entry_bytes()
            for s in sub_pool_specs
            if not isinstance(s, MambaSubPoolSpec)  # mamba is page_size=1
        ]
    )


class UnifiedKVPool:
    """One physical `uint8` byte buffer shared by N sub-pools, each exposing
    per-layer views over its own byte range (contiguous per layer for KV,
    strided for the Mamba state). Two END pools (one grow-up, one grow-down)
    own the buffer's ends; optional "float" MIDDLE pools live between their
    frontiers. Allocators keep byte ranges disjoint; no usage tracking here.
    """

    def __init__(
        self,
        *,
        total_bytes: int,
        sub_pool_specs: List[SubPoolSpec],
        device: str,
        enable_memory_saver: bool,
        page_size: int = 1,
    ):
        assert page_size >= 1, f"page_size must be >= 1; got {page_size}"
        assert len(sub_pool_specs) >= 2, (
            f"UnifiedKVPool needs >= 2 sub-pools; got {len(sub_pool_specs)}"
        )
        names = [s.name for s in sub_pool_specs]
        assert len(set(names)) == len(names), (
            f"sub-pool names must be unique; got {names}"
        )
        # Per-spec direction validity already ran in each spec's __post_init__.
        up_specs = [s for s in sub_pool_specs if s.grow_direction == "up"]
        down_specs = [s for s in sub_pool_specs if s.grow_direction == "down"]
        float_specs = [s for s in sub_pool_specs if s.grow_direction == "float"]
        assert len(up_specs) == 1 and len(down_specs) == 1, (
            f"UnifiedKVPool needs exactly one grow-up and one grow-down END "
            f"sub-pool; got directions "
            f"{[s.grow_direction for s in sub_pool_specs]}"
        )

        self.device = device
        self.total_bytes = total_bytes
        # Canonical chain order, low byte end -> high: grow-up end, float
        # middles (input order preserved), grow-down end. The allocators'
        # neighbour wiring follows it; all other access is by-name.
        self.sub_pool_specs: List[SubPoolSpec] = [
            up_specs[0],
            *float_specs,
            down_specs[0],
        ]
        self._page_size = page_size
        self._specs_by_name: Dict[str, SubPoolSpec] = {
            s.name: s for s in sub_pool_specs
        }

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.view_tail_pad_bytes = max(
            spec.view_tail_pad_bytes(page_size) for spec in sub_pool_specs
        )
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self._raw = torch.empty(
                total_bytes + self.view_tail_pad_bytes, dtype=torch.uint8, device=device
            )
        if envs.SGLANG_DEBUG_POISON_POOL.get():
            # Debug: bf16-NaN-fill so NaN-unsafe reads of never-written bytes
            # fail deterministically.
            self._raw.view(torch.int16).fill_(0x7FC1)
            logger.warning(
                "[unified-memory-pool] POISONED: pool filled with bf16-NaN "
                "patterns (SGLANG_DEBUG_POISON_POOL)"
            )
        else:
            self._raw.zero_()  # unset slots must read as zeros (matches non-shared)

        self._max_slots: Dict[str, int] = {}
        self._anchor_bytes: Dict[str, int] = {}
        self._min_slot_index: Dict[str, int] = {}
        # MHA: (k_buffer, v_buffer); MLA: [per-layer per-layer views];
        # Mamba: (conv_state_list, temporal_state)
        self._mha_views: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
        self._index_views: Dict[str, List[torch.Tensor]] = {}
        self._mla_views: Dict[str, List[torch.Tensor]] = {}
        self._mamba_views: Dict[str, Tuple[List[torch.Tensor], torch.Tensor]] = {}
        self._mamba_sibling_views: Dict[str, List[torch.Tensor]] = {}

        # Slot-0 dummy writes for both pools land in the reserved low-byte sink;
        # each pool's first allocatable slot is chosen so real data starts past it.
        # For a page-aware sub-pool the slot-0 write touches layer blocks spread
        # across the WHOLE page-0 envelope (up to page_size * entry_bytes), not
        # just one slot envelope — reserve the max of both.
        reserved_floor = _reserved_floor_bytes(self.sub_pool_specs, page_size)

        for spec in self.sub_pool_specs:
            entry_bytes = spec.entry_bytes()
            max_slots = total_bytes // entry_bytes
            min_slot_index = (reserved_floor + entry_bytes - 1) // entry_bytes  # ceil
            if max_slots <= min_slot_index:
                raise RuntimeError(
                    f"UnifiedKVPool: sub-pool {spec.name!r} fits only {max_slots} "
                    f"slots in {total_bytes} bytes, but min_slot_index={min_slot_index} "
                    f"leaves no room for real data. Increase total_bytes."
                )
            anchor = 0
            self._max_slots[spec.name] = max_slots
            self._anchor_bytes[spec.name] = anchor
            self._min_slot_index[spec.name] = min_slot_index
            if isinstance(spec, MHASubPoolSpec):
                self._mha_views[spec.name] = self._build_mha_views(
                    spec,
                    anchor,
                    max_slots,
                    page_size=page_size,
                )
                self._index_views[spec.name] = self._build_index_views(
                    spec, anchor, max_slots, page_size=page_size
                )
            elif isinstance(spec, MLASubPoolSpec):
                self._mla_views[spec.name] = self._build_mla_views(
                    spec,
                    anchor,
                    max_slots,
                    page_size=page_size,
                )
            elif isinstance(spec, MambaSubPoolSpec):
                self._mamba_views[spec.name] = self._build_mamba_views(
                    spec, anchor, max_slots
                )
                self._mamba_sibling_views[spec.name] = build_slot_sibling_views(
                    self._raw,
                    layouts=spec.sibling_layouts,
                    entry_bytes=spec.entry_bytes(),
                    first_offset_bytes=spec.sibling_offset_bytes(),
                    max_slots=max_slots,
                    anchor_bytes=anchor,
                )
            else:  # pragma: no cover
                raise TypeError(f"unsupported SubPoolSpec type: {type(spec)}")

        logger.info(
            "[unified-memory-pool] UnifiedKVPool allocated: total_bytes=%.2f GB (=%d B), "
            "%d sub-pool(s)",
            total_bytes / GB,
            total_bytes,
            len(self.sub_pool_specs),
        )
        for s in self.sub_pool_specs:
            logger.info(
                "[unified-memory-pool]   sub-pool %r: kind=%s, layer_num=%d, grow=%s, "
                "entry_bytes=%d, max_slots=%d, min_slot_index=%d (slots [0,%d) reserved)",
                s.name,
                type(s).__name__,
                s.layer_num,
                s.grow_direction,
                s.entry_bytes(),
                self._max_slots[s.name],
                self._min_slot_index[s.name],
                self._min_slot_index[s.name],
            )

    # -- introspection --

    def spec(self, name: str) -> SubPoolSpec:
        return self._specs_by_name[name]

    def mha_spec(self, name: str) -> MHASubPoolSpec:
        s = self._specs_by_name[name]
        assert isinstance(s, MHASubPoolSpec), (
            f"sub-pool {name!r} is {type(s).__name__}, expected MHASubPoolSpec"
        )
        return s

    def mla_spec(self, name: str) -> MLASubPoolSpec:
        s = self._specs_by_name[name]
        assert isinstance(s, MLASubPoolSpec), (
            f"sub-pool {name!r} is {type(s).__name__}, expected MLASubPoolSpec"
        )
        return s

    def mamba_spec(self, name: str) -> MambaSubPoolSpec:
        s = self._specs_by_name[name]
        assert isinstance(s, MambaSubPoolSpec), (
            f"sub-pool {name!r} is {type(s).__name__}, expected MambaSubPoolSpec"
        )
        return s

    def max_slots(self, name: str) -> int:
        return self._max_slots[name]

    def min_slot_index(self, name: str) -> int:
        return self._min_slot_index[name]

    def anchor_bytes(self, name: str) -> int:
        anchor = self._anchor_bytes[name]
        assert anchor == 0, f"current design assumes all anchors are 0; got {anchor}"
        return anchor

    def mha_views_for(self, name: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self._mha_views[name]

    def mla_views_for(self, name: str) -> List[torch.Tensor]:
        return self._mla_views[name]

    def mamba_views_for(self, name: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self._mamba_views[name]

    def index_views_for(self, name: str) -> List[torch.Tensor]:
        """Per-layer ``(rows, index_head_num, index_head_dim)`` index-key views."""
        return self._index_views[name]

    def mamba_sibling_views_for(self, name: str) -> List[torch.Tensor]:
        """``(max_slots, *shape)`` views of each slot's side states."""
        return self._mamba_sibling_views[name]

    def _build_mha_views(
        self,
        spec: MHASubPoolSpec,
        anchor_bytes: int,
        max_slots: int,
        page_size: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        num_pages = max_slots // page_size
        page_rows = spec.kernel_page_stride(page_size)
        _assert_kernel_id_bound(sub_pool_name=spec.name, n_rows=num_pages * page_rows)
        return build_mha_views(
            self._raw,
            layer_num=spec.envelope_layer_num(),
            head_num=spec.head_num,
            head_dim=spec.head_dim,
            v_head_dim=spec.v_head_dim,
            store_dtype=spec.store_dtype,
            page_size=page_size,
            num_pages=num_pages,
            anchor_bytes=anchor_bytes,
            page_rows=page_rows,
        )

    def _build_index_views(
        self,
        spec: MHASubPoolSpec,
        anchor_bytes: int,
        max_slots: int,
        page_size: int,
    ) -> List[torch.Tensor]:
        if spec.index_layer_num == 0:
            return []
        num_pages = max_slots // page_size
        page_bytes = spec.page_bytes(page_size)
        _assert_kernel_id_bound(
            sub_pool_name=spec.name,
            n_rows=num_pages * page_bytes // spec.index_row_bytes(),
        )
        return build_row_block_views(
            self._raw,
            block_offsets_bytes=spec.index_block_offsets_bytes(page_size),
            row_shape=(spec.index_head_num, spec.index_head_dim),
            dtype=spec.index_dtype,
            page_bytes=page_bytes,
            num_pages=num_pages,
            anchor_bytes=anchor_bytes,
        )

    def _build_mla_views(
        self,
        spec: MLASubPoolSpec,
        anchor_bytes: int,
        max_slots: int,
        page_size: int,
    ) -> List[torch.Tensor]:
        num_pages = max_slots // page_size
        _assert_kernel_id_bound(
            sub_pool_name=spec.name,
            n_rows=num_pages * spec.blocks_per_page() * page_size,
        )
        return build_mla_views(
            self._raw,
            layer_num=spec.layer_num,
            kv_cache_dim=spec.kv_cache_dim,
            store_dtype=spec.store_dtype,
            page_size=page_size,
            num_pages=num_pages,
            anchor_bytes=anchor_bytes,
        )

    def _build_mamba_views(
        self, spec: MambaSubPoolSpec, anchor_bytes: int, max_slots: int
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return build_page_major_mamba_views(
            self._raw,
            layer_num=spec.layer_num,
            conv_state_shapes=spec.conv_state_shapes,
            conv_dtype=spec.conv_dtype,
            temporal_state_shape=spec.temporal_state_shape,
            temporal_dtype=spec.temporal_dtype,
            max_slots=max_slots,
            anchor_bytes=anchor_bytes,
            entry_bytes=spec.entry_bytes(),
        )


class UnifiedMHATokenToKVPool(MHATokenToKVPool):
    """MHA KV pool whose per-layer `k_buffer`/`v_buffer` are `build_mha_views`
    views into a `UnifiedKVPool` (requires uniform K/V rows).

    Views are contiguous `(n_rows, head_num, head_dim)`; locs are

        kernel_id(t) = (t // ps) * (ps * 2 * layer_num) + t % ps

    which is layer- and K/V-independent, each view's storage_offset folding in
    its block origin (layer l's K at block 2l, V at 2l+1). `move_kv_cache` is
    the exception: compaction passes REAL physical token ids. When the envelope
    also holds draft layers or index rows the page stride is
    `spec.kernel_page_stride(ps)`, and a draft pool is a second instance over
    envelope layers ``[first_layer, first_layer + layer_count)`` past the target's.
    """

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        sub_pool_name: str,
        page_size: int = 1,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        first_layer: int = 0,
        layer_count: Optional[int] = None,
    ):
        spec = unified_buffer.mha_spec(sub_pool_name)
        k_views, v_views = unified_buffer.mha_views_for(sub_pool_name)
        max_slots = unified_buffer.max_slots(sub_pool_name)
        layer_num = spec.layer_num if layer_count is None else layer_count
        layers = slice(first_layer, first_layer + layer_num)
        assert 0 < layer_num and layers.stop <= spec.envelope_layer_num(), (
            f"sub-pool {sub_pool_name!r} has no envelope layers {layers}"
        )

        self._unified_buffer = unified_buffer
        self._sub_pool_name = sub_pool_name
        self._k_views = k_views[layers]
        self._v_views = v_views[layers]
        self._num_pages = max_slots // page_size
        self._page_bytes = page_size * spec.entry_bytes()
        stride = spec.kernel_page_stride(page_size)
        view_rows = self._num_pages * stride

        super().__init__(
            size=view_rows - page_size,
            page_size=page_size,
            dtype=spec.store_dtype,
            head_num=spec.head_num,
            head_dim=spec.head_dim,
            layer_num=layer_num,
            device=unified_buffer.device,
            enable_memory_saver=False,  # buffer owned by UnifiedKVPool
            v_head_dim=spec.v_head_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=False,
            kv_cache_layout="page_major",
        )
        self.kernel_page_stride = stride
        self.kernel_page_blocks = (
            stride // page_size if stride % page_size == 0 else None
        )

    @property
    def unified_buffer(self) -> UnifiedKVPool:
        return self._unified_buffer

    def _create_buffers(self):
        self.k_buffer = self._k_views
        self.v_buffer = self._v_views
        # This override must initialize the pointers and row strides used by HiCache.
        self._init_data_ptrs_and_strides()

    def _clear_buffers(self):
        # Lifetime owned by UnifiedKVPool; do not delete the views.
        pass

    def get_kv_size_bytes(self):
        return 0, 0  # UnifiedKVPool logs the total; per-sub-pool would double-count

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """Relocate slots by whole page envelope.
        `tgt_loc`/`src_loc` are REAL physical token ids, not kernel-facing ids.
        """
        if tgt_loc.numel() == 0:
            return
        # The envelope view below starts at byte 0, so this sub-pool must be
        # anchored there; a non-zero anchor moves another sub-pool's bytes, and
        # the ids stay in range so nothing downstream notices.
        assert self._unified_buffer.anchor_bytes(self._sub_pool_name) == 0
        ps = self.page_size
        tgt_pages = tgt_loc.view(-1, ps)[:, 0] // ps
        src_pages = src_loc.view(-1, ps)[:, 0] // ps
        with record_function("UnifiedMHA.move_kv_cache"):
            env = self._unified_buffer._raw[: self._num_pages * self._page_bytes].view(
                self._num_pages, self._page_bytes
            )
            env[tgt_pages] = env[src_pages]

    def get_contiguous_buf_infos(self):
        """Register the raw buffer as physical page envelopes for PD transfer.

        Full and SWA expose the same allocation with different envelope sizes;
        the transfer backend preserves both logical entries while deduplicating
        the underlying memory registration.
        """
        assert self._unified_buffer.anchor_bytes(self._sub_pool_name) == 0
        raw = self._unified_buffer._raw
        return [raw.data_ptr()], [raw.numel()], [self._page_bytes]

    def get_page_envelope_buffer(self) -> torch.Tensor:
        """Return this sub-pool's page-strided view of the shared allocation."""
        assert self._unified_buffer.anchor_bytes(self._sub_pool_name) == 0
        raw = self._unified_buffer._raw
        page_count = (
            self._unified_buffer.max_slots(self._sub_pool_name) // self.page_size
        )
        return raw[: page_count * self._page_bytes].view(page_count, self._page_bytes)

    @property
    def grow_direction(self) -> str:
        """Physical growth direction used by this sub-pool's L1 allocator."""
        return self._unified_buffer.spec(self._sub_pool_name).grow_direction

    def _physical_to_kernel_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return (
            indices // self.page_size
        ) * self.kernel_page_stride + indices % self.page_size

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        """Translate physical host-pool ids for the page-major parent path."""
        return super().get_cpu_copy(self._physical_to_kernel_indices(indices))

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        super().load_cpu_copy(kv_cache_cpu, self._physical_to_kernel_indices(indices))

    def set_kv_buffer_prefix_valid(self, *args, **kwargs):
        raise NotImplementedError(
            "prefix-valid commit is unsupported under the unified layout "
            "(_set_kv_buffer_prefix_valid_impl assumes token-id indexing)."
        )


class UnifiedMLATokenToKVPool(MLATokenToKVPool):
    """MLA KV pool whose per-layer `kv_buffer` entries are kernel-facing views into a
    `UnifiedKVPool` (see `build_mla_views`).

    Loc-space contract: every loc this pool receives through the KVCache API
    (`set_kv_buffer` / `set_mla_kv_buffer` / `get_mla_kv_buffer`, and the
    kv_indices consumed by attention kernels reading `get_key_buffer` /
    `get_value_buffer`) is a kernel-facing id — the `translate_kv_loc_for_kernel` output

        kernel_id(t) = (t // ps) * (ps * layer_num) + t % ps

    which is layer-independent (the layer offset is folded into each view's
    storage_offset), so the stock `MLATokenToKVPool` read/write methods work on
    the views unmodified. The ONE exception is `move_kv_cache`: the allocator's
    compaction calls it with REAL physical token ids, and it is overridden to
    relocate whole page envelopes on the raw buffer.
    """

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        sub_pool_name: str,
        kv_cache_dtype: torch.dtype,
        page_size: int = 1,
    ):
        spec = unified_buffer.mla_spec(sub_pool_name)
        store_dtype = _store_dtype_for(kv_cache_dtype)
        assert spec.store_dtype == store_dtype, (
            f"sub-pool {sub_pool_name!r} store dtype {spec.store_dtype} does not "
            f"match kv cache dtype {kv_cache_dtype} (store {store_dtype})"
        )

        self._unified_buffer = unified_buffer
        self._sub_pool_name = sub_pool_name
        self._kv_views = unified_buffer.mla_views_for(sub_pool_name)
        max_slots = unified_buffer.max_slots(sub_pool_name)
        self._num_pages = max_slots // page_size
        self._page_bytes = page_size * spec.entry_bytes()
        self._view_rows = self._num_pages * spec.blocks_per_page() * page_size

        super().__init__(
            # OOB checks bound locs by `size + page_size`; kernel-facing ids run to
            # `_view_rows` (page 0 is the reserved padding sink).
            size=self._view_rows - page_size,
            page_size=page_size,
            dtype=kv_cache_dtype,
            kv_lora_rank=spec.kv_lora_rank,
            qk_rope_head_dim=spec.qk_rope_head_dim,
            layer_num=spec.layer_num,
            device=unified_buffer.device,
            enable_memory_saver=False,  # buffer owned by UnifiedKVPool
        )
        self.kernel_page_blocks = spec.blocks_per_page()
        self.kernel_page_stride = spec.kernel_page_stride(page_size)

    def _create_buffers(self):
        self.kv_buffer = self._kv_views

    def _clear_buffers(self):
        # Lifetime owned by UnifiedKVPool; do not delete the views.
        pass

    # `rebind_write_loc` already collapsed the widened id and sent the rows this
    # rank does not own to the padding sink.
    write_loc_is_dcp_resolved = True

    def get_kv_size_bytes(self):
        return 0  # UnifiedKVPool logs the total; per-sub-pool would double-count

    def get_contiguous_buf_infos(self):
        """PD-transfer registration: ONE entry, the raw buffer, addressed as
        ``raw_ptr + physical_page_id * page_envelope_bytes``.

        The transfer item is the whole page envelope (all layers of one page)
        rather than a per-layer region, because the per-layer per-layer views
        overlap and index in kernel-facing ids. Both sides must therefore build the
        pool with identical specs.
        """
        # The address formula omits the anchor; a nonzero one would mis-address.
        assert self._unified_buffer.anchor_bytes(self._sub_pool_name) == 0
        raw = self._unified_buffer._raw
        return [raw.data_ptr()], [raw.numel()], [self._page_bytes]

    def _physical_to_kernel_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Physical TOKEN ids -> the kernel-facing ids this class's `kv_buffer`
        views are indexed by; the formula is the one in the class docstring."""
        return (indices // self.page_size) * (
            self.page_size * self.kernel_page_blocks
        ) + indices % self.page_size

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        """Translate physical host-pool ids for the page-major parent path."""
        return super().get_cpu_copy(self._physical_to_kernel_indices(indices))

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        super().load_cpu_copy(kv_cache_cpu, self._physical_to_kernel_indices(indices))

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """Relocate whole page envelopes.

        `tgt_loc`/`src_loc` are REAL physical token ids (NOT kernel-facing ids): both
        compaction paths expand page ids into page-major-ordered token runs
        (`pages[:, None] * ps + offsets`), relied on here to recover the page
        lists. One contiguous envelope copy replaces the per-layer strided moves.
        """
        if tgt_loc.numel() == 0:
            return
        ps = self.page_size
        tgt_pages = tgt_loc.view(-1, ps)[:, 0] // ps
        src_pages = src_loc.view(-1, ps)[:, 0] // ps
        with record_function("UnifiedMLA.move_kv_cache"):
            env = self._unified_buffer._raw[: self._num_pages * self._page_bytes].view(
                self._num_pages, self._page_bytes
            )
            env[tgt_pages] = env[src_pages]

    def zero_physical_pages(self, phys_pages: torch.Tensor) -> None:
        """Zero whole page envelopes (PHYSICAL page ids) on allocator
        hand-out."""
        zero_pages(
            self._unified_buffer._raw,
            phys_pages,
            self._num_pages,
            self._page_bytes,
        )


class UnifiedMambaPool(MambaPool):
    """Mamba state pool whose conv/temporal state are strided views into a `UnifiedKVPool`.

    Pure PHYSICAL store: slot lifecycle and the v<->p mapping live in the attached
    `UnifiedMambaSlotAllocator`. Does NOT call `super().__init__()` — replicates the
    minimal `MambaPool` state against the unified buffer so inherited methods work.
    """

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        sub_pool_name: str,
        spec_state_size: int,
        mamba_layer_ids: List[int],
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        spec = unified_buffer.mamba_spec(sub_pool_name)
        assert spec.layer_num == len(mamba_layer_ids)
        # PP disagg state transfer maps entries by global layer id.
        self.mamba_layer_ids = list(mamba_layer_ids)
        conv_views, temporal_view = unified_buffer.mamba_views_for(sub_pool_name)
        max_slots = unified_buffer.max_slots(sub_pool_name)

        self._unified_buffer = unified_buffer
        self._sub_pool_name = sub_pool_name

        # Replicate the state MambaPool.__init__ would have set.
        self._max_size = max_slots - 1  # -1 for reserved slot 0
        self.size = self._max_size
        self.device = unified_buffer.device
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None
        self.num_mamba_layers = spec.layer_num
        # GDN/KDA ReplaySSM / spec unsupported; replicate parent's disabled-state
        # attrs so unconditional reads (e.g. `replayssm_cache_base is not None` in
        # the req-slot alloc path) and `... is not None` guards don't AttributeError.
        self.enable_linear_replayssm = False
        self.linear_replayssm_cache_len = 16
        self.replayssm_write_pos = None
        self.replayssm_is_kda = False
        self.enable_linear_replayssm_spec = False
        self.replayssm_spec_fold = False
        self.replayssm_cache_base = None
        self.replayssm_is_flush = None
        self.replayssm_spec_write_pos = None
        self.debug_memory_pool = False
        self.conv_shard_groups = None
        self.conv_slice_axis = spec.conv_slice_axis

        assert conv_views[0].shape[0] == self.num_mamba_layers, (
            f"conv_views layers={conv_views[0].shape[0]} vs expected {self.num_mamba_layers}"
        )
        assert conv_views[0].shape[1] == self._max_size + 1, (
            f"conv_views slots={conv_views[0].shape[1]} vs expected {self._max_size + 1}"
        )

        # Per-draft-token intermediate buffers have a different outer size
        # (spec_state_size+1), so they're NOT in the shared buffer; allocate locally.
        temporal_state_shape = spec.temporal_state_shape
        conv_state_shape = spec.conv_state_shapes
        conv_dtype = spec.conv_dtype
        ssm_dtype = spec.temporal_dtype
        self._raw = None
        self._intermediate_conv_window_phys = []
        if speculative_num_draft_tokens is not None:
            with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                intermediate_ssm_state_cache = torch.zeros(
                    size=(
                        self.num_mamba_layers,
                        spec_state_size + 1,
                        speculative_num_draft_tokens,
                        temporal_state_shape[0],
                        temporal_state_shape[1],
                        temporal_state_shape[2],
                    ),
                    dtype=ssm_dtype,
                    device=unified_buffer.device,
                )
                intermediate_conv_window_cache = [
                    torch.zeros(
                        size=(
                            self.num_mamba_layers,
                            spec_state_size + 1,
                            speculative_num_draft_tokens,
                            cshape[0],
                            cshape[1],
                        ),
                        dtype=conv_dtype,
                        device=unified_buffer.device,
                    )
                    for cshape in conv_state_shape
                ]
            self._intermediate_conv_window_phys = intermediate_conv_window_cache
            self.mamba_cache = self.SpeculativeState(
                conv=list(conv_views),
                temporal=temporal_view,
                intermediate_ssm=intermediate_ssm_state_cache,
                intermediate_conv_window=intermediate_conv_window_cache,
            )
        else:
            self.mamba_cache = self.State(conv=list(conv_views), temporal=temporal_view)

        self.mem_usage = unified_buffer.total_bytes / GB
        logger.info(
            "[unified-memory-pool] UnifiedMambaPool(%s) wrapped unified buffer: max_slots=%d, "
            "num_mamba_layers=%d",
            sub_pool_name,
            max_slots,
            self.num_mamba_layers,
        )

    # Inherited MambaPool state ops (copy_from/clear_slots/get_cpu_copy/load_cpu_copy)
    # take PHYSICAL slot ids; callers translate via the slot allocator first.

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # Cross-pool physical-move contract, implemented by every pool the
        # MultiEndedAllocator wraps. Ids are PHYSICAL slots; `MambaPool.copy_from`
        # takes (src, dst), hence the swap.
        MambaPool.copy_from(self, src_loc, tgt_loc)

    # -- PD state transfer (StateType.MAMBA) --
    # The transfer item is the whole per-slot envelope, addressed as
    # `raw_ptr + physical_slot * entry_bytes`. An envelope cannot be TP-resliced
    # or PP-subset, so the per-tensor metadata below stays empty and both sides
    # must build identical mamba specs (equal attn TP, pp=1).

    def get_contiguous_buf_infos(self):
        # The address formula omits the anchor; a nonzero one would mis-address.
        assert self._unified_buffer.anchor_bytes(self._sub_pool_name) == 0
        spec = self._unified_buffer.mamba_spec(self._sub_pool_name)
        raw = self._unified_buffer._raw
        return [raw.data_ptr()], [raw.numel()], [spec.entry_bytes()]

    def get_state_dim_per_tensor(self):
        return []

    def get_state_layer_ids(self):
        return []

    def get_state_slice_outer_counts(self):
        return []

    def get_state_conv_shard_groups(self):
        return []


class UnifiedMambaSlotAllocator:
    """Mamba slot allocator (PHYSICAL view) for the unified memory pool.

    Owns slot alloc/free, sizing, and the v<->p mapping (``translate``), presenting the
    upstream ``MambaSlotAllocator`` interface. ``alloc()`` returns VIRTUAL ids and does
    NOT clear state — clearing is deferred to ``UnifiedMambaPool.clear_slots``.
    """

    def __init__(self, mea, max_size: int, device: str):
        self._multi_ended_allocator = mea
        self._max_size = max_size  # excludes reserved slot 0
        self._device = device
        self._alloc_iter = None  # active alloc_group batch iterator

    # -- translation (owns the v<->p mapping) --

    def translate(self, virtual_ids: torch.Tensor) -> torch.Tensor:
        # VIRTUAL -> PHYSICAL slot ids; page_size==1, so a direct v2p gather.
        return self._multi_ended_allocator.virtual_to_physical[virtual_ids]

    @property
    def virtual_to_physical(self) -> torch.Tensor:
        return self._multi_ended_allocator.virtual_to_physical

    # -- sizing / free-list --

    @property
    def size(self) -> int:
        return self._max_size

    def available_size(self) -> int:
        # Slot-conservation count (max - allocated): the leak-check view, NOT the
        # planner value (use schedulable_available_size for that).
        return self._max_size - self._multi_ended_allocator.allocated_count()

    def schedulable_available_size(self) -> int:
        # Byte-coordinated count (>= N => alloc(N) succeeds); credits the peer's
        # drainable holes since alloc flushes the peer before extending.
        return self._multi_ended_allocator.schedulable_available_size()

    @property
    def free_slots(self) -> torch.Tensor:
        # Watermark-derived physical free-list for the invariant checker.
        a = self._multi_ended_allocator
        assert a.page_size == 1, (
            "UnifiedMambaSlotAllocator.free_slots assumes page_size==1; got "
            f"{a.page_size}. Mamba state is per-request, orthogonal to paging."
        )
        if a.grow_direction == "up":
            start, end = a.watermark_physical, a.num_pages
        else:
            start, end = a.min_page_index, a.watermark_physical + 1
        if start >= end:
            return torch.empty((0,), dtype=torch.int64, device=self._device)
        return torch.arange(start, end, dtype=torch.int64, device=self._device)

    # -- slot management (delegates to the MultiEndedAllocator) --

    def alloc(self, need_size: int):
        # alloc_group fast path: single-slot draws from the prefetched batch.
        if self._alloc_iter is not None and need_size == 1:
            slot = next(self._alloc_iter, None)
            if slot is not None:
                return slot
        return self._multi_ended_allocator.alloc(need_size)  # VIRTUAL ids

    def free(self, free_index: torch.Tensor):
        return self._multi_ended_allocator.free(free_index)

    def clear(self):
        self._alloc_iter = None
        return self._multi_ended_allocator.clear()

    def alloc_group_begin(self, num_reqs: int):
        """Pre-allocate a batch that ``alloc(1)`` then draws from."""
        self._alloc_iter = None
        if num_reqs > 0:
            result = self._multi_ended_allocator.alloc(num_reqs)
            if result is not None:
                self._alloc_iter = iter(result.split(1))

    def alloc_group_end(self):
        """Return any unused pre-allocated slots from the current group."""
        if self._alloc_iter is not None:
            remaining = list(self._alloc_iter)
            if remaining:
                self._multi_ended_allocator.free(torch.cat(remaining))
        self._alloc_iter = None

    def is_slot_allocated(self, slot) -> bool:
        return self._multi_ended_allocator.is_slot_allocated(int(slot))

    def allocator_state_str(self) -> str:
        return self._multi_ended_allocator.allocator_state_str()


class UnifiedHybridReqToTokenPool(HybridReqToTokenPool):
    """`HybridReqToTokenPool` whose `mamba_pool` is a `UnifiedMambaPool`. The inherited
    mamba-id state now holds VIRTUAL ids; adds `translate_mamba_indices` for v->p."""

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        mamba_sub_pool_name: str,
        size: int,
        mamba_spec_state_size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params,
        mamba_layer_ids: List[int],
        enable_mamba_extra_buffer: bool,
        enable_mamba_extra_buffer_lazy: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
        enable_overlap_schedule: bool = True,
        start_layer: Optional[int] = None,
        pre_alloc_size: int = 0,
        short_conv_layer_ids: Optional[List[int]] = None,
        short_conv_state_shape: Optional[Tuple[int, int]] = None,
        ngram_context_len: int = 0,
        ngram_eos_token_id: int = 0,
    ):
        self._unified_buffer = unified_buffer
        self._mamba_sub_pool_name = mamba_sub_pool_name
        self._shared_mamba_size = (
            unified_buffer.max_slots(mamba_sub_pool_name) - 1
        )  # reserve slot 0
        super().__init__(
            # `DecodeReqToTokenPool` semantics: rows cover the preallocated
            # requests too, while `self.size` (rebound below) stays the
            # running-request cap the scheduler and leak invariant expect.
            size=size + pre_alloc_size,
            mamba_size=self._shared_mamba_size,
            mamba_spec_state_size=mamba_spec_state_size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
            enable_mamba_extra_buffer_lazy=enable_mamba_extra_buffer_lazy,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            enable_overlap_schedule=enable_overlap_schedule,
            start_layer=start_layer,
            short_conv_layer_ids=short_conv_layer_ids,
            short_conv_state_shape=short_conv_state_shape,
            ngram_context_len=ngram_context_len,
            ngram_eos_token_id=ngram_eos_token_id,
        )
        self.size = size
        self.pre_alloc_size = pre_alloc_size

    def _init_mamba_pool(
        self,
        mamba_size: int,
        mamba_spec_state_size: int,
        cache_params,
        mamba_layer_ids: List[int],
        device: str,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: Optional[int] = None,
        speculative_eagle_topk: Optional[int] = None,
        mamba_envelope_layout: bool = False,
        enable_linear_replayssm: bool = False,
        linear_replayssm_cache_len: int = 16,
        enable_linear_replayssm_spec: bool = False,
        short_conv_layer_ids: Optional[List[int]] = None,
        short_conv_state_shape=None,
        ngram_context_len: int = 0,
        ngram_eos_token_id: int = 0,
    ):
        # mamba_envelope_layout / speculative_eagle_topk / enable_linear_replayssm /
        # linear_replayssm_cache_len / enable_linear_replayssm_spec: accepted to match
        # the parent signature but NOT forwarded — the shared pool's conv/temporal
        # state are fixed-shape views (replayssm/spec are gated off under unified).
        assert mamba_size == self._shared_mamba_size, (
            f"UnifiedHybridReqToTokenPool._init_mamba_pool: mamba_size={mamba_size} "
            f"!= unified_buffer.max_slots({self._mamba_sub_pool_name!r}) - 1 "
            f"= {self._shared_mamba_size}"
        )
        assert len(cache_params.layers) >= len(mamba_layer_ids), (
            f"cache_params.layers ({len(cache_params.layers)}) cannot supply "
            f"{len(mamba_layer_ids)} mamba layer ids"
        )
        self.mamba_pool = UnifiedMambaPool(
            unified_buffer=self._unified_buffer,
            sub_pool_name=self._mamba_sub_pool_name,
            spec_state_size=mamba_spec_state_size,
            mamba_layer_ids=mamba_layer_ids,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        self._init_ple_side_states(
            mamba_size=mamba_size,
            mamba_spec_state_size=mamba_spec_state_size,
            conv_dtype=cache_params.dtype.conv,
            device=device,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            short_conv_layer_ids=short_conv_layer_ids,
            short_conv_state_shape=short_conv_state_shape,
            ngram_context_len=ngram_context_len,
            ngram_eos_token_id=ngram_eos_token_id,
        )
        # Wired in by init_unified_mamba_pools once the mamba allocator exists.
        self.mamba_allocator = None
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}
        self.mamba_ckpt_pool = None  # int8 ckpt pool unused; None = feature off
        self.device = device
        # Sized by req_to_token's first dim (size + 1; row 0 is padding); self.size
        # would under-size by one row.
        req_pool_size = self.req_to_token.shape[0]
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.zeros(
            req_pool_size, dtype=torch.int32, device=self.device
        )
        if enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping: torch.Tensor = (
                torch.zeros(
                    (req_pool_size, self.mamba_ping_pong_track_buffer_size),
                    # int64 to match the parent's uncast index_put source (int32 dest
                    # would dtype-mismatch on the first radix prefill).
                    dtype=torch.int64,
                    device=self.device,
                )
            )

    def _init_ple_side_states(
        self,
        *,
        mamba_size: int,
        mamba_spec_state_size: int,
        conv_dtype: torch.dtype,
        device: str,
        speculative_num_draft_tokens: Optional[int],
        short_conv_layer_ids: Optional[List[int]],
        short_conv_state_shape: Optional[Tuple[int, int]],
        ngram_context_len: int,
        ngram_eos_token_id: int,
    ) -> None:
        """PLE side states live in each PHYSICAL mamba slot's envelope, after
        the state rows (see `_ple_sibling_layouts` for the order), so
        compaction moves them with their slot."""
        from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool

        sibling_views = iter(
            self._unified_buffer.mamba_sibling_views_for(self._mamba_sub_pool_name)
        )
        ngram_context = next(sibling_views) if ngram_context_len > 0 else None
        conv_state = (
            next(sibling_views).transpose(0, 1)
            if short_conv_layer_ids and short_conv_state_shape is not None
            else None
        )
        assert next(sibling_views, None) is None, "unclaimed mamba slot side state"
        self.short_conv_pool = ShortConvPool(
            size=mamba_size,
            spec_state_size=mamba_spec_state_size,
            state_shape=short_conv_state_shape,
            layer_ids=short_conv_layer_ids or [],
            dtype=conv_dtype,
            device=device,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            conv_state=conv_state,
        )
        self.ngram_pool = NGramPool(
            size=mamba_size,
            spec_state_size=mamba_spec_state_size,
            context_len=ngram_context_len,
            eos_token_id=ngram_eos_token_id,
            device=device,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            context=ngram_context,
        )
        if self.short_conv_pool.enabled:
            self.mamba_pool.register_slot_state(self.short_conv_pool)
        if self.ngram_pool.enabled:
            self.mamba_pool.register_slot_state(self.ngram_pool)

    @property
    def mamba_v2p_table(self) -> Optional[torch.Tensor]:
        """This pool's ids ARE virtual; page_size is 1, so the translate is the
        plain gather this table serves, which keeps `mamba_translate_is_fusable`
        true despite the override."""
        if self.mamba_allocator is None:
            return None
        return self.mamba_allocator.virtual_to_physical

    def translate_mamba_indices(self, virtual_ids: torch.Tensor) -> torch.Tensor:
        """Virtual mamba ids -> physical slot ids."""
        return self.mamba_allocator.translate(virtual_ids).to(torch.int32)


class UnifiedHybridLinearKVPool(HybridLinearKVPool):
    """`HybridLinearKVPool` over unified sub-pools (full = Unified{MLA,MHA},
    mamba = UnifiedMambaPool)."""

    def get_kv_layer_ids(self):
        # Empty: the KV component is one whole-envelope entry, so there are no
        # per-layer entries to pair by layer id (the sender falls back to
        # positional pairing).
        return []


class UnifiedQSATokenToKVPool(QSATokenToKVPool):
    """`QSATokenToKVPool` over a unified full sub-pool.

    The full-attention K/V rows and the compressed index-key rows both live in
    the full sub-pool's page envelopes, so compaction moves a page's keys with
    its K/V; `qsa_index_slots` maps a virtual full slot into the index views'
    slot space. An MTP draft is a second instance over the draft layers of the
    same envelopes. The pending ring stays per request slot.
    """

    def get_kv_layer_ids(self):
        return []

    def get_kv_size_bytes(self):
        # UnifiedKVPool accounts every row it holds; only the ring is ours.
        k_size, v_size = HybridLinearKVPool.get_kv_size_bytes(self)
        ring_bytes = (
            sum(t.numel() * t.element_size() for t in self.qsa_key_state_buffer_pool)
            + self.qsa_rope_position_buffer.numel()
            * self.qsa_rope_position_buffer.element_size()
        )
        return k_size + ring_bytes, v_size


def _bind_unified_qsa_pool(
    pool: UnifiedQSATokenToKVPool, *, allocator, host_capacity_tokens: int
) -> None:
    """Address the pool's index rows through the allocator's page table and
    weigh its host tier by the rows it mirrors (the views alias one buffer)."""
    full_kv_pool = pool.full_kv_pool
    spec = full_kv_pool.unified_buffer.mha_spec("full")
    pool.qsa_index_v2p = allocator.full_v2p_page_table
    pool.qsa_index_page_multiplier = spec.index_page_multiplier(pool.page_size)
    # HiCache: the compressed-K sidecar is indexed by the anchor's token ids
    # // page_size, which this lands on the page's rows in the index views.
    pool.host_transfer_translate = pool.qsa_index_slots
    full_kv_pool.host_capacity_tokens = host_capacity_tokens
    full_kv_pool.host_capacity_bytes = host_capacity_tokens * spec.kv_bytes_per_token(
        full_kv_pool.layer_num
    )
    pool.qsa_host_capacity_bytes = host_capacity_tokens * spec.index_bytes_per_token(
        len(pool.qsa_compressed_k_buffer_pool)
    )


def build_unified_qsa_draft_pool(
    *,
    allocator,
    target_pool: UnifiedQSATokenToKVPool,
    full_attention_layer_ids: List[int],
    mamba_pool: MambaPool,
    start_layer: int,
    num_request_slots: int,
    draft_index: int = 0,
) -> UnifiedQSATokenToKVPool:
    """The QSA pool of MTP draft ``draft_index``, over the draft layers the
    target's page envelopes reserve (`MHASubPoolSpec.draft_layer_num`)."""
    target_full = target_pool.full_kv_pool
    shared_pool = target_full.unified_buffer
    spec = shared_pool.mha_spec("full")
    layer_count = len(full_attention_layer_ids)
    first_layer = spec.layer_num + draft_index * layer_count
    assert first_layer + layer_count <= spec.envelope_layer_num(), (
        f"draft {draft_index} needs {layer_count} full-attention layers past "
        f"the target's; the page envelope reserves {spec.draft_layer_num}"
    )
    page_size = target_full.page_size
    # Layer ids arrive dense (0..n-1) from HybridLinearKVPool, as for the
    # static draft pool, so the full pool keeps the default start layer.
    full_kv_pool = UnifiedMHATokenToKVPool(
        unified_buffer=shared_pool,
        sub_pool_name="full",
        page_size=page_size,
        first_layer=first_layer,
        layer_count=layer_count,
    )
    # One translate for target and draft, so HiCache packs the draft's rows
    # with the target's under the same device ids.
    full_kv_pool.host_transfer_translate = target_full.host_transfer_translate
    # The draft shares the envelope geometry, so it shares the target's.
    pool = UnifiedQSATokenToKVPool(
        qsa_index_kv_heads=target_pool.qsa_index_kv_heads,
        qsa_index_head_dim=target_pool.qsa_index_head_dim,
        qsa_compress_ratio=target_pool.qsa_compress_ratio,
        qsa_token_topk=target_pool.qsa_token_topk,
        num_request_slots=num_request_slots,
        full_kv_pool=full_kv_pool,
        compressed_k_views=shared_pool.index_views_for("full")[
            first_layer : first_layer + layer_count
        ],
        page_size=page_size,
        size=target_pool.size,
        dtype=target_pool.dtype,
        head_num=spec.head_num,
        head_dim=spec.head_dim,
        full_attention_layer_ids=full_attention_layer_ids,
        device=shared_pool.device,
        mamba_pool=mamba_pool,
        start_layer=start_layer,
    )
    _bind_unified_qsa_pool(
        pool,
        allocator=allocator,
        host_capacity_tokens=target_full.host_capacity_tokens,
    )
    return pool


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class UnifiedPoolBundle(NamedTuple):
    unified_memory_pool: UnifiedKVPool
    token_to_kv_pool: object  # HybridLinearKVPool
    token_to_kv_pool_allocator: object  # UnifiedMambaTokenToKVPoolAllocator
    req_to_token_pool: object  # UnifiedHybridReqToTokenPool


def _check_bs1_feasibility_floor(
    *,
    total_bytes: int,
    floor_terms: List[Tuple[str, int]],
    factory: str,
) -> None:
    """bs=1 feasibility FLOOR — the retract loop's terminal guarantee.

    The scheduler retracts requests until the LAST one fits; if one worst-case
    request running ALONE does not fit in the buffer, under-sizing is a retract
    LIVELOCK at runtime, not a perf bug. Fail loud at boot, before any pool
    construction, with the itemized requirement.
    """
    floor = sum(b for _, b in floor_terms)
    if total_bytes >= floor:
        return
    detail = " + ".join(f"{name}={b}" for name, b in floor_terms)
    raise RuntimeError(
        f"[unified-memory-pool] {factory}: byte budget {total_bytes} cannot fit "
        f"ONE worst-case request (bs=1 floor {floor} = {detail}). A pool this "
        f"size retract-livelocks at runtime. Raise --mem-fraction-static, lower "
        f"the model context length, or reduce reserved memory."
    )


def _wire_mamba_slot_allocator(
    *,
    mamba_end,
    req_to_token_pool,
    device,
    host_capacity_slots: int,
) -> UnifiedMambaSlotAllocator:
    """Install Mamba slot allocation, host capacities, and transfer translation."""
    slot_allocator = UnifiedMambaSlotAllocator(
        mamba_end,
        max_size=req_to_token_pool._shared_mamba_size,
        device=device,
    )
    req_to_token_pool.mamba_allocator = slot_allocator
    state_pool = req_to_token_pool.mamba_pool
    state_pool.host_transfer_translate = slot_allocator.translate
    # The configured state share, not every slot the shared buffer could hold:
    # the host tier scales with this, and would otherwise spend as much on states
    # as on the KV of the prefixes they belong to.
    state_pool.host_capacity_tokens = min(
        host_capacity_slots, req_to_token_pool._shared_mamba_size
    )
    state_pool.host_capacity_bytes = (
        state_pool.host_capacity_tokens * mamba_end.entry_bytes
    )
    return slot_allocator


def unified_mamba_buffer_bytes(
    *,
    token_budget_bytes: int,
    state_slots: int,
    state_entry_bytes: int,
    full_page_bytes: int,
    full_entry_bytes: int,
    side_bytes_per_token: int = 0,
    page_size: int = 1,
) -> int:
    """Shared-buffer bytes: the static pools' footprint, rearranged.

    The static layout spends the token budget on KV, ``state_slots + 1`` state
    slots (slot 0 is padding) and one padding KV page. The buffer holds exactly
    that: slot 0 is its sink, and the extra page absorbs the misalignment of the
    two ends' grids, so with every state slot in use the full end still fits
    ``token_budget // full_page_bytes`` pages, and every free slot adds to it.

    A side buffer over the allocator's whole virtual id space (a draft pool
    outside the envelope) costs ``side_bytes_per_token`` per ``T // full_entry``
    virtual slot, so the budget then splits as ``T * (1 + side / full_entry)``.
    """
    budget = (
        token_budget_bytes + (state_slots + 1) * state_entry_bytes + full_page_bytes
    )
    if side_bytes_per_token > 0:
        # Covers the draft pool's round-up of the virtual space to a page, plus a page.
        budget -= 2 * page_size * side_bytes_per_token
        budget = budget * full_entry_bytes // (full_entry_bytes + side_bytes_per_token)
    # The factories view the uint8 buffer as the KV/state dtypes; round up so the
    # full end never loses a page to the alignment.
    return -(-budget // 4096) * 4096


def _mha_full_spec(
    *,
    layer_num: int,
    head_num: int,
    head_dim: int,
    store_dtype: torch.dtype,
    qsa_profile: Optional[QSAProfile],
    draft_layer_num: int,
) -> MHASubPoolSpec:
    if qsa_profile is None:
        assert draft_layer_num == 0, (
            "draft layers share the page envelope only for QSA models"
        )
        return MHASubPoolSpec(
            name="full",
            layer_num=layer_num,
            head_num=head_num,
            head_dim=head_dim,
            store_dtype=store_dtype,
            grow_direction="down",
        )
    return MHASubPoolSpec(
        name="full",
        layer_num=layer_num,
        head_num=head_num,
        head_dim=head_dim,
        store_dtype=store_dtype,
        grow_direction="down",
        draft_layer_num=draft_layer_num,
        index_layer_num=layer_num + draft_layer_num,
        index_head_num=qsa_profile.kv_heads,
        index_head_dim=qsa_profile.head_dim,
        index_dtype=QSATokenToKVPool.index_state_dtype,
        index_ratio=qsa_profile.compress_ratio,
    )


def _ple_sibling_layouts(
    ple_req_pool_kwargs: Mapping[str, Any], *, conv_dtype: torch.dtype
) -> Tuple[Tuple[Tuple[int, ...], torch.dtype], ...]:
    """Per-slot PLE side states as mamba slot-envelope siblings: the N-gram
    context (int64) first, then the short-conv window of every PLE layer."""
    layouts = []
    ngram_context_len = ple_req_pool_kwargs.get("ngram_context_len", 0)
    if ngram_context_len > 0:
        layouts.append(((int(ngram_context_len),), torch.int64))
    short_conv_layer_ids = ple_req_pool_kwargs.get("short_conv_layer_ids") or ()
    state_shape = ple_req_pool_kwargs.get("short_conv_state_shape")
    if short_conv_layer_ids and state_shape is not None:
        layouts.append(
            (
                (len(short_conv_layer_ids), *(int(x) for x in state_shape)),
                conv_dtype,
            )
        )
    return tuple(layouts)


def init_unified_mamba_pools(
    *,
    device: str,
    kv_cache_dtype: torch.dtype,
    head_num: int,
    head_dim: int,
    page_size: int,
    start_layer: int,
    end_layer: int,
    is_draft_worker: bool,
    use_mla_backend: bool,
    kv_lora_rank: Optional[int] = None,
    qk_rope_head_dim: Optional[int] = None,
    mamba_layer_ids: List[int],
    full_attention_layer_ids: List[int],
    mamba2_cache_params,
    model_context_len: int,
    extra_max_context_len: int,
    max_total_num_tokens: int,
    max_mamba_cache_size: int,
    max_num_reqs: int,
    enable_memory_saver: bool,
    enable_mamba_extra_buffer: bool,
    speculative_num_draft_tokens: Optional[int],
    disable_overlap_schedule: bool,
    need_sort: bool,
    mamba_full_memory_ratio: Optional[float] = None,  # informational only
    forward_stream: Optional[torch.cuda.Stream] = None,
    lazy_compaction: bool = False,
    decode_pre_alloc_size: int = 0,
    unified_total_bytes: Optional[int] = None,
    enable_mamba_extra_buffer_lazy: bool = False,
    qsa_profile: Optional[QSAProfile] = None,
    token_cell_bytes: Optional[int] = None,
    ple_req_pool_kwargs: Optional[Mapping[str, Any]] = None,
    draft_layer_num: int = 0,
) -> UnifiedPoolBundle:
    """Build the Mamba-hybrid unified-memory-pool stack.

    A QSA model's compressed index keys, and ``draft_layer_num`` MTP draft
    layers (K/V and index keys), share each full page's envelope; PLE side
    states share each state slot's. ``token_cell_bytes`` is what the token
    budget charges per token; whatever it charges beyond the page envelope's
    per-token share is a side buffer over the virtual id space.
    """
    from sglang.srt.mem_cache.allocator.unified_mamba import (
        UnifiedMambaTokenToKVPoolAllocator,
    )

    # Full sub-pool is page-aware; mamba stays page=1 (state is per-request).
    assert page_size >= 1, f"page_size must be >= 1, got {page_size}"

    store_dtype = _store_dtype_for(kv_cache_dtype)
    # full-attn at the high-byte end (grow-down), mamba at the low-byte end (grow-up).
    if use_mla_backend:
        assert kv_lora_rank and qk_rope_head_dim, (
            "init_unified_mamba_pools: MLA-hybrid-Mamba needs kv_lora_rank and "
            f"qk_rope_head_dim; got {kv_lora_rank} / {qk_rope_head_dim}"
        )
        assert not is_draft_worker, (
            "init_unified_mamba_pools: draft workers (speculative decoding) are "
            "not supported with the MLA unified pool"
        )
        full_spec = MLASubPoolSpec(
            name="full",
            layer_num=len(full_attention_layer_ids),
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            store_dtype=store_dtype,
            grow_direction="down",
        )
    else:
        full_spec = _mha_full_spec(
            layer_num=len(full_attention_layer_ids),
            head_num=head_num,
            head_dim=head_dim,
            store_dtype=store_dtype,
            qsa_profile=qsa_profile,
            draft_layer_num=draft_layer_num,
        )
    cp = mamba2_cache_params
    mamba_spec = MambaSubPoolSpec(
        name="mamba",
        layer_num=len(mamba_layer_ids),
        conv_state_shapes=tuple(tuple(int(x) for x in s) for s in cp.shape.conv),
        conv_dtype=cp.dtype.conv,
        temporal_state_shape=tuple(int(x) for x in cp.shape.temporal),
        temporal_dtype=cp.dtype.temporal,
        conv_slice_axis=getattr(cp.shape, "conv_slice_axis", 0),
        grow_direction="up",
        sibling_layouts=_ple_sibling_layouts(
            ple_req_pool_kwargs or {}, conv_dtype=cp.dtype.conv
        ),
    )
    side_bytes_per_token = (
        0
        if token_cell_bytes is None
        else max(token_cell_bytes - full_spec.entry_bytes(), 0)
    )
    # PROFILED byte budget for the token side (captured pre-ratio-floor), else
    # the token-count re-sum; the state pool's bytes ride on top. The token
    # counts stay boot labels / conserve caps -- the runtime split floats.
    token_budget_bytes = (
        unified_total_bytes
        if unified_total_bytes is not None
        else max_total_num_tokens * (full_spec.entry_bytes() + side_bytes_per_token)
    )
    total_bytes = unified_mamba_buffer_bytes(
        token_budget_bytes=token_budget_bytes,
        state_slots=max_mamba_cache_size,
        state_entry_bytes=mamba_spec.entry_bytes(),
        full_page_bytes=page_size * full_spec.entry_bytes(),
        full_entry_bytes=full_spec.entry_bytes(),
        side_bytes_per_token=side_bytes_per_token,
        page_size=page_size,
    )
    # bs=1 floor: the state slots one running request locks (1 active + 2 radix
    # checkpoints, a FLOOR not headroom) + the slot-0 sink. The token side is
    # not charged -- `TpModelWorker.get_worker_info` already clamps max_req_len
    # to the pool, so a too-long request is refused at admission, not livelocked.
    _check_bs1_feasibility_floor(
        total_bytes=total_bytes,
        floor_terms=[
            ("bs1_state_slots", 3 * mamba_spec.entry_bytes()),
            ("sink", _reserved_floor_bytes([full_spec, mamba_spec], page_size)),
        ],
        factory="init_unified_mamba_pools",
    )
    shared_pool = UnifiedKVPool(
        total_bytes=total_bytes,
        sub_pool_specs=[full_spec, mamba_spec],
        device=device,
        enable_memory_saver=enable_memory_saver,
        page_size=page_size,
    )
    req_to_token_pool = UnifiedHybridReqToTokenPool(
        unified_buffer=shared_pool,
        mamba_sub_pool_name="mamba",
        size=max_num_reqs,
        mamba_spec_state_size=max_num_reqs,  # outer dim of spec-decode intermediates
        max_context_len=model_context_len + extra_max_context_len,
        device=device,
        enable_memory_saver=enable_memory_saver,
        cache_params=mamba2_cache_params,
        mamba_layer_ids=mamba_layer_ids,
        enable_mamba_extra_buffer=enable_mamba_extra_buffer,
        enable_mamba_extra_buffer_lazy=enable_mamba_extra_buffer_lazy,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        enable_overlap_schedule=not disable_overlap_schedule,
        start_layer=start_layer,
        pre_alloc_size=decode_pre_alloc_size,
        **(ple_req_pool_kwargs or {}),
    )
    if use_mla_backend:
        # start_layer stays 0: HybridLinearKVPool patches layer ids to the contiguous
        # 0..N-1 index via _transfer_id_context before every MLA pool call.
        unified_full_kv_pool = UnifiedMLATokenToKVPool(
            unified_buffer=shared_pool,
            sub_pool_name="full",
            kv_cache_dtype=kv_cache_dtype,
            page_size=page_size,
        )
    else:
        unified_full_kv_pool = UnifiedMHATokenToKVPool(
            unified_buffer=shared_pool,
            sub_pool_name="full",
            page_size=page_size,
            start_layer=start_layer,
            end_layer=end_layer,
        )
    full_attn_layer_ids_for_pool = (
        [0] if is_draft_worker else list(full_attention_layer_ids)
    )
    hybrid_pool_kwargs = dict(
        page_size=page_size,
        size=max_total_num_tokens,
        dtype=kv_cache_dtype,
        head_num=head_num,
        head_dim=head_dim,
        full_attention_layer_ids=full_attn_layer_ids_for_pool,
        device=device,
        mamba_pool=req_to_token_pool.mamba_pool,
        enable_memory_saver=enable_memory_saver,
        start_layer=start_layer,
        full_kv_pool=unified_full_kv_pool,
    )
    if qsa_profile is None:
        token_to_kv_pool = UnifiedHybridLinearKVPool(
            use_mla=use_mla_backend, **hybrid_pool_kwargs
        )
    else:
        assert not use_mla_backend, "QSA's full-attention KV is MHA"
        token_to_kv_pool = UnifiedQSATokenToKVPool(
            qsa_index_kv_heads=qsa_profile.kv_heads,
            qsa_index_head_dim=qsa_profile.head_dim,
            qsa_compress_ratio=qsa_profile.compress_ratio,
            qsa_token_topk=qsa_profile.budget,
            num_request_slots=req_to_token_pool.req_to_token.shape[0],
            compressed_k_views=shared_pool.index_views_for("full")[
                : full_spec.layer_num
            ],
            **hybrid_pool_kwargs,
        )
    allocator = UnifiedMambaTokenToKVPoolAllocator(
        unified_buffer=shared_pool,
        kvcache=token_to_kv_pool,
        device=device,
        page_size=page_size,
        need_sort=need_sort,
        forward_stream=forward_stream,
        lazy_compaction=lazy_compaction,
    )
    # Size host storage from the configured token cap, not the dynamic buffer view.
    if qsa_profile is not None:
        _bind_unified_qsa_pool(
            token_to_kv_pool,
            allocator=allocator,
            host_capacity_tokens=max_total_num_tokens,
        )
    else:
        full_pool = token_to_kv_pool.full_kv_pool
        full_pool.host_capacity_tokens = max_total_num_tokens
        full_pool.host_capacity_bytes = (
            max_total_num_tokens * allocator.full_attn_allocator.entry_bytes
        )

    mamba_slot_allocator = _wire_mamba_slot_allocator(
        mamba_end=allocator.mamba_allocator,
        req_to_token_pool=req_to_token_pool,
        device=device,
        host_capacity_slots=max_mamba_cache_size,
    )
    # Only HybridLinearKVPool's retraction CPU-copy path uses this hook.
    token_to_kv_pool._mamba_translate = mamba_slot_allocator.translate
    # No full-KV translate hook is wired: both MLA doors now receive
    # KERNEL-FACING ids -- writes from the ForwardBatch rebind, reads
    # translated at their production sites.

    logger.info(
        "[unified-memory-pool] ============================================================"
    )
    logger.info(
        "[unified-memory-pool] UNIFIED MEMORY POOL ENABLED -- path=Mamba hybrid (%s full side)",
        "MLA" if use_mla_backend else "MHA",
    )
    if use_mla_backend:
        logger.info(
            "[unified-memory-pool]   full_layers=%d, mamba_layers=%d, kv_lora_rank=%d, "
            "qk_rope_head_dim=%d, page_size=%d (per-layer views, kernel_page_multiplier=%d, "
            "view_tail_pad=%d B)",
            len(full_attention_layer_ids),
            len(mamba_layer_ids),
            kv_lora_rank,
            qk_rope_head_dim,
            page_size,
            len(full_attention_layer_ids),
            shared_pool.view_tail_pad_bytes,
        )
    else:
        logger.info(
            "[unified-memory-pool]   full_layers=%d, mamba_layers=%d, head_num=%d, head_dim=%d, "
            "page_size=%d, is_draft_worker=%s (%s)",
            len(full_attention_layer_ids),
            len(mamba_layer_ids),
            head_num,
            head_dim,
            page_size,
            is_draft_worker,
            "per-layer views, kernel_page_stride=%d, draft_layers=%d, "
            "index_layers=%d, view_tail_pad=%d B"
            % (
                full_spec.kernel_page_stride(page_size),
                full_spec.draft_layer_num,
                full_spec.index_layer_num,
                shared_pool.view_tail_pad_bytes,
            ),
        )
    logger.info(
        "[unified-memory-pool]   total_bytes=%d, max_total_num_tokens=%d, max_mamba_cache_size=%d, "
        "max_num_reqs=%d, speculative_num_draft_tokens=%s",
        total_bytes,
        max_total_num_tokens,
        max_mamba_cache_size,
        max_num_reqs,
        speculative_num_draft_tokens,
    )
    if mamba_full_memory_ratio is not None:
        logger.info(
            "[unified-memory-pool]   mamba_full_memory_ratio=%s governs the total budget only, "
            "not the runtime split.",
            mamba_full_memory_ratio,
        )
    logger.info(
        "[unified-memory-pool] ============================================================"
    )
    return UnifiedPoolBundle(
        unified_memory_pool=shared_pool,
        token_to_kv_pool=token_to_kv_pool,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=req_to_token_pool,
    )


# ---------------------------------------------------------------------------
# UnifiedSWAKVPool — hybrid SWA on the shared byte buffer
# ---------------------------------------------------------------------------


class UnifiedSWAKVPool(SWAKVPool):
    """Shared-buffer replacement for `SWAKVPool`.

    Composes two `UnifiedMHATokenToKVPool` instances (full + swa) aliasing the same
    byte buffer. Inherits from `SWAKVPool` only for `isinstance`; does NOT call the
    parent `__init__` (it would build static-partition pools). The per-sub-pool v2p
    table IS the full->swa mapping, so `register_mapping` is a no-op.
    """

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        swa_attention_layer_ids: List[int],
        full_attention_layer_ids: List[int],
        page_size: int = 1,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_memory_saver: bool = False,
    ):
        # Do NOT call super().__init__ — it would allocate static-partition pools.
        self.unified_buffer = unified_buffer
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        self.layer_num = self.full_layer_nums + self.swa_layer_nums
        self.start_layer = start_layer if start_layer is not None else 0
        self.page_size = page_size
        self.layer_transfer_counter = None

        self.size = unified_buffer.max_slots("full") - 1
        self.size_swa = unified_buffer.max_slots("swa") - 1

        full_spec = unified_buffer.mha_spec("full")
        swa_spec = unified_buffer.mha_spec("swa")
        assert full_spec.store_dtype == swa_spec.store_dtype, (
            "UnifiedSWAKVPool: full and swa sub-pools must share store_dtype; got "
            f"full={full_spec.store_dtype}, swa={swa_spec.store_dtype}"
        )
        self.dtype = full_spec.store_dtype
        self.head_num = full_spec.head_num
        self.head_dim = full_spec.head_dim
        self.device = unified_buffer.device

        self.full_kv_pool = UnifiedMHATokenToKVPool(
            unified_buffer=unified_buffer,
            sub_pool_name="full",
            page_size=page_size,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        self.swa_kv_pool = UnifiedMHATokenToKVPool(
            unified_buffer=unified_buffer,
            sub_pool_name="swa",
            page_size=page_size,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        # disagg/nvlink disabled; keep attrs present to avoid AttributeError.
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None
        self.kernel_page_blocks = 1
        self.kernel_page_stride = page_size
        self.store_dtype = full_spec.store_dtype
        self.end_layer = end_layer if end_layer is not None else self.layer_num - 1
        self.allocation_label = None
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.cpu_offloading_chunk_size = 8192
        self.dsa_kv_cache_store_fp8 = False
        self.kv_cache_dim = None
        self.index_head_dim = None

        # {global_layer_id: (per-pool index, is_swa_layer)}
        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        for idx, gid in enumerate(full_attention_layer_ids):
            self.layers_mapping[gid] = (idx, False)
        for idx, gid in enumerate(swa_attention_layer_ids):
            self.layers_mapping[gid] = (idx, True)

        # None so dispatch routes through our v2p-table overrides, not a registered mapping.
        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None

        self.mem_usage = 0.0  # cosmetic; UnifiedKVPool logs the real size

        # Wired in via attach_allocators.
        self._full_allocator = None
        self._swa_allocator = None

        logger.info(
            "[unified-memory-pool] UnifiedSWAKVPool wrapped unified buffer: "
            "full_layers=%d (max_slots=%d), swa_layers=%d (max_slots=%d), "
            "head_num=%d, head_dim=%d",
            self.full_layer_nums,
            unified_buffer.max_slots("full"),
            self.swa_layer_nums,
            unified_buffer.max_slots("swa"),
            self.head_num,
            self.head_dim,
        )

    # -- allocator wiring --

    def attach_allocators(self, *, full_allocator, swa_allocator) -> None:
        """Wire the two `MultiEndedAllocator`s whose v2p tables translate slot ids."""
        self._full_allocator = full_allocator
        self._swa_allocator = swa_allocator

    # -- BaseSWAKVPool ABC surface --

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
        return  # no-op in shared mode (the swa-side v2p IS the mapping)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        """Virtual token ids -> swa kernel-facing ids (int64)."""
        assert self._swa_allocator is not None, (
            "UnifiedSWAKVPool.translate_loc_from_full_to_swa called before "
            "attach_allocators"
        )
        return self._swa_allocator.translate_kv_loc_for_kernel(kv_indices)

    def get_state_buf_infos(self):
        return self.swa_kv_pool.get_contiguous_buf_infos()

    # -- size/info --

    def get_kv_size_bytes(self):
        return 0, 0  # UnifiedKVPool logs the total; per-side would double-count

    def get_contiguous_buf_infos(self):
        return self.full_kv_pool.get_contiguous_buf_infos()

    # -- buffer accessors --

    def get_key_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        return pool.get_key_buffer(pool_layer_id)

    def get_value_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        return pool.get_value_buffer(pool_layer_id)

    def get_kv_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa else self.full_kv_pool
        return pool.get_kv_buffer(pool_layer_id)

    # -- kv writing --

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        """Route to the right sub-pool. Both `swa_loc` and `full_loc` are PHYSICAL
        (pre-translated once per forward by the attention backend); never translates here.
        """
        loc, swa_loc, full_loc = unwrap_write_loc(loc_info)
        layer_id = layer.layer_id
        pool_layer_id, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            # swa_loc is ALREADY swa-physical. Routed through the UnifiedMHATokenToKVPool
            # override (its 4-D layer-major view can't take the parent's view(-1, row_dim)).
            assert swa_loc is not None, (
                "UnifiedSWAKVPool.set_kv_buffer: SWA layer received no swa_loc; the "
                "attention backend must bundle forward_metadata.swa_out_cache_loc."
            )
            self.swa_kv_pool.set_kv_buffer(
                None,
                swa_loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=pool_layer_id,
            )
            return
        # Full layer: `loc` is already the full-side kernel-facing id, so an
        # explicit full_loc is a same-space alias -- only triton's captured path
        # passes one (its capture-stable buffer).
        if full_loc is None:
            full_loc = loc
        self.full_kv_pool.set_kv_buffer(
            None,
            full_loc,
            cache_k,
            cache_v,
            k_scale,
            v_scale,
            layer_id_override=pool_layer_id,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # Never called on the composite — compaction runs per-sub-pool via
        # UnifiedMHATokenToKVPool.move_kv_cache.
        raise NotImplementedError(
            "UnifiedSWAKVPool.move_kv_cache should not be called; compaction "
            "operates per-sub-pool via UnifiedMHATokenToKVPool.move_kv_cache."
        )

    # -- HiCache shims (translate virtual->physical, then delegate) --

    @staticmethod
    def _virt_tokens_to_phys_tokens(
        virt_tokens: torch.Tensor, allocator
    ) -> torch.Tensor:
        """Virtual TOKEN ids -> physical TOKEN ids (page-aware). Unbound pages yield
        negatives; callers filter via `swa_phys >= 0`."""
        ps = allocator.page_size
        if ps == 1:
            return allocator.virtual_to_physical[virt_tokens]
        virt_pages = virt_tokens // ps
        offsets = virt_tokens % ps
        phys_pages = allocator.virtual_to_physical[virt_pages]
        return phys_pages * ps + offsets

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        assert self._full_allocator is not None
        assert self._swa_allocator is not None
        # `indices` are virtual TOKEN ids; translate per sub-pool.
        full_phys = self._virt_tokens_to_phys_tokens(indices, self._full_allocator)
        swa_phys = self._virt_tokens_to_phys_tokens(indices, self._swa_allocator)
        full_cpu = self.full_kv_pool.get_cpu_copy(
            full_phys, req_pool_index=req_pool_index
        )
        valid = swa_phys >= 0
        swa_cpu = None
        if bool(valid.any().item()):
            swa_cpu = self.swa_kv_pool.get_cpu_copy(swa_phys[valid])
        return {"full": full_cpu, "swa": swa_cpu, "swa_mask": valid.cpu()}

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        assert self._full_allocator is not None
        full_phys = self._virt_tokens_to_phys_tokens(indices, self._full_allocator)
        self.full_kv_pool.load_cpu_copy(
            kv_cache_cpu["full"], full_phys, req_pool_index=req_pool_index
        )
        if kv_cache_cpu.get("swa") is not None:
            assert self._swa_allocator is not None
            swa_phys = self._virt_tokens_to_phys_tokens(indices, self._swa_allocator)
            old_swa_mask = kv_cache_cpu["swa_mask"].to(indices.device)
            assert old_swa_mask.shape == indices.shape
            row_mask = (swa_phys >= 0)[old_swa_mask].cpu()
            swa_phys = swa_phys[old_swa_mask][row_mask.to(indices.device)]
            if swa_phys.numel() == 0:
                return
            swa_cpu = self._filter_swa_cpu_copy(
                swa_kv_cpu=kv_cache_cpu["swa"], row_mask=row_mask
            )
            self.swa_kv_pool.load_cpu_copy(swa_cpu, swa_phys)


class UnifiedSWAPoolBundle(NamedTuple):
    unified_memory_pool: UnifiedKVPool
    token_to_kv_pool: object  # UnifiedSWAKVPool
    token_to_kv_pool_allocator: object  # UnifiedSWATokenToKVPoolAllocator


def init_unified_swa_pools(
    *,
    device: str,
    kv_cache_dtype: torch.dtype,
    head_num: int,
    head_dim: int,
    v_head_dim: int,
    swa_head_num: int,
    swa_head_dim: int,
    swa_v_head_dim: int,
    page_size: int,
    start_layer: int,
    end_layer: int,
    swa_attention_layer_ids: List[int],
    full_attention_layer_ids: List[int],
    full_max_total_num_tokens: Optional[int] = None,
    swa_max_total_num_tokens: Optional[int] = None,
    total_bytes: Optional[int] = None,
    enable_memory_saver: bool,
    need_sort: bool,
    forward_stream: Optional[torch.cuda.Stream] = None,
    lazy_compaction: bool = False,
    model_context_len: Optional[int] = None,
    sliding_window_size: Optional[int] = None,
) -> UnifiedSWAPoolBundle:
    """Build the SWA-hybrid unified-memory-pool stack."""
    from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
        UnifiedSWATokenToKVPoolAllocator,
    )

    # Both sub-allocators are page-aware: one virtual ID space at PAGE granularity,
    # two physical sub-pools compacting pages independently.
    assert page_size >= 1, f"page_size must be >= 1, got {page_size}"
    assert len(full_attention_layer_ids) > 0, (
        "SWA-hybrid with zero full-attention layers is degenerate"
    )
    assert len(swa_attention_layer_ids) > 0, (
        "SWA-hybrid with zero SWA-attention layers is degenerate"
    )

    store_dtype = _store_dtype_for(kv_cache_dtype)
    # full-attn at the high-byte end (grow-down), swa at the low-byte end (grow-up).
    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=len(full_attention_layer_ids),
        head_num=head_num,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        store_dtype=store_dtype,
        grow_direction="down",
    )
    swa_spec = MHASubPoolSpec(
        name="swa",
        layer_num=len(swa_attention_layer_ids),
        head_num=swa_head_num,
        head_dim=swa_head_dim,
        v_head_dim=swa_v_head_dim,
        store_dtype=store_dtype,
        grow_direction="up",
    )
    legacy_allocator_capacities = {}
    if total_bytes is None:
        if full_max_total_num_tokens is None or swa_max_total_num_tokens is None:
            raise ValueError(
                "total_bytes or both legacy full/SWA capacities must be provided"
            )
        total_bytes = (
            full_max_total_num_tokens * full_spec.entry_bytes()
            + swa_max_total_num_tokens * swa_spec.entry_bytes()
        )
        legacy_allocator_capacities = {
            "full_max_total_num_tokens": full_max_total_num_tokens,
            "swa_max_total_num_tokens": swa_max_total_num_tokens,
        }
    if model_context_len is not None:
        # bs=1 floor: ONE sliding window of swa KV (+ a page of slack for the
        # page-granular walk) + the slot-0 sink. The full side is not charged
        # (max_req_len clamps it); the swa sub-pool is sized independently of
        # that clamp, which is why the window term stays.
        swa_bs1_tokens = (
            min(model_context_len, sliding_window_size + page_size)
            if sliding_window_size is not None
            else model_context_len
        )
        _check_bs1_feasibility_floor(
            total_bytes=total_bytes,
            floor_terms=[
                ("swa_window_kv", swa_bs1_tokens * swa_spec.entry_bytes()),
                ("sink", _reserved_floor_bytes([full_spec, swa_spec], page_size)),
            ],
            factory="init_unified_swa_pools",
        )
    if total_bytes <= 0:
        raise ValueError(f"total_bytes must be positive, got {total_bytes}")
    shared_pool = UnifiedKVPool(
        total_bytes=total_bytes,
        sub_pool_specs=[full_spec, swa_spec],
        device=device,
        enable_memory_saver=enable_memory_saver,
        page_size=page_size,
    )
    token_to_kv_pool = UnifiedSWAKVPool(
        unified_buffer=shared_pool,
        swa_attention_layer_ids=swa_attention_layer_ids,
        full_attention_layer_ids=full_attention_layer_ids,
        page_size=page_size,
        start_layer=start_layer,
        end_layer=end_layer,
        enable_memory_saver=enable_memory_saver,
    )
    allocator = UnifiedSWATokenToKVPoolAllocator(
        unified_buffer=shared_pool,
        kvcache=token_to_kv_pool,
        device=device,
        page_size=page_size,
        need_sort=need_sort,
        forward_stream=forward_stream,
        lazy_compaction=lazy_compaction,
        **legacy_allocator_capacities,
    )

    logger.info(
        "[unified-memory-pool] ============================================================"
    )
    logger.info("[unified-memory-pool] UNIFIED MEMORY POOL ENABLED -- path=SWA hybrid")
    logger.info(
        "[unified-memory-pool]   %s",
        "per-layer views, kernel_page_multiplier full=%d swa=%d, view_tail_pad=%d B"
        % (
            full_spec.blocks_per_page(),
            swa_spec.blocks_per_page(),
            shared_pool.view_tail_pad_bytes,
        ),
    )
    logger.info(
        "[unified-memory-pool]   full_layers=%d, swa_layers=%d, head_num=%d, head_dim=%d, "
        "v_head_dim=%d, swa_head_num=%d, swa_head_dim=%d, swa_v_head_dim=%d, "
        "page_size=%d",
        len(full_attention_layer_ids),
        len(swa_attention_layer_ids),
        head_num,
        head_dim,
        v_head_dim,
        swa_head_num,
        swa_head_dim,
        swa_v_head_dim,
        page_size,
    )
    logger.info(
        "[unified-memory-pool]   total_bytes=%d (=%.2f GB), "
        "full_capacity=%d, swa_capacity=%d, joint_available=%d slots",
        total_bytes,
        total_bytes / GB,
        allocator.size_full,
        allocator.size_swa,
        allocator.available_size(),
    )
    logger.info(
        "[unified-memory-pool] ============================================================"
    )
    return UnifiedSWAPoolBundle(
        unified_memory_pool=shared_pool,
        token_to_kv_pool=token_to_kv_pool,
        token_to_kv_pool_allocator=allocator,
    )


def init_unified_mamba_swa_pools(
    *,
    device: str,
    kv_cache_dtype: torch.dtype,
    head_num: int,
    head_dim: int,
    v_head_dim: int,
    swa_head_num: int,
    swa_head_dim: int,
    swa_v_head_dim: int,
    page_size: int,
    start_layer: int,
    end_layer: int,
    swa_attention_layer_ids: List[int],
    full_attention_layer_ids: List[int],
    mamba_layer_ids: List[int],
    mamba2_cache_params,
    full_max_total_num_tokens: int,
    swa_max_total_num_tokens: int,
    max_mamba_cache_size: int,
    model_context_len: int,
    extra_max_context_len: int,
    max_num_reqs: int,
    enable_memory_saver: bool,
    enable_mamba_extra_buffer: bool,
    disable_overlap_schedule: bool,
    need_sort: bool,
    speculative_num_draft_tokens: Optional[int] = None,
    forward_stream: Optional[torch.cuda.Stream] = None,
    lazy_compaction: bool = False,
    unified_total_bytes: Optional[int] = None,
    sliding_window_size: Optional[int] = None,
    decode_pre_alloc_size: int = 0,
) -> UnifiedPoolBundle:
    """Build the TRI-pool unified-memory-pool stack for models with full KV +
    SWA KV + mamba/conv state (Inkling-class: `mambaish_config` AND
    `is_hybrid_swa` simultaneously — Inkling's SConv state is conv-only but
    rides the mamba machinery, so "mamba" here == the conv state pool).

    Chain: ``[mamba (up END) | swa (FLOAT) | full (down END)]``. The KV side
    is a `UnifiedSWAKVPool` (per-layer full/swa routing, asymmetric head
    geometry supported); the state side is a `UnifiedHybridReqToTokenPool`
    whose `mamba_pool` the model reads directly (sconv:
    `req_to_token_pool.mamba2_layer_cache(layer).conv[...]` with
    `translate_mamba_indices` for v->p).

    Sizing inputs are the same token counts the 2-pool factories take (ratio-
    fed until the byte configurator lands); the buffer budget is their byte
    sum and the runtime split floats.
    """
    from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
        UnifiedMambaSWATokenToKVPoolAllocator,
    )

    assert page_size >= 1, f"page_size must be >= 1, got {page_size}"
    assert len(full_attention_layer_ids) > 0, (
        "tri-pool with zero full-attention layers is degenerate"
    )
    assert len(swa_attention_layer_ids) > 0, (
        "tri-pool with zero SWA-attention layers is degenerate"
    )
    assert len(mamba_layer_ids) > 0, "tri-pool with zero state layers is degenerate"

    store_dtype = _store_dtype_for(kv_cache_dtype)
    # mamba/conv at the LOWEST bytes, full KV at the HIGHEST, SWA floating
    # between: ends never relocate, and SWA's window-capped span is cheapest to move.
    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=len(full_attention_layer_ids),
        head_num=head_num,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        store_dtype=store_dtype,
        grow_direction="down",
    )
    swa_spec = MHASubPoolSpec(
        name="swa",
        layer_num=len(swa_attention_layer_ids),
        head_num=swa_head_num,
        head_dim=swa_head_dim,
        v_head_dim=swa_v_head_dim,
        store_dtype=store_dtype,
        grow_direction="float",
    )
    cp = mamba2_cache_params
    mamba_spec = MambaSubPoolSpec(
        name="mamba",
        layer_num=len(mamba_layer_ids),
        conv_state_shapes=tuple(tuple(int(x) for x in s) for s in cp.shape.conv),
        conv_dtype=cp.dtype.conv,
        temporal_state_shape=tuple(int(x) for x in cp.shape.temporal),
        temporal_dtype=cp.dtype.temporal,
        grow_direction="up",
    )
    if unified_total_bytes is not None:
        # PROFILED byte budget for the token side (captured pre-ratio-floor);
        # the state pool's bytes ride on top. The token counts stay boot
        # labels / conserve caps -- the runtime split floats.
        total_bytes = (
            unified_total_bytes + max_mamba_cache_size * mamba_spec.entry_bytes()
        )
    else:
        total_bytes = (
            full_max_total_num_tokens * full_spec.entry_bytes()
            + swa_max_total_num_tokens * swa_spec.entry_bytes()
            + max_mamba_cache_size * mamba_spec.entry_bytes()
        )
    # bs=1 floor: ONE sliding window of swa KV (+ a page of slack, clamped to
    # the context) + the state slots one running request locks (1 active + 2
    # radix checkpoints, a FLOOR not headroom) + the slot-0 sink. The
    # full-attention side is not charged: `max_req_len` already clamps to the pool.
    swa_bs1_tokens = (
        min(model_context_len, sliding_window_size + page_size)
        if sliding_window_size is not None
        else model_context_len
    )
    _check_bs1_feasibility_floor(
        total_bytes=total_bytes,
        floor_terms=[
            ("swa_window_kv", swa_bs1_tokens * swa_spec.entry_bytes()),
            ("bs1_state_slots", 3 * mamba_spec.entry_bytes()),
            (
                "sink",
                _reserved_floor_bytes([full_spec, swa_spec, mamba_spec], page_size),
            ),
        ],
        factory="init_unified_mamba_swa_pools",
    )
    shared_pool = UnifiedKVPool(
        total_bytes=total_bytes,
        sub_pool_specs=[full_spec, swa_spec, mamba_spec],
        device=device,
        enable_memory_saver=enable_memory_saver,
        page_size=page_size,
    )
    token_to_kv_pool = UnifiedSWAKVPool(
        unified_buffer=shared_pool,
        swa_attention_layer_ids=swa_attention_layer_ids,
        full_attention_layer_ids=full_attention_layer_ids,
        page_size=page_size,
        start_layer=start_layer,
        end_layer=end_layer,
        enable_memory_saver=enable_memory_saver,
    )
    req_to_token_pool = UnifiedHybridReqToTokenPool(
        unified_buffer=shared_pool,
        mamba_sub_pool_name="mamba",
        size=max_num_reqs,
        mamba_spec_state_size=max_num_reqs,
        max_context_len=model_context_len + extra_max_context_len,
        device=device,
        enable_memory_saver=enable_memory_saver,
        cache_params=mamba2_cache_params,
        mamba_layer_ids=mamba_layer_ids,
        enable_mamba_extra_buffer=enable_mamba_extra_buffer,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        enable_overlap_schedule=not disable_overlap_schedule,
        start_layer=start_layer,
        pre_alloc_size=decode_pre_alloc_size,
    )
    allocator = UnifiedMambaSWATokenToKVPoolAllocator(
        unified_buffer=shared_pool,
        kvcache=token_to_kv_pool,
        mamba_kvcache=req_to_token_pool.mamba_pool,
        device=device,
        full_max_total_num_tokens=full_max_total_num_tokens,
        swa_max_total_num_tokens=swa_max_total_num_tokens,
        page_size=page_size,
        need_sort=need_sort,
        forward_stream=forward_stream,
        lazy_compaction=lazy_compaction,
    )
    _wire_mamba_slot_allocator(
        mamba_end=allocator.mamba_allocator,
        req_to_token_pool=req_to_token_pool,
        device=device,
        host_capacity_slots=max_mamba_cache_size,
    )

    logger.info(
        "[unified-memory-pool] ============================================================"
    )
    logger.info(
        "[unified-memory-pool] UNIFIED MEMORY POOL ENABLED -- path=SWA+Mamba tri-pool"
    )
    logger.info(
        "[unified-memory-pool]   full_layers=%d, swa_layers=%d, state_layers=%d, "
        "head_num=%d/%d, head_dim=%d/%d, page_size=%d",
        len(full_attention_layer_ids),
        len(swa_attention_layer_ids),
        len(mamba_layer_ids),
        head_num,
        swa_head_num,
        head_dim,
        swa_head_dim,
        page_size,
    )
    logger.info(
        "[unified-memory-pool]   total_bytes=%d (=%.2f GB), full_max=%d, swa_max=%d, "
        "max_mamba_cache_size=%d, max_num_reqs=%d, joint_available=%d",
        total_bytes,
        total_bytes / GB,
        full_max_total_num_tokens,
        swa_max_total_num_tokens,
        max_mamba_cache_size,
        max_num_reqs,
        allocator.available_size(),
    )
    logger.info(
        "[unified-memory-pool] ============================================================"
    )
    return UnifiedPoolBundle(
        unified_memory_pool=shared_pool,
        token_to_kv_pool=token_to_kv_pool,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=req_to_token_pool,
    )
