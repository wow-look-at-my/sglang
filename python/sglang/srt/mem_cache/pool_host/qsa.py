"""Host mirror of the QSA compressed index-K cache."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.pool_host.base import (
    HostMemoryBudgetError,
    host_memory_budget_bytes,
)
from sglang.srt.mem_cache.pool_host.common import get_allocator_from_storage
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost

if TYPE_CHECKING:
    from sglang.srt.mem_cache.pool_host.base import HostKVCache
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

logger = logging.getLogger(__name__)


def qsa_compressed_page_views(pool: QSATokenToKVPool) -> list[torch.Tensor]:
    """Per-layer ``[num_full_pages, page_bytes]`` uint8 views of the compressed cache.

    Row ``p`` is full-KV page ``p``'s compressed keys: compressed slot is
    ``full_slot // ratio`` and a full page is a whole number of groups, so
    the page's ``page_size // ratio`` compressed rows are contiguous.
    """
    rows_per_page = pool.qsa_compressed_page_size
    views = []
    for buffer in pool.qsa_compressed_k_buffer_pool:
        num_pages = buffer.shape[0] // rows_per_page
        views.append(
            buffer[: num_pages * rows_per_page].reshape(num_pages, -1).view(torch.uint8)
        )
    return views


def qsa_compressed_bytes(pool: QSATokenToKVPool) -> int:
    """Device bytes of the compressed-K cache (the pending ring is per request)."""
    return sum(buffer.nbytes for buffer in pool.qsa_compressed_k_buffer_pool)


class QSACompressedKPoolHost(DSAIndexerPoolHost):
    """Host-side QSA compressed-K pages; slots follow the anchor KV host pool.

    One host item is one full-KV page's compressed keys, so the anchor's
    page-aligned host/device token indices address it directly
    (``index // page_size``). Packed MTP draft layers follow the target layers.
    """

    def __init__(
        self,
        device_pool: QSATokenToKVPool,
        anchor_host: HostKVCache,
        layout: str,
        mtp_draft_device_pools: tuple[QSATokenToKVPool, ...] = (),
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        self._is_dummy = False
        self.device_pool = device_pool
        self.page_size = anchor_host.page_size
        if self.page_size != device_pool.page_size:
            raise ValueError(
                "QSA compressed-K HiCache needs the host page to be the device "
                f"page: host={self.page_size}, device={device_pool.page_size}"
            )
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.indexer_dtype = torch.uint8
        self.dtype = self.indexer_dtype
        self.mtp_draft_device_pools = tuple(mtp_draft_device_pools)
        self._page_views = {id(device_pool): qsa_compressed_page_views(device_pool)}
        for draft_pool in self.mtp_draft_device_pools:
            self._check_draft_geometry(draft_pool)
            self._page_views[id(draft_pool)] = qsa_compressed_page_views(draft_pool)
        self.target_layer_num = len(device_pool.qsa_compressed_k_buffer_pool)
        self.layer_num = self.target_layer_num + len(self.mtp_draft_device_pools)
        self.start_layer = device_pool.start_layer
        self.end_layer = self.start_layer + self.target_layer_num

        self.indexer_page_stride_size = (
            device_pool.qsa_compressed_page_size
            * device_pool.qsa_index_kv_heads
            * device_pool.qsa_index_head_dim
            * device_pool.index_state_dtype.itemsize
        )
        self.indexer_layout_dim = self.indexer_page_stride_size * self.layer_num
        self.size = anchor_host.size
        self.page_num = anchor_host.page_num
        self.indexer_page_num = (self.size + self.page_size + 1) // self.page_size
        self.size_per_token = (
            self.indexer_page_stride_size * self.layer_num // self.page_size
        )

        self.can_use_jit = False
        self.can_use_write_back_jit = False
        requested_bytes = (
            self.indexer_page_num * self.layer_num * self.indexer_page_stride_size
        )
        available_bytes = host_memory_budget_bytes(requested_bytes)
        if requested_bytes > available_bytes:
            raise HostMemoryBudgetError(
                "Not enough host memory for the QSA compressed-K hierarchical "
                f"cache. Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )
        logger.info(
            "Allocating %.2f GB host memory for QSA compressed keys (layout=%s): "
            "target_layers=%d, draft_layers=%d.",
            requested_bytes / 1e9,
            layout,
            self.target_layer_num,
            len(self.mtp_draft_device_pools),
        )
        self.init_kv_buffer()
        # HostKVCache.destroy unregisters kv_buffer.
        self.kv_buffer = self.index_k_with_scale_buffer
        self._init_write_back_staging_buffers()
        self.lock = threading.RLock()
        self.clear()

    def _check_draft_geometry(self, draft_pool: QSATokenToKVPool) -> None:
        target = self.device_pool
        if len(draft_pool.qsa_compressed_k_buffer_pool) != 1:
            raise ValueError(
                "Packed MTP QSA HiCache expects one compressed-K layer per draft "
                f"pool, got {len(draft_pool.qsa_compressed_k_buffer_pool)}"
            )
        geometry = (
            "qsa_compress_ratio",
            "qsa_index_kv_heads",
            "qsa_index_head_dim",
            "page_size",
        )
        mismatched = [
            name
            for name in geometry
            if getattr(draft_pool, name) != getattr(target, name)
        ]
        if mismatched:
            raise ValueError(
                "Packed MTP QSA HiCache needs the draft compressed-K geometry to "
                f"match the target's; mismatched: {mismatched}"
            )

    def get_size_per_token(self):
        return self.size_per_token

    def _device_index_buffers(self, device_pool) -> list[torch.Tensor]:
        return self._page_views[id(device_pool)]

    def _device_owned_layer_range(self, device_pool=None) -> tuple[int, int]:
        device_pool = device_pool or self.device_pool
        return 0, len(device_pool.qsa_compressed_k_buffer_pool)

    def _is_device_layer_sharded(self, device_pool=None) -> bool:
        return False
