"""Page-granularity envelope (page-major, layer-major within a page) cache views.

A pool of this layout keeps all layers of all slots in one contiguous byte
buffer. The buffer is split into pages of ``page_size`` slots; within a page,
each layer's K and V (or each Mamba conv/temporal tensor) are grouped together:

    page bytes = [L0_K * ps | L0_V * ps | L1_K * ps | L1_V * ps | ...]

Across pages the layout is envelope-major (one ``page_bytes`` block per page).
At ``page_size == 1`` a page is a single slot, so the within-page block is the
per-slot ``[L0_K | L0_V | L1_K | L1_V | ...]`` envelope (token-granularity).

These builders produce per-layer views into a raw ``uint8`` buffer; they hold
no allocator/ownership state. ``anchor_bytes`` is the byte offset of the
pool's region inside the raw buffer (0 for a standalone pool).
"""

from typing import List, Optional, Sequence, Tuple

import torch


def _prod(shape: Sequence[int]) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


def mha_entry_bytes(
    *, layer_num: int, head_num: int, head_dim: int, v_head_dim: int, itemsize: int
) -> int:
    """Bytes occupied by one slot across all layers (K and V)."""
    k_row_bytes = head_num * head_dim * itemsize
    v_row_bytes = head_num * v_head_dim * itemsize
    return layer_num * (k_row_bytes + v_row_bytes)


def build_mha_views(
    raw: torch.Tensor,
    *,
    layer_num: int,
    head_num: int,
    head_dim: int,
    v_head_dim: int,
    store_dtype: torch.dtype,
    page_size: int,
    num_pages: int,
    anchor_bytes: int = 0,
    page_rows: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Per-layer K/V views over ``raw`` for uniform-row MHA.

    ``page_rows`` is the page envelope in K/V rows when it holds more than the
    K/V blocks (default ``2 * layer_num * ps``); ids then step by it per page.

    The page envelope ``[L0_K*ps | L0_V*ps | L1_K*ps | ...]`` is a uniform
    array of ``2*layer_num`` row-blocks when K and V rows are equally wide, so
    it is a valid paged pool under

        kernel_id(t) = (t // ps) * (ps * 2 * layer_num) + t % ps

    with layer ``l``'s K at block ``2l`` and its V at block ``2l+1``. Each view
    is a contiguous ``(num_pages * 2 * layer_num * ps, head_num, head_dim)``.

    Views overlap by ``ps`` rows per block, safe because an id always resolves
    inside its own block; the last view runs ``(2*layer_num - 1) * ps`` rows
    past the envelope, so ``raw`` needs ``UnifiedKVPool.view_tail_pad_bytes``.
    """
    assert head_dim == v_head_dim, (
        f"build_mha_views requires uniform rows (head_dim == v_head_dim); "
        f"got head_dim={head_dim}, v_head_dim={v_head_dim}. Asymmetric-KV "
        "models cannot use the unified pool (screened out at startup)."
    )
    itemsize = store_dtype.itemsize
    row_elems = head_num * head_dim
    row_bytes = row_elems * itemsize
    blocks = 2 * layer_num
    if page_rows is None:
        page_rows = blocks * page_size
    assert page_rows >= blocks * page_size
    page_bytes = page_rows * row_bytes
    n_rows = num_pages * page_rows
    assert anchor_bytes % itemsize == 0
    last_view_end = (
        anchor_bytes + (blocks - 1) * page_size * row_bytes + n_rows * row_bytes
    )
    assert last_view_end <= raw.numel() * raw.itemsize, (
        f"build_mha_views: block {blocks - 1}'s view ends at byte "
        f"{last_view_end} but the raw buffer holds only "
        f"{raw.numel() * raw.itemsize} bytes; allocate the tail pad "
        f"(one page envelope = {page_bytes} B) via view_tail_pad_bytes"
    )

    as_dtype_view = raw.view(store_dtype)
    k_buffer: List[torch.Tensor] = []
    v_buffer: List[torch.Tensor] = []
    for layer in range(layer_num):
        k_base_bytes = anchor_bytes + (2 * layer) * page_size * row_bytes
        v_base_bytes = k_base_bytes + page_size * row_bytes
        for base_bytes, out in ((k_base_bytes, k_buffer), (v_base_bytes, v_buffer)):
            assert base_bytes % itemsize == 0
            out.append(
                torch.as_strided(
                    as_dtype_view,
                    size=(n_rows, head_num, head_dim),
                    stride=(row_elems, head_dim, 1),
                    storage_offset=base_bytes // itemsize,
                )
            )
    return k_buffer, v_buffer


def mla_entry_bytes(*, layer_num: int, kv_cache_dim: int, itemsize: int) -> int:
    """Bytes occupied by one MLA slot across all layers (single latent row, no V)."""
    return layer_num * kv_cache_dim * itemsize


def build_mla_views(
    raw: torch.Tensor,
    *,
    layer_num: int,
    kv_cache_dim: int,
    store_dtype: torch.dtype,
    page_size: int,
    num_pages: int,
    anchor_bytes: int = 0,
) -> List[torch.Tensor]:
    """Per-layer views over ``raw`` for MLA in the page-major layout.

    The page envelope is ``[L0_latent * ps | L1_latent * ps | ...]``. Because all
    MLA layers share one uniform row size (``kv_cache_dim``), the envelope is
    itself a valid paged pool under a re-numbered index space: folding the
    layer offset ``l * ps * kv_cache_dim`` into each view's storage_offset makes
    every per-layer view a plain CONTIGUOUS ``(num_pages * layer_num * ps, 1,
    kv_cache_dim)`` tensor, addressed by the layer-independent kernel-facing id

        kernel_id(t) = (t // ps) * (ps * layer_num) + t % ps      (t = physical token)

    so one shared block table (entry = page * layer_num) serves every layer, and
    kernels that require ``.view(-1, page_size, kv_cache_dim)`` (trtllm/cutlass/
    flashmla) work on the views natively.

    The views overlap each other (view ``l+1`` is view ``l`` shifted by ``ps``
    rows); that is safe because layer ``l`` is only ever indexed at kernel-facing ids,
    which always resolve to layer-``l`` bytes relative to view ``l``'s origin.
    Layer ``layer_num-1``'s view extends ``(layer_num-1) * ps`` rows past the
    last page envelope, so ``raw`` must carry at least one extra page envelope
    of tail padding (``UnifiedKVPool``'s ``view_tail_pad_bytes``).
    """
    itemsize = store_dtype.itemsize
    row_bytes = kv_cache_dim * itemsize
    page_bytes = page_size * layer_num * row_bytes
    n_rows = num_pages * layer_num * page_size
    assert anchor_bytes % itemsize == 0
    last_view_end = (
        anchor_bytes + (layer_num - 1) * page_size * row_bytes + (n_rows * row_bytes)
    )
    assert last_view_end <= raw.numel() * raw.itemsize, (
        f"build_mla_views: layer {layer_num - 1}'s view ends at byte "
        f"{last_view_end} but the raw buffer holds only "
        f"{raw.numel() * raw.itemsize} bytes; allocate the tail pad "
        f"(one page envelope = {page_bytes} B) via view_tail_pad_bytes"
    )

    as_dtype_view = raw.view(store_dtype)
    views: List[torch.Tensor] = []
    for layer in range(layer_num):
        base_bytes = anchor_bytes + layer * page_size * row_bytes
        assert base_bytes % itemsize == 0
        views.append(
            torch.as_strided(
                as_dtype_view,
                size=(n_rows, 1, kv_cache_dim),
                stride=(kv_cache_dim, kv_cache_dim, 1),
                storage_offset=base_bytes // itemsize,
            )
        )
    return views


def build_row_block_views(
    raw: torch.Tensor,
    *,
    block_offsets_bytes: Sequence[int],
    row_shape: Sequence[int],
    dtype: torch.dtype,
    page_bytes: int,
    num_pages: int,
    anchor_bytes: int = 0,
) -> List[torch.Tensor]:
    """Contiguous ``(num_pages * page_bytes / row_bytes, *row_shape)`` views, one
    per block of rows at ``block_offsets_bytes`` inside every page envelope.

    A row of page ``P`` at in-block row ``r`` has id ``P * page_bytes / row_bytes
    + r``; like the K/V views, the views overlap and each id resolves inside its
    own block, so ``raw`` needs one page envelope of tail pad.
    """
    itemsize = dtype.itemsize
    row_elems = _prod(row_shape)
    row_bytes = row_elems * itemsize
    assert page_bytes % row_bytes == 0, (
        f"page envelope of {page_bytes} B is not a whole number of {row_bytes} B rows"
    )
    n_rows = num_pages * page_bytes // row_bytes
    as_dtype_view = raw.view(dtype)
    views = []
    for offset in block_offsets_bytes:
        base_bytes = anchor_bytes + offset
        assert base_bytes % itemsize == 0
        assert base_bytes + n_rows * row_bytes <= raw.numel() * raw.itemsize, (
            "build_row_block_views: allocate one page envelope of tail pad"
        )
        strides = [1]
        for size in reversed(list(row_shape)[1:]):
            strides.insert(0, strides[0] * int(size))
        views.append(
            torch.as_strided(
                as_dtype_view,
                size=(n_rows, *row_shape),
                stride=(row_elems, *strides),
                storage_offset=base_bytes // itemsize,
            )
        )
    return views


def build_slot_sibling_views(
    raw: torch.Tensor,
    *,
    layouts: Sequence[Tuple[Tuple[int, ...], torch.dtype]],
    entry_bytes: int,
    first_offset_bytes: int,
    max_slots: int,
    anchor_bytes: int = 0,
) -> List[torch.Tensor]:
    """``(max_slots, *shape)`` views of per-slot side tensors packed after the
    Mamba state in each slot envelope (slot stride ``entry_bytes``)."""
    views = []
    offset = anchor_bytes + first_offset_bytes
    for shape, dtype in layouts:
        itemsize = dtype.itemsize
        assert entry_bytes % itemsize == 0 and offset % itemsize == 0, (
            f"misaligned side state {dtype} at byte {offset} in a "
            f"{entry_bytes} B slot envelope"
        )
        strides = [1]
        for size in reversed(list(shape)[1:]):
            strides.insert(0, strides[0] * int(size))
        views.append(
            torch.as_strided(
                raw.view(dtype),
                size=(max_slots, *shape),
                stride=(entry_bytes // itemsize, *strides),
                storage_offset=offset // itemsize,
            )
        )
        offset += _prod(shape) * itemsize
    return views


def mamba_entry_bytes(
    *,
    layer_num: int,
    conv_state_shapes: Sequence[Sequence[int]],
    conv_dtype: torch.dtype,
    temporal_state_shape: Sequence[int],
    temporal_dtype: torch.dtype,
) -> int:
    """Bytes occupied by one Mamba slot across all layers (conv + temporal)."""
    total = 0
    for shape in conv_state_shapes:
        total += layer_num * _prod(shape) * conv_dtype.itemsize
    total += layer_num * _prod(temporal_state_shape) * temporal_dtype.itemsize
    return total


def build_page_major_mamba_views(
    raw: torch.Tensor,
    *,
    layer_num: int,
    conv_state_shapes: Sequence[Sequence[int]],
    conv_dtype: torch.dtype,
    temporal_state_shape: Sequence[int],
    temporal_dtype: torch.dtype,
    max_slots: int,
    anchor_bytes: int = 0,
    entry_bytes: Optional[int] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Per-slot envelope views over ``raw`` for Mamba state.

    Layout per slot: ``[conv[0] rows × layers][conv[1] rows × layers]...
    [temporal rows × layers]``. Each returned view has shape
    ``(num_layers, max_slots, *inner_shape)`` matching ``MambaPool.State.conv[i]``
    / ``.temporal``. Mamba state is always token-granular (page_size == 1).
    ``entry_bytes`` is the slot stride when per-slot side states follow the
    temporal rows (default: conv + temporal only).
    """
    state_bytes = mamba_entry_bytes(
        layer_num=layer_num,
        conv_state_shapes=conv_state_shapes,
        conv_dtype=conv_dtype,
        temporal_state_shape=temporal_state_shape,
        temporal_dtype=temporal_dtype,
    )
    if entry_bytes is None:
        entry_bytes = state_bytes
    assert entry_bytes >= state_bytes, (
        f"slot envelope of {entry_bytes} B cannot hold {state_bytes} B of state"
    )

    def contiguous_strides(shape: Sequence[int]) -> Tuple[int, ...]:
        strides = []
        acc = 1
        for s in reversed(shape):
            strides.append(acc)
            acc *= int(s)
        return tuple(reversed(strides))

    conv_itemsize = conv_dtype.itemsize
    assert entry_bytes % conv_itemsize == 0, (
        f"misaligned mamba spec: per-slot entry_bytes={entry_bytes} is not a "
        f"multiple of the conv-state itemsize {conv_itemsize} B"
    )
    assert anchor_bytes % conv_itemsize == 0, (
        f"misaligned mamba spec: anchor_bytes={anchor_bytes} is not a multiple "
        f"of the conv-state itemsize {conv_itemsize} B"
    )
    as_conv_dtype = raw.view(conv_dtype)
    conv_slot_stride_elems = entry_bytes // conv_itemsize

    offset_bytes_within_entry = 0
    conv_views: List[torch.Tensor] = []
    for shape in conv_state_shapes:
        inner_shape_bytes = _prod(shape) * conv_itemsize
        assert inner_shape_bytes % conv_itemsize == 0
        offset_elems = (anchor_bytes + offset_bytes_within_entry) // conv_itemsize
        stride = (
            inner_shape_bytes // conv_itemsize,
            conv_slot_stride_elems,
        ) + contiguous_strides(shape)
        conv_views.append(
            torch.as_strided(
                as_conv_dtype,
                size=(layer_num, max_slots) + tuple(shape),
                stride=stride,
                storage_offset=offset_elems,
            )
        )
        offset_bytes_within_entry += layer_num * inner_shape_bytes

    # The temporal view's storage_offset is computed in temporal-dtype elements
    # by integer-dividing a byte offset by itemsize, so every term of that byte
    # offset (entry stride, anchor, the conv region) must be a whole multiple of
    # itemsize or the offset truncates and mis-places the view.
    itemsize = temporal_dtype.itemsize
    assert entry_bytes % itemsize == 0, (
        f"misaligned mamba spec: per-slot entry_bytes={entry_bytes} is not a "
        f"multiple of the temporal-state itemsize {itemsize} B; the temporal "
        f"view's storage_offset would truncate and mis-place the state"
    )
    assert anchor_bytes % itemsize == 0, (
        f"misaligned mamba spec: anchor_bytes={anchor_bytes} is not a multiple "
        f"of the temporal-state itemsize {itemsize} B"
    )
    inner_shape_bytes = _prod(temporal_state_shape) * itemsize
    assert inner_shape_bytes % itemsize == 0, (
        f"misaligned mamba spec: temporal inner_shape_bytes={inner_shape_bytes} "
        f"is not a multiple of the temporal-state itemsize {itemsize} B"
    )
    assert (anchor_bytes + offset_bytes_within_entry) % itemsize == 0, (
        f"misaligned mamba spec: temporal region byte offset "
        f"{anchor_bytes + offset_bytes_within_entry} is not a multiple of the "
        f"temporal-state itemsize {itemsize} B"
    )
    offset_elems = (anchor_bytes + offset_bytes_within_entry) // itemsize
    as_temporal_dtype = raw.view(temporal_dtype)
    stride = (
        inner_shape_bytes // itemsize,
        entry_bytes // itemsize,
    ) + contiguous_strides(temporal_state_shape)
    temporal_view = torch.as_strided(
        as_temporal_dtype,
        size=(layer_num, max_slots) + tuple(temporal_state_shape),
        stride=stride,
        storage_offset=offset_elems,
    )
    return conv_views, temporal_view
