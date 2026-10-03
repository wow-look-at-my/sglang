//! The model-family seam of the server MM pipeline.

/// Typed tensor payload. Grows a variant per dtype actually produced by a
/// family — not speculatively.
pub enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
    /// Raw BF16 bits, exposed to numpy as u16 without copying the allocation.
    Bf16(Vec<u16>),
}

impl TensorData {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(data) => data.len(),
            Self::I64(data) => data.len(),
            Self::Bf16(data) => data.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: TensorData,
}

/// Named auxiliary tensors that reach the model runner as kwargs — the Rust analogue.
pub type NamedTensors = Vec<(String, Tensor)>;

/// One decoded media item handed to [`MmFamilyProcessor::process_item`].
pub enum DecodedMedia {
    /// HWC u8 RGB.
    Image {
        rgb: Vec<u8>,
        height: usize,
        width: usize,
    },
}

/// Family-internal geometry of one processed item, consumed.
#[derive(Clone, Debug)]
pub enum Geometry {
    /// `[t, h, w]` patch grid (`t` = 1 for still images).
    Grid([u32; 3]),
}

/// One processed media item, mirroring Python's `MultimodalDataItem`: the
/// primary feature tensor, named auxiliary tensors.
pub struct ProcessedItem {
    /// The model's feature tensor for this item (qwen: `pixel_values`).
    pub feature: Tensor,
    pub aux: NamedTensors,
    pub geometry: Geometry,
}

/// The tokens one media item occupies in the expanded prompt.
pub enum TokenPattern {
    /// N copies of one placeholder id (qwen-style).
    Repeat { id: i32, n: usize },
    /// An explicit id sequence — tile markers, row separators.
    Explicit(Vec<i32>),
}

/// One span of the expanded prompt.
pub enum Segment {
    /// Copy `src` (a range into the original ids) verbatim.
    Text(std::ops::Range<usize>),
    /// Media item `item`'s token span.
    Media { item: usize, pattern: TokenPattern },
}

/// Prompt geometry as data: the family *describes* the expansion, the driver
/// *applies* it (`common::token_layout::apply_layout`).
pub struct TokenLayout {
    pub segments: Vec<Segment>,
}

/// Modalities a family accepts; the server's message layer rejects anything a family does not declare.
#[derive(Clone, Copy, Debug, Default)]
pub struct Capabilities {
    pub video: bool,
    pub audio: bool,
}

/// Position scheme of the expanded prompt.
pub enum PositionOutput {
    /// Plain sequential positions — the scheduler needs nothing extra.
    Rope1D,
    /// M-RoPE: flattened row-major `[3, input_len]` positions + the position delta (`max + 1 - input_len`).
    MRope { positions: Vec<i64>, delta: i64 },
}

/// The per-model-family hooks of the server pipeline.
/// [`crate::registry::pipeline_from_spec`].
/// runtime spec JSON (resolved from the HF config on the Python side);
/// nothing is hardcoded per model.
pub trait MmFamilyProcessor: Send + Sync {
    /// Modalities beyond images this family accepts. Default: images only.
    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    /// Preprocess one decoded media item.
    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem, String>;

    /// Describe how the prompt expands around the processed items (in prompt order).
    fn layout(&self, input_ids: &[i32], items: &[Geometry]) -> Result<TokenLayout, String>;

    /// Positions for the expanded prompt. Families without a custom scheme
    /// keep the default.
    fn positions(
        &self,
        input_len: usize,
        offsets: &[(u32, u32)],
        items: &[Geometry],
    ) -> Result<PositionOutput, String> {
        let _ = (input_len, offsets, items);
        Ok(PositionOutput::Rope1D)
    }
}
