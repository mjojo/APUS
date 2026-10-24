use crate::{ApusConverter, ConverterError, ImatrixCalibrator, QuantizerInt4, SourceExpert};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, Read};

/// Magic bytes for GGUF v2/v3 model files (`"GGUF"`)
pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// Parsed GGUF file header summary
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHeaderSummary {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

impl GgufHeaderSummary {
    /// Parses the 24-byte binary GGUF header from any reader
    pub fn read_from<R: Read>(mut reader: R) -> io::Result<Self> {
        let mut buf = [0u8; 24];
        reader.read_exact(&mut buf)?;
        if &buf[0..4] != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid GGUF magic bytes (expected 'GGUF')",
            ));
        }
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let tensor_count = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        let metadata_kv_count = u64::from_le_bytes(buf[16..24].try_into().unwrap());
        Ok(Self {
            version,
            tensor_count,
            metadata_kv_count,
        })
    }
}

/// Tensor metadata entry inside a Hugging Face `.safetensors` JSON header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorDescriptor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

/// Classification of a tensor in an MoE architecture
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoETensorRole {
    /// Resident in VRAM/RAM (Attention, Shared Experts, Embeddings, RMSNorm, Router Gate, LM Head)
    DenseBackbone,
    /// Routed Expert weight block (`layer_id`, `expert_id`) to be quantized and streamed from NVMe
    RoutedExpert { layer_id: u16, expert_id: u16 },
}

/// Classifies a tensor name (Hugging Face Safetensors or GGUF convention) into Dense vs Routed Expert
pub fn classify_moe_tensor_name(name: &str) -> MoETensorRole {
    // Hugging Face convention: "model.layers.12.mlp.experts.45.gate_proj.weight"
    if let Some(layers_pos) = name.find("layers.") {
        let after_layers = &name[layers_pos + 7..];
        if let Some(dot1) = after_layers.find('.') {
            if let Ok(layer_id) = after_layers[..dot1].parse::<u16>() {
                if let Some(exp_pos) = after_layers.find("experts.") {
                    // Exclude shared_experts (which must stay pinned in the dense backbone)
                    if !after_layers.contains("shared_experts") {
                        let after_exp = &after_layers[exp_pos + 8..];
                        let end = after_exp.find('.').unwrap_or(after_exp.len());
                        if let Ok(expert_id) = after_exp[..end].parse::<u16>() {
                            return MoETensorRole::RoutedExpert { layer_id, expert_id };
                        }
                    }
                }
            }
        }
    }
    MoETensorRole::DenseBackbone
}

/// Zero-copy parser for Hugging Face `.safetensors` buffers that automatically separates
/// dense backbone tensors from routed MoE experts and compiles them into an `ApusConverter`.
pub struct SafetensorsMoEParser {
    pub descriptors: HashMap<String, SafetensorDescriptor>,
    pub payload_offset: usize,
}

impl SafetensorsMoEParser {
    /// Parses the 8-byte header size and JSON metadata table from a `.safetensors` byte buffer
    pub fn parse_header(raw_bytes: &[u8]) -> io::Result<Self> {
        if raw_bytes.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Safetensors buffer smaller than 8 bytes",
            ));
        }
        let header_len = u64::from_le_bytes(raw_bytes[0..8].try_into().unwrap()) as usize;
        let payload_offset = 8 + header_len;
        if raw_bytes.len() < payload_offset {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Truncated Safetensors JSON header",
            ));
        }

        let header_json = &raw_bytes[8..payload_offset];
        let descriptors: HashMap<String, SafetensorDescriptor> =
            serde_json::from_slice(header_json).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid Safetensors JSON header: {e}"),
                )
            })?;

        Ok(Self {
            descriptors,
            payload_offset,
        })
    }

    /// Splits a `.safetensors` buffer into dense backbone bytes and INT4-quantized routed experts
    pub fn into_apus_converter(
        &self,
        raw_bytes: &[u8],
        model_id: &str,
        total_layers: u32,
        experts_per_layer: u32,
        imatrix: Option<&ImatrixCalibrator>,
    ) -> Result<ApusConverter, ConverterError> {
        let payload = &raw_bytes[self.payload_offset..];
        let mut dense_weights = Vec::new();
        let mut routed_map: HashMap<(u16, u16), Vec<f32>> = HashMap::new();

        let mut sorted_names: Vec<&String> = self.descriptors.keys().collect();
        sorted_names.sort();

        for name in sorted_names {
            let desc = &self.descriptors[name];
            let [start, end] = desc.data_offsets;
            if end > payload.len() || start > end {
                continue;
            }
            let slice = &payload[start..end];

            match classify_moe_tensor_name(name) {
                MoETensorRole::DenseBackbone => {
                    dense_weights.extend_from_slice(slice);
                }
                MoETensorRole::RoutedExpert { layer_id, expert_id } => {
                    let entry = routed_map.entry((layer_id, expert_id)).or_default();
                    for chunk in slice.chunks_exact(4) {
                        entry.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                    }
                }
            }
        }

        if dense_weights.is_empty() {
            dense_weights.resize(4096, 0u8);
        }

        let mut converter =
            ApusConverter::new(model_id, total_layers, experts_per_layer, dense_weights);

        let mut sorted_experts: Vec<((u16, u16), Vec<f32>)> = routed_map.into_iter().collect();
        sorted_experts.sort_by_key(|&(k, _)| k);

        for ((layer_id, expert_id), fp32_weights) in sorted_experts {
            let quantized = QuantizerInt4::quantize(&fp32_weights, imatrix);
            converter.push_expert(SourceExpert {
                layer_id,
                expert_id,
                raw_quantized_weights: quantized,
            });
        }

        Ok(converter)
    }
}
