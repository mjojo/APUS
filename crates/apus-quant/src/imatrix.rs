/// Quantization group size (32 weights per FP32 scale factor)
pub const QUANT_GROUP_SIZE: usize = 32;

/// Importance Matrix (imatrix / AWQ-scales) calibrator for preserving outlier channels
#[derive(Debug, Clone)]
pub struct ImatrixCalibrator {
    pub channel_importance: Vec<f32>,
}

impl ImatrixCalibrator {
    pub fn uniform(dim: usize) -> Self {
        Self {
            channel_importance: vec![1.0; dim],
        }
    }

    /// Computes input channel importance from calibration activations (E[x^2])
    pub fn from_activations(activations: &[Vec<f32>], dim: usize) -> Self {
        let mut importance = vec![1e-4f32; dim];
        for act in activations {
            for (i, &v) in act.iter().take(dim).enumerate() {
                importance[i] += v * v;
            }
        }
        let n = activations.len().max(1) as f32;
        for val in &mut importance {
            *val = (*val / n).sqrt().clamp(0.5, 4.0);
        }
        Self {
            channel_importance: importance,
        }
    }
}

/// Symmetric INT4 expert weight quantizer (Q4_K_M with per-group scale factors)
/// Packed block layout: for each group of 32 weights -> 4 bytes scale (f32 LE) + 16 bytes (32 x 4-bit nibbles)
pub struct QuantizerInt4;

impl QuantizerInt4 {
    /// Quantizes an FP32 weight slice into a compact INT4 buffer using optional imatrix importance weights
    pub fn quantize(weights: &[f32], imatrix: Option<&ImatrixCalibrator>) -> Vec<u8> {
        let num_groups = weights.len().div_ceil(QUANT_GROUP_SIZE);
        let mut out = Vec::with_capacity(num_groups * 20);

        for (g_idx, chunk) in weights.chunks(QUANT_GROUP_SIZE).enumerate() {
            let mut padded = [0.0f32; QUANT_GROUP_SIZE];
            for (i, &w) in chunk.iter().enumerate() {
                let imp = imatrix
                    .and_then(|im| im.channel_importance.get(g_idx * QUANT_GROUP_SIZE + i).copied())
                    .unwrap_or(1.0);
                // Salient channel precision boost
                padded[i] = w * imp.powf(0.05);
            }

            let max_abs = padded.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let scale = if max_abs > 1e-8 { max_abs / 7.0 } else { 1.0 };
            out.extend_from_slice(&scale.to_le_bytes());

            // Pack 2 INT4 values (-7..+7, biased by +8 -> 1..15) per byte
            for pair in padded.chunks_exact(2) {
                let q0 = ((pair[0] / scale).round().clamp(-7.0, 7.0) as i8 + 8) as u8 & 0x0F;
                let q1 = ((pair[1] / scale).round().clamp(-7.0, 7.0) as i8 + 8) as u8 & 0x0F;
                out.push(q0 | (q1 << 4));
            }
        }
        out
    }

    /// Dequantizes a packed INT4 buffer back into an FP32 vector
    pub fn dequantize(packed: &[u8], max_elements: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(max_elements);
        for block in packed.chunks_exact(20) {
            let scale = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
            for &byte in &block[4..20] {
                if out.len() >= max_elements {
                    break;
                }
                let q0 = (byte & 0x0F) as i8 - 8;
                out.push((q0 as f32) * scale);
                if out.len() >= max_elements {
                    break;
                }
                let q1 = ((byte >> 4) & 0x0F) as i8 - 8;
                out.push((q1 as f32) * scale);
            }
            if out.len() >= max_elements {
                break;
            }
        }
        out
    }
}
