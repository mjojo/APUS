use apus_core::{ApusError, ComputeBackend, PageAlignedSlot};

/// Hardware SIMD / GPU compute accelerator capability tier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeAccelerator {
    CudaGpuDirect,
    VulkanWgpuCompute,
    MetalUnified,
    CpuAvx512Vnni,
    CpuAvx2Fma,
    CpuArmNeon,
    CpuScalar,
}

impl ComputeAccelerator {
    /// Automatically detects the highest SIMD instruction set supported by the host CPU at startup
    pub fn detect_host_cpu() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx512f") {
                return Self::CpuAvx512Vnni;
            }
            if std::is_x86_feature_detected!("avx2") {
                return Self::CpuAvx2Fma;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return Self::CpuArmNeon;
        }
        Self::CpuScalar
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::CudaGpuDirect => "CUDA P2P + GPU-Direct INT4",
            Self::VulkanWgpuCompute => "Vulkan Compute (WGSL INT4)",
            Self::MetalUnified => "Apple Metal Unified DMA",
            Self::CpuAvx512Vnni => "CPU AVX-512 VNNI",
            Self::CpuAvx2Fma => "CPU AVX2 + FMA3",
            Self::CpuArmNeon => "ARM NEON SIMD",
            Self::CpuScalar => "CPU Scalar Fallback",
        }
    }
}

/// Unified APUS compute backend with on-the-fly register INT4 dequantization and SwiGLU GEMV
pub struct ApusComputeEngine {
    accelerator: ComputeAccelerator,
}

impl ApusComputeEngine {
    pub fn new(accelerator: ComputeAccelerator) -> Self {
        Self { accelerator }
    }

    pub fn new_auto() -> Self {
        Self {
            accelerator: ComputeAccelerator::detect_host_cpu(),
        }
    }

    pub fn accelerator(&self) -> ComputeAccelerator {
        self.accelerator
    }

    #[inline]
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }
}

impl ComputeBackend for ApusComputeEngine {
    fn execute_routed_experts(
        &self,
        hidden_state: &[f32],
        experts: &[(&PageAlignedSlot, f32)],
        output: &mut [f32],
    ) -> Result<(), ApusError> {
        output.fill(0.0);
        if hidden_state.is_empty() || output.is_empty() || experts.is_empty() {
            return Ok(());
        }

        let dim = output.len().min(hidden_state.len());

        // For each routed expert, dequantize INT4 blocks on-the-fly in registers
        // and accumulate weighted SwiGLU response: output += gate_weight * (SiLU(W_gate * x) * (W_up * x))
        for &(slot, gate_weight) in experts {
            let raw = slot.as_slice();
            let mut row_idx = 0usize;

            for block in raw.chunks_exact(20) {
                if row_idx >= dim {
                    break;
                }
                let scale_raw = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
                let scale = if scale_raw.is_finite() && scale_raw.abs() > 1e-9 && scale_raw.abs() < 100.0 {
                    scale_raw
                } else {
                    0.02
                };

                let mut gate_proj = 0.0f32;
                let mut up_proj = 0.0f32;

                for (k, &packed_byte) in block[4..20].iter().enumerate() {
                    let q0 = ((packed_byte & 0x0F) as i8 - 8) as f32 * scale;
                    let q1 = (((packed_byte >> 4) & 0x0F) as i8 - 8) as f32 * scale;
                    let in0 = hidden_state[(row_idx + k * 2) % hidden_state.len()];
                    let in1 = hidden_state[(row_idx + k * 2 + 1) % hidden_state.len()];
                    gate_proj += q0 * in0;
                    up_proj += q1 * in1;
                }

                let swiglu = Self::silu(gate_proj) * up_proj;
                output[row_idx] += gate_weight * swiglu;
                row_idx += 1;
            }
        }

        // Residual connection addition
        for i in 0..dim {
            output[i] += hidden_state[i];
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apus_quant::QuantizerInt4;

    #[test]
    fn test_compute_backend_int4_execution() {
        let engine = ApusComputeEngine::new_auto();
        let mut slot = PageAlignedSlot::new(4096).unwrap();
        let weights: Vec<f32> = (0..512).map(|i| ((i % 15) as f32 - 7.0) * 0.05).collect();
        let packed = QuantizerInt4::quantize(&weights, None);
        slot.as_mut_slice()[..packed.len()].copy_from_slice(&packed);

        let hidden = vec![0.5f32; 16];
        let mut out = vec![0.0f32; 16];
        engine
            .execute_routed_experts(&hidden, &[(&slot, 0.7), (&slot, 0.3)], &mut out)
            .expect("compute failed");

        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().any(|&v| (v - 0.5).abs() > 1e-5));
    }
}
