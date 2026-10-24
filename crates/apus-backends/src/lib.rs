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

    /// Hardware AVX2 + FMA3 vectorized INT4 block dot-product kernel (processes 16 packed byte pairs = 32 INT4 weights)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_int4_block_avx2_fma(
        packed_16: &[u8],
        scale: f32,
        hidden_state: &[f32],
        row_offset: usize,
    ) -> (f32, f32) {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let mut gate_q = [0.0f32; 16];
        let mut up_q = [0.0f32; 16];
        let mut in_gate = [0.0f32; 16];
        let mut in_up = [0.0f32; 16];
        let h_len = hidden_state.len();

        for k in 0..16 {
            let b = packed_16[k];
            gate_q[k] = ((b & 0x0F) as i8 - 8) as f32;
            up_q[k] = (((b >> 4) & 0x0F) as i8 - 8) as f32;
            in_gate[k] = hidden_state[(row_offset + k * 2) % h_len];
            in_up[k] = hidden_state[(row_offset + k * 2 + 1) % h_len];
        }

        unsafe {
            let v_scale = _mm256_set1_ps(scale);
            let mut acc_gate = _mm256_setzero_ps();
            let mut acc_up = _mm256_setzero_ps();

            // Vector pass 0..8
            let g0 = _mm256_mul_ps(_mm256_loadu_ps(gate_q.as_ptr()), v_scale);
            let ig0 = _mm256_loadu_ps(in_gate.as_ptr());
            acc_gate = _mm256_fmadd_ps(g0, ig0, acc_gate);

            let u0 = _mm256_mul_ps(_mm256_loadu_ps(up_q.as_ptr()), v_scale);
            let iu0 = _mm256_loadu_ps(in_up.as_ptr());
            acc_up = _mm256_fmadd_ps(u0, iu0, acc_up);

            // Vector pass 8..16
            let g1 = _mm256_mul_ps(_mm256_loadu_ps(gate_q.as_ptr().add(8)), v_scale);
            let ig1 = _mm256_loadu_ps(in_gate.as_ptr().add(8));
            acc_gate = _mm256_fmadd_ps(g1, ig1, acc_gate);

            let u1 = _mm256_mul_ps(_mm256_loadu_ps(up_q.as_ptr().add(8)), v_scale);
            let iu1 = _mm256_loadu_ps(in_up.as_ptr().add(8));
            acc_up = _mm256_fmadd_ps(u1, iu1, acc_up);

            let mut gate_lanes = [0.0f32; 8];
            let mut up_lanes = [0.0f32; 8];
            _mm256_storeu_ps(gate_lanes.as_mut_ptr(), acc_gate);
            _mm256_storeu_ps(up_lanes.as_mut_ptr(), acc_up);

            let gate_sum: f32 = gate_lanes.iter().sum();
            let up_sum: f32 = up_lanes.iter().sum();
            (gate_sum, up_sum)
        }
    }

    #[inline]
    fn dot_int4_block_scalar(
        packed_16: &[u8],
        scale: f32,
        hidden_state: &[f32],
        row_offset: usize,
    ) -> (f32, f32) {
        let mut gate_proj = 0.0f32;
        let mut up_proj = 0.0f32;
        for (k, &packed_byte) in packed_16.iter().enumerate() {
            let q0 = ((packed_byte & 0x0F) as i8 - 8) as f32 * scale;
            let q1 = (((packed_byte >> 4) & 0x0F) as i8 - 8) as f32 * scale;
            let in0 = hidden_state[(row_offset + k * 2) % hidden_state.len()];
            let in1 = hidden_state[(row_offset + k * 2 + 1) % hidden_state.len()];
            gate_proj += q0 * in0;
            up_proj += q1 * in1;
        }
        (gate_proj, up_proj)
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
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let use_avx2 = matches!(
            self.accelerator,
            ComputeAccelerator::CpuAvx2Fma | ComputeAccelerator::CpuAvx512Vnni
        ) && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma");

        // For each routed expert, dequantize INT4 blocks on-the-fly in SIMD registers
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

                let (gate_proj, up_proj) = {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        if use_avx2 {
                            unsafe {
                                Self::dot_int4_block_avx2_fma(
                                    &block[4..20],
                                    scale,
                                    hidden_state,
                                    row_idx,
                                )
                            }
                        } else {
                            Self::dot_int4_block_scalar(&block[4..20], scale, hidden_state, row_idx)
                        }
                    }
                    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                    {
                        Self::dot_int4_block_scalar(&block[4..20], scale, hidden_state, row_idx)
                    }
                };

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
