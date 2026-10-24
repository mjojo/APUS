pub mod autodetect;
pub mod cache;
pub mod config;
pub mod grammar;
pub mod mla;
pub mod mtp;
pub mod profiles;
pub mod sampler;
pub mod slab;
pub mod stress;
pub mod topo;

use std::alloc::{Layout, alloc, dealloc};
use std::ptr::NonNull;
use thiserror::Error;

pub use autodetect::{AutoDetectError, HardwareProfile, detect_system_hardware};
pub use cache::{ArcCache, ArcTelemetry, CacheTier};
pub use config::ApusConfig;
pub use grammar::{GrammarConstraintEngine, GrammarMode};
pub use mla::{GqaKvCache, KV_PAGE_TOKENS, LatentKvToken, MLA_LATENT_DIM, MlaKvCache};
pub use mtp::{
    ExternalDraftSpeculativeVerifier, LookAheadPrefetcher, MoERouter, MtpSpeculativeEngine,
    RoutedExpertSelection,
};
pub use profiles::MoEModelPreset;
pub use sampler::{BpeTokenizer, GenerationSampler, SamplerConfig};
pub use slab::SlabArena;
pub use stress::{ConcurrencyStressRunner, StressReport};
pub use topo::{HardwareTopologyProfile, HardwareTopologyScanner, PcieLinkTopology};

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn VirtualLock(lpAddress: *mut std::ffi::c_void, dwSize: usize) -> i32;
    fn VirtualUnlock(lpAddress: *mut std::ffi::c_void, dwSize: usize) -> i32;
}

#[cfg(unix)]
unsafe extern "C" {
    fn mlock(addr: *const std::ffi::c_void, len: usize) -> i32;
    fn munlock(addr: *const std::ffi::c_void, len: usize) -> i32;
}

/// Author signature of the APUS runtime engine
pub const ENGINE_AUTHOR: &str = "mjojo";

/// Page alignment size (4096 bytes) required for O_DIRECT (Linux) and FILE_FLAG_NO_BUFFERING (Windows)
pub const PAGE_ALIGNMENT: usize = 4096;

#[derive(Debug, Error)]
pub enum ApusError {
    #[error("Page-aligned slab memory allocation failed: {0}")]
    AllocationFailed(String),

    #[error("Storage I/O error at layer {layer}, expert {expert}: {source}")]
    StorageIo {
        layer: u16,
        expert: u16,
        #[source]
        source: std::io::Error,
    },

    #[error("BLAKE3 cryptographic checksum mismatch for expert ({layer}, {expert})")]
    CorruptedExpertWeights { layer: u16, expert: u16 },

    #[error(
        "System NVMe thermal threshold exceeded ({temp_c}°C) — emergency thermal brake engaged"
    )]
    ThermalThrottleTripped { temp_c: u8 },
}

/// Unique coordinate identifier for a routed MoE expert
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct ExpertId {
    pub layer: u16,
    pub expert: u16,
}

/// 4KB page-aligned RAM buffer (Pinned Slab Slot) protected against OS swap eviction
pub struct PageAlignedSlot {
    ptr: NonNull<u8>,
    layout: Layout,
    pinned: bool,
    os_locked: bool,
}

unsafe impl Send for PageAlignedSlot {}
unsafe impl Sync for PageAlignedSlot {}

impl PageAlignedSlot {
    pub fn new(size_bytes: usize) -> Result<Self, ApusError> {
        let safe_size = size_bytes.max(PAGE_ALIGNMENT);
        let aligned_size = (safe_size + PAGE_ALIGNMENT - 1) & !(PAGE_ALIGNMENT - 1);
        let layout = Layout::from_size_align(aligned_size, PAGE_ALIGNMENT)
            .map_err(|e| ApusError::AllocationFailed(e.to_string()))?;

        let raw_ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw_ptr)
            .ok_or_else(|| ApusError::AllocationFailed("Out of physical RAM".into()))?;

        unsafe {
            std::ptr::write_bytes(ptr.as_ptr(), 0, aligned_size);
        }

        Ok(Self {
            ptr,
            layout,
            pinned: false,
            os_locked: false,
        })
    }

    /// Pins the buffer in physical RAM using OS-native VirtualLock (Windows) or mlock (Unix)
    /// to prevent the OS kernel from swapping cached MoE expert weights to the pagefile.
    pub fn pin_memory(&mut self) -> bool {
        self.pinned = true;
        if self.os_locked {
            return true;
        }
        #[cfg(windows)]
        {
            let res = unsafe {
                VirtualLock(
                    self.ptr.as_ptr() as *mut std::ffi::c_void,
                    self.layout.size(),
                )
            };
            self.os_locked = res != 0;
        }
        #[cfg(unix)]
        {
            let res = unsafe {
                mlock(
                    self.ptr.as_ptr() as *const std::ffi::c_void,
                    self.layout.size(),
                )
            };
            self.os_locked = res == 0;
        }
        self.os_locked
    }

    pub fn is_pinned(&self) -> bool {
        self.pinned
    }

    /// Returns true if the OS kernel confirmed physical page locking (`VirtualLock` / `mlock`)
    pub fn is_os_locked(&self) -> bool {
        self.os_locked
    }

    /// Verifies that the memory address is strictly a multiple of 4096 bytes
    #[inline]
    pub fn is_page_aligned(&self) -> bool {
        (self.ptr.as_ptr() as usize).is_multiple_of(PAGE_ALIGNMENT)
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.layout.size()
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.layout.size()) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.layout.size()) }
    }
}

impl Drop for PageAlignedSlot {
    fn drop(&mut self) {
        if self.os_locked {
            #[cfg(windows)]
            unsafe {
                let _ = VirtualUnlock(
                    self.ptr.as_ptr() as *mut std::ffi::c_void,
                    self.layout.size(),
                );
            }
            #[cfg(unix)]
            unsafe {
                let _ = munlock(
                    self.ptr.as_ptr() as *const std::ffi::c_void,
                    self.layout.size(),
                );
            }
        }
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

/// Pluggable storage engine abstraction (.apus 4K block, mmap, or compressed KV store)
pub trait ExpertStorage: Send + Sync {
    /// Asynchronously streams an expert from disk directly into a 4KB-aligned L2 RAM slot
    fn read_expert_into_slot(
        &self,
        id: ExpertId,
        slot: &mut PageAlignedSlot,
    ) -> impl std::future::Future<Output = Result<(), ApusError>> + Send;

    /// Issues look-ahead prefetch hints to the NVMe controller for layers N+1 and N+2
    fn prefetch_experts(&self, upcoming: &[ExpertId]);
}

/// Hardware compute backend abstraction (CUDA, Metal, Vulkan/wgpu, AVX-512/AVX2)
pub trait ComputeBackend: Send + Sync {
    /// Streams compressed INT4/INT2 expert weights to the accelerator with on-the-fly register dequantization
    fn execute_routed_experts(
        &self,
        hidden_state: &[f32],
        experts: &[(&PageAlignedSlot, f32)], // (Expert weights slot, router gate weight)
        output: &mut [f32],
    ) -> Result<(), ApusError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_aligned_slot_alignment() {
        let mut slot = PageAlignedSlot::new(5000).expect("alloc failed");
        assert!(slot.is_page_aligned());
        assert_eq!(slot.size(), 8192);
        slot.pin_memory();
        assert!(slot.is_pinned());
        slot.as_mut_slice()[0] = 0xAA;
        assert_eq!(slot.as_slice()[0], 0xAA);
    }

    #[test]
    fn test_arc_cache_recency_and_frequency() {
        let cache = ArcCache::new(4, 2);
        let e1 = ExpertId {
            layer: 0,
            expert: 1,
        };
        let e2 = ExpertId {
            layer: 0,
            expert: 2,
        };
        assert_eq!(cache.access(e1).0, CacheTier::L3NvmeMiss);
        // Second access promotes the expert to T2 and elevates it into L1 VRAM
        assert_eq!(cache.access(e1).0, CacheTier::L2SlabRam);
        assert_eq!(cache.access(e1).0, CacheTier::L1Vram);
        assert_eq!(cache.access(e2).0, CacheTier::L3NvmeMiss);
        let stats = cache.telemetry();
        assert_eq!(stats.l1_hits, 1);
        assert_eq!(stats.l2_hits, 1);
        assert_eq!(stats.ssd_misses, 2);
    }

    #[test]
    fn test_mla_kv_cache_roundtrip() {
        let mut kv = MlaKvCache::new(4, 16);
        let state = vec![0.25f32; MLA_LATENT_DIM];
        for tok in 0..30 {
            kv.append_token(tok, &state);
        }
        assert_eq!(kv.total_context_tokens(), 30);
        let serialized = kv.to_bytes();
        let restored = MlaKvCache::from_bytes(&serialized, 4, 16);
        assert_eq!(restored.total_context_tokens(), 30);

        // Verify GqaKvCache and ExternalDraftSpeculativeVerifier (Sprints 16 & 17)
        let mut gqa = GqaKvCache::new(8, 64, 4, 16);
        let gqa_vec = vec![0.15f32; gqa.kv_vector_dim()];
        gqa.append_gqa_token(1, &gqa_vec);
        assert_eq!(gqa.total_context_tokens(), 1);

        let draft_verifier = ExternalDraftSpeculativeVerifier::new(4, 0.8);
        let accepted = draft_verifier.verify_draft_batch(
            &[(10, 0.9), (20, 0.9), (30, 0.9)],
            &[(10, 0.95), (20, 0.85), (99, 0.10)],
        );
        assert_eq!(accepted, vec![10, 20, 99]);
    }

    #[test]
    fn test_bpe_tokenizer_and_sampler() {
        let tok = BpeTokenizer::new_default();
        let text = "pub struct PageAlignedSlot // mjojo";
        let encoded = tok.encode(text);
        assert!(!encoded.is_empty());
        assert!(encoded.len() < text.len());
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, text);

        let mut sampler = GenerationSampler::new(SamplerConfig::default(), 42);
        let mut logits = vec![-2.0f32; 128];
        logits[17] = 10.0;
        let sampled = sampler.sample(&logits, &[1, 2, 3]);
        assert_eq!(sampled, 17);

        // Verify live RAM auto-detection via sysinfo
        let hw = detect_system_hardware().expect("sysinfo memory detection failed");
        assert!(hw.total_ram_gb > 0.0);
        assert!(hw.recommended_l2_slab_gb >= 8.0);
        assert!((4.0..=12.0).contains(&hw.os_reserved_gb));
    }
}
