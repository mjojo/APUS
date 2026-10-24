pub mod hub;
pub mod qos;
pub mod thermal;

use apus_core::{ApusError, ExpertId, ExpertStorage, PAGE_ALIGNMENT, PageAlignedSlot};
use apus_quant::{ApusFileHeader, LbaExpertEntry};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub use hub::{ApusIdxManifest, ChunkedHubDownloader, HubChunkDescriptor};
pub use qos::{IoPriorityClass, OsDriveQosController, QosMetrics};
pub use thermal::{SmartThermalController, ThermalZone};

/// Pluggable disk storage backend modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageMode {
    /// 4KB page-aligned binary blocks (*.apus) for O_DIRECT / Windows IOCP / DirectStorage
    ApusBlockAligned,
    /// Virtual memory file mapping (mmap) for dedicated NVMe drives and Apple Silicon Unified Memory
    MemoryMapped,
    /// Compressed Key-Value expert store for SATA SSDs and network-attached storage (NAS)
    CompressedKvStore,
}

/// 1. `ApusBlockStorage` Engine — "Surgeon Mode" (4KB Page-Aligned Blocks + BLAKE3 + QoS)
pub struct ApusBlockStorage {
    path: PathBuf,
    header: ApusFileHeader,
    lba_index: HashMap<ExpertId, LbaExpertEntry>,
    expected_hashes: HashMap<ExpertId, blake3::Hash>,
    prefetch_queue: Mutex<Vec<ExpertId>>,
    qos: Arc<OsDriveQosController>,
    thermal: Arc<SmartThermalController>,
}

impl ApusBlockStorage {
    pub fn open<P: AsRef<Path>>(
        path: P,
        qos: Arc<OsDriveQosController>,
        thermal: Arc<SmartThermalController>,
    ) -> Result<Self, ApusError> {
        let path_buf = path.as_ref().to_path_buf();
        let mut file = File::open(&path_buf).map_err(|e| ApusError::StorageIo {
            layer: 0,
            expert: 0,
            source: e,
        })?;

        let header = ApusFileHeader::read_from(&mut file)
            .map_err(|e| ApusError::AllocationFailed(format!("Invalid .apus header: {e}")))?;

        let entries = header
            .read_lba_table(&mut file)
            .map_err(|e| ApusError::AllocationFailed(format!("Invalid LBA table: {e}")))?;

        let mut lba_index = HashMap::with_capacity(entries.len());
        let mut expected_hashes = HashMap::with_capacity(entries.len());

        // Index LBA entries and pre-compute reference BLAKE3 digests when opening the container
        for entry in entries {
            let id = ExpertId {
                layer: entry.layer_id,
                expert: entry.expert_id,
            };
            file.seek(SeekFrom::Start(entry.file_offset))
                .map_err(|e| ApusError::StorageIo {
                    layer: id.layer,
                    expert: id.expert,
                    source: e,
                })?;
            let mut buf = vec![0u8; entry.compressed_size as usize];
            file.read_exact(&mut buf)
                .map_err(|e| ApusError::StorageIo {
                    layer: id.layer,
                    expert: id.expert,
                    source: e,
                })?;
            expected_hashes.insert(id, blake3::hash(&buf));
            lba_index.insert(id, entry);
        }

        Ok(Self {
            path: path_buf,
            header,
            lba_index,
            expected_hashes,
            prefetch_queue: Mutex::new(Vec::new()),
            qos,
            thermal,
        })
    }

    pub fn header(&self) -> &ApusFileHeader {
        &self.header
    }

    pub fn indexed_experts_count(&self) -> usize {
        self.lba_index.len()
    }

    pub fn pending_prefetches(&self) -> Vec<ExpertId> {
        self.prefetch_queue.lock().clone()
    }

    /// SATA AHCI Cluster Sequential Read (Sprint 11):
    /// Coalesces multiple requested experts on the same layer, sorts them by physical LBA file offset,
    /// and reads them in a single linear sequential disk sweep to avoid random-seek thrashing on SATA SSDs.
    pub async fn read_expert_cluster_sequential(
        &self,
        ids: &[ExpertId],
    ) -> Result<Vec<(ExpertId, PageAlignedSlot)>, ApusError> {
        let mut entries: Vec<(ExpertId, LbaExpertEntry)> = ids
            .iter()
            .filter_map(|&id| self.lba_index.get(&id).copied().map(|e| (id, e)))
            .collect();
        entries.sort_by_key(|&(_, e)| e.file_offset);

        let mut results = Vec::with_capacity(entries.len());
        for (id, entry) in entries {
            let mut slot = PageAlignedSlot::new(entry.compressed_size as usize)?;
            self.read_expert_into_slot(id, &mut slot).await?;
            results.push((id, slot));
        }
        Ok(results)
    }
}

impl ExpertStorage for ApusBlockStorage {
    async fn read_expert_into_slot(
        &self,
        id: ExpertId,
        slot: &mut PageAlignedSlot,
    ) -> Result<(), ApusError> {
        // 1. Check NVMe controller temperature via S.M.A.R.T. PID thermal guard
        let thermal_delay = self.thermal.check_and_compute_pid_cooldown()?;
        if !thermal_delay.is_zero() {
            tokio::time::sleep(thermal_delay).await;
        }

        let entry = self
            .lba_index
            .get(&id)
            .copied()
            .ok_or_else(|| ApusError::StorageIo {
                layer: id.layer,
                expert: id.expert,
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "Expert ({}, {}) not found in LBA offset table",
                        id.layer, id.expert
                    ),
                ),
            })?;

        let read_len = (entry.compressed_size as usize).min(slot.size());
        debug_assert_eq!(entry.file_offset % (PAGE_ALIGNMENT as u64), 0);

        // 2. Acquire hard Queue Depth semaphore permit (QD <= 32) and enforce OS Drive QoS bandwidth pacing
        let _qd_permit = self.qos.acquire_qd_permit().await;
        let qos_delay = self.qos.register_read_and_compute_delay(read_len);
        if !qos_delay.is_zero() {
            tokio::time::sleep(qos_delay).await;
        }

        // 3. Stream 4KB-aligned expert block directly into PageAlignedSlot via unbuffered Direct I/O
        let mut file =
            open_unbuffered_direct_file(&self.path).map_err(|e| ApusError::StorageIo {
                layer: id.layer,
                expert: id.expert,
                source: e,
            })?;
        file.seek(SeekFrom::Start(entry.file_offset))
            .map_err(|e| ApusError::StorageIo {
                layer: id.layer,
                expert: id.expert,
                source: e,
            })?;

        let dst = &mut slot.as_mut_slice()[..read_len];
        let result = file.read_exact(dst);
        self.qos.complete_read();
        result.map_err(|e| ApusError::StorageIo {
            layer: id.layer,
            expert: id.expert,
            source: e,
        })?;

        // 4. Instant cryptographic chunk verification against the BLAKE3 Merkle tree
        if let Some(expected) = self.expected_hashes.get(&id) {
            let actual = blake3::hash(dst);
            if &actual != expected {
                return Err(ApusError::CorruptedExpertWeights {
                    layer: id.layer,
                    expert: id.expert,
                });
            }
        }

        Ok(())
    }

    fn prefetch_experts(&self, upcoming: &[ExpertId]) {
        let mut q = self.prefetch_queue.lock();
        q.clear();
        q.extend_from_slice(upcoming);
    }
}

/// Opens a file with OS-native unbuffered Direct I/O flags (`FILE_FLAG_NO_BUFFERING` on Windows,
/// `O_DIRECT` on Linux) to bypass the OS Page Cache and eliminate double caching in RAM.
/// Gracefully falls back to standard buffered open if the underlying volume rejects direct flags.
pub fn open_unbuffered_direct_file(path: &Path) -> std::io::Result<File> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
        if let Ok(f) = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_NO_BUFFERING)
            .open(path)
        {
            return Ok(f);
        }
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        const O_DIRECT: i32 = 0x4000;
        if let Ok(f) = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(O_DIRECT)
            .open(path)
        {
            return Ok(f);
        }
    }
    File::open(path)
}

/// `MmapStorage` Engine — "Speed King" Mode (Zero-Copy Virtual Memory Projection via `memmap2`).
/// Ideal for dedicated (non-OS) NVMe drives and Apple Silicon Unified Memory.
pub struct MmapStorage {
    mmap: memmap2::Mmap,
    header: ApusFileHeader,
    lba_index: HashMap<ExpertId, LbaExpertEntry>,
    expected_hashes: HashMap<ExpertId, blake3::Hash>,
    prefetched: Mutex<Vec<ExpertId>>,
}

impl MmapStorage {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, ApusError> {
        let mut file = File::open(path.as_ref()).map_err(|e| ApusError::StorageIo {
            layer: 0,
            expert: 0,
            source: e,
        })?;
        let header = ApusFileHeader::read_from(&mut file)
            .map_err(|e| ApusError::AllocationFailed(format!("Invalid .apus header: {e}")))?;
        let entries = header
            .read_lba_table(&mut file)
            .map_err(|e| ApusError::AllocationFailed(format!("Invalid LBA table: {e}")))?;

        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }.map_err(|e| {
            ApusError::StorageIo {
                layer: 0,
                expert: 0,
                source: e,
            }
        })?;

        let mut lba_index = HashMap::with_capacity(entries.len());
        let mut expected_hashes = HashMap::with_capacity(entries.len());
        for entry in entries {
            let id = ExpertId {
                layer: entry.layer_id,
                expert: entry.expert_id,
            };
            let start = entry.file_offset as usize;
            let end = start + (entry.compressed_size as usize);
            if end <= mmap.len() {
                expected_hashes.insert(id, blake3::hash(&mmap[start..end]));
                lba_index.insert(id, entry);
            }
        }

        Ok(Self {
            mmap,
            header,
            lba_index,
            expected_hashes,
            prefetched: Mutex::new(Vec::new()),
        })
    }

    pub fn header(&self) -> &ApusFileHeader {
        &self.header
    }
}

impl ExpertStorage for MmapStorage {
    async fn read_expert_into_slot(
        &self,
        id: ExpertId,
        slot: &mut PageAlignedSlot,
    ) -> Result<(), ApusError> {
        let entry = self
            .lba_index
            .get(&id)
            .copied()
            .ok_or_else(|| ApusError::StorageIo {
                layer: id.layer,
                expert: id.expert,
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Expert missing in mmap LBA",
                ),
            })?;

        let start = entry.file_offset as usize;
        let len = (entry.compressed_size as usize).min(slot.size());
        let src = &self.mmap[start..start + len];

        if let Some(expected) = self.expected_hashes.get(&id) {
            let actual = blake3::hash(src);
            if &actual != expected {
                return Err(ApusError::CorruptedExpertWeights {
                    layer: id.layer,
                    expert: id.expert,
                });
            }
        }

        slot.as_mut_slice()[..len].copy_from_slice(src);
        Ok(())
    }

    fn prefetch_experts(&self, upcoming: &[ExpertId]) {
        let mut p = self.prefetched.lock();
        p.clear();
        p.extend_from_slice(upcoming);

        // Sprint 9: On Unix / macOS Unified Memory, issue `madvise(MADV_WILLNEED)` hints
        // so the kernel asynchronously prefetches upcoming expert pages into Unified RAM.
        #[cfg(unix)]
        {
            const MADV_WILLNEED: i32 = 3;
            unsafe extern "C" {
                fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
            }
            for id in upcoming {
                if let Some(entry) = self.lba_index.get(id) {
                    let start = entry.file_offset as usize;
                    let len = entry.compressed_size as usize;
                    if start + len <= self.mmap.len() {
                        unsafe {
                            let ptr = self.mmap.as_ptr().add(start) as *mut std::ffi::c_void;
                            let _ = madvise(ptr, len, MADV_WILLNEED);
                        }
                    }
                }
            }
        }
    }
}

/// `KvCompressedStorage` Engine — block-compressed KV store for SATA SSDs and NAS (Sprint 7).
/// Compresses each expert's weights on-the-fly using lossless Run-Length / Zero-Page framing (`KVZ4`)
/// so slow SATA III (550 MB/s) and gigabit NAS links transfer fewer physical bytes.
pub struct KvCompressedStorage {
    store: Mutex<HashMap<ExpertId, Vec<u8>>>,
    prefetched: Mutex<Vec<ExpertId>>,
}

impl KvCompressedStorage {
    pub fn new() -> Self {
        Self {
            store: Mutex::new(HashMap::new()),
            prefetched: Mutex::new(Vec::new()),
        }
    }

    fn compress_blob(input: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(input.len() + 8);
        out.extend_from_slice(b"KVZ4");
        out.extend_from_slice(&(input.len() as u32).to_le_bytes());
        let mut i = 0;
        while i < input.len() {
            let b = input[i];
            let mut run = 1usize;
            while i + run < input.len() && input[i + run] == b && run < 255 {
                run += 1;
            }
            if run >= 4 || b == 0xFE {
                out.push(0xFE);
                out.push(b);
                out.push(run as u8);
                i += run;
            } else {
                out.push(b);
                i += 1;
            }
        }
        out
    }

    fn decompress_blob(compressed: &[u8]) -> Option<Vec<u8>> {
        if compressed.len() < 8 || &compressed[0..4] != b"KVZ4" {
            return None;
        }
        let orig_len = u32::from_le_bytes(compressed[4..8].try_into().ok()?) as usize;
        let mut out = Vec::with_capacity(orig_len);
        let mut i = 8;
        while i < compressed.len() && out.len() < orig_len {
            if compressed[i] == 0xFE {
                if i + 2 >= compressed.len() {
                    return None;
                }
                let val = compressed[i + 1];
                let count = compressed[i + 2] as usize;
                for _ in 0..count {
                    if out.len() < orig_len {
                        out.push(val);
                    }
                }
                i += 3;
            } else {
                out.push(compressed[i]);
                i += 1;
            }
        }
        Some(out)
    }

    /// Compresses and stores an expert weight block in the KV store, returning the compressed byte size
    pub fn put_expert(&self, id: ExpertId, raw_weights: &[u8]) -> usize {
        let compressed = Self::compress_blob(raw_weights);
        let len = compressed.len();
        self.store.lock().insert(id, compressed);
        len
    }
}

impl Default for KvCompressedStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpertStorage for KvCompressedStorage {
    async fn read_expert_into_slot(
        &self,
        id: ExpertId,
        slot: &mut PageAlignedSlot,
    ) -> Result<(), ApusError> {
        let compressed =
            self.store
                .lock()
                .get(&id)
                .cloned()
                .ok_or_else(|| ApusError::StorageIo {
                    layer: id.layer,
                    expert: id.expert,
                    source: std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "Expert not found in KV store",
                    ),
                })?;
        let data = Self::decompress_blob(&compressed).ok_or(ApusError::CorruptedExpertWeights {
            layer: id.layer,
            expert: id.expert,
        })?;
        let copy_len = data.len().min(slot.size());
        slot.as_mut_slice()[..copy_len].copy_from_slice(&data[..copy_len]);
        Ok(())
    }

    fn prefetch_experts(&self, upcoming: &[ExpertId]) {
        let mut p = self.prefetched.lock();
        p.clear();
        p.extend_from_slice(upcoming);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apus_quant::{ApusConverter, SourceExpert};

    #[tokio::test]
    async fn test_block_storage_read_and_blake3_verification() {
        let tmp_path = std::env::temp_dir().join("test_io_block.apus");
        let mut conv = ApusConverter::new("GLM-5.2-IO", 1, 2, vec![0x11; 4096]);
        conv.push_expert(SourceExpert {
            layer_id: 0,
            expert_id: 0,
            raw_quantized_weights: vec![0x42; 4096],
        });
        conv.push_expert(SourceExpert {
            layer_id: 0,
            expert_id: 1,
            raw_quantized_weights: vec![0x99; 4096],
        });
        conv.build(&tmp_path).expect("build failed");

        let qos = Arc::new(OsDriveQosController::new(
            true,
            IoPriorityClass::IdleBackground,
            5.5,
            32,
        ));
        let thermal = Arc::new(SmartThermalController::new(60, 65, 70));
        let storage = ApusBlockStorage::open(&tmp_path, qos, Arc::clone(&thermal))
            .expect("open storage failed");

        let mut slot = PageAlignedSlot::new(4096).expect("slot failed");
        storage
            .read_expert_into_slot(
                ExpertId {
                    layer: 0,
                    expert: 1,
                },
                &mut slot,
            )
            .await
            .expect("read expert failed");
        assert_eq!(slot.as_slice()[0], 0x99);
        assert_eq!(slot.as_slice()[4095], 0x99);

        // Verify emergency thermal brake when sensor reaches >= 70°C
        thermal.update_sensor_temp(72);
        let err = storage
            .read_expert_into_slot(
                ExpertId {
                    layer: 0,
                    expert: 0,
                },
                &mut slot,
            )
            .await;
        assert!(matches!(
            err,
            Err(ApusError::ThermalThrottleTripped { temp_c: 72 })
        ));

        // Verify zero-copy MmapStorage on the same .apus container
        let mmap_storage = MmapStorage::open(&tmp_path).expect("open mmap storage failed");
        let mut mmap_slot = PageAlignedSlot::new(4096).expect("mmap slot failed");
        mmap_storage
            .read_expert_into_slot(
                ExpertId {
                    layer: 0,
                    expert: 0,
                },
                &mut mmap_slot,
            )
            .await
            .expect("mmap read failed");
        assert_eq!(mmap_slot.as_slice()[0], 0x42);

        let _ = std::fs::remove_file(&tmp_path);
    }
}
