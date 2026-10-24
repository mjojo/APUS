use apus_core::ApusError;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Entry for a single 4KB-aligned expert chunk inside a `*.apus.idx` distribution manifest
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HubChunkDescriptor {
    pub chunk_index: u32,
    pub layer_id: u16,
    pub expert_id: u16,
    pub file_offset: u64,
    pub size_bytes: u32,
    pub blake3_hex: String,
}

/// Cryptographic distribution manifest (`*.apus.idx`) for resumable multi-hundred-gigabyte model downloads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApusIdxManifest {
    pub model_id: String,
    pub author: String,
    pub total_size_bytes: u64,
    pub root_blake3_hex: String,
    pub chunks: Vec<HubChunkDescriptor>,
}

impl ApusIdxManifest {
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ApusError> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| ApusError::AllocationFailed(format!("Failed to serialize .apus.idx: {e}")))?;
        fs::write(path, json)
            .map_err(|e| ApusError::AllocationFailed(format!("Failed to write .apus.idx: {e}")))
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, ApusError> {
        let text = fs::read_to_string(path)
            .map_err(|e| ApusError::AllocationFailed(format!("Failed to read .apus.idx: {e}")))?;
        serde_json::from_str(&text)
            .map_err(|e| ApusError::AllocationFailed(format!("Invalid .apus.idx JSON: {e}")))
    }
}

/// Resumable chunk downloader and verifier (`apus-hub` Sprint 8)
/// Supports HTTP Range byte-offset resumption and per-chunk BLAKE3 integrity verification
pub struct ChunkedHubDownloader;

impl ChunkedHubDownloader {
    /// Generates a `*.apus.idx` manifest from an existing `*.apus` model container
    pub fn generate_manifest_from_container<P: AsRef<Path>>(
        apus_path: P,
    ) -> Result<ApusIdxManifest, ApusError> {
        let mut file = fs::File::open(apus_path.as_ref()).map_err(|e| ApusError::StorageIo {
            layer: 0,
            expert: 0,
            source: e,
        })?;
        let total_size_bytes = file.metadata().map(|m| m.len()).unwrap_or(0);
        let header = apus_quant::ApusFileHeader::read_from(&mut file)
            .map_err(|e| ApusError::AllocationFailed(format!("Header error: {e}")))?;
        let lba = header
            .read_lba_table(&mut file)
            .map_err(|e| ApusError::AllocationFailed(format!("LBA error: {e}")))?;

        let mut chunks = Vec::with_capacity(lba.len());
        let mut tree = blake3::Hasher::new();

        for (idx, entry) in lba.iter().enumerate() {
            file.seek(SeekFrom::Start(entry.file_offset))
                .map_err(|e| ApusError::StorageIo {
                    layer: entry.layer_id,
                    expert: entry.expert_id,
                    source: e,
                })?;
            let mut buf = vec![0u8; entry.compressed_size as usize];
            file.read_exact(&mut buf).map_err(|e| ApusError::StorageIo {
                layer: entry.layer_id,
                expert: entry.expert_id,
                source: e,
            })?;
            let h = blake3::hash(&buf);
            tree.update(h.as_bytes());
            chunks.push(HubChunkDescriptor {
                chunk_index: idx as u32,
                layer_id: entry.layer_id,
                expert_id: entry.expert_id,
                file_offset: entry.file_offset,
                size_bytes: entry.compressed_size,
                blake3_hex: h.to_hex().to_string(),
            });
        }

        let root_blake3_hex = tree.finalize().to_hex().to_string();
        Ok(ApusIdxManifest {
            model_id: header.model_id_str(),
            author: header.author_str(),
            total_size_bytes,
            root_blake3_hex,
            chunks,
        })
    }

    /// Inspects a partially downloaded `.apus` file against its `.apus.idx` manifest
    /// and returns the list of missing or corrupted chunks that require HTTP Range resumption.
    pub fn find_missing_or_corrupted_chunks<P: AsRef<Path>>(
        target_path: P,
        manifest: &ApusIdxManifest,
    ) -> Vec<HubChunkDescriptor> {
        let mut file = match fs::File::open(target_path.as_ref()) {
            Ok(f) => f,
            Err(_) => return manifest.chunks.clone(),
        };
        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
        let mut needed = Vec::new();

        for chunk in &manifest.chunks {
            let end_offset = chunk.file_offset + (chunk.size_bytes as u64);
            if end_offset > file_len {
                needed.push(chunk.clone());
                continue;
            }
            if file.seek(SeekFrom::Start(chunk.file_offset)).is_err() {
                needed.push(chunk.clone());
                continue;
            }
            let mut buf = vec![0u8; chunk.size_bytes as usize];
            if file.read_exact(&mut buf).is_err() {
                needed.push(chunk.clone());
                continue;
            }
            let actual_hex = blake3::hash(&buf).to_hex().to_string();
            if actual_hex != chunk.blake3_hex {
                needed.push(chunk.clone());
            }
        }
        needed
    }

    /// Writes and cryptographically verifies a downloaded chunk at its exact 4KB-aligned LBA file offset
    pub fn write_verified_chunk<P: AsRef<Path>>(
        target_path: P,
        descriptor: &HubChunkDescriptor,
        payload: &[u8],
    ) -> Result<(), ApusError> {
        let actual_hex = blake3::hash(payload).to_hex().to_string();
        if actual_hex != descriptor.blake3_hex {
            return Err(ApusError::CorruptedExpertWeights {
                layer: descriptor.layer_id,
                expert: descriptor.expert_id,
            });
        }

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(target_path.as_ref())
            .map_err(|e| ApusError::StorageIo {
                layer: descriptor.layer_id,
                expert: descriptor.expert_id,
                source: e,
            })?;

        file.seek(SeekFrom::Start(descriptor.file_offset))
            .map_err(|e| ApusError::StorageIo {
                layer: descriptor.layer_id,
                expert: descriptor.expert_id,
                source: e,
            })?;
        file.write_all(payload).map_err(|e| ApusError::StorageIo {
            layer: descriptor.layer_id,
            expert: descriptor.expert_id,
            source: e,
        })?;
        file.flush().map_err(|e| ApusError::StorageIo {
            layer: descriptor.layer_id,
            expert: descriptor.expert_id,
            source: e,
        })?;
        Ok(())
    }
}
