use crate::format::{ApusFileHeader, LbaExpertEntry, APUS_MAGIC, AUTHOR_SIGNATURE};
use std::fs::{File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::Path;
use thiserror::Error;

pub const PAGE_ALIGNMENT: u64 = 4096;

#[derive(Debug, Error)]
pub enum ConverterError {
    #[error("I/O error during conversion: {0}")]
    Io(#[from] io::Error),
    #[error("Invalid model topology: layer or expert count is zero")]
    InvalidTopology,
}

/// Input payload of a single routed expert for conversion
#[derive(Debug, Clone)]
pub struct SourceExpert {
    pub layer_id: u16,
    pub expert_id: u16,
    pub raw_quantized_weights: Vec<u8>, // INT4/INT2 quantized weights
}

pub struct ApusConverter {
    model_id: String,
    total_layers: u32,
    experts_per_layer: u32,
    dense_weights: Vec<u8>,
    experts: Vec<SourceExpert>,
}

impl ApusConverter {
    pub fn new(
        model_id: &str,
        total_layers: u32,
        experts_per_layer: u32,
        dense_weights: Vec<u8>,
    ) -> Self {
        Self {
            model_id: model_id.to_string(),
            total_layers,
            experts_per_layer,
            dense_weights,
            experts: Vec::new(),
        }
    }

    /// Enqueues a quantized expert block for conversion
    pub fn push_expert(&mut self, expert: SourceExpert) {
        self.experts.push(expert);
    }

    /// Compiles and writes the 4KB page-aligned `*.apus` container to disk (returns the hex BLAKE3 Merkle root)
    pub fn build<P: AsRef<Path>>(self, output_path: P) -> Result<String, ConverterError> {
        if self.total_layers == 0 || self.experts_per_layer == 0 || self.experts.is_empty() {
            return Err(ConverterError::InvalidTopology);
        }

        let total_experts_count = self.experts.len();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(output_path)?;

        // 1. Reserve the first 512 bytes for ApusFileHeader, written at the end.
        file.seek(SeekFrom::Start(512))?;

        // 2. Write the dense model base (Attention, Shared Experts, Embeddings)
        let dense_offset = file.stream_position()?;
        file.write_all(&self.dense_weights)?;

        // Align file cursor after dense weights to a 4KB boundary
        let current_pos = file.stream_position()?;
        let aligned_dense_pos = align_to_page(current_pos);
        pad_file_to(&mut file, aligned_dense_pos)?;
        let dense_size = aligned_dense_pos - dense_offset;

        // 3. Reserve space for the LBA Offset Table immediately following dense weights
        let lba_table_offset = file.stream_position()?;
        let lba_table_size = (total_experts_count * std::mem::size_of::<LbaExpertEntry>()) as u64;

        // Zero-fill reserved bytes for the LBA table
        let dummy_lba_bytes = vec![0u8; lba_table_size as usize];
        file.write_all(&dummy_lba_bytes)?;

        let aligned_after_lba = align_to_page(file.stream_position()?);
        pad_file_to(&mut file, aligned_after_lba)?;

        // 4. Sequentially write each expert with 4KB alignment and BLAKE3 hashing
        let mut lba_entries = Vec::with_capacity(total_experts_count);
        let mut blake3_hasher_tree = blake3::Hasher::new();

        for exp in &self.experts {
            let expert_file_offset = file.stream_position()?;

            // Write expert weights and pad to the next 4096-byte page boundary
            file.write_all(&exp.raw_quantized_weights)?;
            let next_pos = align_to_page(file.stream_position()?);
            pad_file_to(&mut file, next_pos)?;

            let written_size = (next_pos - expert_file_offset) as u32;

            // Compute BLAKE3 digest of the 4KB-aligned expert block
            let mut aligned_chunk = exp.raw_quantized_weights.clone();
            aligned_chunk.resize(written_size as usize, 0u8);
            let expert_hash = blake3::hash(&aligned_chunk);
            blake3_hasher_tree.update(expert_hash.as_bytes());

            lba_entries.push(LbaExpertEntry {
                layer_id: exp.layer_id,
                expert_id: exp.expert_id,
                file_offset: expert_file_offset,
                compressed_size: written_size,
            });
        }

        // Finalize the BLAKE3 Merkle root hash across all experts
        let blake3_root_hash_bytes = blake3_hasher_tree.finalize();
        let blake3_root_u64 =
            u64::from_ne_bytes(blake3_root_hash_bytes.as_bytes()[0..8].try_into().unwrap());

        // 5. Seek back and write the populated LBA table into the reserved region
        file.seek(SeekFrom::Start(lba_table_offset))?;
        for entry in &lba_entries {
            let entry_bytes = unsafe {
                std::slice::from_raw_parts(
                    entry as *const LbaExpertEntry as *const u8,
                    std::mem::size_of::<LbaExpertEntry>(),
                )
            };
            file.write_all(entry_bytes)?;
        }

        // 6. Populate and write the final 512-byte header into sector 0 (0x0)
        let mut author_sig_array = [0u8; 32];
        author_sig_array[..AUTHOR_SIGNATURE.len()].copy_from_slice(AUTHOR_SIGNATURE);

        let mut model_id_array = [0u8; 32];
        let model_bytes = self.model_id.as_bytes();
        let copy_len = model_bytes.len().min(31);
        model_id_array[..copy_len].copy_from_slice(&model_bytes[..copy_len]);

        let header = ApusFileHeader {
            magic: *APUS_MAGIC,
            version: 0x00000001,
            author_sig: author_sig_array,
            model_id: model_id_array,
            total_layers: self.total_layers,
            total_experts: self.experts.len() as u32,
            dense_size,
            lba_table_offset,
            blake3_root_hash: blake3_root_u64,
            reserved: [0u8; 376],
        };

        file.seek(SeekFrom::Start(0))?;
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const ApusFileHeader as *const u8,
                std::mem::size_of::<ApusFileHeader>(),
            )
        };
        file.write_all(header_bytes)?;
        file.flush()?;

        Ok(blake3_root_hash_bytes.to_hex().to_string())
    }
}

/// Rounds a byte offset up to the nearest multiple of 4096 (memory page / NVMe sector)
#[inline]
pub const fn align_to_page(position: u64) -> u64 {
    (position + PAGE_ALIGNMENT - 1) & !(PAGE_ALIGNMENT - 1)
}

/// Zero-pads a file up to the target aligned byte offset
pub fn pad_file_to(file: &mut File, target_pos: u64) -> io::Result<()> {
    let current_pos = file.stream_position()?;
    if target_pos > current_pos {
        let diff = (target_pos - current_pos) as usize;
        let zeros = vec![0u8; diff];
        file.write_all(&zeros)?;
    }
    Ok(())
}
