use apus_core::ExpertId;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use thiserror::Error;

pub const APUS_MAGIC: &[u8; 4] = b"APUS";
pub const ROOST_MAGIC: &[u8; 4] = b"ROST";
pub const AUTHOR_SIGNATURE: &[u8] = b"apus-moe // authored by mjojo";
pub const ROOST_AUTHOR_SIGNATURE: &[u8] = b"roost // created by mjojo";

#[derive(Debug, Error)]
pub enum FormatError {
    #[error("Invalid file magic bytes (expected APUS/ROST)")]
    InvalidMagic,
    #[error("Author signature missing or corrupted (mjojo signature required)")]
    InvalidAuthorSignature,
    #[error("I/O error while reading header: {0}")]
    Io(#[from] io::Error),
}

/// Fixed 512-byte header for `*.apus` model container files
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct ApusFileHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub author_sig: [u8; 32],
    pub model_id: [u8; 32],
    pub total_layers: u32,
    pub total_experts: u32,
    pub dense_size: u64,
    pub lba_table_offset: u64,
    pub blake3_root_hash: u64,
    pub reserved: [u8; 376],
}

/// 16-byte record in the LBA Expert Offset Table
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct LbaExpertEntry {
    pub layer_id: u16,
    pub expert_id: u16,
    pub file_offset: u64,
    pub compressed_size: u32,
}

impl ApusFileHeader {
    pub fn read_from(file: &mut File) -> Result<Self, FormatError> {
        file.seek(SeekFrom::Start(0))?;
        let mut header_buf = [0u8; 512];
        file.read_exact(&mut header_buf)?;

        // Safe byte-by-byte verification to prevent undefined behavior on unaligned reads
        if &header_buf[0..4] != APUS_MAGIC {
            return Err(FormatError::InvalidMagic);
        }

        // Verify the mandatory mjojo author signature within header bounds
        if !header_buf[8..40].starts_with(AUTHOR_SIGNATURE) {
            return Err(FormatError::InvalidAuthorSignature);
        }

        // Materialize packed header struct from buffer
        let header = unsafe {
            std::ptr::read_unaligned(header_buf.as_ptr() as *const ApusFileHeader)
        };
        Ok(header)
    }

    pub fn model_id_str(&self) -> String {
        let end = self
            .model_id
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(self.model_id.len());
        String::from_utf8_lossy(&self.model_id[..end]).into_owned()
    }

    pub fn author_str(&self) -> String {
        let end = self
            .author_sig
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(self.author_sig.len());
        String::from_utf8_lossy(&self.author_sig[..end]).into_owned()
    }

    /// Reads the LBA expert offset table from an open `*.apus` file
    pub fn read_lba_table(&self, file: &mut File) -> Result<Vec<LbaExpertEntry>, FormatError> {
        let offset = self.lba_table_offset;
        let count = self.total_experts as usize;
        file.seek(SeekFrom::Start(offset))?;
        let mut entries = Vec::with_capacity(count);
        let mut buf = [0u8; std::mem::size_of::<LbaExpertEntry>()];
        for _ in 0..count {
            file.read_exact(&mut buf)?;
            let entry = unsafe {
                std::ptr::read_unaligned(buf.as_ptr() as *const LbaExpertEntry)
            };
            entries.push(entry);
        }
        Ok(entries)
    }
}

/// Fixed 128-byte header for `*.roost` session snapshot files
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct RoostSessionHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub author_tag: [u8; 24],
    pub timestamp: u64,
    pub context_tokens: u64,
    pub warm_experts_count: u32,
    pub kv_payload_size: u32,
    pub reserved: [u8; 72],
}

/// Complete `*.roost` session snapshot (instant freeze/resume of MLA KV-cache and warm expert profile)
#[derive(Debug, Clone)]
pub struct RoostSnapshot {
    pub timestamp: u64,
    pub context_tokens: u64,
    pub warm_experts: Vec<ExpertId>,
    pub compressed_mla_kv: Vec<u8>,
}

impl RoostSnapshot {
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), FormatError> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        let mut author_tag = [0u8; 24];
        let copy_len = ROOST_AUTHOR_SIGNATURE.len().min(24);
        author_tag[..copy_len].copy_from_slice(&ROOST_AUTHOR_SIGNATURE[..copy_len]);

        let header = RoostSessionHeader {
            magic: *ROOST_MAGIC,
            version: 1,
            author_tag,
            timestamp: self.timestamp,
            context_tokens: self.context_tokens,
            warm_experts_count: self.warm_experts.len() as u32,
            kv_payload_size: self.compressed_mla_kv.len() as u32,
            reserved: [0u8; 72],
        };

        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const RoostSessionHeader as *const u8,
                std::mem::size_of::<RoostSessionHeader>(),
            )
        };
        file.write_all(header_bytes)?;

        for exp in &self.warm_experts {
            file.write_all(&exp.layer.to_le_bytes())?;
            file.write_all(&exp.expert.to_le_bytes())?;
        }
        file.write_all(&self.compressed_mla_kv)?;
        file.flush()?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, FormatError> {
        let mut file = File::open(path)?;
        let mut header_buf = [0u8; 128];
        file.read_exact(&mut header_buf)?;

        if &header_buf[0..4] != ROOST_MAGIC {
            return Err(FormatError::InvalidMagic);
        }
        if !header_buf[8..32].starts_with(&ROOST_AUTHOR_SIGNATURE[..24.min(ROOST_AUTHOR_SIGNATURE.len())]) {
            return Err(FormatError::InvalidAuthorSignature);
        }

        let header = unsafe {
            std::ptr::read_unaligned(header_buf.as_ptr() as *const RoostSessionHeader)
        };

        let warm_count = header.warm_experts_count as usize;
        let mut warm_experts = Vec::with_capacity(warm_count);
        for _ in 0..warm_count {
            let mut b = [0u8; 4];
            file.read_exact(&mut b)?;
            let layer = u16::from_le_bytes([b[0], b[1]]);
            let expert = u16::from_le_bytes([b[2], b[3]]);
            warm_experts.push(ExpertId { layer, expert });
        }

        let kv_len = header.kv_payload_size as usize;
        let mut compressed_mla_kv = vec![0u8; kv_len];
        file.read_exact(&mut compressed_mla_kv)?;

        Ok(Self {
            timestamp: header.timestamp,
            context_tokens: header.context_tokens,
            warm_experts,
            compressed_mla_kv,
        })
    }
}
