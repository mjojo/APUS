use serde::{Deserialize, Serialize};

/// Dimensionality of the compressed MLA latent vector c_t^{KV} in GLM-5.2 (576 floats vs 32,768)
pub const MLA_LATENT_DIM: usize = 576;
/// PagedAttention page size (in tokens)
pub const KV_PAGE_TOKENS: usize = 64;

/// Compressed latent vector for a single token (quantized to FP8/INT8 for VRAM/RAM and .roost storage)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentKvToken {
    pub token_id: u32,
    pub scale: f32,
    pub latent_fp8: Vec<i8>,
}

impl LatentKvToken {
    /// Compresses 576 FP32/FP16 latent state values into a compact FP8/INT8 representation
    pub fn compress(token_id: u32, latent_state: &[f32]) -> Self {
        let max_abs = latent_state
            .iter()
            .fold(0.0f32, |acc, &v| acc.max(v.abs()))
            .max(1e-6);
        let scale = max_abs / 127.0;
        let latent_fp8 = latent_state
            .iter()
            .take(MLA_LATENT_DIM)
            .map(|&v| (v / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();
        Self {
            token_id,
            scale,
            latent_fp8,
        }
    }

    /// Decompresses the latent vector back to FP32 for attention score calculation
    pub fn decompress(&self) -> Vec<f32> {
        self.latent_fp8
            .iter()
            .map(|&q| (q as f32) * self.scale)
            .collect()
    }
}

/// Paged Multi-Head Latent Attention (MLA) KV-cache manager with Attention Sinks
/// and automatic cold page offloading from VRAM to system RAM.
pub struct MlaKvCache {
    sink_tokens_keep: usize,
    vram_hot_window_tokens: usize,
    hot_vram_tokens: Vec<LatentKvToken>,
    cold_ram_pages: Vec<Vec<LatentKvToken>>,
}

impl MlaKvCache {
    pub fn new(sink_tokens_keep: usize, vram_hot_window_tokens: usize) -> Self {
        Self {
            sink_tokens_keep: sink_tokens_keep.max(4),
            vram_hot_window_tokens: vram_hot_window_tokens.max(64),
            hot_vram_tokens: Vec::new(),
            cold_ram_pages: Vec::new(),
        }
    }

    /// Appends a new token into the latent KV-cache. When the hot VRAM window overflows,
    /// older pages (past the preserved Attention Sinks) are offloaded into cold system RAM without context loss.
    pub fn append_token(&mut self, token_id: u32, latent_state: &[f32]) {
        let compressed = LatentKvToken::compress(token_id, latent_state);
        self.hot_vram_tokens.push(compressed);

        let max_hot = self.sink_tokens_keep + self.vram_hot_window_tokens;
        if self.hot_vram_tokens.len() >= max_hot + KV_PAGE_TOKENS {
            let drain_end = self.sink_tokens_keep + KV_PAGE_TOKENS;
            let page: Vec<LatentKvToken> = self
                .hot_vram_tokens
                .drain(self.sink_tokens_keep..drain_end)
                .collect();
            self.cold_ram_pages.push(page);
        }
    }

    pub fn total_context_tokens(&self) -> usize {
        self.hot_vram_tokens.len() + self.cold_ram_pages.iter().map(|p| p.len()).sum::<usize>()
    }

    pub fn vram_tokens_count(&self) -> usize {
        self.hot_vram_tokens.len()
    }

    pub fn cold_pages_count(&self) -> usize {
        self.cold_ram_pages.len()
    }

    /// Estimated memory footprint in bytes (~584 bytes per token in FP8/INT8)
    pub fn memory_footprint_bytes(&self) -> usize {
        self.total_context_tokens() * (MLA_LATENT_DIM + 8)
    }

    /// Serializes the entire latent KV-cache into a byte buffer for persistence in `*.roost`
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut all_tokens = Vec::with_capacity(self.total_context_tokens());
        let sinks_count = self.sink_tokens_keep.min(self.hot_vram_tokens.len());
        all_tokens.extend_from_slice(&self.hot_vram_tokens[..sinks_count]);
        for page in &self.cold_ram_pages {
            all_tokens.extend_from_slice(page);
        }
        if self.hot_vram_tokens.len() > sinks_count {
            all_tokens.extend_from_slice(&self.hot_vram_tokens[sinks_count..]);
        }

        let mut buf = Vec::with_capacity(all_tokens.len() * (MLA_LATENT_DIM + 8));
        for tok in &all_tokens {
            buf.extend_from_slice(&tok.token_id.to_le_bytes());
            buf.extend_from_slice(&tok.scale.to_le_bytes());
            let len = (tok.latent_fp8.len() as u16).to_le_bytes();
            buf.extend_from_slice(&len);
            for &b in &tok.latent_fp8 {
                buf.push(b as u8);
            }
        }
        buf
    }

    /// Restores the latent KV-cache from a `*.roost` binary payload
    pub fn from_bytes(
        bytes: &[u8],
        sink_tokens_keep: usize,
        vram_hot_window_tokens: usize,
    ) -> Self {
        let mut cache = Self::new(sink_tokens_keep, vram_hot_window_tokens);
        let mut cursor = 0;
        while cursor + 10 <= bytes.len() {
            let token_id = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            let scale = f32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap());
            let len =
                u16::from_le_bytes(bytes[cursor + 8..cursor + 10].try_into().unwrap()) as usize;
            cursor += 10;
            if cursor + len > bytes.len() {
                break;
            }
            let latent_fp8: Vec<i8> = bytes[cursor..cursor + len]
                .iter()
                .map(|&b| b as i8)
                .collect();
            cursor += len;
            cache.hot_vram_tokens.push(LatentKvToken {
                token_id,
                scale,
                latent_fp8,
            });
        }
        cache
    }
}
