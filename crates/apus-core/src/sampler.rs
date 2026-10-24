use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sampling parameters for token generation (Temperature, Top-K, Top-P, Min-P, Repetition & DRY Penalty)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    /// DRY (Don't Repeat Yourself) n-gram sequence penalty multiplier (Sprint 25)
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: usize,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.90,
            min_p: 0.05,
            repetition_penalty: 1.12,
            dry_multiplier: 0.8,
            dry_base: 1.75,
            dry_allowed_length: 2,
        }
    }
}

/// Production token sampler supporting Temperature, Repetition Penalty, Top-K, Top-P (Nucleus), and Min-P filtering
pub struct GenerationSampler {
    config: SamplerConfig,
    rng_state: u64,
}

impl GenerationSampler {
    pub fn new(config: SamplerConfig, seed: u64) -> Self {
        Self {
            config,
            rng_state: seed.max(1),
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        // SplitMix64 fast deterministic PRNG
        self.rng_state = self.rng_state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    /// Samples the next token ID from raw vocabulary logits and recent context history
    pub fn sample(&mut self, logits: &[f32], context_tokens: &[u32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }

        let mut candidates: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(id, &l)| (id as u32, l))
            .collect();

        // 1. Apply Repetition Penalty over recent context tokens
        if (self.config.repetition_penalty - 1.0).abs() > 1e-4 {
            for &prev_tok in context_tokens.iter().rev().take(128) {
                if let Some((_, score)) = candidates.get_mut(prev_tok as usize) {
                    if *score > 0.0 {
                        *score /= self.config.repetition_penalty;
                    } else {
                        *score *= self.config.repetition_penalty;
                    }
                }
            }
        }

        // 1b. Apply DRY (Don't Repeat Yourself) n-gram suffix sequence penalty (Sprint 25):
        // Scans context history for suffixes matching the tail of `context_tokens` and exponentially
        // penalizes the token that immediately followed that suffix in earlier positions.
        if self.config.dry_multiplier > 0.0 && context_tokens.len() >= 2 {
            let tail_tok = *context_tokens.last().unwrap();
            let n = context_tokens.len();
            for end_idx in 1..(n - 1) {
                if context_tokens[end_idx - 1] == tail_tok {
                    // Measure contiguous matching suffix length backwards
                    let mut match_len = 1usize;
                    while match_len < end_idx
                        && match_len < n - 1
                        && context_tokens[end_idx - 1 - match_len]
                            == context_tokens[n - 1 - match_len]
                    {
                        match_len += 1;
                    }
                    if match_len >= self.config.dry_allowed_length {
                        let next_repeated_tok = context_tokens[end_idx] as usize;
                        if let Some((_, score)) = candidates.get_mut(next_repeated_tok) {
                            let exp_pow = (match_len - self.config.dry_allowed_length) as i32;
                            let penalty =
                                self.config.dry_multiplier * self.config.dry_base.powi(exp_pow);
                            *score -= penalty;
                        }
                    }
                }
            }
        }

        // Greedy argmax when temperature <= 0.0
        if self.config.temperature <= 1e-5 {
            return candidates
                .into_iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(id, _)| id)
                .unwrap_or(0);
        }

        // 2. Sort descending by logit and apply Top-K truncation
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if self.config.top_k > 0 && candidates.len() > self.config.top_k {
            candidates.truncate(self.config.top_k);
        }

        // 3. Temperature scaling & Softmax conversion
        let inv_temp = 1.0 / self.config.temperature;
        let max_logit = candidates[0].1 * inv_temp;
        let mut probs: Vec<(u32, f32)> = candidates
            .into_iter()
            .map(|(id, l)| (id, ((l * inv_temp) - max_logit).exp()))
            .collect();

        let sum_exp: f32 = probs.iter().map(|(_, p)| *p).sum::<f32>().max(1e-9);
        for (_, p) in &mut probs {
            *p /= sum_exp;
        }

        // 4. Min-P filtering relative to the top token probability
        let max_prob = probs[0].1;
        let min_p_threshold = max_prob * self.config.min_p;
        if self.config.min_p > 0.0 && probs.len() > 1 {
            probs.retain(|&(_, p)| p >= min_p_threshold);
            if probs.is_empty() {
                return 0;
            }
        }

        // 5. Top-P (Nucleus) cumulative probability cutoff
        if self.config.top_p < 1.0 && probs.len() > 1 {
            let mut cum = 0.0f32;
            let mut cutoff = probs.len();
            for (i, &(_, p)) in probs.iter().enumerate() {
                cum += p;
                if cum >= self.config.top_p {
                    cutoff = (i + 1).max(1);
                    break;
                }
            }
            probs.truncate(cutoff);
        }

        // 6. Renormalize and sample from the filtered categorical distribution
        let final_sum: f32 = probs.iter().map(|(_, p)| *p).sum::<f32>().max(1e-9);
        let r = self.next_f32() * final_sum;
        let mut acc = 0.0f32;
        for &(id, p) in &probs {
            acc += p;
            if r <= acc {
                return id;
            }
        }
        probs.last().map(|(id, _)| *id).unwrap_or(0)
    }
}

/// Lightweight Byte-Pair Encoding (BPE) Tokenizer for APUS runtime pipelines
pub struct BpeTokenizer {
    vocab_to_id: HashMap<String, u32>,
    id_to_vocab: Vec<String>,
}

impl BpeTokenizer {
    /// Initializes a built-in technical BPE vocabulary (covering ASCII bytes + common Rust/systems tokens)
    pub fn new_default() -> Self {
        let mut id_to_vocab = Vec::with_capacity(512);
        let mut vocab_to_id = HashMap::with_capacity(512);

        // Base 256 byte-level tokens
        for b in 0u8..=255 {
            let s = if b.is_ascii_graphic() || b == b' ' {
                (b as char).to_string()
            } else {
                format!("<0x{:02X}>", b)
            };
            vocab_to_id.insert(s.clone(), id_to_vocab.len() as u32);
            id_to_vocab.push(s);
        }

        // High-frequency systems programming subword tokens
        let subwords = [
            "pub ",
            "fn ",
            "struct ",
            "impl ",
            "async ",
            "await",
            "unsafe ",
            "mut ",
            "let ",
            "Result",
            "Apus",
            "Expert",
            "NVMe",
            "DirectStorage",
            "io_uring",
            "O_DIRECT",
            "BLAKE3",
            "PageAlignedSlot",
            "SlabArena",
            "ArcCache",
            "MlaKvCache",
            "mjojo",
            " -> ",
            "::",
            "    ",
            "\n",
            "4096",
            "INT4",
            "VRAM",
            "RAM",
            "QoS",
            "DMA",
        ];
        for sw in subwords {
            if !vocab_to_id.contains_key(sw) {
                let id = id_to_vocab.len() as u32;
                vocab_to_id.insert(sw.to_string(), id);
                id_to_vocab.push(sw.to_string());
            }
        }

        Self {
            vocab_to_id,
            id_to_vocab,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_vocab.len()
    }

    pub fn token_id(&self, piece: &str) -> Option<u32> {
        self.vocab_to_id.get(piece).copied()
    }

    /// Greedy longest-match subword encoding with byte fallback
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let mut matched = None;
            // Check multi-char tokens (starting from token index 256)
            for (id, piece) in self.id_to_vocab.iter().enumerate().skip(256) {
                if rest.starts_with(piece) && matched.is_none_or(|(_, len)| piece.len() > len) {
                    matched = Some((id as u32, piece.len()));
                }
            }
            if let Some((id, len)) = matched {
                tokens.push(id);
                rest = &rest[len..];
            } else {
                let b = rest.as_bytes()[0];
                tokens.push(b as u32);
                let step = rest.chars().next().map(|c| c.len_utf8()).unwrap_or(1);
                rest = &rest[step..];
            }
        }
        tokens
    }

    /// Decodes token IDs back into a UTF-8 string
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            let idx = (id as usize) % self.id_to_vocab.len();
            out.push_str(&self.id_to_vocab[idx]);
        }
        out
    }
}
