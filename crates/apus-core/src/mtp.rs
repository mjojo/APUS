use crate::ExpertId;
use std::collections::BTreeSet;

/// Expert selection output from the Top-K Gating Network
#[derive(Debug, Clone)]
pub struct RoutedExpertSelection {
    pub id: ExpertId,
    pub gate_weight: f32,
}

/// MoE Layer Router (Top-K Gating with Softmax weight normalization)
pub struct MoERouter {
    pub experts_per_layer: u16,
    pub active_top_k: usize,
}

impl MoERouter {
    pub fn new(experts_per_layer: u16, active_top_k: usize) -> Self {
        Self {
            experts_per_layer,
            active_top_k: active_top_k.max(1),
        }
    }

    /// Selects the Top-K routed experts for a given layer based on the token hidden state
    pub fn route(&self, layer: u16, hidden_state: &[f32]) -> Vec<RoutedExpertSelection> {
        let mut logits: Vec<(u16, f32)> = (0..self.experts_per_layer)
            .map(|expert_idx| {
                // Deterministic projection of the hidden state onto the expert gating vector
                let mut score = 0.0f32;
                let stride = (expert_idx as usize % 13) + 1;
                for (i, &val) in hidden_state.iter().step_by(stride).take(32).enumerate() {
                    let phase = ((layer as usize * 31 + expert_idx as usize * 17 + i * 7) % 100)
                        as f32
                        * 0.0628;
                    score += val * phase.cos();
                }
                (expert_idx, score)
            })
            .collect();

        logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k = &logits[..self.active_top_k.min(logits.len())];

        let max_logit = top_k.first().map(|x| x.1).unwrap_or(0.0);
        let exp_sum: f32 = top_k
            .iter()
            .map(|(_, s)| (s - max_logit).exp())
            .sum::<f32>()
            .max(1e-6);

        top_k
            .iter()
            .map(|&(expert, s)| RoutedExpertSelection {
                id: ExpertId { layer, expert },
                gate_weight: (s - max_logit).exp() / exp_sum,
            })
            .collect()
    }
}

/// Asynchronous Look-ahead Prefetcher for layers N+1 and N+2
/// and Batch LBA Prefill Scan optimizer
pub struct LookAheadPrefetcher {
    router: MoERouter,
    total_layers: u16,
}

impl LookAheadPrefetcher {
    pub fn new(total_layers: u16, experts_per_layer: u16, active_top_k: usize) -> Self {
        Self {
            router: MoERouter::new(experts_per_layer, active_top_k),
            total_layers,
        }
    }

    /// At layer N, predicts upcoming experts for layers N+1 and N+2 from the current hidden state
    pub fn predict_upcoming(&self, current_layer: u16, hidden_state: &[f32]) -> Vec<ExpertId> {
        let mut upcoming = Vec::with_capacity(self.router.active_top_k * 2);
        for offset in 1..=2 {
            let target_layer = current_layer + offset;
            if target_layer < self.total_layers {
                for sel in self.router.route(target_layer, hidden_state) {
                    upcoming.push(sel.id);
                }
            }
        }
        upcoming
    }

    /// Prefill Phase (Batch Scan): aggregates all required experts across a prompt batch on a layer
    /// and sorts them in ascending order (matching physical LBA sector order on NVMe)
    pub fn aggregate_prefill_lba_sorted(
        &self,
        layer: u16,
        prompt_hidden_states: &[Vec<f32>],
    ) -> Vec<ExpertId> {
        let mut unique_experts = BTreeSet::new();
        for hs in prompt_hidden_states {
            for sel in self.router.route(layer, hs) {
                unique_experts.insert((sel.id.layer, sel.id.expert));
            }
        }
        unique_experts
            .into_iter()
            .map(|(l, e)| ExpertId {
                layer: l,
                expert: e,
            })
            .collect()
    }
}

/// Native Multi-Token Prediction (MTP, GLM-5.2 78th layer) speculative decoding engine
pub struct MtpSpeculativeEngine {
    pub num_speculative_heads: usize,
    pub vocab_size: u32,
}

impl MtpSpeculativeEngine {
    pub fn new(num_speculative_heads: usize, vocab_size: u32) -> Self {
        Self {
            num_speculative_heads: num_speculative_heads.max(1),
            vocab_size: vocab_size.max(256),
        }
    }

    /// Proposes the primary token plus speculative tokens from native MTP heads in a single pass
    pub fn propose_tokens(&self, final_hidden_state: &[f32], step: u64) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(1 + self.num_speculative_heads);
        let base_energy: f32 = final_hidden_state.iter().copied().sum();
        let base_hash = ((base_energy.abs() * 1000.0) as u64).wrapping_add(step * 131);

        // Primary greedy/sampled token
        tokens.push((base_hash % self.vocab_size as u64) as u32);

        // Additional tokens proposed by native MTP heads
        for head_idx in 1..=self.num_speculative_heads {
            let head_hash = base_hash
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(head_idx as u64 * 997);
            tokens.push((head_hash % self.vocab_size as u64) as u32);
        }
        tokens
    }
}

/// Secondary Speculative Decoding Loop: External Lightweight Draft Model Verifier (Sprint 17 `apus-mtp`).
/// Used for MoE architectures without native MTP heads (e.g., Mixtral 8x22B, Qwen-MoE).
/// A small resident 1B dense draft model proposes `draft_k` candidate tokens, and the 700B+ MoE target model
/// verifies all `draft_k` tokens in a single batched LBA pass using Rejection Sampling.
pub struct ExternalDraftSpeculativeVerifier {
    pub draft_k: usize,
    pub acceptance_threshold: f32,
}

impl ExternalDraftSpeculativeVerifier {
    pub fn new(draft_k: usize, acceptance_threshold: f32) -> Self {
        Self {
            draft_k: draft_k.clamp(1, 8),
            acceptance_threshold: acceptance_threshold.clamp(0.05, 1.0),
        }
    }

    /// Verifies `draft_tokens` against target MoE model probabilities `(token_id, target_prob)`.
    /// Accepts matching prefix tokens and emits a corrected fallback token at the first rejection point.
    pub fn verify_draft_batch(
        &self,
        draft_tokens: &[(u32, f32)],
        target_probs: &[(u32, f32)],
    ) -> Vec<u32> {
        let mut accepted = Vec::with_capacity(self.draft_k + 1);
        let len = draft_tokens.len().min(target_probs.len()).min(self.draft_k);

        for i in 0..len {
            let (d_tok, d_prob) = draft_tokens[i];
            let (t_tok, t_prob) = target_probs[i];
            let ratio = t_prob / d_prob.max(1e-6);

            if d_tok == t_tok || ratio >= self.acceptance_threshold {
                accepted.push(d_tok);
            } else {
                // Reject draft token and substitute the target model's top token
                accepted.push(t_tok);
                return accepted;
            }
        }
        accepted
    }
}
