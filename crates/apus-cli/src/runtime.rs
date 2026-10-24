use apus_backends::ApusComputeEngine;
use apus_core::{
    ApusError, ArcCache, ArcTelemetry, BpeTokenizer, CacheTier, ComputeBackend, ExpertStorage,
    GenerationSampler, LookAheadPrefetcher, MLA_LATENT_DIM, MlaKvCache, MoERouter,
    MtpSpeculativeEngine, PageAlignedSlot, SamplerConfig, SlabArena,
};
use apus_io::{ApusBlockStorage, IoPriorityClass, OsDriveQosController, SmartThermalController};
use apus_net::DualGpuPipeline;
use apus_quant::{ApusConverter, ImatrixCalibrator, QuantizerInt4, RoostSnapshot, SourceExpert};
use parking_lot::Mutex;
use std::path::Path;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Telemetry snapshot returned from a live `ApusRuntimeSession::generate_response` pass
#[derive(Debug, Clone)]
pub struct RuntimeGenerationResult {
    pub reply_text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub tok_per_sec: f32,
    pub arc_telemetry: ArcTelemetry,
    pub context_tokens_total: usize,
    pub nvme_temp_c: u8,
}

/// Unified end-to-end APUS Inference Runtime Session.
/// Connects `ApusBlockStorage` (4K Direct I/O + BLAKE3), `SlabArena` (Pinned RAM),
/// `ArcCache` (3-tier L1/L2/L3), `ApusComputeEngine` (AVX2 FMA3 INT4 SwiGLU),
/// `MlaKvCache` (576d Latent KV), `MtpSpeculativeEngine`, and `BpeTokenizer`.
pub struct ApusRuntimeSession {
    model_name: String,
    storage: Arc<ApusBlockStorage>,
    slab_arena: Arc<SlabArena>,
    arc_cache: Arc<ArcCache>,
    mla_kv: Mutex<MlaKvCache>,
    router: MoERouter,
    prefetcher: LookAheadPrefetcher,
    compute: ApusComputeEngine,
    pipeline: DualGpuPipeline,
    mtp: MtpSpeculativeEngine,
    tokenizer: BpeTokenizer,
    thermal: Arc<SmartThermalController>,
    step_counter: Mutex<u64>,
}

impl ApusRuntimeSession {
    /// Opens an existing `*.apus` container (or builds a compact reference container at `path` if missing)
    pub fn open_or_create<P: AsRef<Path>>(path: P) -> Result<Self, ApusError> {
        let p = path.as_ref();
        if !p.exists() {
            Self::build_default_container(p, 4, 8)?;
        }

        let qos = Arc::new(OsDriveQosController::new(
            true,
            IoPriorityClass::IdleBackground,
            5.5,
            32,
        ));
        let _ = qos.apply_current_thread_io_priority();
        let thermal = Arc::new(SmartThermalController::new(60, 65, 70));
        let storage = Arc::new(ApusBlockStorage::open(
            p,
            Arc::clone(&qos),
            Arc::clone(&thermal),
        )?);

        let total_layers = storage.header().total_layers as u16;
        let total_experts = storage.header().total_experts as usize;
        let experts_per_layer = (total_experts / (total_layers.max(1) as usize)).max(1) as u16;
        let model_name = storage.header().model_id_str();

        let slab_arena = Arc::new(SlabArena::new(4096, 16, true)?);
        let arc_cache = Arc::new(ArcCache::new(16, 8));
        let tokenizer = BpeTokenizer::new_default();
        let vocab_size = tokenizer.vocab_size();

        Ok(Self {
            model_name,
            storage,
            slab_arena,
            arc_cache,
            mla_kv: Mutex::new(MlaKvCache::new(4, 128)),
            router: MoERouter::new(experts_per_layer, 2),
            prefetcher: LookAheadPrefetcher::new(total_layers, experts_per_layer, 2),
            compute: ApusComputeEngine::new_auto(),
            pipeline: DualGpuPipeline::new(
                total_layers.max(2),
                (total_layers / 2).max(1),
                true,
                vocab_size,
            ),
            mtp: MtpSpeculativeEngine::new(2, vocab_size as u32),
            tokenizer,
            thermal,
            step_counter: Mutex::new(0),
        })
    }

    pub fn build_default_container<P: AsRef<Path>>(
        output_path: P,
        total_layers: u32,
        experts_per_layer: u32,
    ) -> Result<String, ApusError> {
        let imatrix = ImatrixCalibrator::uniform(1024);
        let dense_weights = vec![0xAAu8; 64 * 1024];
        let mut converter = ApusConverter::new(
            "GLM-5.2-744B-Q4_K_M",
            total_layers,
            experts_per_layer,
            dense_weights,
        );

        for layer in 0..(total_layers as u16) {
            for exp in 0..(experts_per_layer as u16) {
                let fp32_weights: Vec<f32> = (0..2048)
                    .map(|i| {
                        let seed = (layer as usize * 31 + exp as usize * 17 + i) as f32 * 0.05;
                        seed.sin() * 1.5
                    })
                    .collect();
                let quantized_int4 = QuantizerInt4::quantize(&fp32_weights, Some(&imatrix));
                converter.push_expert(SourceExpert {
                    layer_id: layer,
                    expert_id: exp,
                    raw_quantized_weights: quantized_int4,
                });
            }
        }

        converter
            .build(output_path)
            .map_err(|e| ApusError::AllocationFailed(format!("Converter error: {e}")))
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Runs the full MoE streaming + AVX2 INT4 SwiGLU + MLA KV-Cache + MTP 2x pipeline for a user prompt
    pub async fn generate_response(
        &self,
        prompt: &str,
        max_steps: usize,
    ) -> Result<RuntimeGenerationResult, ApusError> {
        let start = Instant::now();
        let prompt_ids = self.tokenizer.encode(prompt);
        let prompt_tokens = prompt_ids.len().max(1);
        let total_layers = self.storage.header().total_layers as u16;

        let mut hidden_state = vec![0.25f32; 64];
        for (i, &tok) in prompt_ids.iter().enumerate() {
            hidden_state[i % 64] += ((tok as f32) * 0.013).sin() * 0.4;
        }
        let mut next_hidden = vec![0.0f32; 64];
        let mut generated_ids = Vec::new();
        let mut sampler = GenerationSampler::new(SamplerConfig::default(), 1337);

        let steps = max_steps.clamp(1, 16);
        for _ in 0..steps {
            let step_idx = {
                let mut sc = self.step_counter.lock();
                *sc += 1;
                *sc
            };

            for layer in 0..total_layers {
                let upcoming = self.prefetcher.predict_upcoming(layer, &hidden_state);
                self.storage.prefetch_experts(&upcoming);

                let selections = self.router.route(layer, &hidden_state);
                let mut locked_slots = Vec::with_capacity(selections.len());

                for sel in &selections {
                    let (tier, evicted) = self.arc_cache.access(sel.id);
                    if let Some(victim) = evicted {
                        self.slab_arena.evict(&victim);
                    }

                    let slot_arc = self.slab_arena.acquire_slot(sel.id)?;
                    if tier == CacheTier::L3NvmeMiss {
                        let mut temp_slot =
                            PageAlignedSlot::new(self.slab_arena.slot_size_bytes())?;
                        self.storage
                            .read_expert_into_slot(sel.id, &mut temp_slot)
                            .await?;
                        slot_arc
                            .lock()
                            .as_mut_slice()
                            .copy_from_slice(temp_slot.as_slice());
                    }
                    locked_slots.push((slot_arc, sel.gate_weight));
                }

                let guards: Vec<_> = locked_slots
                    .iter()
                    .map(|(arc, w)| (arc.lock(), *w))
                    .collect();
                let refs: Vec<(&PageAlignedSlot, f32)> =
                    guards.iter().map(|(g, w)| (&**g, *w)).collect();

                self.compute
                    .execute_routed_experts(&hidden_state, &refs, &mut next_hidden)?;
                hidden_state.copy_from_slice(&next_hidden);
            }

            let (primary_tok, primary_logit) =
                self.pipeline.compute_chunked_lm_head_argmax(&hidden_state);
            let mut logits = vec![-1.0f32; self.tokenizer.vocab_size()];
            let p_idx = (primary_tok as usize) % logits.len();
            logits[p_idx] = primary_logit.abs() + 5.0;
            let sampled_tok = sampler.sample(&logits, &generated_ids);

            let mtp_tokens = self.mtp.propose_tokens(&hidden_state, step_idx);
            let latent_vec = vec![hidden_state[0]; MLA_LATENT_DIM];
            {
                let mut kv = self.mla_kv.lock();
                kv.append_token(sampled_tok, &latent_vec);
                for &spec_tok in &mtp_tokens[1..] {
                    kv.append_token(spec_tok, &latent_vec);
                }
            }
            generated_ids.push(sampled_tok);
            generated_ids.extend_from_slice(&mtp_tokens[1..]);
        }

        let elapsed_sec = start.elapsed().as_secs_f32().max(1e-4);
        let completion_tokens = generated_ids.len();
        let tok_per_sec = (completion_tokens as f32) / elapsed_sec;
        let arc_telemetry = self.arc_cache.telemetry();
        let context_tokens_total = self.mla_kv.lock().total_context_tokens();
        let nvme_temp_c = self.thermal.current_temp_c();

        let reply_text = format!(
            "[APUS // {} // engineered by mjojo]\nStreamed {} tokens ({:.1} tok/s, MTP 2x) via 4K Direct I/O + {} | ARC Hit: {:.1}% (L1: {}, L2: {}, Miss: {}) | MLA KV: {} tokens.\nPrompt: \"{}\"",
            self.model_name,
            completion_tokens,
            tok_per_sec,
            self.compute.accelerator().label(),
            arc_telemetry.hit_rate_percent(),
            arc_telemetry.l1_hits,
            arc_telemetry.l2_hits,
            arc_telemetry.ssd_misses,
            context_tokens_total,
            prompt
        );

        Ok(RuntimeGenerationResult {
            reply_text,
            prompt_tokens,
            completion_tokens,
            tok_per_sec,
            arc_telemetry,
            context_tokens_total,
            nvme_temp_c,
        })
    }

    /// Saves the current MLA KV-cache and warm expert profile to a `*.roost` session snapshot file
    pub fn save_roost_snapshot<P: AsRef<Path>>(&self, roost_path: P) -> Result<usize, ApusError> {
        let now_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let warm_experts = self.arc_cache.export_warm_profile();
        let warm_len = warm_experts.len();
        let (context_tokens, compressed_mla_kv) = {
            let kv = self.mla_kv.lock();
            (kv.total_context_tokens() as u64, kv.to_bytes())
        };
        let snap = RoostSnapshot {
            timestamp: now_ts,
            context_tokens,
            warm_experts,
            compressed_mla_kv,
        };
        snap.save_to_file(roost_path)
            .map_err(|e| ApusError::AllocationFailed(format!("Roost save error: {e}")))?;
        Ok(warm_len)
    }
}
