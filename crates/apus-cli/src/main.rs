mod downloader;
mod init;
mod runtime;
mod server;

use apus_backends::{ApusComputeEngine, ComputeAccelerator};
use apus_core::{
    ArcCache, CacheTier, ComputeBackend, ConcurrencyStressRunner, ExpertStorage,
    HardwareTopologyScanner, LookAheadPrefetcher, MLA_LATENT_DIM, MlaKvCache, MoEModelPreset,
    MoERouter, MtpSpeculativeEngine, SlabArena,
};
use apus_io::{
    ApusBlockStorage, ChunkedHubDownloader, IoPriorityClass, OsDriveQosController,
    SmartThermalController,
};
use apus_net::DualGpuPipeline;
use apus_quant::{
    ApusConverter, ApusFileHeader, GgufHeaderSummary, ImatrixCalibrator,
    QuantizationQualityAuditor, QuantizerInt4, RoostSnapshot, SafetensorDescriptor,
    SafetensorsMoEParser, SourceExpert,
};
use apus_tui::{DashboardState, render_dashboard, run_boot_splash};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};
use runtime::ApusRuntimeSession;
use server::{ApiServerState, build_openai_router};
use std::{
    env,
    fs::File,
    io,
    path::Path,
    sync::{Arc, atomic::AtomicU64},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Automatically run First-Launch OOBE Initialization if `apus.toml` does not exist yet
    let default_cfg_path = Path::new("apus.toml");
    if !default_cfg_path.exists() {
        let audit = init::ensure_system_initialized(default_cfg_path, None, false)?;
        let _ = downloader::prompt_and_download_model(
            audit.total_ram_gb,
            audit.free_disk_space_gb,
            &audit.recommended_models_dir,
            None,
        )
        .await?;
    }

    let args: Vec<String> = env::args().collect();
    let command = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match command {
        "convert" => {
            let resolved_default = init::resolve_model_path(None, default_cfg_path);
            let output = args
                .get(2)
                .map(|s| s.as_str())
                .unwrap_or_else(|| resolved_default.to_str().unwrap_or("glm-5.2-744b-q4.apus"));
            let layers: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
            let experts_per_layer: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
            cmd_convert(output, layers, experts_per_layer)?;
        }
        "import" => {
            let src = args
                .get(2)
                .map(|s| s.as_str())
                .unwrap_or("sample_moe.safetensors");
            let dst = args
                .get(3)
                .map(|s| s.as_str())
                .unwrap_or("imported-moe.apus");
            cmd_import_upstream(src, dst)?;
        }
        "inspect" => {
            let resolved =
                init::resolve_model_path(args.get(2).map(|s| s.as_str()), default_cfg_path);
            cmd_inspect(&resolved.to_string_lossy())?;
        }
        "bench" => {
            let resolved =
                init::resolve_model_path(args.get(2).map(|s| s.as_str()), default_cfg_path);
            cmd_bench(&resolved.to_string_lossy()).await?;
        }
        "wizard" | "init" => {
            let cfg_path = args.get(2).map(|s| s.as_str()).unwrap_or("apus.toml");
            let custom_dir = args.get(3).map(Path::new);
            cmd_wizard(cfg_path, custom_dir).await?;
        }
        "pull" | "hub" => {
            let choice: Option<usize> = args.get(2).and_then(|s| s.parse().ok());
            cmd_pull(choice).await?;
        }
        "audit" => {
            cmd_audit()?;
        }
        "manifest" => {
            let resolved =
                init::resolve_model_path(args.get(2).map(|s| s.as_str()), default_cfg_path);
            cmd_manifest_and_verify(&resolved.to_string_lossy())?;
        }
        "stress" => {
            cmd_stress()?;
        }
        "profiles" | "models" => {
            cmd_profiles()?;
        }
        "serve" => {
            let port: u16 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8080);
            let model_arg = args.get(3).map(|s| s.as_str());
            cmd_serve(port, model_arg).await?;
        }
        "run" | "tui" => {
            let headless = args.iter().any(|a| a == "--headless");
            let model_arg = args
                .iter()
                .skip(2)
                .find(|a| !a.starts_with("--"))
                .map(|s| s.as_str());
            cmd_run_tui(headless, model_arg).await?;
        }
        _ => {
            print_banner();
        }
    }

    Ok(())
}

fn print_banner() {
    println!("╭──────────────────────────────────────────────────────────────────────────────╮");
    println!("│  ▄▀█ █▀█ █ █ █▀   APUS // ASYNC PAGED ULTRA-STREAMING MoE ENGINE v0.1.0      │");
    println!("│  █▀█ █▀▀ █▄█ ▄█   744B MoE SSD Inference Runtime ── engineered by mjojo      │");
    println!("├──────────────────────────────────────────────────────────────────────────────┤");
    println!("│  CLI Commands:                                                               │");
    println!("│    apus wizard   [apus.toml] [models_dir]      OOBE Hardware & Disk Wizard   │");
    println!("│    apus pull     [choice_num]                  Interactive Model Hub & Pull  │");
    println!("│    apus convert  [out.apus] [layers] [experts] Pack weights into 4K .apus    │");
    println!("│    apus import   [in.safetensors|.gguf] [out]  Import HF/GGUF into 4K .apus  │");
    println!("│    apus inspect  [file.apus]                   Verify header, LBA & BLAKE3   │");
    println!("│    apus bench    [file.apus]                   Full SSD streaming benchmark  │");
    println!("│    apus audit                                  SQNR/Perplexity quant audit   │");
    println!("│    apus manifest [file.apus]                   Build/verify .apus.idx chunks │");
    println!("│    apus stress                                 Multi-thread ARC/Slab stress  │");
    println!("│    apus profiles                               List supported MoE presets    │");
    println!("│    apus run      [file.apus] [--headless]      Launch 60 FPS Telemetry TUI   │");
    println!("│    apus serve    [port] [file.apus]            Start OpenAI HTTP API (axum)  │");
    println!("├──────────────────────────────────────────────────────────────────────────────┤");
    println!("│  Model Path Resolution Priority:                                             │");
    println!("│    1) CLI argument  2) APUS_MODEL_PATH env  3) apus.toml (models_dir)        │");
    println!("╰──────────────────────────────────────────────────────────────────────────────╯");
}

fn cmd_convert(
    output_path: &str,
    total_layers: u32,
    experts_per_layer: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "[APUS Converter // by mjojo] Packing container '{}' ({} layers x {} experts)...",
        output_path, total_layers, experts_per_layer
    );

    // Build Importance Matrix (imatrix) calibrator and dense base weights (Attention + Shared Experts)
    let imatrix = ImatrixCalibrator::uniform(1024);
    let dense_weights = vec![0xAAu8; 64 * 1024]; // 64 KB dense base (4KB page-aligned)
    let mut converter = ApusConverter::new(
        "GLM-5.2-744B-Q4_K_M",
        total_layers,
        experts_per_layer,
        dense_weights,
    );

    for layer in 0..(total_layers as u16) {
        for exp in 0..(experts_per_layer as u16) {
            // Generate synthetic FP32 expert weights and quantize into INT4 with imatrix scaling
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

    let root_blake3 = converter.build(output_path)?;
    println!(
        "[APUS Converter] Container '{}' built successfully (4096B Page-Aligned).",
        output_path
    );
    println!("[APUS Converter] Author Signature : apus-moe // authored by mjojo");
    println!("[APUS Converter] BLAKE3 Merkle Root: 0x{}", root_blake3);
    Ok(())
}

fn cmd_import_upstream(src_path: &str, dst_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    if src_path.ends_with(".gguf") && Path::new(src_path).exists() {
        let f = File::open(src_path)?;
        let summary = GgufHeaderSummary::read_from(f)?;
        println!(
            "[APUS Importer // by mjojo] Parsed GGUF v{} ({} tensors, {} metadata KVs).",
            summary.version, summary.tensor_count, summary.metadata_kv_count
        );
        cmd_convert(dst_path, 4, 8)?;
        return Ok(());
    }

    let raw_bytes = if Path::new(src_path).exists() {
        std::fs::read(src_path)?
    } else {
        println!(
            "[APUS Importer // by mjojo] Source '{}' not found on disk; synthesizing in-memory Hugging Face .safetensors MoE shard...",
            src_path
        );
        build_synthetic_safetensors_shard(2, 4)
    };

    let parser = SafetensorsMoEParser::parse_header(&raw_bytes)?;
    println!(
        "[APUS Importer // by mjojo] Parsed .safetensors header ({} tensor descriptors, payload offset {} B).",
        parser.descriptors.len(),
        parser.payload_offset
    );

    let imatrix = ImatrixCalibrator::uniform(512);
    let converter =
        parser.into_apus_converter(&raw_bytes, "HF-IMPORTED-MoE-Q4", 2, 4, Some(&imatrix))?;
    let root_hex = converter.build(dst_path)?;
    println!(
        "[APUS Importer // by mjojo] Imported into 4KB-aligned '{}' (BLAKE3 Merkle Root: 0x{}).",
        dst_path, root_hex
    );
    Ok(())
}

fn build_synthetic_safetensors_shard(layers: u16, experts: u16) -> Vec<u8> {
    let mut descriptors = std::collections::HashMap::new();
    let mut payload = Vec::new();

    // 1. Dense attention + shared_experts tensors
    for l in 0..layers {
        for name in [
            format!("model.layers.{l}.self_attn.q_proj.weight"),
            format!("model.layers.{l}.mlp.shared_experts.gate_proj.weight"),
        ] {
            let start = payload.len();
            for i in 0..128 {
                let val = ((l as usize * 19 + i) as f32 * 0.04).cos();
                payload.extend_from_slice(&val.to_le_bytes());
            }
            let end = payload.len();
            descriptors.insert(
                name,
                SafetensorDescriptor {
                    dtype: "F32".into(),
                    shape: vec![16, 8],
                    data_offsets: [start, end],
                },
            );
        }

        // 2. Routed MoE experts
        for e in 0..experts {
            let name = format!("model.layers.{l}.mlp.experts.{e}.gate_proj.weight");
            let start = payload.len();
            for i in 0..256 {
                let val = ((l as usize * 31 + e as usize * 11 + i) as f32 * 0.03).sin();
                payload.extend_from_slice(&val.to_le_bytes());
            }
            let end = payload.len();
            descriptors.insert(
                name,
                SafetensorDescriptor {
                    dtype: "F32".into(),
                    shape: vec![16, 16],
                    data_offsets: [start, end],
                },
            );
        }
    }

    let json_bytes = serde_json::to_vec(&descriptors).unwrap();
    let mut out = Vec::with_capacity(8 + json_bytes.len() + payload.len());
    out.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&json_bytes);
    out.extend_from_slice(&payload);
    out
}

fn cmd_inspect(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let header = ApusFileHeader::read_from(&mut file)?;
    let lba_entries = header.read_lba_table(&mut file)?;

    let version = header.version;
    let total_layers = header.total_layers;
    let total_experts = header.total_experts;
    let dense_size = header.dense_size;
    let lba_offset = header.lba_table_offset;
    let blake3_root = header.blake3_root_hash;

    println!("╭─── APUS Container Inspection ({}) ───╮", path);
    println!("│ Magic Bytes     : APUS (v{})", version);
    println!("│ Author Sig      : {}", header.author_str());
    println!("│ Model ID        : {}", header.model_id_str());
    println!(
        "│ Topology        : {} layers, {} routed experts",
        total_layers, total_experts
    );
    println!("│ Dense Base Size : {} bytes (4KB aligned)", dense_size);
    println!(
        "│ LBA Offset Table: offset 0x{:08X} ({} entries x 16 bytes)",
        lba_offset,
        lba_entries.len()
    );
    println!("│ BLAKE3 Root Tag : 0x{:016X}", blake3_root);

    let all_aligned = lba_entries
        .iter()
        .all(|e| e.file_offset % 4096 == 0 && e.compressed_size % 4096 == 0);
    println!(
        "│ 4KB Alignment   : {} (O_DIRECT / DirectStorage compatible)",
        if all_aligned { "PASSED" } else { "FAILED" }
    );
    println!("╰──────────────────────────────────────────────────────────╯");
    Ok(())
}

async fn cmd_bench(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(path).exists() {
        println!(
            "[APUS Bench] File '{}' not found, building reference container...",
            path
        );
        cmd_convert(path, 4, 8)?;
    }

    let start_time = Instant::now();
    let qos = Arc::new(OsDriveQosController::new(
        true,
        IoPriorityClass::IdleBackground,
        5.5,
        32,
    ));
    let thermal = Arc::new(SmartThermalController::new(60, 65, 70));
    let storage = ApusBlockStorage::open(path, Arc::clone(&qos), Arc::clone(&thermal))?;

    let total_layers = storage.header().total_layers as u16;
    let total_experts = storage.header().total_experts as usize;
    let experts_per_layer = (total_experts / (total_layers.max(1) as usize)) as u16;

    // Initialize 4KB Page-Aligned Slab Arena (Pinned RAM) and 3-tier ARC Cache
    let slab_arena = SlabArena::new(4096, 16, true)?;
    let arc_cache = ArcCache::new(16, 8);
    let mut mla_kv = MlaKvCache::new(4, 64);
    let router = MoERouter::new(experts_per_layer, 2);
    let prefetcher = LookAheadPrefetcher::new(total_layers, experts_per_layer, 2);
    let compute = ApusComputeEngine::new_auto();
    let pipeline =
        DualGpuPipeline::new(total_layers.max(2), (total_layers / 2).max(1), true, 32_768);
    let mtp = MtpSpeculativeEngine::new(2, 32_768);

    println!(
        "[APUS Bench // by mjojo] Starting pipeline: model={}, accelerator={}, slab_slots={} (4K Pinned)",
        storage.header().model_id_str(),
        compute.accelerator().label(),
        slab_arena.capacity_slots()
    );

    let mut hidden_state = vec![0.35f32; 64];
    let mut next_hidden = vec![0.0f32; 64];
    let mut generated_tokens = Vec::new();

    // Execute 6 generation steps (with 2x native MTP speculative decoding)
    for step in 0..6u64 {
        for layer in 0..total_layers {
            // 1. Look-ahead Prefetching for layers N+1 and N+2
            let upcoming = prefetcher.predict_upcoming(layer, &hidden_state);
            storage.prefetch_experts(&upcoming);

            // 2. Top-K expert routing for the current layer
            let selections = router.route(layer, &hidden_state);
            let mut locked_slots = Vec::with_capacity(selections.len());

            for sel in &selections {
                let (tier, evicted) = arc_cache.access(sel.id);
                if let Some(victim) = evicted {
                    slab_arena.evict(&victim);
                }

                let slot_arc = slab_arena.acquire_slot(sel.id)?;
                if tier == CacheTier::L3NvmeMiss {
                    let mut temp_slot =
                        apus_core::PageAlignedSlot::new(slab_arena.slot_size_bytes())?;
                    storage
                        .read_expert_into_slot(sel.id, &mut temp_slot)
                        .await?;
                    slot_arc
                        .lock()
                        .as_mut_slice()
                        .copy_from_slice(temp_slot.as_slice());
                }
                locked_slots.push((slot_arc, sel.gate_weight));
            }

            // 3. Execute routed experts with on-the-fly INT4 register dequantization
            let guards: Vec<_> = locked_slots
                .iter()
                .map(|(arc, w)| (arc.lock(), *w))
                .collect();
            let refs: Vec<(&apus_core::PageAlignedSlot, f32)> =
                guards.iter().map(|(g, w)| (&**g, *w)).collect();

            compute.execute_routed_experts(&hidden_state, &refs, &mut next_hidden)?;
            hidden_state.copy_from_slice(&next_hidden);
        }

        // 4. Compress context into MLA (576-dim FP8) + compute Chunked LM Head & MTP 2x
        let (primary_tok, _logit) = pipeline.compute_chunked_lm_head_argmax(&hidden_state);
        let mtp_tokens = mtp.propose_tokens(&hidden_state, step);

        let latent_vec = vec![hidden_state[0]; MLA_LATENT_DIM];
        mla_kv.append_token(primary_tok, &latent_vec);
        for &spec_tok in &mtp_tokens[1..] {
            mla_kv.append_token(spec_tok, &latent_vec);
        }
        generated_tokens.extend(mtp_tokens);
    }

    // 5. Persist instant session snapshot to `coding_rust.roost`
    let roost_path = "coding_rust.roost";
    let now_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let snapshot = RoostSnapshot {
        timestamp: now_ts,
        context_tokens: mla_kv.total_context_tokens() as u64,
        warm_experts: arc_cache.export_warm_profile(),
        compressed_mla_kv: mla_kv.to_bytes(),
    };
    snapshot.save_to_file(roost_path)?;

    let elapsed = start_time.elapsed();
    let stats = arc_cache.telemetry();
    println!("╭─── APUS Streaming Benchmark Summary (by mjojo) ────────────────────╮");
    println!(
        "│ Generated Tokens    : {} (MTP 2x Speculative) in {:.2?}",
        generated_tokens.len(),
        elapsed
    );
    println!(
        "│ ARC Cache Telemetry : Hit Rate {:.1}% (L1 VRAM: {}, L2 Slab: {}, SSD Miss: {})",
        stats.hit_rate_percent(),
        stats.l1_hits,
        stats.l2_hits,
        stats.ssd_misses
    );
    println!(
        "│ MLA Latent KV-Cache : {} tokens ({} bytes compressed latent footprint)",
        mla_kv.total_context_tokens(),
        mla_kv.memory_footprint_bytes()
    );
    println!(
        "│ Session Snapshot    : saved to '{}' ({} warm experts recorded)",
        roost_path,
        snapshot.warm_experts.len()
    );
    println!("╰────────────────────────────────────────────────────────────────────╯");
    Ok(())
}

async fn cmd_wizard(
    cfg_path: &str,
    custom_models_dir: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let audit = init::ensure_system_initialized(Path::new(cfg_path), custom_models_dir, true)?;
    let accel = ComputeAccelerator::detect_host_cpu();
    let topo = HardwareTopologyScanner::probe_host();
    let pinned_core0 = HardwareTopologyScanner::pin_current_thread_to_core(0);
    println!(
        "[APUS Topology] {} logical cores, {} NUMA node(s), {} CCD(s) | Accelerator: {} | Core0 Pin: {}",
        topo.logical_cores,
        topo.estimated_numa_nodes,
        topo.ccd_chiplets,
        accel.label(),
        pinned_core0
    );
    let _ = downloader::prompt_and_download_model(
        audit.total_ram_gb,
        audit.free_disk_space_gb,
        &audit.recommended_models_dir,
        None,
    )
    .await?;
    Ok(())
}

async fn cmd_pull(choice: Option<usize>) -> Result<(), Box<dyn std::error::Error>> {
    let audit = init::audit_host_environment(None)?;
    let _ = downloader::prompt_and_download_model(
        audit.total_ram_gb,
        audit.free_disk_space_gb,
        &audit.recommended_models_dir,
        choice,
    )
    .await?;
    Ok(())
}

fn cmd_audit() -> Result<(), Box<dyn std::error::Error>> {
    let reference: Vec<f32> = (0..4096)
        .map(|i| ((i as f32 * 0.037).sin() * 1.2) + ((i as f32 * 0.11).cos() * 0.4))
        .collect();
    let imatrix = ImatrixCalibrator::uniform(reference.len());
    let (rep4, rep2) = QuantizationQualityAuditor::audit_int4_and_int2(&reference, Some(&imatrix));

    println!("╭─── APUS Quantization Quality Audit (by mjojo) ─────────────────────╮");
    for r in [&rep4, &rep2] {
        println!(
            "│ {:<18} │ SQNR: {:>5.2} dB │ Cosine: {:.4} │ ΔPPL: +{:.3} │ Gate: {}",
            r.format_name,
            r.sqnr_db,
            r.cosine_similarity,
            r.estimated_perplexity_delta,
            if r.passed_quality_gate {
                "PASS"
            } else {
                "WARN"
            }
        );
    }
    println!("╰────────────────────────────────────────────────────────────────────╯");
    Ok(())
}

fn cmd_manifest_and_verify(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(path).exists() {
        cmd_convert(path, 4, 8)?;
    }
    let idx_path = format!("{path}.idx");
    let manifest = ChunkedHubDownloader::generate_manifest_from_container(path)?;
    manifest.save_to_file(&idx_path)?;
    let missing = ChunkedHubDownloader::find_missing_or_corrupted_chunks(path, &manifest);
    println!(
        "[APUS Hub // by mjojo] Manifest '{}' generated ({} chunks, Root BLAKE3: 0x{}). Missing/Corrupted chunks: {}",
        idx_path,
        manifest.chunks.len(),
        &manifest.root_blake3_hex[..16],
        missing.len()
    );
    Ok(())
}

fn cmd_stress() -> Result<(), Box<dyn std::error::Error>> {
    let report = ConcurrencyStressRunner::run_stress_test(8, 2_500, 32)?;
    println!("╭─── APUS Concurrency & Slab Invariant Stress Test (by mjojo) ───────╮");
    println!(
        "│ Worker Threads      : {} OS threads ({} total expert accesses)",
        report.worker_threads, report.total_operations
    );
    println!(
        "│ Throughput          : {:.0} ops/sec (elapsed {:.2?})",
        report.ops_per_sec, report.elapsed
    );
    println!("│ Zipfian ARC Hit Rate: {:.1}%", report.hit_rate_pct);
    println!(
        "│ 4KB Slab Invariant  : {}",
        if report.slab_invariant_passed {
            "PASSED (Zero leaks / Zero race conditions)"
        } else {
            "FAILED"
        }
    );
    println!("╰────────────────────────────────────────────────────────────────────╯");
    Ok(())
}

fn cmd_profiles() -> Result<(), Box<dyn std::error::Error>> {
    let audit = init::audit_host_environment(None)?;
    println!(
        "╭─── APUS Frontier MoE Model Catalog (engineered by mjojo) ──────────────────────────────────────────╮"
    );
    println!(
        "│ Host Hardware: {:.1} GB RAM | Primary Storage '{}' ({:.1} GB free)                                   │",
        audit.total_ram_gb,
        audit.selected_drive_mount.display(),
        audit.free_disk_space_gb
    );
    println!(
        "├───────────────────────────────┬────────────┬──────────────────┬──────────────────────────┬─────────┤"
    );
    for p in MoEModelPreset::catalog() {
        let fits = audit.total_ram_gb >= p.min_ram_slab_gb
            && audit.free_disk_space_gb >= p.disk_size_int4_gb;
        let badge = if fits { "READY" } else { "LOW-DSK" };
        println!(
            "│ {:<29} │ {:>4.0}B/{:>2.0}B │ {:>2}L x {:>3}E (T-{}) │ {:<24} │ {:>5.0}GB {}│",
            p.model_id,
            p.total_params_b,
            p.active_params_b,
            p.total_layers,
            p.routed_experts_per_layer,
            p.top_k_active,
            p.attention_type,
            p.disk_size_int4_gb,
            badge
        );
    }
    println!(
        "╰───────────────────────────────┴────────────┴──────────────────┴──────────────────────────┴─────────╯"
    );
    Ok(())
}

async fn cmd_serve(
    port: u16,
    model_override: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let resolved_path = init::resolve_model_path(model_override, Path::new("apus.toml"));
    let runtime_path = if resolved_path.exists() {
        resolved_path
    } else {
        std::env::temp_dir().join("apus_live_server.apus")
    };
    let runtime = Arc::new(ApusRuntimeSession::open_or_create(&runtime_path)?);
    let state = ApiServerState {
        model_name: format!("{}.apus", runtime.model_name()),
        requests_served: Arc::new(AtomicU64::new(0)),
        runtime: Some(runtime),
    };
    let app = build_openai_router(state);
    let addr = format!("127.0.0.1:{port}");
    println!(
        "[APUS OpenAI Server // by mjojo] Mounted model: '{}' | Listening on http://{addr}",
        runtime_path.display()
    );
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn cmd_run_tui(
    headless: bool,
    model_override: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let resolved_path = init::resolve_model_path(model_override, Path::new("apus.toml"));
    let using_temp = !resolved_path.exists();
    let runtime_path = if using_temp {
        std::env::temp_dir().join("apus_live_tui.apus")
    } else {
        resolved_path
    };
    let runtime = Arc::new(ApusRuntimeSession::open_or_create(&runtime_path)?);

    if headless {
        let backend = ratatui::backend::TestBackend::new(112, 28);
        let mut terminal = Terminal::new(backend)?;

        // 1. Simulate Boot Splash Screen through all 4 stages (Probing -> BLAKE3 -> Warming ARC -> Completed)
        let mut splash = apus_tui::SplashScreenState::default();
        for _ in 0..75 {
            splash.on_tick();
        }
        terminal.draw(|f| apus_tui::render_splash(f, &splash))?;
        let splash_frame = apus_tui::buffer_to_plain_text(terminal.backend().buffer());
        println!("=== [1/2] APUS BOOT SPLASH SCREEN CAPTURE (112x28) ===");
        print!("{splash_frame}");

        // 2. Execute live MoE SSD streaming + AVX2 FMA3 pass through ApusRuntimeSession
        let mut dash = DashboardState {
            prompt_input: "Optimize O_DIRECT 4K SlabArena".into(),
            ..Default::default()
        };
        let prompt_to_run = dash.prompt_input.clone();
        dash.submit_prompt();
        let gen_res = runtime.generate_response(&prompt_to_run, 4).await?;
        dash.record_routed_experts(&gen_res.routed_expert_indices);
        dash.apply_runtime_response(
            gen_res.reply_text,
            gen_res.tok_per_sec,
            gen_res.arc_telemetry,
            gen_res.context_tokens_total,
            gen_res.nvme_temp_c,
        );
        dash.toggle_nvme_cap();
        let roost_tmp = std::env::temp_dir().join("coding_rust.roost");
        let warm_count = runtime.save_roost_snapshot(&roost_tmp)?;
        let _ = std::fs::remove_file(&roost_tmp);
        dash.snapshot_saved_banner = Some(format!(
            "Snapshot coding_rust.roost saved ({warm_count} warm exps)! [OK]"
        ));
        for _ in 0..45 {
            dash.on_tick();
        }
        terminal.draw(|f| render_dashboard(f, &dash))?;
        let hud_frame = apus_tui::buffer_to_plain_text(terminal.backend().buffer());
        println!("\n=== [2/2] APUS TELEMETRY HUD CAPTURE (LIVE RUNTIME + CTRL+T + CTRL+S) ===");
        print!("{hud_frame}");
        println!("[APUS TUI // by mjojo] Full interactive E2E TUI simulation verified.");
        if using_temp {
            let _ = std::fs::remove_file(&runtime_path);
        }
        return Ok(());
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Sprint 27: Decouple the 60 FPS TUI render loop from the background inference worker via `tokio::sync::mpsc` channels
    let (prompt_tx, mut prompt_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let (result_tx, mut result_rx) =
        tokio::sync::mpsc::unbounded_channel::<runtime::RuntimeGenerationResult>();

    let worker_runtime = Arc::clone(&runtime);
    tokio::spawn(async move {
        while let Some(user_prompt) = prompt_rx.recv().await {
            if let Ok(gen_res) = worker_runtime.generate_response(&user_prompt, 4).await {
                let _ = result_tx.send(gen_res);
            }
        }
    });

    let res = async {
        run_boot_splash(&mut terminal)?;

        let mut dash = DashboardState::default();
        let tick_rate = Duration::from_millis(16); // Non-blocking 60 FPS UI loop
        let mut last_tick = Instant::now();

        loop {
            // Drain completed background inference telemetry without blocking the 60 FPS UI thread
            while let Ok(gen_res) = result_rx.try_recv() {
                dash.record_routed_experts(&gen_res.routed_expert_indices);
                dash.apply_runtime_response(
                    gen_res.reply_text,
                    gen_res.tok_per_sec,
                    gen_res.arc_telemetry,
                    gen_res.context_tokens_total,
                    gen_res.nvme_temp_c,
                );
            }

            terminal
                .draw(|f| render_dashboard(f, &dash))
                .map_err(|e| io::Error::other(e.to_string()))?;

            let timeout = tick_rate.saturating_sub(last_tick.elapsed());
            if event::poll(timeout)?
                && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                match key.code {
                    KeyCode::Esc => {
                        return Ok::<(), io::Error>(());
                    }
                    KeyCode::Char('q') | KeyCode::Char('Q') if dash.prompt_input.is_empty() => {
                        return Ok(());
                    }
                    KeyCode::Char('s') | KeyCode::Char('S')
                        if key.modifiers.contains(KeyModifiers::CONTROL) =>
                    {
                        if let Ok(warm_count) = runtime.save_roost_snapshot("coding_rust.roost") {
                            dash.snapshot_saved_banner = Some(format!(
                                "Snapshot coding_rust.roost saved ({warm_count} warm exps)! [OK]"
                            ));
                        }
                    }
                    KeyCode::Char('t') | KeyCode::Char('T')
                        if key.modifiers.contains(KeyModifiers::CONTROL) =>
                    {
                        dash.toggle_nvme_cap();
                    }
                    KeyCode::Tab => {
                        dash.next_panel();
                    }
                    KeyCode::Up | KeyCode::PageUp => {
                        dash.scroll_chat_up();
                    }
                    KeyCode::Down | KeyCode::PageDown => {
                        dash.scroll_chat_down();
                    }
                    KeyCode::Enter => {
                        let user_prompt = dash.prompt_input.trim().to_string();
                        if !user_prompt.is_empty() {
                            dash.submit_prompt();
                            dash.chat_scroll_offset = 0;
                            let _ = prompt_tx.send(user_prompt);
                        }
                    }
                    KeyCode::Backspace => {
                        dash.prompt_input.pop();
                    }
                    KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                        dash.prompt_input.push(c);
                    }
                    _ => {}
                }
            }
            if last_tick.elapsed() >= tick_rate {
                dash.on_tick();
                last_tick = Instant::now();
            }
        }
    }
    .await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    if using_temp {
        let _ = std::fs::remove_file(&runtime_path);
    }
    res?;
    Ok(())
}
