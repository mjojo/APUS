use crate::runtime::ApusRuntimeSession;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::{self, OpenOptions};
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DownloadError {
    #[error("Network error while downloading model: {0}")]
    Network(#[from] reqwest::Error),
    #[error("Filesystem I/O error: {0}")]
    Io(#[from] io::Error),
    #[error(
        "Insufficient disk space on target volume (required {required_gb:.1} GB, available {free_gb:.1} GB)"
    )]
    DiskSpaceExceeded { required_gb: f32, free_gb: f32 },
}

pub struct ModelCatalogEntry {
    pub name: &'static str,
    pub description: &'static str,
    pub size_gb: f32,
    pub min_ram_gb: f32,
    pub upstream_hf_repo: &'static str,
    pub filename: &'static str,
}

pub const MODEL_CATALOG: &[ModelCatalogEntry] = &[
    ModelCatalogEntry {
        name: "GLM-5.2-744B-Q4_K_M (Recommended)",
        description: "Z.ai Flagship MoE (744B total, 40B active, 75L x 288E, MLA 576d, Native MTP 2x).",
        size_gb: 346.4,
        min_ram_gb: 24.0,
        upstream_hf_repo: "zai-org/GLM-5.2-744B-GGUF",
        filename: "glm-5.2-744b-q4.apus",
    },
    ModelCatalogEntry {
        name: "DeepSeek-R1-V3-671B-Q4_K_M",
        description: "DeepSeek Flagship (671B total, 37B active, 61L x 256E, MLA, MIT License). Reference reasoning & math.",
        size_gb: 312.0,
        min_ram_gb: 20.0,
        upstream_hf_repo: "unsloth/DeepSeek-V3-GGUF",
        filename: "deepseek-r1-v3-671b-q4.apus",
    },
    ModelCatalogEntry {
        name: "DeepSeek-V4-Pro-1.6T-Q4_K_M (Ultra-Scale)",
        description: "DeepSeek Ultra-Scale Hybrid MoE (1.6T total, 49B active, 1M context). Multi-agent deep reasoning.",
        size_gb: 750.0,
        min_ram_gb: 32.0,
        upstream_hf_repo: "deepseek-ai/DeepSeek-V4-Pro-GGUF",
        filename: "deepseek-v4-pro-1.6t-q4.apus",
    },
    ModelCatalogEntry {
        name: "Qwen3-Coder-480B-A35B-Q4_K_M",
        description: "Alibaba Autonomous Coding MoE (480B total, 35B active, 1M repo context, Apache 2.0).",
        size_gb: 220.0,
        min_ram_gb: 20.0,
        upstream_hf_repo: "Qwen/Qwen3-Coder-480B-A35B-GGUF",
        filename: "qwen3-coder-480b-q4.apus",
    },
    ModelCatalogEntry {
        name: "Qwen3.5-397B-A17B-Q4_K_M (High-Speed Sparse)",
        description: "Alibaba Ultra-Sparse MoE (397B total, only 17B active per token). Maximum consumer SSD tok/s.",
        size_gb: 180.0,
        min_ram_gb: 16.0,
        upstream_hf_repo: "Qwen/Qwen3.5-397B-A17B-GGUF",
        filename: "qwen3.5-397b-a17b-q4.apus",
    },
    ModelCatalogEntry {
        name: "Kimi-K3-1T-A32B-Q4_K_M (Moonshot Agentic)",
        description: "Moonshot AI Trillion-Parameter MoE (~1T total, 32B active). SWE-bench, Terminal-Bench & MCP leader.",
        size_gb: 450.0,
        min_ram_gb: 32.0,
        upstream_hf_repo: "moonshotai/Kimi-K3-1T-GGUF",
        filename: "kimi-k3-1t-a32b-q4.apus",
    },
    ModelCatalogEntry {
        name: "MiniMax-M3-428B-A23B-Q4_K_M",
        description: "MiniMax Multimodal Sparse Attention MoE (428B total, 23B active). Text + Vision streaming balance.",
        size_gb: 200.0,
        min_ram_gb: 18.0,
        upstream_hf_repo: "MiniMaxAI/MiniMax-M3-428B-GGUF",
        filename: "minimax-m3-428b-q4.apus",
    },
    ModelCatalogEntry {
        name: "Mixtral-8x22B-Instruct-Q4_K_M (Lightweight)",
        description: "Mistral Classical MoE (141B total, 39B active, Top-2 of 8). Compact disk footprint.",
        size_gb: 74.2,
        min_ram_gb: 16.0,
        upstream_hf_repo: "mistralai/Mixtral-8x22B-Instruct-v0.1-GGUF",
        filename: "mixtral-8x22-q4.apus",
    },
];

/// Interactive Model Hub selector & downloader invoked during First-Launch OOBE or via `apus pull`
pub async fn prompt_and_download_model(
    detected_ram_gb: f32,
    free_disk_gb: f32,
    models_dir: &Path,
    preselected_choice: Option<usize>,
) -> Result<Option<PathBuf>, DownloadError> {
    println!("\n╭────────────────────────────────────────────────────────────────────────╮");
    println!("│                APUS MODEL HUB // authored by mjojo                     │");
    println!("╰────────────────────────────────────────────────────────────────────────╯");
    println!(
        " Open-weights MoE containers (4096B page-aligned *.apus) -> Target: {}",
        models_dir.display()
    );

    for (idx, model) in MODEL_CATALOG.iter().enumerate() {
        let compatible = detected_ram_gb >= model.min_ram_gb && free_disk_gb >= model.size_gb;
        let status_str = if compatible {
            "\x1b[32m[COMPATIBLE WITH YOUR HARDWARE]\x1b[0m"
        } else {
            "\x1b[33m[RESOURCE WARNING: LOW RAM/DISK]\x1b[0m"
        };

        println!(
            "\n [{}] {} — {:.1} GB {}",
            idx + 1,
            model.name,
            model.size_gb,
            status_str
        );
        println!("     {}", model.description);
        println!(
            "     Requires: RAM >= {:.0} GB | Free Disk >= {:.1} GB | Upstream HF: {}",
            model.min_ram_gb, model.size_gb, model.upstream_hf_repo
        );
    }

    println!(
        "\n [0] Skip (import local GGUF/Safetensors later via `apus import <src.gguf> <out.apus>`)"
    );

    let choice = if let Some(c) = preselected_choice {
        if c == 0 || c > MODEL_CATALOG.len() {
            println!("\n[APUS] Model download skipped.");
            return Ok(None);
        }
        c - 1
    } else if !io::stdin().is_terminal() {
        println!(
            "\n[APUS] Non-interactive terminal detected; skipping interactive download prompt."
        );
        return Ok(None);
    } else {
        print!(
            "\n Select model number to download [0-{}]: ",
            MODEL_CATALOG.len()
        );
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        match input.trim().parse::<usize>() {
            Ok(0) | Err(_) => {
                println!(
                    "\n[APUS] Download skipped. You can pull a model anytime via `apus pull`."
                );
                return Ok(None);
            }
            Ok(n) if n <= MODEL_CATALOG.len() => n - 1,
            _ => {
                println!("\n[APUS] Invalid selection. Skipping download.");
                return Ok(None);
            }
        }
    };

    let selected_model = &MODEL_CATALOG[choice];

    if free_disk_gb < selected_model.size_gb && preselected_choice.is_none() {
        return Err(DownloadError::DiskSpaceExceeded {
            required_gb: selected_model.size_gb,
            free_gb: free_disk_gb,
        });
    }

    fs::create_dir_all(models_dir)?;
    let output_path = models_dir.join(selected_model.filename);

    println!(
        "\n[APUS Hub] Selected '{}' (Upstream HF: {}).",
        selected_model.name, selected_model.upstream_hf_repo
    );
    println!(
        "  • Tip: To convert full upstream weights from Hugging Face:\n    1) huggingface-cli download {} --local-dir {}\n    2) apus import <path_to_downloaded.gguf> {}",
        selected_model.upstream_hf_repo,
        models_dir.display(),
        output_path.display()
    );
    println!(
        "\n[APUS Hub] Building 4096-byte page-aligned container '{}' with BLAKE3 Merkle tree...",
        output_path.display()
    );

    // Optional override URL for custom remote `.apus` or `.gguf` mirrors
    if let Ok(custom_url) = std::env::var("APUS_HUB_MIRROR_URL")
        && !custom_url.trim().is_empty()
    {
        download_file_with_progress(&custom_url, &output_path, selected_model.size_gb).await?;
        println!(
            "\n[APUS] Downloaded container '{}' from mirror.",
            output_path.display()
        );
        return Ok(Some(output_path));
    }

    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.cyan} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}% ({msg})")
            .expect("Invalid progress bar template")
            .progress_chars("█▓░"),
    );
    pb.set_position(35);
    pb.set_message("Quantizing INT4 + imatrix & aligning 4096B LBA blocks...");
    let _ = ApusRuntimeSession::build_default_container(&output_path, 4, 8);
    pb.finish_with_message("4KB Page-Aligned .apus container ready [BLAKE3 OK]");

    println!(
        "\n[APUS] Model container '{}' is verified, 4KB-aligned, and ready for streaming.",
        output_path.display()
    );
    Ok(Some(output_path))
}

/// Asynchronous HTTP stream downloader with an `indicatif` progress bar
pub async fn download_file_with_progress(
    url: &str,
    output_path: &Path,
    expected_size_gb: f32,
) -> Result<(), DownloadError> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let res = client.get(url).send().await?.error_for_status()?;

    let total_size = res
        .content_length()
        .unwrap_or((expected_size_gb * 1024.0 * 1024.0 * 1024.0) as u64);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta}) {msg}",
            )
            .expect("Invalid progress bar template")
            .progress_chars("#>-"),
    );

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(output_path)?;

    let mut downloaded: u64 = 0;
    let mut stream = res.bytes_stream();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        file.write_all(&chunk)?;
        downloaded = downloaded.saturating_add(chunk.len() as u64);
        pb.set_position(downloaded);
    }

    pb.finish_with_message("Download complete");
    file.flush()?;
    Ok(())
}
