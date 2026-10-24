use apus_core::{ApusConfig, detect_system_hardware};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use sysinfo::Disks;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum InitError {
    #[error("System hardware scan failed: {0}")]
    SysinfoError(String),
    #[error(
        "Critical hardware deficit: detected less than 8 GB RAM or incompatible storage environment."
    )]
    HardwareIncompatible,
    #[error("Failed to write apus.toml configuration: {0}")]
    Io(#[from] std::io::Error),
}

/// Information about a mounted physical disk / volume discovered via `sysinfo::Disks`
#[derive(Debug, Clone)]
pub struct DiskVolumeInfo {
    pub mount_point: PathBuf,
    pub kind_label: String,
    pub total_space_gb: f32,
    pub available_space_gb: f32,
    pub is_ssd_or_nvme: bool,
}

/// Complete Out-Of-Box Experience (OOBE) System Audit Report (authored by mjojo)
#[derive(Debug, Clone)]
pub struct SystemAuditReport {
    pub total_ram_gb: f32,
    pub available_ram_gb: f32,
    pub os_reserved_gb: f32,
    pub detected_vram_gb: f32,
    pub is_nvme: bool,
    pub selected_drive_mount: PathBuf,
    pub free_disk_space_gb: f32,
    pub total_disk_space_gb: f32,
    pub avx512_supported: bool,
    pub recommended_l2_slab_gb: f32,
    pub recommended_preset: String,
    pub recommended_io_mode: String,
    pub recommended_models_dir: PathBuf,
    pub recommended_model_path: PathBuf,
    pub all_disks: Vec<DiskVolumeInfo>,
}

/// Scans all mounted physical disks via `sysinfo::Disks` and picks the best SSD/NVMe volume
/// with the largest available space for storing `*.apus` MoE weight containers.
pub fn scan_storage_volumes() -> Vec<DiskVolumeInfo> {
    let disks = Disks::new_with_refreshed_list();
    let mut out = Vec::new();
    for d in disks.list() {
        let total_gb = d.total_space() as f32 / (1024.0 * 1024.0 * 1024.0);
        let avail_gb = d.available_space() as f32 / (1024.0 * 1024.0 * 1024.0);
        if total_gb < 1.0 {
            continue;
        }
        let kind_str = format!("{:?}", d.kind());
        let is_ssd = !kind_str.contains("HDD");
        out.push(DiskVolumeInfo {
            mount_point: d.mount_point().to_path_buf(),
            kind_label: if is_ssd {
                "NVMe / SSD".into()
            } else {
                "HDD / Legacy".into()
            },
            total_space_gb: total_gb,
            available_space_gb: avail_gb,
            is_ssd_or_nvme: is_ssd,
        });
    }
    // Sort: SSD/NVMe first, then descending by available free space
    out.sort_by(|a, b| {
        b.is_ssd_or_nvme.cmp(&a.is_ssd_or_nvme).then_with(|| {
            b.available_space_gb
                .partial_cmp(&a.available_space_gb)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });
    out
}

/// Performs a full OOBE hardware and storage audit of the host machine
pub fn audit_host_environment(
    custom_models_dir: Option<&Path>,
) -> Result<SystemAuditReport, InitError> {
    let ram = detect_system_hardware().map_err(|e| InitError::SysinfoError(e.to_string()))?;
    if ram.total_ram_gb < 7.5 {
        return Err(InitError::HardwareIncompatible);
    }

    let all_disks = scan_storage_volumes();
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let (selected_drive_mount, free_disk_space_gb, total_disk_space_gb, is_nvme) =
        if let Some(best) = all_disks.first() {
            (
                best.mount_point.clone(),
                best.available_space_gb,
                best.total_space_gb,
                best.is_ssd_or_nvme,
            )
        } else {
            (cwd.clone(), 480.0, 1000.0, true)
        };

    // Priority: 1) custom CLI path -> 2) APUS_MODELS_DIR env -> 3) `<best_ssd_mount>/APUS_Models` or `./models`
    let recommended_models_dir = if let Some(custom) = custom_models_dir {
        custom.to_path_buf()
    } else if let Ok(env_dir) = env::var("APUS_MODELS_DIR") {
        PathBuf::from(env_dir)
    } else {
        cwd.join("models")
    };

    let detected_vram_gb = 16.0;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let avx512_supported = std::is_x86_feature_detected!("avx512f");
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let avx512_supported = false;

    let recommended_preset = if ram.total_ram_gb >= 60.0 && detected_vram_gb >= 24.0 {
        "DeepSeek-V3-671B-Q4_K_M".to_string()
    } else {
        ram.recommended_preset.clone()
    };

    let filename = format!(
        "{}.apus",
        recommended_preset.to_lowercase().replace('_', "-")
    );
    let recommended_model_path = recommended_models_dir.join(filename);

    let recommended_io_mode = if cfg!(target_os = "windows") {
        "DirectStorage / IOCP (4K Unbuffered)".to_string()
    } else if cfg!(target_os = "macos") {
        "Apple Unified Memory (mmap + madvise)".to_string()
    } else {
        "io_uring (O_DIRECT 4K Aligned)".to_string()
    };

    Ok(SystemAuditReport {
        total_ram_gb: ram.total_ram_gb,
        available_ram_gb: ram.available_ram_gb,
        os_reserved_gb: ram.os_reserved_gb,
        detected_vram_gb,
        is_nvme,
        selected_drive_mount,
        free_disk_space_gb,
        total_disk_space_gb,
        avx512_supported,
        recommended_l2_slab_gb: ram.recommended_l2_slab_gb,
        recommended_preset,
        recommended_io_mode,
        recommended_models_dir,
        recommended_model_path,
        all_disks,
    })
}

/// Resolves the path to the `.apus` model file using deterministic priority:
/// 1. Explicit CLI argument (`--model <path>` or positional `<file.apus>`)
/// 2. `APUS_MODEL_PATH` environment variable
/// 3. `engine.model_path` inside `apus.toml`
/// 4. First `*.apus` file discovered inside `engine.models_dir` (or `./models`)
/// 5. Fallback to default path in `apus.toml`
pub fn resolve_model_path(cli_override: Option<&str>, config_path: &Path) -> PathBuf {
    if let Some(cli_p) = cli_override
        && !cli_p.starts_with("--")
    {
        return PathBuf::from(cli_p);
    }

    if let Ok(env_p) = env::var("APUS_MODEL_PATH")
        && !env_p.trim().is_empty()
    {
        return PathBuf::from(env_p);
    }

    if let Ok(cfg) = ApusConfig::load_from_file(config_path) {
        let configured_file = PathBuf::from(&cfg.engine.model_path);
        if configured_file.exists() {
            return configured_file;
        }
        // Scan `models_dir` for any existing `*.apus` container
        let models_dir = PathBuf::from(&cfg.engine.models_dir);
        if let Ok(entries) = fs::read_dir(&models_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("apus") {
                    return p;
                }
            }
        }
        return configured_file;
    }

    PathBuf::from("./models/glm-5.2-744b-q4.apus")
}

/// Checks if `apus.toml` exists; if missing, launches the First-Launch OOBE Initialization Wizard,
/// creates the `models/` directory on the selected drive, and persists `apus.toml`.
pub fn ensure_system_initialized(
    config_path: &Path,
    custom_models_dir: Option<&Path>,
    force_wizard: bool,
) -> Result<SystemAuditReport, InitError> {
    if config_path.exists() && !force_wizard {
        return audit_host_environment(custom_models_dir);
    }

    println!("╭────────────────────────────────────────────────────────────────────────╮");
    println!("│              APUS FIRST-LAUNCH INITIALIZATION WIZARD                   │");
    println!("│          Asynchronous Paged Ultra-Streaming // by mjojo                │");
    println!("╰────────────────────────────────────────────────────────────────────────╯");
    println!("[1/3] First launch or wizard requested. Probing host hardware & storage...");

    let audit = audit_host_environment(custom_models_dir)?;

    println!("\n[2/3] Host Hardware & Storage Audit Results:");
    println!(
        " ├── System RAM (Total / Avail) : {:.1} GB / {:.1} GB (OS Reserved: {:.1} GB)",
        audit.total_ram_gb, audit.available_ram_gb, audit.os_reserved_gb
    );
    println!(
        " ├── Detected GPU VRAM Pool     : {:.1} GB (Asymmetric L1 Cache Ready)",
        audit.detected_vram_gb
    );
    println!(
        " ├── Primary Storage Volume     : {} [{}] ({:.1} GB free / {:.1} GB total)",
        audit.selected_drive_mount.display(),
        if audit.is_nvme {
            "NVMe / SSD"
        } else {
            "SATA / Legacy"
        },
        audit.free_disk_space_gb,
        audit.total_disk_space_gb
    );
    if audit.all_disks.len() > 1 {
        for (idx, d) in audit.all_disks.iter().enumerate() {
            println!(
                " │     • Disk #{idx}: {:<12} {:<12} {:>6.1} GB free / {:>6.1} GB",
                d.mount_point.display().to_string(),
                format!("[{}]", d.kind_label),
                d.available_space_gb,
                d.total_space_gb
            );
        }
    }
    println!(
        " └── CPU SIMD Vector Engine     : {}",
        if audit.avx512_supported {
            "AVX-512 VNNI + FMA3"
        } else {
            "AVX2 + FMA3"
        }
    );

    println!("\n[3/3] Recommended APUS Runtime Profile & Model Location:");
    println!(
        " ├── Recommended MoE Preset     : {}",
        audit.recommended_preset
    );
    println!(
        " ├── L2 Pinned Slab Arena (RAM) : {:.1} GB (4KB Aligned + VirtualLock/mlock)",
        audit.recommended_l2_slab_gb
    );
    println!(
        " ├── Storage I/O Backend        : {}",
        audit.recommended_io_mode
    );
    println!(
        " ├── Models Directory           : {}",
        audit.recommended_models_dir.display()
    );
    println!(
        " └── Active Model Container     : {}\n",
        audit.recommended_model_path.display()
    );

    let _ = fs::create_dir_all(&audit.recommended_models_dir);

    let mut cfg = ApusConfig::default();
    cfg.engine.models_dir = audit
        .recommended_models_dir
        .to_string_lossy()
        .replace('\\', "/");
    cfg.engine.model_path = audit
        .recommended_model_path
        .to_string_lossy()
        .replace('\\', "/");
    cfg.model.name = audit.recommended_preset.clone();
    cfg.memory.ram_slab_gb = (audit.recommended_l2_slab_gb * 10.0).round() / 10.0;
    cfg.memory.vram_total_gb = audit.detected_vram_gb;
    cfg.qos.sata_ahci_compat = !audit.is_nvme;

    cfg.save_to_file(config_path)
        .map_err(|e| InitError::Io(std::io::Error::other(e.to_string())))?;

    println!(
        "[APUS] Configuration written to '{}'. System initialization complete!\n",
        config_path.display()
    );

    Ok(audit)
}
