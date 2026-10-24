use crate::ApusError;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApusConfig {
    pub engine: EngineSection,
    pub model: ModelSection,
    pub memory: MemorySection,
    pub qos: QosSection,
    pub thermal: ThermalSection,
    pub topology: TopologySection,
    pub server: ServerSection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineSection {
    pub author: String,
    #[serde(default = "default_models_dir")]
    pub models_dir: String,
    pub model_path: String,
    pub session_roost_path: String,
    pub storage_backend: String,
    pub quant_format: String,
}

fn default_models_dir() -> String {
    "./models".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSection {
    pub name: String,
    pub total_layers: u32,
    pub total_experts: u32,
    pub experts_per_layer: u32,
    pub active_experts_per_tok: usize,
    pub shared_experts_per_layer: usize,
    pub hidden_size: usize,
    pub mla_latent_dim: usize,
    pub mtp_heads: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySection {
    pub vram_total_gb: f32,
    pub vram_dense_reserved_gb: f32,
    pub vram_l1_cache_gb: f32,
    pub ram_slab_gb: f32,
    pub expert_slot_size_bytes: usize,
    pub lock_pages_in_ram: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QosSection {
    pub os_drive_mode: bool,
    pub io_priority: String,
    pub max_bandwidth_gbs: f32,
    #[serde(alias = "max_nvme_Potential_gbs", default = "default_nvme_potential")]
    pub max_nvme_potential_gbs: f32,
    pub max_queue_depth: usize,
    pub sata_ahci_compat: bool,
}

fn default_nvme_potential() -> f32 {
    7.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalSection {
    pub enabled: bool,
    pub target_temp_c: u8,
    pub yellow_zone_temp_c: u8,
    pub red_zone_temp_c: u8,
    pub poll_interval_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologySection {
    pub num_gpus: usize,
    pub pipeline_split_layer: u16,
    pub enable_p2p_dma: bool,
    pub pinned_host_fallback: bool,
    pub numa_node: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSection {
    pub host: String,
    pub port: u16,
    pub max_batch_size: usize,
    pub continuous_batching: bool,
}

impl Default for ApusConfig {
    fn default() -> Self {
        Self {
            engine: EngineSection {
                author: "mjojo".into(),
                models_dir: "./models".into(),
                model_path: "./models/glm-5.2-744b-q4.apus".into(),
                session_roost_path: "coding_rust.roost".into(),
                storage_backend: "block".into(),
                quant_format: "int4_imatrix".into(),
            },
            model: ModelSection {
                name: "GLM-5.2-744B-Q4_K_M".into(),
                total_layers: 75,
                total_experts: 21_504,
                experts_per_layer: 288,
                active_experts_per_tok: 8,
                shared_experts_per_layer: 2,
                hidden_size: 4096,
                mla_latent_dim: 576,
                mtp_heads: 2,
            },
            memory: MemorySection {
                vram_total_gb: 16.0,
                vram_dense_reserved_gb: 10.0,
                vram_l1_cache_gb: 4.2,
                ram_slab_gb: 24.0,
                expert_slot_size_bytes: 19 * 1024 * 1024,
                lock_pages_in_ram: true,
            },
            qos: QosSection {
                os_drive_mode: true,
                io_priority: "idle".into(),
                max_bandwidth_gbs: 5.5,
                max_nvme_potential_gbs: 7.0,
                max_queue_depth: 32,
                sata_ahci_compat: false,
            },
            thermal: ThermalSection {
                enabled: true,
                target_temp_c: 60,
                yellow_zone_temp_c: 65,
                red_zone_temp_c: 70,
                poll_interval_ms: 500,
            },
            topology: TopologySection {
                num_gpus: 2,
                pipeline_split_layer: 37,
                enable_p2p_dma: true,
                pinned_host_fallback: true,
                numa_node: 0,
            },
            server: ServerSection {
                host: "127.0.0.1".into(),
                port: 8080,
                max_batch_size: 8,
                continuous_batching: true,
            },
        }
    }
}

impl ApusConfig {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, ApusError> {
        let content = fs::read_to_string(path)
            .map_err(|e| ApusError::AllocationFailed(format!("Failed to read apus.toml: {e}")))?;
        toml::from_str(&content)
            .map_err(|e| ApusError::AllocationFailed(format!("Failed to parse apus.toml: {e}")))
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ApusError> {
        let content = toml::to_string_pretty(self).map_err(|e| {
            ApusError::AllocationFailed(format!("Failed to serialize apus.toml: {e}"))
        })?;
        fs::write(path, content)
            .map_err(|e| ApusError::AllocationFailed(format!("Failed to write apus.toml: {e}")))
    }
}
