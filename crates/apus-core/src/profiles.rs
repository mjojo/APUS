use serde::{Deserialize, Serialize};

/// Pre-configured architectural profile for flagship open-weights MoE LLMs (Sprint 32 `apus-release`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoEModelPreset {
    pub model_id: &'static str,
    pub total_params_b: f32,
    pub active_params_b: f32,
    pub total_layers: u32,
    pub routed_experts_per_layer: u32,
    pub shared_experts_per_layer: u32,
    pub top_k_active: usize,
    pub attention_type: &'static str,
    pub has_native_mtp: bool,
    pub disk_size_int4_gb: f32,
    pub min_ram_slab_gb: f32,
}

impl MoEModelPreset {
    /// Returns the built-in catalog of supported frontier MoE architectures
    pub fn catalog() -> &'static [MoEModelPreset] {
        &[
            MoEModelPreset {
                model_id: "GLM-5.2-744B-Q4_K_M",
                total_params_b: 744.0,
                active_params_b: 40.0,
                total_layers: 75,
                routed_experts_per_layer: 288,
                shared_experts_per_layer: 2,
                top_k_active: 8,
                attention_type: "MLA (576d Latent KV)",
                has_native_mtp: true,
                disk_size_int4_gb: 346.4,
                min_ram_slab_gb: 24.0,
            },
            MoEModelPreset {
                model_id: "DeepSeek-V3-671B-Q4_K_M",
                total_params_b: 671.0,
                active_params_b: 37.0,
                total_layers: 61,
                routed_experts_per_layer: 256,
                shared_experts_per_layer: 1,
                top_k_active: 8,
                attention_type: "MLA (576d Latent KV)",
                has_native_mtp: true,
                disk_size_int4_gb: 312.0,
                min_ram_slab_gb: 20.0,
            },
            MoEModelPreset {
                model_id: "Qwen3-235B-A22B-Q4_K_M",
                total_params_b: 235.0,
                active_params_b: 22.0,
                total_layers: 94,
                routed_experts_per_layer: 128,
                shared_experts_per_layer: 0,
                top_k_active: 8,
                attention_type: "GQA (PagedAttention)",
                has_native_mtp: false,
                disk_size_int4_gb: 118.5,
                min_ram_slab_gb: 16.0,
            },
            MoEModelPreset {
                model_id: "Mixtral-8x22B-Instruct-Q4_K_M",
                total_params_b: 141.0,
                active_params_b: 39.0,
                total_layers: 56,
                routed_experts_per_layer: 8,
                shared_experts_per_layer: 0,
                top_k_active: 2,
                attention_type: "GQA (PagedAttention)",
                has_native_mtp: false,
                disk_size_int4_gb: 74.2,
                min_ram_slab_gb: 16.0,
            },
        ]
    }
}
