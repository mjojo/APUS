pub mod converter;
pub mod format;
pub mod imatrix;

pub use converter::{align_to_page, pad_file_to, ApusConverter, ConverterError, SourceExpert};
pub use format::{
    ApusFileHeader, FormatError, LbaExpertEntry, RoostSessionHeader, RoostSnapshot, APUS_MAGIC,
    AUTHOR_SIGNATURE, ROOST_AUTHOR_SIGNATURE, ROOST_MAGIC,
};
pub use imatrix::{ImatrixCalibrator, QuantizerInt4, QUANT_GROUP_SIZE};

#[cfg(test)]
mod tests {
    use super::*;
    use apus_core::ExpertId;
    use std::fs::File;

    #[test]
    fn test_apus_converter_and_header_verification() {
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("test_mjojo_model.apus");

        let mut converter = ApusConverter::new("GLM-5.2-744B-TEST", 2, 2, vec![0x55; 1024]);
        for layer in 0..2 {
            for exp in 0..2 {
                converter.push_expert(SourceExpert {
                    layer_id: layer,
                    expert_id: exp,
                    raw_quantized_weights: vec![(layer * 2 + exp + 1) as u8; 5000],
                });
            }
        }

        let root_hex = converter.build(&path).expect("build failed");
        assert_eq!(root_hex.len(), 64);

        let mut f = File::open(&path).expect("open failed");
        let header = ApusFileHeader::read_from(&mut f).expect("header read failed");
        assert_eq!(header.model_id_str(), "GLM-5.2-744B-TEST");
        assert_eq!(header.author_str(), "apus-moe // authored by mjojo");
        let total_layers = header.total_layers;
        let total_experts = header.total_experts;
        assert_eq!(total_layers, 2);
        assert_eq!(total_experts, 4);

        let lba = header.read_lba_table(&mut f).expect("lba read failed");
        assert_eq!(lba.len(), 4);
        for entry in &lba {
            let offset = entry.file_offset;
            let size = entry.compressed_size;
            assert_eq!(offset % 4096, 0, "Offset must be 4KB page-aligned");
            assert_eq!(size % 4096, 0, "Size must be 4KB page-aligned");
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_roost_snapshot_roundtrip() {
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("test_session.roost");

        let snap = RoostSnapshot {
            timestamp: 1_761_318_000,
            context_tokens: 48_200,
            warm_experts: vec![
                ExpertId { layer: 0, expert: 7 },
                ExpertId { layer: 42, expert: 128 },
            ],
            compressed_mla_kv: vec![1, 2, 3, 4, 5, 6, 7, 8],
        };
        snap.save_to_file(&path).expect("save roost failed");
        let loaded = RoostSnapshot::load_from_file(&path).expect("load roost failed");
        assert_eq!(loaded.context_tokens, 48_200);
        assert_eq!(loaded.warm_experts.len(), 2);
        assert_eq!(loaded.warm_experts[1], ExpertId { layer: 42, expert: 128 });
        assert_eq!(loaded.compressed_mla_kv, vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_int4_quantize_dequantize_accuracy() {
        let original: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.1).collect();
        let packed = QuantizerInt4::quantize(&original, None);
        let unpacked = QuantizerInt4::dequantize(&packed, original.len());
        assert_eq!(unpacked.len(), original.len());
        for (a, b) in original.iter().zip(unpacked.iter()) {
            assert!((a - b).abs() < 0.35, "mismatch: {a} vs {b}");
        }
    }
}
