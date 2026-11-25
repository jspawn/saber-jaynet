use std::path::Path;

use saber_model::qwen3_5_moe::spike::{kernel_coverage, layer_tensors, Coverage, WeightClass};
use saber_model::qwen3_5_moe::Qwen35MoeConfig;
use saber_model::qwen4_exp::st::ShardSet;

const OCCAMY: &str = include_str!("data/qwen3_5_moe_occamy.config.json");

fn config() -> Qwen35MoeConfig {
    Qwen35MoeConfig::from_config_json(OCCAMY).unwrap()
}

#[test]
fn gdn_layer_manifest_covers_every_weight() {
    let tensors = layer_tensors(&config(), 0).unwrap();
    assert_eq!(tensors.len(), 784);
    assert_eq!(
        tensors
            .iter()
            .filter(|tensor| tensor.class == WeightClass::RoutedExpert)
            .count(),
        256 * 3
    );
    let qkv = tensors
        .iter()
        .find(|tensor| tensor.name.ends_with("linear_attn.in_proj_qkv.weight"))
        .unwrap();
    assert_eq!(qkv.shape, vec![8192, 2048]);
    let out = tensors
        .iter()
        .find(|tensor| tensor.name.ends_with("linear_attn.out_proj.weight"))
        .unwrap();
    assert_eq!(out.shape, vec![2048, 4096]);
}

#[test]
fn full_attention_layer_manifest_covers_every_weight() {
    let tensors = layer_tensors(&config(), 3).unwrap();
    assert_eq!(tensors.len(), 781);
    for (suffix, shape) in [
        ("self_attn.q_proj.weight", vec![8192, 2048]),
        ("self_attn.k_proj.weight", vec![512, 2048]),
        ("self_attn.v_proj.weight", vec![512, 2048]),
        ("self_attn.o_proj.weight", vec![2048, 4096]),
        ("self_attn.q_norm.weight", vec![256]),
        ("self_attn.k_norm.weight", vec![256]),
    ] {
        let tensor = tensors
            .iter()
            .find(|tensor| tensor.name.ends_with(suffix))
            .unwrap();
        assert_eq!(tensor.shape, shape, "{suffix}");
    }
}

#[test]
fn one_layer_manifests_match_the_real_safetensors_headers() {
    let root = Path::new("/srv/models/radconv/occamy-1.0");
    if !root.join("model.safetensors.index.json").exists() {
        eprintln!("SKIP: Occamy checkpoint absent");
        return;
    }
    let mut shards = ShardSet::open(root).unwrap();
    for layer in [0, 3] {
        for tensor in layer_tensors(&config(), layer).unwrap() {
            let info = shards
                .info(&tensor.name)
                .unwrap_or_else(|| panic!("{}: missing from the checkpoint index", tensor.name));
            assert_eq!(info.dtype, tensor.dtype, "{}", tensor.name);
            let expected_shape: Vec<u64> = tensor.shape.iter().map(|&dim| u64::from(dim)).collect();
            assert_eq!(info.shape, expected_shape, "{}", tensor.name);
        }
    }
}

#[test]
fn gdn_kernels_cover_occamy_at_tp2() {
    let probes = kernel_coverage(&config()).unwrap();
    for probe in probes
        .iter()
        .filter(|probe| probe.operation.starts_with("gdn."))
    {
        match &probe.coverage {
            Coverage::Existing(symbol) => {
                assert!(symbol.starts_with("saber_"), "{}", probe.operation)
            }
            Coverage::Missing(reason) => panic!("{}: {reason}", probe.operation),
        }
    }
}

#[test]
fn attention_uses_the_occamy_gqa8_instantiation() {
    let probes = kernel_coverage(&config()).unwrap();
    for operation in ["attention.prefill", "attention.decode"] {
        let probe = probes
            .iter()
            .find(|probe| probe.operation == operation)
            .unwrap();
        match &probe.coverage {
            Coverage::Existing(symbol) => {
                assert!(symbol.contains("gqa8_bf16kv"), "{symbol}")
            }
            Coverage::Missing(reason) => panic!("{operation}: {reason}"),
        }
    }
}

#[test]
fn moe_needs_occamy_geometry_variants() {
    let probes = kernel_coverage(&config()).unwrap();
    for probe in probes
        .iter()
        .filter(|probe| probe.operation.starts_with("moe."))
    {
        match &probe.coverage {
            Coverage::Missing(reason) => {
                assert!(
                    reason.contains("e256 top-k8 hidden2048 intermediate512"),
                    "{reason}"
                )
            }
            Coverage::Existing(_) => panic!("{} unexpectedly reused Flash-Next", probe.operation),
        }
    }
}

#[test]
fn manifest_refuses_a_layer_past_the_model() {
    let error = layer_tensors(&config(), 40).unwrap_err();
    assert!(error.contains("outside"), "{error}");
}
