use std::collections::BTreeSet;
use std::path::Path;

use saber_model::graph::{DeviceFinalRole, Dtype, ShardKind};
use saber_model::kolibri::{
    spike::kernel_coverage, KolibriConfig, KolibriLayerPackPlan, LayerKind,
};
use serde_json::{json, Value};

const CHECKPOINT: &str = "/srv/models/saber-models/models/Kolibri-1";

fn config_json() -> Value {
    json!({
        "architectures": ["Kolibri1ForCausalLM"],
        "model_type": "kolibri1",
        "hidden_size": 256,
        "num_hidden_layers": 5,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "hidden_act": "silu",
        "max_position_embeddings": 1024,
        "rms_norm_eps": 1e-6,
        "vocab_size": 512,
        "rope_theta": 10000.0,
        "num_experts": 8,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 256,
        "shared_expert_intermediate_size": 256,
        "norm_topk_prob": false,
        "attention_bias": false,
        "attention_dropout": 0.0,
        "tie_word_embeddings": false,
        "transformers_version": "5.20.0",
        "use_cache": true,
        "use_sliding_window": true,
        "sliding_window": 513,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        "bos_token_id": null,
        "eos_token_id": 500,
        "pad_token_id": 501,
        "dtype": "bfloat16",
        "head_dtype": "float32",
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["lm_head", "model.layers.0.mlp.gate", "model.layers.1.mlp.gate", "model.layers.2.mlp.gate", "model.layers.3.mlp.gate", "model.layers.4.mlp.gate"]
        }
    })
}

fn config() -> KolibriConfig {
    KolibriConfig::from_config_json(&config_json().to_string()).unwrap()
}

fn plan(tp: u32) -> KolibriLayerPackPlan {
    KolibriLayerPackPlan::new(&config(), tp, 0).unwrap()
}

#[test]
fn names_attention_moe_and_norm_facts() {
    let cfg = config();
    assert_eq!(cfg.q_proj_rows(), 512);
    assert_eq!(cfg.kv_proj_rows(), 256);
    assert_eq!(cfg.full_attention_layers(), 1);
    assert!(cfg.uses_rope(0));
    assert!(!cfg.uses_rope(4));
    assert_eq!(cfg.layer_kind(4), LayerKind::FullAttention);
    assert_eq!(cfg.kv_cache_bytes_total(1024, 1), 1_574_912);
    cfg.validate_tp(2).unwrap();
}

#[test]
fn real_config_has_native_256k_and_ten_global_layers() {
    let path = Path::new(CHECKPOINT).join("config.json");
    if !path.exists() {
        eprintln!("SKIP: {} absent", path.display());
        return;
    }
    let cfg = KolibriConfig::from_config_json(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(cfg.num_hidden_layers, 50);
    assert_eq!(cfg.full_attention_layers(), 10);
    assert_eq!(cfg.max_position_embeddings, 262_144);
    assert_eq!(cfg.kv_cache_bytes_total(262_144, 1), 2_705_367_040);
    cfg.validate_tp(2).unwrap();
}

#[test]
fn layer_plan_preserves_router_and_splits_every_expert() {
    let plan = plan(2);
    assert_eq!(plan.records.len(), 66);
    let names: BTreeSet<_> = plan
        .records
        .iter()
        .map(|record| record.name.as_str())
        .collect();
    assert_eq!(names.len(), plan.records.len());
    let q = plan
        .records
        .iter()
        .find(|record| record.name == "model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(q.dtype, Dtype::Bf16);
    assert_eq!(q.layout_id, 12);
    assert_eq!(q.shard, ShardKind::Dim0);
    assert_eq!(
        q.sources[0].scale.as_deref(),
        Some("model.layers.0.self_attn.q_proj.weight_scale_inv")
    );
    let gate_up = plan
        .records
        .iter()
        .find(|record| record.name == "layer.0.expert.3.gate_up.w")
        .unwrap();
    assert_eq!(gate_up.shape, [256, 256]);
    assert_eq!(gate_up.sources.len(), 2);
    let down = plan
        .records
        .iter()
        .find(|record| record.name == "layer.0.expert.3.down.ws")
        .unwrap();
    assert_eq!(down.role, DeviceFinalRole::BlockScale);
    assert_eq!(down.shape, [4, 256]);
    let bias = plan
        .records
        .iter()
        .find(|record| record.name == "model.layers.0.moe.router.expert_bias")
        .unwrap();
    assert_eq!(bias.dtype, Dtype::F32);
    assert_eq!(bias.layout_id, 5);
}

#[test]
fn real_layer_zero_index_is_covered_exactly() {
    let path = Path::new(CHECKPOINT).join("model.safetensors.index.json");
    if !path.exists() {
        eprintln!("SKIP: {} absent", path.display());
        return;
    }
    let cfg = KolibriConfig::from_config_json(
        &std::fs::read_to_string(Path::new(CHECKPOINT).join("config.json")).unwrap(),
    )
    .unwrap();
    let index: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let names: Vec<_> = index["weight_map"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .filter(|name| name.starts_with("model.layers.0."))
        .collect();
    KolibriLayerPackPlan::new(&cfg, 2, 0)
        .unwrap()
        .validate_tensor_names(names)
        .unwrap();
}

#[test]
fn rejects_wrong_router_normalization_and_attention_schedule() {
    let mut value = config_json();
    value["norm_topk_prob"] = json!(true);
    assert!(KolibriConfig::from_config_json(&value.to_string())
        .unwrap_err()
        .to_string()
        .contains("routing"));
    let mut value = config_json();
    value["layer_types"][3] = json!("full_attention");
    assert!(KolibriConfig::from_config_json(&value.to_string())
        .unwrap_err()
        .to_string()
        .contains("4:1"));
}

#[test]
fn stock_kernel_coverage_is_explicit_about_new_glue() {
    let probes = kernel_coverage(&config(), 2).unwrap();
    assert!(
        probes
            .iter()
            .filter(|probe| probe.operation.starts_with("bf16.")
                || probe.operation.starts_with("mxfp4.")
                || probe.operation.starts_with("attention."))
            .all(|probe| probe.coverage.is_ok()),
        "{probes:#?}"
    );
    assert!(probes.iter().any(
        |probe| probe.operation == "router.sigmoid_top6_correction_bias" && probe.coverage.is_err()
    ));
}

#[test]
fn full_plan_has_unique_records_and_exact_index_coverage() {
    let cfg = config();
    let plan = saber_model::kolibri::KolibriPackPlan::new(&cfg, 2).unwrap();
    assert_eq!(plan.records.len(), 5 * 66 + 3);
    let names: BTreeSet<_> = plan
        .records
        .iter()
        .map(|record| record.name.as_str())
        .collect();
    assert_eq!(names.len(), plan.records.len());

    let path = Path::new(CHECKPOINT).join("model.safetensors.index.json");
    if !path.exists() {
        eprintln!("SKIP: {} absent", path.display());
        return;
    }
    let cfg = KolibriConfig::from_config_json(
        &std::fs::read_to_string(Path::new(CHECKPOINT).join("config.json")).unwrap(),
    )
    .unwrap();
    let index: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let names: Vec<_> = index["weight_map"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    saber_model::kolibri::KolibriPackPlan::new(&cfg, 2)
        .unwrap()
        .validate_tensor_names(names)
        .unwrap();
}
