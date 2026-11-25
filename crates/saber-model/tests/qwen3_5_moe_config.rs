use std::path::Path;

use saber_model::qwen3_5_moe::{LayerKind, QuantConfig, Qwen35MoeConfig};
use serde_json::Value;

const OCCAMY: &str = include_str!("data/qwen3_5_moe_occamy.config.json");

fn parse(source: &str) -> Qwen35MoeConfig {
    Qwen35MoeConfig::from_config_json(source).expect("the Occamy config parses")
}

fn doctor(f: impl FnOnce(&mut Value)) -> String {
    let mut value: Value = serde_json::from_str(OCCAMY).unwrap();
    f(&mut value);
    match Qwen35MoeConfig::from_config_json(&value.to_string()) {
        Ok(_) => panic!("a doctored config was ACCEPTED"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn fixture_is_the_local_checkpoint_config() {
    let path = Path::new("/srv/models/radconv/occamy-1.0/config.json");
    if !path.exists() {
        eprintln!("SKIP: {} absent; the fixture stands alone", path.display());
        return;
    }
    assert_eq!(std::fs::read_to_string(path).unwrap(), OCCAMY);
}

#[test]
fn occamy_config_names_every_text_forward_fact() {
    let config = parse(OCCAMY);
    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.num_layers(), 40);
    let full: Vec<u32> = (0..40)
        .filter(|&index| config.layers[index as usize] == LayerKind::FullAttention)
        .collect();
    assert_eq!(full, (0..10).map(|index| 4 * index + 3).collect::<Vec<_>>());
    assert_eq!(
        config
            .layers
            .iter()
            .filter(|&&kind| kind == LayerKind::Gdn)
            .count(),
        30
    );
    assert_eq!(config.full_attention_interval, 4);
    assert_eq!(config.vocab_size, 248_320);
    assert!(!config.tie_word_embeddings);
    assert_eq!(config.rms_norm_eps, 1e-6);
    assert_eq!(config.rms_norm_weight_add(), 1.0);
    assert_eq!(config.gdn_norm_weight_add(), 0.0);
    assert_eq!(
        (config.bos_token_id, config.eos_token_id),
        (248_044, 248_044)
    );
    assert_eq!(config.quant, QuantConfig::Bf16);
}

#[test]
fn attention_and_rope_match_the_checkpoint_shapes() {
    let config = parse(OCCAMY);
    assert_eq!(
        (
            config.attention.num_heads,
            config.attention.num_kv_heads,
            config.attention.head_dim
        ),
        (16, 2, 256)
    );
    assert!(config.attention.output_gate_sigmoid);
    assert!(!config.attention.bias);
    assert_eq!(config.attention.max_position_embeddings, 262_144);
    assert_eq!(config.q_proj_rows(), 8192);
    assert_eq!(config.kv_proj_rows(), 512);
    assert_eq!(config.rope.rotary_dim(256), 64);
    assert_eq!(config.rope.mrope_section, vec![11, 11, 10]);
    assert!(config.rope.mrope_interleaved);
    assert_eq!(config.rope.theta, 1e7);
}

#[test]
fn gdn_and_moe_match_the_checkpoint_shapes() {
    let config = parse(OCCAMY);
    assert_eq!(
        (
            config.gdn.num_key_heads,
            config.gdn.num_value_heads,
            config.gdn.key_head_dim,
            config.gdn.value_head_dim,
            config.gdn.conv_kernel
        ),
        (16, 32, 128, 128, 4)
    );
    assert_eq!(config.gdn.qkv_rows(), 8192);
    assert_eq!(config.gdn.key_dim(), 2048);
    assert_eq!(config.gdn.value_dim(), 4096);
    assert!(config.gdn.ssm_dtype_f32);
    assert_eq!(
        (
            config.moe.num_experts,
            config.moe.top_k,
            config.moe.intermediate,
            config.moe.shared_intermediate
        ),
        (256, 8, 512, 512)
    );
    assert!(config.moe.norm_topk_prob);
    assert!(config.moe.act_silu);
    config.validate_tp2().unwrap();
}

#[test]
fn mtp_is_off_and_vision_is_described_but_not_enabled() {
    let config = parse(OCCAMY);
    assert_eq!(config.mtp.num_layers, 0);
    assert!(!config.mtp.dedicated_embeddings);
    assert_eq!(config.vision.depth, 27);
    assert_eq!(config.vision.hidden_size, 1152);
    assert_eq!(config.vision.out_hidden_size, 2048);
    assert!(config.vision.deepstack_visual_indexes.is_empty());
    assert_eq!(config.vision_tokens, [248_056, 248_057, 248_053, 248_054]);
}

#[test]
fn refuses_unknown_and_missing_fields_by_path() {
    let error = doctor(|value| {
        value["text_config"]["hc_count"] = Value::from(4);
    });
    assert!(error.contains("text_config.hc_count"), "{error}");
    assert!(error.contains("unknown field"), "{error}");

    let error = doctor(|value| {
        value["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("linear_num_value_heads");
    });
    assert!(
        error.contains("text_config.linear_num_value_heads"),
        "{error}"
    );
    assert!(error.contains("missing"), "{error}");
}

#[test]
fn refuses_a_layer_schedule_that_disagrees_with_the_interval() {
    let error = doctor(|value| {
        value["text_config"]["layer_types"][3] = Value::from("linear_attention");
    });
    assert!(error.contains("text_config.layer_types[3]"), "{error}");
    assert!(error.contains("full_attention_interval"), "{error}");
}

#[test]
fn refuses_inconsistent_rope_geometry() {
    let error = doctor(|value| {
        value["text_config"]["rope_parameters"]["mrope_section"] = serde_json::json!([11, 11, 11]);
    });
    assert!(error.contains("mrope_section"), "{error}");

    let error = doctor(|value| {
        value["text_config"]["partial_rotary_factor"] = Value::from(0.5);
    });
    assert!(error.contains("partial_rotary_factor"), "{error}");
}

#[test]
fn refuses_mtp_and_unsupported_quantization() {
    let error = doctor(|value| {
        value["text_config"]["mtp_num_hidden_layers"] = Value::from(1);
    });
    assert!(error.contains("text_config MTP"), "{error}");

    let error = doctor(|value| {
        value["quantization_config"] = serde_json::json!({"quant_method": "fp8"});
    });
    assert!(error.contains("quantization_config"), "{error}");
}

#[test]
fn accepts_the_official_block128_fp8_config() {
    let mut value: Value = serde_json::from_str(OCCAMY).unwrap();
    value["quantization_config"] = serde_json::json!({
        "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
        "weight_block_size": [128, 128], "modules_to_not_convert": ["lm_head"]
    });
    let config = Qwen35MoeConfig::from_config_json(&value.to_string()).unwrap();
    assert_eq!(config.quant, QuantConfig::Fp8Block128);

    value["quantization_config"]["weight_block_size"] = serde_json::json!([64, 128]);
    let error = Qwen35MoeConfig::from_config_json(&value.to_string()).unwrap_err();
    assert!(error.to_string().contains("weight_block_size"));
}

#[test]
fn refuses_gdn_geometry_the_initial_kernels_do_not_cover() {
    let error = doctor(|value| {
        value["text_config"]["linear_value_head_dim"] = Value::from(64);
    });
    assert!(error.contains("GDN geometry"), "{error}");
}

#[test]
fn refuses_attention_geometry_that_cannot_split_over_two_ranks() {
    let error = doctor(|value| {
        value["text_config"]["num_attention_heads"] = Value::from(15);
        value["text_config"]["num_key_value_heads"] = Value::from(3);
    });
    assert!(error.contains("TP2"), "{error}");
}
