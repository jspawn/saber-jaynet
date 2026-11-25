use saber_model::ripple::{LayerKind, RippleConfig};
use serde_json::Value;

const NVFP4: &str = include_str!("data/ripple_65b_nvfp4.config.json");

fn parse() -> RippleConfig {
    RippleConfig::from_config_json(NVFP4).expect("the real Ripple config parses")
}

fn doctor(f: impl FnOnce(&mut Value)) -> String {
    let mut value: Value = serde_json::from_str(NVFP4).unwrap();
    f(&mut value);
    match RippleConfig::from_config_json(&value.to_string()) {
        Ok(_) => panic!("a doctored config was accepted"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn ripple_65b_names_every_forward_fact() {
    let c = parse();
    assert_eq!(c.model_type, "ricursive");
    assert_eq!(c.hidden_size, 5120);
    assert_eq!(c.num_hidden_layers, 80);
    assert_eq!(c.num_attention_heads, 80);
    assert_eq!(c.num_key_value_heads, 8);
    assert_eq!(c.head_dim, 128);
    assert_eq!(c.intermediate_size, 40960);
    assert_eq!(c.vocab_size, 100352);
    assert_eq!(c.q_proj_rows(), 10240);
    assert_eq!(c.kv_proj_rows(), 1024);
    assert_eq!(c.full_attention_layers(), 16);
    assert_eq!(c.sliding_window, 1024);
    assert_eq!(c.ffn_clamp_threshold, Some(20.0));
    assert_eq!(c.embed_scale, Some(71.55417527999327));
    assert!(c.use_valres && c.use_gated_attention && c.gate_full_precision);
    assert!(c.nvfp4_mlp());
    for layer in 0..80 {
        let expected = if layer % 5 == 4 {
            LayerKind::FullAttention
        } else {
            LayerKind::SlidingAttention
        };
        assert_eq!(c.layer_kind(layer), expected);
        assert_eq!(c.uses_rope(layer), expected == LayerKind::SlidingAttention);
    }
    c.validate_tp(2).unwrap();
}

#[test]
fn hybrid_kv_accounting_caps_only_local_layers() {
    let c = parse();
    assert_eq!(c.kv_cache_bytes_total(1024, 2), 335_544_320);
    assert_eq!(c.kv_cache_bytes_total(65536, 1), 2_281_701_376);
    assert_eq!(c.kv_cache_bytes_total(65536, 2), 4_563_402_752);
}

#[test]
fn bf16_config_without_quantization_is_accepted() {
    let mut value: Value = serde_json::from_str(NVFP4).unwrap();
    value.as_object_mut().unwrap().remove("quantization_config");
    let c = RippleConfig::from_config_json(&value.to_string()).unwrap();
    assert!(!c.nvfp4_mlp());
}

#[test]
fn rejects_an_unknown_field() {
    let e = doctor(|v| {
        v["unexpected"] = Value::from(true);
    });
    assert!(e.contains("unexpected"), "{e}");
}

#[test]
fn rejects_a_hybrid_schedule_disagreement() {
    let e = doctor(|v| v["hybrid_layer_pattern"][0] = Value::from(0));
    assert!(e.contains("hybrid_layer_pattern[0]"), "{e}");
}

#[test]
fn rejects_an_attention_schedule_length_mismatch() {
    let e = doctor(|v| {
        v["layer_types"].as_array_mut().unwrap().pop();
    });
    assert!(e.contains("layer_types has 79 entries"), "{e}");
}

#[test]
fn rejects_an_unquantized_attention_omission() {
    let e = doctor(|v| {
        let ignore = v["quantization_config"]["ignore"].as_array_mut().unwrap();
        ignore.retain(|x| x != "model.layers.0.self_attn*");
    });
    assert!(e.contains("NVFP4 ignore"), "{e}");
}

#[test]
fn rejects_the_wrong_nvfp4_group_size() {
    let e = doctor(|v| {
        v["quantization_config"]["config_groups"]["group_0"]["weights"]["group_size"] =
            Value::from(32)
    });
    assert!(e.contains("group_size 16"), "{e}");
}

#[test]
fn rejects_disabled_value_residuals() {
    let e = doctor(|v| v["use_valres"] = Value::from(false));
    assert!(e.contains("value residuals"), "{e}");
}

#[test]
fn rejects_a_tp_degree_that_does_not_divide_the_heads() {
    let e = parse().validate_tp(3).unwrap_err().to_string();
    assert!(e.contains("num_attention_heads=80"), "{e}");
}
