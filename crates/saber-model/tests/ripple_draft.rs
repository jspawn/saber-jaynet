use saber_model::ripple::draft::RippleDraftConfig;
use saber_model::ripple::RippleConfig;

const TARGET: &str = include_str!("data/ripple_65b_nvfp4.config.json");
const DRAFT: &str = include_str!("data/ripple_65b_dflash2.config.json");

#[test]
fn ripple_dflash2_maps_its_real_geometry() {
    let target = RippleConfig::from_config_json(TARGET).unwrap();
    let draft = RippleDraftConfig::from_config_json(DRAFT, &target).unwrap();
    let cfg = draft.dflash_config(2);
    assert_eq!(cfg.hidden, 5120);
    assert_eq!(cfg.layers, 5);
    assert_eq!(cfg.intermediate, 40960);
    assert_eq!(cfg.q_heads, 80);
    assert_eq!(cfg.kv_heads, 8);
    assert_eq!(cfg.block_size, 16);
    assert_eq!(cfg.conv_taps, 4);
    assert_eq!(cfg.selector_top_k, 32);
    assert_eq!(cfg.target_layers, 5);
    assert_eq!(cfg.vocab, 100352);
    assert_eq!(cfg.q_heads_local(), 40);
    assert_eq!(cfg.kv_heads_local(), 4);
    assert_eq!(cfg.qkv_n(), 6144);
    assert_eq!(cfg.kernel_projection_n(), 2560);
}

#[test]
fn draft_rejects_a_target_tap_outside_the_target() {
    let target = RippleConfig::from_config_json(TARGET).unwrap();
    let mut value: serde_json::Value = serde_json::from_str(DRAFT).unwrap();
    value["dflash_config"]["target_layer_ids"][4] = 80.into();
    let e = RippleDraftConfig::from_config_json(&value.to_string(), &target).unwrap_err();
    assert!(e.to_string().contains("target taps"), "{e}");
}

#[test]
fn draft_rejects_a_target_geometry_mismatch() {
    let target = RippleConfig::from_config_json(TARGET).unwrap();
    let mut value: serde_json::Value = serde_json::from_str(DRAFT).unwrap();
    value["intermediate_size"] = 17408.into();
    let e = RippleDraftConfig::from_config_json(&value.to_string(), &target).unwrap_err();
    assert!(e.to_string().contains("geometry disagree"), "{e}");
}

#[test]
fn draft_plan_covers_the_real_81_tensors() {
    let path = std::path::Path::new(
        "/srv/models/saber-models/models/Ripple-65B-Preview-DFlash2/model.safetensors",
    );
    if !path.exists() {
        eprintln!("SKIP: {} absent", path.display());
        return;
    }
    let header = saber_model::checkpoint_head::read_header(path).unwrap();
    let target = RippleConfig::from_config_json(TARGET).unwrap();
    let draft = RippleDraftConfig::from_config_json(DRAFT, &target).unwrap();
    saber_model::ripple::draft::validate_draft_tensors(
        &draft,
        2,
        131072,
        header
            .as_object()
            .unwrap()
            .keys()
            .filter(|name| name.as_str() != "__metadata__")
            .map(String::as_str),
    )
    .unwrap();
}
