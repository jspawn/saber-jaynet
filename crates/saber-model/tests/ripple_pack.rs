use std::collections::BTreeSet;
use std::path::Path;

use saber_model::graph::{DeviceFinalRole, Dtype, ShardKind};
use saber_model::ripple::{RippleConfig, RipplePackPlan};

const CONFIG: &str = include_str!("data/ripple_65b_nvfp4.config.json");
const CHECKPOINT: &str = "/srv/models/saber-models/models/Ripple-65B-Preview-NVFP4";

fn plan(tp: u32) -> RipplePackPlan {
    let cfg = RippleConfig::from_config_json(CONFIG).unwrap();
    RipplePackPlan::new(&cfg, tp).unwrap()
}

#[test]
fn tp2_plan_has_unique_records_and_the_official_precision_split() {
    let p = plan(2);
    assert_eq!(p.records.len(), 1443);
    let names: BTreeSet<_> = p.records.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(names.len(), p.records.len());
    let gate_up = p
        .records
        .iter()
        .find(|r| r.name == "layer.0.mlp.gate_up.w")
        .unwrap();
    assert_eq!(gate_up.shape, [40960, 5120]);
    assert_eq!(gate_up.dtype, Dtype::Fp4E2m1X2);
    assert_eq!(gate_up.shard, ShardKind::Dim0Segments);
    assert_eq!(gate_up.sources.len(), 2);
    assert_eq!(
        gate_up.sources[0].weight,
        "model.layers.0.mlp.gate_proj.weight"
    );
    assert_eq!(
        gate_up.sources[0].global_scale.as_deref(),
        Some("model.layers.0.mlp.gate_proj.weight_scale_2")
    );
    let down = p
        .records
        .iter()
        .find(|r| r.name == "layer.0.mlp.down_proj.ws")
        .unwrap();
    assert_eq!(down.shape, [640, 5120]);
    assert_eq!(down.role, DeviceFinalRole::BlockScale);
    for name in ["q_proj", "g_proj", "k_proj", "v_proj", "o_proj"] {
        let record = p
            .records
            .iter()
            .find(|r| r.name == format!("model.layers.0.self_attn.{name}.weight"))
            .unwrap();
        assert_eq!(record.dtype, Dtype::Bf16);
        assert_eq!(record.layout_id, 0);
    }
    assert!(p.bytes_per_rank() > 20_000_000_000);
}

#[test]
fn plan_covers_the_real_checkpoint_index_exactly() {
    let path = Path::new(CHECKPOINT).join("model.safetensors.index.json");
    if !path.exists() {
        eprintln!("SKIP: {} absent", path.display());
        return;
    }
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let names = index["weight_map"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str);
    plan(2).validate_tensor_names(names).unwrap();
}

#[test]
fn coverage_refuses_missing_and_unused_tensors() {
    let p = plan(2);
    let mut names: Vec<String> = p
        .records
        .iter()
        .flat_map(|r| {
            r.sources.iter().flat_map(|s| {
                let mut names = vec![s.weight.clone()];
                names.extend(s.scale.clone());
                names.extend(s.global_scale.clone());
                names.extend(s.input_scale.clone());
                names
            })
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    names.pop();
    names.push("model.layers.0.unused".into());
    let e = p
        .validate_tensor_names(names.iter().map(String::as_str))
        .unwrap_err()
        .to_string();
    assert!(e.contains("missing"), "{e}");
    assert!(e.contains("model.layers.0.unused"), "{e}");
}
