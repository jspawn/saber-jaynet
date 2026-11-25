use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use saber_model::qwen3_5_moe::pack_plan::{PackPlan, Placement, SourceKind};
use saber_model::qwen3_5_moe::spike::layer_tensors;
use saber_model::qwen3_5_moe::Qwen35MoeConfig;
use saber_model::qwen4_exp::st::ShardSet;

const OCCAMY: &str = include_str!("data/qwen3_5_moe_occamy.config.json");

fn plan() -> PackPlan {
    PackPlan::spike(&Qwen35MoeConfig::from_config_json(OCCAMY).unwrap()).unwrap()
}

#[test]
fn spike_has_unique_records_and_expected_placement() {
    let plan = plan();
    assert_eq!(plan.layers, vec![0, 3]);
    let mut names = BTreeSet::new();
    for record in &plan.records {
        assert!(names.insert((record.placement, record.name.clone())));
    }
    for (placement, count) in [
        (Placement::Rank(0), 522),
        (Placement::Rank(1), 522),
        (Placement::Shared, 15),
    ] {
        assert_eq!(
            plan.records
                .iter()
                .filter(|record| record.placement == placement)
                .count(),
            count
        );
    }
    assert_eq!(plan.records.len(), 1059);
}

#[test]
fn every_source_tensor_is_consumed_exactly_once() {
    let config = Qwen35MoeConfig::from_config_json(OCCAMY).unwrap();
    let plan = plan();
    let expected: BTreeMap<String, u64> = [0, 3]
        .into_iter()
        .flat_map(|layer| layer_tensors(&config, layer).unwrap())
        .map(|tensor| {
            let elems = tensor.shape.iter().map(|&dim| u64::from(dim)).product();
            (tensor.name, elems)
        })
        .collect();
    let mut actual = BTreeMap::<String, u64>::new();
    for record in &plan.records {
        for source in &record.sources {
            *actual.entry(source.tensor.clone()).or_default() += source.elems();
        }
    }
    assert_eq!(actual, expected);
}

#[test]
fn gdn_slices_follow_the_kernel_layout() {
    let plan = plan();
    let qkvz = plan
        .records
        .iter()
        .find(|record| {
            record.name == "layer.0.gdn.in_proj_qkvz.w" && record.placement == Placement::Rank(1)
        })
        .unwrap();
    assert_eq!(qkvz.shape, vec![6144, 2048]);
    let ranges: Vec<_> = qkvz.sources.iter().map(|source| source.rows).collect();
    assert_eq!(
        ranges,
        vec![(1024, 2048), (3072, 4096), (6144, 8192), (2048, 4096)]
    );
    let a_log = plan
        .records
        .iter()
        .find(|record| {
            record.name == "model.language_model.layers.0.linear_attn.A_log"
                && record.placement == Placement::Rank(1)
        })
        .unwrap();
    assert_eq!(a_log.kind, SourceKind::F32FromBf16);
    assert_eq!(a_log.dtype, "F32");
    assert_eq!(a_log.shape, vec![16]);
    assert_eq!(a_log.bytes, 64);
    assert_eq!(a_log.sources[0].rows, (16, 32));
}

#[test]
fn attention_and_experts_use_the_expected_tp2_slices() {
    let plan = plan();
    let q = plan
        .records
        .iter()
        .find(|record| {
            record.name == "model.language_model.layers.3.self_attn.q_proj.weight"
                && record.placement == Placement::Rank(1)
        })
        .unwrap();
    assert_eq!(q.shape, vec![4096, 2048]);
    assert_eq!(q.sources[0].rows, (4096, 8192));
    let o = plan
        .records
        .iter()
        .find(|record| {
            record.name == "model.language_model.layers.3.self_attn.o_proj.weight"
                && record.placement == Placement::Rank(1)
        })
        .unwrap();
    assert_eq!(o.shape, vec![2048, 2048]);
    assert_eq!(o.sources[0].cols, (2048, 4096));
    for (rank, first, last) in [(0, 0, 127), (1, 128, 255)] {
        for (expert, present) in [(first, true), (last, true), ((last + 1) % 256, false)] {
            assert_eq!(
                plan.records.iter().any(|record| {
                    record.name == format!("layer.0.expert.{expert}.gate_up.w")
                        && record.placement == Placement::Rank(rank)
                }),
                present,
                "rank {rank}, expert {expert}"
            );
        }
    }
}

#[test]
fn spike_sources_match_the_real_checkpoint_headers() {
    let root = Path::new("/srv/models/radconv/occamy-1.0");
    if !root.join("model.safetensors.index.json").exists() {
        eprintln!("SKIP: Occamy checkpoint absent");
        return;
    }
    let plan = plan();
    let mut shards = ShardSet::open(root).unwrap();
    let mut seen = BTreeSet::new();
    for record in &plan.records {
        for source in &record.sources {
            if !seen.insert(&source.tensor) {
                continue;
            }
            let info = shards.info(&source.tensor).unwrap();
            assert_eq!(info.dtype, "BF16", "{}", source.tensor);
            assert_eq!(info.shape, source.shape, "{}", source.tensor);
        }
    }
    assert_eq!(seen.len(), 1565);
}

#[test]
fn all_layers_cover_every_source_once_with_unique_records() {
    let config = Qwen35MoeConfig::from_config_json(OCCAMY).unwrap();
    let plan = PackPlan::all_layers(&config).unwrap();
    assert_eq!(plan.layers, (0..40).collect::<Vec<_>>());
    let mut names = BTreeSet::new();
    for record in &plan.records {
        assert!(names.insert((record.placement, record.name.clone())));
    }
    for (placement, count) in [
        (Placement::Rank(0), 10_460),
        (Placement::Rank(1), 10_460),
        (Placement::Shared, 290),
    ] {
        assert_eq!(
            plan.records
                .iter()
                .filter(|record| record.placement == placement)
                .count(),
            count
        );
    }
    assert_eq!(plan.records.len(), 21_210);
    let expected: BTreeMap<String, u64> = (0..40)
        .flat_map(|layer| layer_tensors(&config, layer).unwrap())
        .map(|tensor| {
            let elems = tensor.shape.iter().map(|&dim| u64::from(dim)).product();
            (tensor.name, elems)
        })
        .collect();
    let mut actual = BTreeMap::<String, u64>::new();
    for record in &plan.records {
        for source in &record.sources {
            *actual.entry(source.tensor.clone()).or_default() += source.elems();
        }
    }
    assert_eq!(actual.len(), 31_330);
    assert_eq!(actual, expected);
}

#[test]
fn globals_shard_embedding_and_head_and_replicate_the_final_norm() {
    let config = Qwen35MoeConfig::from_config_json(OCCAMY).unwrap();
    let plan = PackPlan::globals(&config).unwrap();
    assert!(plan.layers.is_empty());
    assert_eq!(plan.records.len(), 5);
    for name in ["model.language_model.embed_tokens.weight", "lm_head.weight"] {
        for rank in 0..2 {
            let record = plan
                .records
                .iter()
                .find(|record| record.name == name && record.placement == Placement::Rank(rank))
                .unwrap();
            assert_eq!(record.shape, [124_160, 2048]);
            assert_eq!(record.sources[0].rows.1 - record.sources[0].rows.0, 124_160);
        }
    }
    let norm = plan
        .records
        .iter()
        .find(|record| record.name == "model.language_model.norm.weight")
        .unwrap();
    assert_eq!(norm.placement, Placement::Shared);
    assert_eq!(norm.shape, [2048]);
}
