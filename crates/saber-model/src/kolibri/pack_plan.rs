use std::collections::BTreeSet;

use crate::graph::{
    DeviceFinalRecord, DeviceFinalRole, Dtype, ShardKind, SourceQuant, SourceTensor,
    LAYOUT_BF16_FROM_FP8_BLOCK, LAYOUT_MXFP4_BLOCK_SCALE, LAYOUT_MXFP4_ROW_REF, LAYOUT_MXFP4_WPERM,
    LAYOUT_MXFP4_WPERM_MERGED, LAYOUT_RAW,
};
use crate::ModelError;

use super::KolibriConfig;

#[derive(Clone, Debug)]
pub struct KolibriLayerPackPlan {
    pub layer: u32,
    pub tp: u32,
    pub records: Vec<DeviceFinalRecord>,
}

impl KolibriLayerPackPlan {
    pub fn new(cfg: &KolibriConfig, tp: u32, layer: u32) -> Result<Self, ModelError> {
        cfg.validate_tp(tp)?;
        if layer >= cfg.num_hidden_layers {
            return Err(ModelError::Config(format!(
                "layer {layer} is outside {} layers",
                cfg.num_hidden_layers
            )));
        }
        let fp8 = cfg.quantization_config.is_some();
        let base = format!("model.layers.{layer}");
        let mut records = Vec::new();
        for norm in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "post_attn_norm.weight",
            "post_ffn_norm.weight",
        ] {
            records.push(raw(
                format!("{base}.{norm}"),
                ShardKind::Replicated,
                &[u64::from(cfg.hidden_size)],
                Dtype::Bf16,
            ));
        }
        for norm in ["q_norm.weight", "k_norm.weight"] {
            records.push(raw(
                format!("{base}.self_attn.{norm}"),
                ShardKind::Replicated,
                &[u64::from(cfg.head_dim)],
                Dtype::Bf16,
            ));
        }
        for (projection, rows) in [
            ("q_proj", cfg.q_proj_rows()),
            ("k_proj", cfg.kv_proj_rows()),
            ("v_proj", cfg.kv_proj_rows()),
        ] {
            records.push(bf16_matrix(
                format!("{base}.self_attn.{projection}"),
                ShardKind::Dim0,
                rows / tp,
                cfg.hidden_size,
                fp8,
            ));
        }
        records.push(bf16_matrix(
            format!("{base}.self_attn.o_proj"),
            ShardKind::Dim1,
            cfg.hidden_size,
            cfg.q_proj_rows() / tp,
            fp8,
        ));
        records.push(raw(
            format!("{base}.mlp.gate.weight"),
            ShardKind::Replicated,
            &[u64::from(cfg.num_experts), u64::from(cfg.hidden_size)],
            Dtype::Bf16,
        ));
        let mut router_bias = raw(
            format!("{base}.moe.router.expert_bias"),
            ShardKind::Replicated,
            &[u64::from(cfg.num_experts)],
            Dtype::F32,
        );
        router_bias.layout_id = crate::graph::LAYOUT_F32_FROM_BF16;
        records.push(router_bias);
        let shared = format!("{base}.mlp.shared_experts");
        records.extend(mxfp4(
            format!("layer.{layer}.mlp.shared_expert.gate_up"),
            2 * cfg.shared_expert_intermediate_size / tp,
            cfg.hidden_size,
            ShardKind::Dim0Segments,
            &[format!("{shared}.gate_proj"), format!("{shared}.up_proj")],
            fp8,
        ));
        records.extend(mxfp4(
            format!("layer.{layer}.mlp.shared_expert.down"),
            cfg.hidden_size,
            cfg.shared_expert_intermediate_size / tp,
            ShardKind::Dim1,
            &[format!("{shared}.down_proj")],
            fp8,
        ));
        for expert in 0..cfg.num_experts {
            let source = format!("{base}.mlp.experts.{expert}");
            records.extend(mxfp4(
                format!("layer.{layer}.expert.{expert}.gate_up"),
                2 * cfg.moe_intermediate_size / tp,
                cfg.hidden_size,
                ShardKind::Dim0Segments,
                &[format!("{source}.gate_proj"), format!("{source}.up_proj")],
                fp8,
            ));
            records.extend(mxfp4(
                format!("layer.{layer}.expert.{expert}.down"),
                cfg.hidden_size,
                cfg.moe_intermediate_size / tp,
                ShardKind::Dim1,
                &[format!("{source}.down_proj")],
                fp8,
            ));
        }
        Ok(Self { layer, tp, records })
    }

    pub fn validate_tensor_names<'a>(
        &self,
        names: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), ModelError> {
        let actual: BTreeSet<&str> = names.into_iter().collect();
        let mut used = BTreeSet::new();
        for record in &self.records {
            for source in &record.sources {
                used.insert(source.weight.as_str());
                if let Some(scale) = &source.scale {
                    used.insert(scale.as_str());
                }
            }
        }
        let missing: Vec<_> = used.difference(&actual).copied().collect();
        let unused: Vec<_> = actual.difference(&used).copied().collect();
        if !missing.is_empty() || !unused.is_empty() {
            return Err(ModelError::Config(format!(
                "Kolibri layer-{} tensor coverage: missing {missing:?}; unused {unused:?}",
                self.layer
            )));
        }
        Ok(())
    }

    pub fn add_global_records(&mut self, cfg: &KolibriConfig) {
        for (name, shard, shape) in [
            (
                "model.embed_tokens.weight",
                ShardKind::Dim0,
                vec![
                    u64::from(cfg.vocab_size / self.tp),
                    u64::from(cfg.hidden_size),
                ],
            ),
            (
                "model.norm.weight",
                ShardKind::Replicated,
                vec![u64::from(cfg.hidden_size)],
            ),
            (
                "lm_head.weight",
                ShardKind::Dim0,
                vec![
                    u64::from(cfg.vocab_size / self.tp),
                    u64::from(cfg.hidden_size),
                ],
            ),
        ] {
            self.records
                .push(raw(name.into(), shard, &shape, Dtype::Bf16));
        }
    }

    #[must_use]
    pub fn bytes_per_rank(&self) -> u64 {
        self.records
            .iter()
            .map(|record| record.shape.iter().product::<u64>() * u64::from(record.dtype.bits()) / 8)
            .sum()
    }

    #[must_use]
    pub fn estimated_model_bytes_per_rank(&self, cfg: &KolibriConfig) -> u64 {
        let global = 2 * (u64::from(cfg.vocab_size / self.tp) * u64::from(cfg.hidden_size) * 2)
            + u64::from(cfg.hidden_size) * 2;
        global + self.bytes_per_rank() * u64::from(cfg.num_hidden_layers)
    }
}

#[derive(Clone, Debug)]
pub struct KolibriPackPlan {
    pub tp: u32,
    pub records: Vec<DeviceFinalRecord>,
}

impl KolibriPackPlan {
    pub fn new(cfg: &KolibriConfig, tp: u32) -> Result<Self, ModelError> {
        let mut first = KolibriLayerPackPlan::new(cfg, tp, 0)?;
        first.add_global_records(cfg);
        let mut records = first.records;
        for layer in 1..cfg.num_hidden_layers {
            records.extend(KolibriLayerPackPlan::new(cfg, tp, layer)?.records);
        }
        Ok(Self { tp, records })
    }

    pub fn validate_tensor_names<'a>(
        &self,
        names: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), ModelError> {
        let actual: BTreeSet<&str> = names.into_iter().collect();
        let mut used = BTreeSet::new();
        for record in &self.records {
            for source in &record.sources {
                used.insert(source.weight.as_str());
                if let Some(scale) = &source.scale {
                    used.insert(scale.as_str());
                }
            }
        }
        let missing: Vec<_> = used.difference(&actual).copied().collect();
        let unused: Vec<_> = actual.difference(&used).copied().collect();
        if !missing.is_empty() || !unused.is_empty() {
            return Err(ModelError::Config(format!(
                "Kolibri full tensor coverage: missing {missing:?}; unused {unused:?}"
            )));
        }
        Ok(())
    }

    #[must_use]
    pub fn bytes_per_rank(&self) -> u64 {
        self.records
            .iter()
            .map(|record| record.shape.iter().product::<u64>() * u64::from(record.dtype.bits()) / 8)
            .sum()
    }
}

fn source(name: String, fp8: bool) -> SourceTensor {
    SourceTensor {
        weight: format!("{name}.weight"),
        quant: if fp8 {
            SourceQuant::Fp8Block
        } else {
            SourceQuant::Bf16
        },
        scale: fp8.then(|| format!("{name}.weight_scale_inv")),
        global_scale: None,
        input_scale: None,
    }
}

fn stored(name: String, fp8: bool) -> SourceTensor {
    if fp8 {
        source(name, true)
    } else {
        source(name, false)
    }
}

fn raw(name: String, shard: ShardKind, shape: &[u64], dtype: Dtype) -> DeviceFinalRecord {
    DeviceFinalRecord {
        name: name.clone(),
        layout_id: LAYOUT_RAW,
        role: DeviceFinalRole::Passthrough,
        op: None,
        dtype,
        shard,
        shape: shape.to_vec(),
        sources: vec![SourceTensor {
            weight: name,
            quant: SourceQuant::AsStored,
            scale: None,
            global_scale: None,
            input_scale: None,
        }],
    }
}

fn bf16_matrix(name: String, shard: ShardKind, n: u32, k: u32, fp8: bool) -> DeviceFinalRecord {
    DeviceFinalRecord {
        name: format!("{name}.weight"),
        layout_id: LAYOUT_BF16_FROM_FP8_BLOCK,
        role: DeviceFinalRole::Passthrough,
        op: Some(name.clone()),
        dtype: Dtype::Bf16,
        shard,
        shape: vec![u64::from(n), u64::from(k)],
        sources: vec![stored(name, fp8)],
    }
}

fn mxfp4(
    op: String,
    n: u32,
    k: u32,
    shard: ShardKind,
    sources: &[String],
    fp8: bool,
) -> Vec<DeviceFinalRecord> {
    let sources: Vec<_> = sources
        .iter()
        .cloned()
        .map(|name| source(name, fp8))
        .collect();
    let merged = sources.len() > 1;
    vec![
        DeviceFinalRecord {
            name: format!("{op}.w"),
            layout_id: if merged {
                LAYOUT_MXFP4_WPERM_MERGED
            } else {
                LAYOUT_MXFP4_WPERM
            },
            role: DeviceFinalRole::Weight,
            op: Some(op.clone()),
            dtype: Dtype::Fp4E2m1X2,
            shard,
            shape: vec![u64::from(n), u64::from(k)],
            sources: sources.clone(),
        },
        DeviceFinalRecord {
            name: format!("{op}.ws"),
            layout_id: LAYOUT_MXFP4_BLOCK_SCALE,
            role: DeviceFinalRole::BlockScale,
            op: Some(op.clone()),
            dtype: Dtype::E8M0,
            shard,
            shape: vec![u64::from(k / 32), u64::from(n)],
            sources: sources.clone(),
        },
        DeviceFinalRecord {
            name: format!("{op}.wref"),
            layout_id: LAYOUT_MXFP4_ROW_REF,
            role: DeviceFinalRole::RowRef,
            op: Some(op),
            dtype: Dtype::E8M0,
            shard,
            shape: vec![u64::from(n)],
            sources,
        },
    ]
}
