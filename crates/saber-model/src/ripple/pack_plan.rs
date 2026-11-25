use std::collections::BTreeSet;

use crate::graph::{
    DeviceFinalRecord, DeviceFinalRole, Dtype, ShardKind, SourceQuant, SourceTensor,
    LAYOUT_MXFP4_BLOCK_SCALE, LAYOUT_MXFP4_ROW_REF, LAYOUT_MXFP4_WPERM, LAYOUT_MXFP4_WPERM_MERGED,
    LAYOUT_RAW,
};
use crate::ModelError;

use super::RippleConfig;

#[derive(Clone, Debug)]
pub struct RipplePackPlan {
    pub tp: u32,
    pub records: Vec<DeviceFinalRecord>,
}

impl RipplePackPlan {
    pub fn new(cfg: &RippleConfig, tp: u32) -> Result<Self, ModelError> {
        cfg.validate_tp(tp)?;
        if !cfg.nvfp4_mlp() {
            return Err(ModelError::Config(
                "the initial Ripple pack planner requires the official NVFP4 checkpoint".into(),
            ));
        }
        let mut records = Vec::new();
        records.push(raw(
            "model.embed_tokens.weight",
            ShardKind::Dim0,
            &[u64::from(cfg.vocab_size / tp), u64::from(cfg.hidden_size)],
            None,
        ));
        for layer in 0..cfg.num_hidden_layers {
            let base = format!("model.layers.{layer}");
            for norm in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
            ] {
                records.push(raw(
                    format!("{base}.{norm}"),
                    ShardKind::Replicated,
                    &[u64::from(cfg.hidden_size)],
                    None,
                ));
            }
            for norm in ["q_norm.weight", "k_norm.weight"] {
                records.push(raw(
                    format!("{base}.self_attn.{norm}"),
                    ShardKind::Replicated,
                    &[u64::from(cfg.head_dim)],
                    None,
                ));
            }
            records.push(raw(
                format!("{base}.self_attn.value_res_lambda"),
                ShardKind::Replicated,
                &[1],
                None,
            ));
            for (projection, rows) in [
                ("q_proj", cfg.q_proj_rows()),
                ("g_proj", cfg.q_proj_rows()),
                ("k_proj", cfg.kv_proj_rows()),
                ("v_proj", cfg.kv_proj_rows()),
            ] {
                records.push(raw(
                    format!("{base}.self_attn.{projection}.weight"),
                    ShardKind::Dim0,
                    &[u64::from(rows / tp), u64::from(cfg.hidden_size)],
                    Some(format!("layer.{layer}.attn.{projection}")),
                ));
            }
            records.push(raw(
                format!("{base}.self_attn.o_proj.weight"),
                ShardKind::Dim1,
                &[
                    u64::from(cfg.hidden_size),
                    u64::from(cfg.q_proj_rows() / tp),
                ],
                Some(format!("layer.{layer}.attn.o_proj")),
            ));
            let gate_up = format!("layer.{layer}.mlp.gate_up");
            let n = 2 * cfg.intermediate_size / tp;
            let k = cfg.hidden_size;
            records.extend(mxfp4(
                gate_up,
                n,
                k,
                ShardKind::Dim0Segments,
                &[
                    format!("{base}.mlp.gate_proj"),
                    format!("{base}.mlp.up_proj"),
                ],
            ));
            records.extend(mxfp4(
                format!("layer.{layer}.mlp.down_proj"),
                cfg.hidden_size,
                cfg.intermediate_size / tp,
                ShardKind::Dim1,
                &[format!("{base}.mlp.down_proj")],
            ));
        }
        records.push(raw(
            "model.norm.weight",
            ShardKind::Replicated,
            &[u64::from(cfg.hidden_size)],
            None,
        ));
        records.push(raw(
            "lm_head.weight",
            ShardKind::Dim0,
            &[u64::from(cfg.vocab_size / tp), u64::from(cfg.hidden_size)],
            Some("lm_head".into()),
        ));
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
                for name in [&source.scale, &source.global_scale, &source.input_scale]
                    .into_iter()
                    .flatten()
                {
                    used.insert(name.as_str());
                }
            }
        }
        let missing: Vec<_> = used.difference(&actual).copied().collect();
        let unused: Vec<_> = actual.difference(&used).copied().collect();
        if !missing.is_empty() || !unused.is_empty() {
            return Err(ModelError::Config(format!(
                "Ripple tensor coverage: missing {missing:?}; unused {unused:?}"
            )));
        }
        Ok(())
    }

    #[must_use]
    pub fn bytes_per_rank(&self) -> u64 {
        self.records
            .iter()
            .map(|record| {
                let elements = record.shape.iter().product::<u64>();
                elements * u64::from(record.dtype.bits()) / 8
            })
            .sum()
    }
}

fn stored(name: String) -> SourceTensor {
    SourceTensor {
        weight: name,
        quant: SourceQuant::AsStored,
        scale: None,
        global_scale: None,
        input_scale: None,
    }
}

fn nvfp4(base: String) -> SourceTensor {
    SourceTensor {
        weight: format!("{base}.weight"),
        quant: SourceQuant::Nvfp4ModelOpt,
        scale: Some(format!("{base}.weight_scale")),
        global_scale: Some(format!("{base}.weight_scale_2")),
        input_scale: Some(format!("{base}.input_scale")),
    }
}

fn raw(
    name: impl Into<String>,
    shard: ShardKind,
    shape: &[u64],
    op: Option<String>,
) -> DeviceFinalRecord {
    let name = name.into();
    DeviceFinalRecord {
        name: name.clone(),
        layout_id: LAYOUT_RAW,
        role: DeviceFinalRole::Passthrough,
        op,
        dtype: Dtype::Bf16,
        shard,
        shape: shape.to_vec(),
        sources: vec![stored(name)],
    }
}

fn mxfp4(
    op: String,
    n: u32,
    k: u32,
    shard: ShardKind,
    sources: &[String],
) -> Vec<DeviceFinalRecord> {
    let sources: Vec<_> = sources.iter().cloned().map(nvfp4).collect();
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
