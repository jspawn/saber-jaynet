//! The one-layer Occamy go/no-go spike: source tensors and existing-kernel coverage.

use saber_kernels::moe;
use saber_kernels::saber::{Geometry, Registry, SaberRules};

use super::config::{LayerKind, Qwen35MoeConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightClass {
    Norm,
    Attention,
    Gdn,
    Router,
    SharedExpert,
    RoutedExpert,
    Embedding,
    Head,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<u32>,
    pub class: WeightClass,
}

fn tensor(output: &mut Vec<TensorSpec>, name: String, shape: &[u32], class: WeightClass) {
    output.push(TensorSpec {
        name,
        dtype: "BF16",
        shape: shape.to_vec(),
        class,
    });
}

/// Every checkpoint tensor needed by one text layer, including all routed and shared experts.
pub fn layer_tensors(config: &Qwen35MoeConfig, layer: u32) -> Result<Vec<TensorSpec>, String> {
    let Some(kind) = config.layers.get(layer as usize) else {
        return Err(format!(
            "layer {layer}: outside the model's {} layers",
            config.num_layers()
        ));
    };
    let h = config.hidden_size;
    let prefix = format!("model.language_model.layers.{layer}");
    let mut output = Vec::new();
    for name in ["input_layernorm", "post_attention_layernorm"] {
        tensor(
            &mut output,
            format!("{prefix}.{name}.weight"),
            &[h],
            WeightClass::Norm,
        );
    }

    match kind {
        LayerKind::Gdn => {
            let g = &config.gdn;
            let base = format!("{prefix}.linear_attn");
            let nv = g.num_value_heads;
            for (suffix, shape) in [
                ("in_proj_qkv.weight", vec![g.qkv_rows(), h]),
                ("in_proj_z.weight", vec![g.value_dim(), h]),
                ("in_proj_a.weight", vec![nv, h]),
                ("in_proj_b.weight", vec![nv, h]),
                ("conv1d.weight", vec![g.qkv_rows(), 1, g.conv_kernel]),
                ("A_log", vec![nv]),
                ("dt_bias", vec![nv]),
                ("norm.weight", vec![g.value_head_dim]),
                ("out_proj.weight", vec![h, g.value_dim()]),
            ] {
                tensor(
                    &mut output,
                    format!("{base}.{suffix}"),
                    &shape,
                    WeightClass::Gdn,
                );
            }
        }
        LayerKind::FullAttention => {
            let a = &config.attention;
            let base = format!("{prefix}.self_attn");
            let query_rows = a.num_heads * a.head_dim;
            for (suffix, shape) in [
                ("q_proj.weight", vec![config.q_proj_rows(), h]),
                ("k_proj.weight", vec![config.kv_proj_rows(), h]),
                ("v_proj.weight", vec![config.kv_proj_rows(), h]),
                ("o_proj.weight", vec![h, query_rows]),
                ("q_norm.weight", vec![a.head_dim]),
                ("k_norm.weight", vec![a.head_dim]),
            ] {
                tensor(
                    &mut output,
                    format!("{base}.{suffix}"),
                    &shape,
                    WeightClass::Attention,
                );
            }
        }
    }

    let mlp = format!("{prefix}.mlp");
    tensor(
        &mut output,
        format!("{mlp}.gate.weight"),
        &[config.moe.num_experts, h],
        WeightClass::Router,
    );
    tensor(
        &mut output,
        format!("{mlp}.shared_expert_gate.weight"),
        &[1, h],
        WeightClass::Router,
    );
    for (name, rows, cols) in [
        ("gate_proj", config.moe.shared_intermediate, h),
        ("up_proj", config.moe.shared_intermediate, h),
        ("down_proj", h, config.moe.shared_intermediate),
    ] {
        tensor(
            &mut output,
            format!("{mlp}.shared_expert.{name}.weight"),
            &[rows, cols],
            WeightClass::SharedExpert,
        );
    }
    for expert in 0..config.moe.num_experts {
        for (name, rows, cols) in [
            ("gate_proj", config.moe.intermediate, h),
            ("up_proj", config.moe.intermediate, h),
            ("down_proj", h, config.moe.intermediate),
        ] {
            tensor(
                &mut output,
                format!("{mlp}.experts.{expert}.{name}.weight"),
                &[rows, cols],
                WeightClass::RoutedExpert,
            );
        }
    }
    Ok(output)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Coverage {
    Existing(String),
    Missing(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelProbe {
    pub operation: &'static str,
    pub coverage: Coverage,
}

fn probe_registry(
    registry: &Registry<'_>,
    operation: &'static str,
    op: &str,
    geometry: Geometry,
) -> KernelProbe {
    let coverage = match registry.select(op, &geometry) {
        Ok(row) => Coverage::Existing(format!("saber_{}", row.entry)),
        Err(reason) => Coverage::Missing(reason),
    };
    KernelProbe {
        operation,
        coverage,
    }
}

/// Ask the real 27B registry which core kernels cover Occamy at TP2, and compare the MoE family's
/// fixed geometry. Projection-rule rows and pack layouts are separate follow-up work.
pub fn kernel_coverage(config: &Qwen35MoeConfig) -> Result<Vec<KernelProbe>, String> {
    config.validate_tp2().map_err(|error| error.to_string())?;
    let rules = SaberRules::occamy().map_err(|error| error.to_string())?;
    let registry = Registry::new(&rules);
    let mut output = Vec::new();

    let gdn = Geometry::new()
        .num("head_k", i64::from(config.gdn.key_head_dim))
        .num("head_v", i64::from(config.gdn.value_head_dim))
        .num("conv_width", i64::from(config.gdn.conv_kernel))
        .num("chunk", 64);
    for (operation, op) in [
        ("gdn.conv_prep", "gdn_conv_prep"),
        ("gdn.kkt_solve", "gdn_kkt_solve"),
        ("gdn.chunk_scan", "gdn_chunk_scan"),
        ("gdn.recurrent_update", "gdn_recurrent_update"),
        ("gdn.fused_update", "gdn_fused_update"),
    ] {
        output.push(probe_registry(&registry, operation, op, gdn.clone()));
    }
    output.push(probe_registry(
        &registry,
        "gdn.gated_rmsnorm",
        "gdn_gated_rmsnorm",
        Geometry::new().num("channels", i64::from(config.gdn.value_head_dim)),
    ));

    let gqa = config.attention.num_heads / config.attention.num_kv_heads;
    for (operation, op) in [
        ("attention.prefill", "attn_prefill_paged"),
        ("attention.decode", "attn_decode_paged"),
    ] {
        let mut geometry = Geometry::new()
            .num("head_dim", i64::from(config.attention.head_dim))
            .num("gqa", i64::from(gqa))
            .num("block_size", 16)
            .num("causal", 1)
            .dtype("q_dtype", "bf16")
            .dtype("kv_dtype", "bf16");
        if op == "attn_decode_paged" {
            geometry = geometry.num("q_len", 1);
        }
        output.push(probe_registry(&registry, operation, op, geometry));
    }

    let moe_geometry_matches = moe::EXPERTS == config.moe.num_experts
        && moe::TOPK == config.moe.top_k
        && moe::HIDDEN == config.hidden_size
        && moe::INTER == config.moe.intermediate;
    for operation in ["moe.route", "moe.gate_up", "moe.down", "moe.combine"] {
        let coverage = if moe_geometry_matches {
            Coverage::Existing("Flash-Next MoE geometry".into())
        } else {
            Coverage::Missing(format!(
                "needs Occamy e{} top-k{} hidden{} intermediate{} variants; existing MoE is e{} top-k{} hidden{} intermediate{}",
                config.moe.num_experts,
                config.moe.top_k,
                config.hidden_size,
                config.moe.intermediate,
                moe::EXPERTS,
                moe::TOPK,
                moe::HIDDEN,
                moe::INTER
            ))
        };
        output.push(KernelProbe {
            operation,
            coverage,
        });
    }
    Ok(output)
}
