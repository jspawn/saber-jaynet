use saber_kernels::dflash::{self, DflashOp, DflashRules, DflashShape};
use saber_kernels::rules::{self, LaunchRules, Shape};
use saber_kernels::saber::{Geometry, Registry, SaberRules};

use super::KolibriConfig;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelProbe {
    pub operation: String,
    pub coverage: Result<String, String>,
}

fn probe(
    operation: impl Into<String>,
    result: Result<saber_kernels::rules::LaunchDescriptor, saber_kernels::Error>,
) -> KernelProbe {
    KernelProbe {
        operation: operation.into(),
        coverage: result
            .map(|descriptor| descriptor.summary())
            .map_err(|error| error.to_string()),
    }
}

pub fn kernel_coverage(cfg: &KolibriConfig, tp: u32) -> Result<Vec<KernelProbe>, String> {
    cfg.validate_tp(tp).map_err(|error| error.to_string())?;
    let mut out = Vec::new();
    let rules = SaberRules::prod_saber().map_err(|error| error.to_string())?;
    let registry = Registry::new(&rules);
    let gqa = cfg.num_attention_heads / cfg.num_key_value_heads;
    for (operation, op) in [
        ("attention.prefill", "attn_prefill_paged"),
        ("attention.decode", "attn_decode_paged"),
    ] {
        if cfg.head_dim == 128 {
            out.push(KernelProbe {
                operation: operation.into(),
                coverage: Ok("ripple_spike h128 kernel (single-request spike)".into()),
            });
            continue;
        }
        let mut geometry = Geometry::new()
            .num("head_dim", i64::from(cfg.head_dim))
            .num("gqa", i64::from(gqa))
            .num("block_size", 16)
            .num("causal", 1)
            .dtype("q_dtype", "bf16")
            .dtype("kv_dtype", "fp8");
        if op == "attn_decode_paged" {
            geometry = geometry.num("q_len", 1);
        }
        out.push(KernelProbe {
            operation: operation.into(),
            coverage: registry
                .select(op, &geometry)
                .map(|row| format!("saber_{}", row.entry))
                .map_err(|error| error.to_string()),
        });
    }

    let draft = DflashRules::prod_dflash().map_err(|error| error.to_string())?;
    for (name, n, k) in [
        ("q_proj", cfg.q_proj_rows() / tp, cfg.hidden_size),
        ("k_proj", cfg.kv_proj_rows() / tp, cfg.hidden_size),
        ("v_proj", cfg.kv_proj_rows() / tp, cfg.hidden_size),
        ("o_proj", cfg.hidden_size, cfg.q_proj_rows() / tp),
        (
            "expert_gate_up",
            2 * cfg.moe_intermediate_size / tp,
            cfg.hidden_size,
        ),
        (
            "expert_down",
            cfg.hidden_size,
            cfg.moe_intermediate_size / tp,
        ),
    ] {
        out.push(probe(
            format!("bf16.{name}.decode"),
            dflash::resolve(DflashOp::BgemmBf16, DflashShape::new(1, n, k), &draft),
        ));
        out.push(probe(
            format!("bf16.{name}.verify16"),
            dflash::resolve(DflashOp::BgemmBf16Mt, DflashShape::new(16, n, k), &draft),
        ));
    }
    out.push(probe(
        "bf16.lm_head.decode",
        dflash::resolve_head_bgemm(1, cfg.vocab_size / tp, cfg.hidden_size, &draft),
    ));

    let mx = LaunchRules::prod_mxfp4().map_err(|error| error.to_string())?;
    for m in [1, 16] {
        out.push(probe(
            format!("mxfp4.expert_gate_up.m{m}"),
            rules::resolve(
                rules::Op::gemm(),
                Shape::new(m, 2 * cfg.moe_intermediate_size / tp, cfg.hidden_size),
                &mx,
            ),
        ));
        out.push(probe(
            format!("mxfp4.expert_down.m{m}"),
            rules::resolve(
                rules::Op::gemm(),
                Shape::new(m, cfg.hidden_size, cfg.moe_intermediate_size / tp),
                &mx,
            ),
        ));
        out.push(probe(
            format!("swiglu_quant.m{m}"),
            rules::resolve(
                rules::Op::SiluMulQuant,
                Shape::new(m, cfg.moe_intermediate_size / tp, 0),
                &mx,
            ),
        ));
    }
    for operation in [
        "router.sigmoid_top6_correction_bias",
        "moe.shared_expert_without_gate",
        "perinorm.post_self_attn",
        "perinorm.post_mlp",
    ] {
        out.push(KernelProbe {
            operation: operation.into(),
            coverage: Err("requires Kolibri-specific glue".into()),
        });
    }
    Ok(out)
}
