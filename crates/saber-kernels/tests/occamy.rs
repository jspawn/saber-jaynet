use saber_kernels::manifest::Manifest;
use saber_kernels::prebuilt;
use saber_kernels::saber::{
    attn_decode_combine, plugin_for, resolve, AttnShape, KvDtype, Op, SaberRules, Shape,
};

fn shape(q_len: u32) -> AttnShape {
    AttnShape {
        num_seqs: 1,
        q_len,
        live_rows: q_len,
        q_heads: 8,
        kv_heads: 1,
        head_dim: 256,
        block_size: 16,
        max_blocks: 64,
        max_ctx: 4096,
        splits: 1,
        kv_dtype: KvDtype::Bf16,
        kv_block_stride: 4096,
        kv_head_stride: 4096,
        scale_bits: 1.0f32.to_bits(),
        has_descales: false,
        ragged_rows: 0,
    }
}

#[test]
fn occamy_resolves_gqa8_prefill_at_48_query_rows_per_block() {
    let rules = SaberRules::occamy().unwrap();
    assert_eq!(rules.knobs.attn_gqa, 8);
    assert_eq!(rules.knobs.prefill_bq, 48);
    let descriptor = resolve(
        Op::AttnPrefill { fp8_mode: 0 },
        Shape::Attn(shape(49)),
        &rules,
    )
    .unwrap();
    assert_eq!(descriptor.kernel, "occamy_attn_prefill_h256_gqa8_bf16kv");
    assert_eq!(descriptor.plugin, "occamy_attn");
    assert_eq!(descriptor.grid, [2, 1, 1]);
    assert_eq!(descriptor.block, [768, 1, 1]);
}

#[test]
fn occamy_decode_covers_one_through_eight_query_tokens() {
    let rules = SaberRules::occamy().unwrap();
    for (q_len, nw) in [
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 2),
        (5, 3),
        (6, 3),
        (7, 4),
        (8, 4),
    ] {
        let descriptor = resolve(Op::AttnDecode, Shape::Attn(shape(q_len)), &rules).unwrap();
        assert_eq!(
            descriptor.kernel,
            format!("occamy_attn_decode_h256_gqa8_bf16kv_nw{nw}")
        );
        assert_eq!(descriptor.block, [nw * 32, 1, 1]);
        assert_eq!(descriptor.n_launches, 2);
        let combine = attn_decode_combine(shape(q_len), &rules).unwrap();
        assert_eq!(combine.kernel, "saber_attn_splitkv_combine_h256_pf1");
        assert_eq!(combine.grid[0], q_len * 8);
    }
    assert!(resolve(Op::AttnDecode, Shape::Attn(shape(9)), &rules).is_err());
}

#[test]
fn occamy_symbols_are_in_their_own_code_object() {
    let rules = SaberRules::occamy().unwrap();
    let manifest = Manifest::load(prebuilt::dir().join("occamy_attn.manifest.toml")).unwrap();
    let symbols: Vec<_> = rules
        .attn_prefill
        .iter()
        .map(|row| &row.symbol)
        .chain(rules.attn_decode.iter().map(|row| &row.symbol))
        .filter(|symbol| symbol.starts_with("occamy_"))
        .collect();
    for symbol in symbols {
        assert_eq!(plugin_for(symbol).unwrap(), "occamy_attn");
        assert!(manifest.kernel.iter().any(|kernel| &kernel.name == symbol));
    }
    assert_eq!(
        saber_kernels::rule_table_for_plugin("occamy_attn"),
        Some("occamy_gfx1201.toml")
    );
}

#[test]
fn occamy_bf16_moe_kernels_cover_the_runtime_geometry() {
    use saber_kernels::occamy;

    let manifest = Manifest::load(prebuilt::dir().join("occamy_moe.manifest.toml")).unwrap();
    for symbol in [occamy::ROUTE_SYMBOL, occamy::COMBINE_SYMBOL] {
        assert_eq!(plugin_for(symbol).unwrap(), "occamy_moe");
        assert!(saber_kernels::abi::signature_for(symbol).is_some());
        assert!(manifest.kernel.iter().any(|kernel| kernel.name == symbol));
    }
    for (plugin, symbol) in [
        ("flashnext_moe", occamy::GATE_UP_SYMBOL),
        ("flashnext_glue", occamy::SILU_SYMBOL),
        ("flashnext_moe", occamy::DOWN_SYMBOL),
    ] {
        let manifest =
            Manifest::load(prebuilt::dir().join(format!("{plugin}.manifest.toml"))).unwrap();
        assert!(manifest.kernel.iter().any(|kernel| kernel.name == symbol));
    }
    occamy::validate_gemm_geometry(2 * occamy::INTERMEDIATE, occamy::HIDDEN).unwrap();
    occamy::validate_gemm_geometry(occamy::HIDDEN, occamy::INTERMEDIATE).unwrap();
    assert!(occamy::validate_gemm_geometry(512, 2047).is_err());
    assert_eq!(occamy::route_grid(33), [2, 1, 1]);
    assert_eq!(occamy::grouped_grid(1024, 7), [16, 7, 1]);
}

#[test]
fn occamy_dense_projections_cover_tp2_shapes_and_every_row_band() {
    use saber_kernels::occamy;

    for n in [6144, 32, 2048, 4096, 256, 1] {
        let launch = occamy::dense_projection(3, n, 2048).unwrap();
        assert_eq!(launch.symbol, "fn_bgemm_sk_m16");
        assert_eq!(launch.grid, [n.div_ceil(64), 1, 4]);
        assert_eq!(launch.threads, 128);
        assert_eq!(launch.partial_bytes, u64::from(4 * 3 * n * 4));
        assert_eq!(launch.counter_bytes, u64::from(n.div_ceil(64) * 4));
    }
    for (m, symbol, splits) in [
        (17, "fn_bgemm_sk_m32", 2),
        (33, "fn_bgemm_sk_m64", 2),
        (65, "fn_bgemm_lds_l128x128", 1),
        (8192, "fn_bgemm_lds_l128x128", 1),
    ] {
        let launch = occamy::dense_projection(m, 256, 2048).unwrap();
        assert_eq!(launch.symbol, symbol);
        assert_eq!(launch.splits, splits);
        assert_eq!(launch.grid[2], splits);
        if splits == 1 {
            assert_eq!(launch.partial_bytes, 0);
            assert_eq!(launch.counter_bytes, 0);
        }
    }
    assert!(occamy::dense_projection(0, 256, 2048).is_err());
    assert!(occamy::dense_projection(3, 256, 1024).is_err());
    assert!(occamy::dense_projection(65, 32, 2048).is_err());
    assert!(occamy::dense_projection(8193, 256, 2048).is_err());
    assert_eq!(occamy::KV_APPEND_SYMBOL, "fn_kv_append_ring_h256_bf16");
    assert_eq!(occamy::GDN_VALUE_HEADS / occamy::GDN_KEY_HEADS, 2);
    assert!(occamy::GDN_KKT_SYMBOL.ends_with("_r2"));
    let manifest =
        Manifest::load(prebuilt::dir().join("saber_gdn_kkt_solve.manifest.toml")).unwrap();
    assert!(manifest
        .kernel
        .iter()
        .any(|kernel| kernel.name == occamy::GDN_KKT_SYMBOL));
}
