pub const EXPERTS: u32 = 256;
pub const LOCAL_EXPERTS: u32 = 128;
pub const TOPK: u32 = 8;
pub const HIDDEN: u32 = 2048;
pub const INTERMEDIATE: u32 = 512;

pub const ROUTE_SYMBOL: &str = "occamy_moe_route_topk8_sort_e256";
pub const COMBINE_SYMBOL: &str = "occamy_moe_combine_topk8_bf16";
pub const GATE_UP_SYMBOL: &str = "fn_moe_bgemm_sk_m16_g1";
pub const SILU_SYMBOL: &str = "fn_moe_silu_mul_bf16_t256";
pub const DOWN_SYMBOL: &str = "fn_moe_bgemm_sk_m16_g0";

#[must_use]
pub fn route_grid(tokens: u32) -> [u32; 3] {
    [tokens.div_ceil(32), 1, 1]
}

#[must_use]
pub fn grouped_grid(n: u32, tile_capacity: u32) -> [u32; 3] {
    [n.div_ceil(64), tile_capacity, 1]
}

pub fn validate_gemm_geometry(n: u32, k: u32) -> Result<(), String> {
    if n == 0 || k == 0 || n % 16 != 0 || k % 16 != 0 {
        return Err(format!(
            "BF16 grouped GEMM needs positive multiples of 16, got N {n}, K {k}"
        ));
    }
    Ok(())
}

pub const KV_APPEND_SYMBOL: &str = "fn_kv_append_ring_h256_bf16";
pub const GDN_VALUE_HEADS: u32 = 16;
pub const GDN_KEY_HEADS: u32 = 8;
pub const GDN_KKT_SYMBOL: &str = "saber_gdn_kkt_solve_k128_c64_r2";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseProjection {
    pub symbol: &'static str,
    pub grid: [u32; 3],
    pub threads: u32,
    pub splits: u32,
    pub partial_bytes: u64,
    pub counter_bytes: u64,
}

pub fn dense_projection(m: u32, n: u32, k: u32) -> Result<DenseProjection, String> {
    if m == 0 || n == 0 || k != HIDDEN {
        return Err(format!(
            "Occamy dense projection needs positive M/N and K {HIDDEN}, got M {m}, N {n}, K {k}"
        ));
    }
    let (symbol, rows, columns, threads, splits) = match m {
        1..=16 => ("fn_bgemm_sk_m16", 16, 64, 128, 4),
        17..=32 => ("fn_bgemm_sk_m32", 32, 64, 128, 2),
        33..=64 => ("fn_bgemm_sk_m64", 64, 64, 128, 2),
        65..=8192 if n % 128 == 0 => ("fn_bgemm_lds_l128x128", 128, 128, 256, 1),
        65..=8192 => {
            return Err(format!(
                "Occamy prefill projection needs N divisible by 128, got {n}"
            ));
        }
        _ => return Err(format!("Occamy dense projection has no band for M {m}")),
    };
    if k % (16 * splits) != 0 {
        return Err(format!("K {k} is not divisible across {splits} splits"));
    }
    let grid = [n.div_ceil(columns), m.div_ceil(rows), splits];
    Ok(DenseProjection {
        symbol,
        grid,
        threads,
        splits,
        partial_bytes: if splits > 1 {
            u64::from(splits) * u64::from(m) * u64::from(n) * 4
        } else {
            0
        },
        counter_bytes: if splits > 1 {
            u64::from(grid[0]) * u64::from(grid[1]) * 4
        } else {
            0
        },
    })
}
