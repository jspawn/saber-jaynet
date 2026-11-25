use std::path::Path;
use std::rc::Rc;

use anyhow::ensure;
use saber_exec::{LaunchCtx, Launcher, RankPeer};
use saber_hip::{DevicePtr, HipThread, Stream};
use saber_kernels::abi::{self, SigArg};
use saber_kernels::dflash::{self, DflashOp, DflashRules, DflashShape};
use saber_kernels::rules::Shape;
use saber_mem::ledger::{Ledger, Pool};
use saber_mem::upload::{HipSink, PinnedRing, Upload, UploadConfig};
use saber_model::ripple::RippleDraftConfig;

use super::super::reference::rope_table_values;
use super::{
    buffer, descriptor, dispatch, float, gemm_bf16, scalar, Buffers, Collective,
    LOCAL_CACHE_CAPACITY,
};

pub struct Drafter {
    rank: u32,
    config: RippleDraftConfig,
    buffers: Buffers,
    trace: bool,
    fp8: bool,
    cache_capacity: u32,
    _weight_arena: Rc<DevicePtr>,
}

impl Drafter {
    pub(super) fn load(
        hip: &HipThread,
        stream: &Stream,
        pack: &Path,
        target: &Buffers,
        config: RippleDraftConfig,
        rank: u32,
        max_ctx: u32,
        trace: bool,
    ) -> anyhow::Result<Self> {
        let upload = Upload::open(
            pack,
            rank,
            UploadConfig {
                expect_tp: Some(2),
                ..UploadConfig::default()
            },
        )?;
        let plan = upload.plan();
        let mut ledger = Ledger::new();
        ledger.declare(Pool::Vram(rank), plan.arena_bytes.max(1))?;
        ledger.declare(Pool::HostPinned, plan.staging_bytes.max(1))?;
        let reserved = upload.reserve(&mut ledger)?;
        let mut staging = hip.host_malloc(reserved.staging_bytes())?;
        let ring = PinnedRing::new(&mut staging, &reserved)?;
        let arena = Rc::new(hip.malloc(plan.arena_bytes)?);
        let mut sink = HipSink::over(hip, &reserved, Rc::clone(&arena))?;
        let mut buffers = Buffers::new();
        let mut records = 0;
        let mut fp8 = false;
        upload.run(&mut sink, &ring, |record| {
            fp8 |= record.name == "aux_hidden_projection.ws";
            buffers.insert_external(&format!("draft.{}", record.name), record.addr);
            records += 1;
        })?;
        ensure!(records == plan.records);
        buffers.insert_external("target_embed", target.addr("model.embed_tokens.weight")?);
        buffers.insert_external("target_head", target.addr("lm_head.weight")?);
        let mut draft = Self {
            rank,
            config,
            buffers,
            trace,
            fp8,
            cache_capacity: max_ctx.min(LOCAL_CACHE_CAPACITY),
            _weight_arena: arena,
        };
        draft.allocate(hip, stream, max_ctx)?;
        println!(
            "ripple rank{rank}: DFlash2 {records} records, {:.3} GiB, {}",
            plan.payload_bytes as f64 / (1u64 << 30) as f64,
            if fp8 { "FP8" } else { "BF16" }
        );
        Ok(draft)
    }

    fn mark(&self, hip: &HipThread, stream: &Stream, stage: &str) -> anyhow::Result<()> {
        if self.trace {
            hip.sync_stream(stream)?;
            println!("ripple rank{}: draft {stage}", self.rank);
        }
        Ok(())
    }

    fn allocate(&mut self, hip: &HipThread, stream: &Stream, max_ctx: u32) -> anyhow::Result<()> {
        let h = self.config.hidden_size as usize;
        let nq = self.config.num_attention_heads as usize / 2;
        let nkv = self.config.num_key_value_heads as usize / 2;
        let qw = nq * 128;
        let kvw = nkv * 128;
        let intermediate = self.config.intermediate_size as usize / 2;
        let local_vocab = self.config.vocab_size as usize / 2;
        let context_width = self.config.num_hidden_layers as usize * 2 * kvw;
        let kernel_width = 2 * 4 * (h / 16);
        for (name, elements) in [
            ("d.ctx_proj", 64 * h),
            ("d.ctx_normed", 64 * h),
            ("d.ctx_kv", 64 * context_width),
            ("d.emb0", 16 * h),
            ("d.emb1", 16 * h),
            ("d.h0", 16 * h),
            ("d.h1", 16 * h),
            ("d.res0", 16 * h),
            ("d.res1", 16 * h),
            ("d.coeff", 16 * kernel_width),
            ("d.prepared", 16 * h),
            ("d.conv_out", 16 * h),
            ("d.qkv", 16 * (qw + 2 * kvw)),
            ("d.qn", 16 * qw),
            ("d.kn", 16 * kvw),
            ("d.v", 16 * kvw),
            ("d.attn", 16 * qw),
            ("d.o", 16 * h),
            ("d.o_sum", 16 * h),
            ("d.gate_up", 16 * 2 * intermediate),
            ("d.act_bf16", 16 * intermediate),
            ("d.down", 16 * h),
            ("d.down_sum", 16 * h),
            ("d.logits", 15 * local_vocab),
            ("d.selector_hidden", 15 * 256),
        ] {
            self.buffers.alloc(hip, name, elements * 2)?;
        }
        for (name, elements) in [
            ("d.aux_q", 64 * 5 * h),
            ("d.q", 16 * h),
            ("d.attn_q", 16 * qw),
            ("d.act_q", 16 * intermediate),
        ] {
            self.buffers.alloc(hip, name, elements)?;
        }
        for (name, elements) in [
            ("d.aux_s", 64 * 5 * h / 128),
            ("d.q_s", 16 * h / 128),
            ("d.attn_s", 16 * qw / 128),
            ("d.act_s", 16 * intermediate / 128),
            ("d.top_vals", 15 * 32),
            ("d.unary", 15 * 32),
            ("d.scores", 15 * 32 * 32),
        ] {
            self.buffers.alloc(hip, name, elements * 4)?;
        }
        self.buffers.alloc(hip, "d.top_ids", 15 * 32 * 8)?;
        self.buffers.alloc(hip, "d.candidates", 15 * 32 * 8)?;
        self.buffers.alloc(hip, "d.ids", 16 * 4)?;
        self.buffers.alloc(hip, "d.positions", 64 * 8)?;
        let table = rope_table_values(
            self.config.rope_theta,
            self.config.head_dim,
            max_ctx as usize + 16,
        );
        self.buffers.upload_bf16(hip, stream, "d.rope", &table)?;
        for layer in 0..self.config.num_hidden_layers {
            let slots = if layer == 4 {
                max_ctx as usize + 16
            } else {
                self.cache_capacity as usize
            };
            for name in [format!("d.kv.{layer}.k"), format!("d.kv.{layer}.v")] {
                self.buffers.alloc(hip, &name, slots * kvw)?;
            }
            for name in [format!("d.kv.{layer}.ks"), format!("d.kv.{layer}.vs")] {
                self.buffers.alloc(hip, &name, slots * nkv * 4)?;
            }
        }
        Ok(())
    }

    fn custom(
        &self,
        name: &str,
        grid: [u32; 3],
        block: u32,
        signature: &'static [SigArg],
        values: Vec<saber_kernels::rules::ArgValue>,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
    ) -> anyhow::Result<()> {
        let desc = descriptor("ripple_draft", name, grid, block, signature, values)?;
        dispatch(
            stream,
            ctx,
            launcher,
            &desc,
            &self.buffers,
            Shape::new(0, 0, 0),
        )
    }

    fn gemm(
        &mut self,
        rules: &DflashRules,
        input: &str,
        scales: &str,
        weight: &str,
        output: &str,
        rows: u32,
        n: u32,
        k: u32,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
    ) -> anyhow::Result<()> {
        if !self.fp8 {
            self.buffers.alias("a", input)?;
            self.buffers.alias("w", &format!("draft.{weight}.w"))?;
            self.buffers.alias("c", output)?;
            let desc = dflash::resolve_head_bgemm(rows, n, k, rules)?;
            return dispatch(
                stream,
                ctx,
                launcher,
                &desc,
                &self.buffers,
                Shape::new(rows, n, k),
            );
        }
        self.buffers.alias("a", input)?;
        self.buffers.alias("as", scales)?;
        self.buffers.alias("w", &format!("draft.{weight}.w"))?;
        self.buffers.alias("ws", &format!("draft.{weight}.ws"))?;
        self.buffers.alias("c", output)?;
        let desc = dflash::resolve(
            DflashOp::blockscale_for(rows, rules),
            DflashShape::new(rows, n, k),
            rules,
        )?;
        dispatch(
            stream,
            ctx,
            launcher,
            &desc,
            &self.buffers,
            Shape::new(rows, n, k),
        )
    }

    pub(super) fn context(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        bf16: &DflashRules,
        rules: &DflashRules,
        target: &Buffers,
        rows: usize,
        start: usize,
    ) -> anyhow::Result<()> {
        let h = self.config.hidden_size as usize;
        let nkv = self.config.num_key_value_heads as usize / 2;
        let kvw = nkv * 128;
        let context_width = self.config.num_hidden_layers as usize * 2 * kvw;
        self.buffers
            .insert_external("d.target_aux", target.addr("aux")?);
        let positions: Vec<u8> = (start..start + rows)
            .flat_map(|p| (p as i64).to_le_bytes())
            .collect();
        self.buffers
            .upload(hip, stream, "d.positions", &positions)?;
        if self.fp8 {
            self.custom(
                "ripple_act_quant_bf16",
                [rows as u32, (5 * h / 128) as u32, 1],
                256,
                abi::SIG_DFLASH_ACT_QUANT,
                vec![
                    buffer("d.target_aux"),
                    buffer("d.aux_q"),
                    buffer("d.aux_s"),
                    scalar(rows as i64),
                    scalar((5 * h) as i64),
                ],
                stream,
                ctx,
                launcher,
            )?;
        }
        let aux_input = if self.fp8 { "d.aux_q" } else { "d.target_aux" };
        self.gemm(
            rules,
            aux_input,
            "d.aux_s",
            "aux_hidden_projection",
            "d.ctx_proj",
            rows as u32,
            h as u32,
            (5 * h) as u32,
            stream,
            ctx,
            launcher,
        )?;
        self.custom(
            "ripple_rmsnorm_bf16",
            [rows as u32, 1, 1],
            256,
            abi::SIG_RIPPLE_RMSNORM,
            vec![
                buffer("d.ctx_proj"),
                buffer("draft.context_kv.hidden_norm.weight"),
                buffer("d.ctx_normed"),
                scalar(rows as i64),
                scalar(h as i64),
                float(self.config.rms_norm_eps as f32),
            ],
            stream,
            ctx,
            launcher,
        )?;
        gemm_bf16(
            bf16,
            stream,
            ctx,
            launcher,
            &mut self.buffers,
            "d.ctx_normed",
            "draft.context_kv.kv_projection.w",
            "d.ctx_kv",
            rows as u32,
            context_width as u32,
            h as u32,
        )?;
        for layer in 0..self.config.num_hidden_layers {
            let rope = i64::from(layer != 4);
            let window = if layer == 4 {
                0
            } else {
                self.config.sliding_window as i64
            };
            let cache_capacity = if layer == 4 {
                0
            } else {
                i64::from(self.cache_capacity)
            };
            self.custom(
                "ripple_draft_ctx_append_fp8",
                [rows as u32, nkv as u32, 1],
                32,
                abi::SIG_RIPPLE_DRAFT_CTX_APPEND,
                vec![
                    buffer("d.ctx_kv"),
                    buffer(&format!("draft.layer.{layer}.attn.k_norm.weight")),
                    buffer("d.rope"),
                    buffer("d.positions"),
                    buffer(&format!("d.kv.{layer}.k")),
                    buffer(&format!("d.kv.{layer}.v")),
                    buffer(&format!("d.kv.{layer}.ks")),
                    buffer(&format!("d.kv.{layer}.vs")),
                    scalar(rows as i64),
                    scalar(nkv as i64),
                    scalar(context_width as i64),
                    scalar((layer as usize * 2 * kvw) as i64),
                    float(self.config.rms_norm_eps as f32),
                    scalar(rope),
                    scalar(window),
                    scalar(cache_capacity),
                ],
                stream,
                ctx,
                launcher,
            )?;
        }
        Ok(())
    }

    pub(super) fn propose(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        collective: &Collective,
        peer: &RankPeer,
        bf16: &DflashRules,
        rules: &DflashRules,
        anchor: u32,
        position: usize,
    ) -> anyhow::Result<Vec<u32>> {
        let h = self.config.hidden_size as usize;
        let nq = self.config.num_attention_heads as usize / 2;
        let nkv = self.config.num_key_value_heads as usize / 2;
        let qw = nq * 128;
        let kvw = nkv * 128;
        let qkv_width = qw + 2 * kvw;
        let intermediate = self.config.intermediate_size as usize / 2;
        let local_vocab = self.config.vocab_size as usize / 2;
        let eps = self.config.rms_norm_eps as f32;
        let mut ids = [self.config.dflash_config.mask_token_id; 16];
        ids[0] = anchor;
        let id_bytes: Vec<u8> = ids
            .iter()
            .flat_map(|id| (*id as i32).to_le_bytes())
            .collect();
        self.buffers.upload(hip, stream, "d.ids", &id_bytes)?;
        let positions: Vec<u8> = (position..position + 16)
            .flat_map(|p| (p as i64).to_le_bytes())
            .collect();
        self.buffers
            .upload(hip, stream, "d.positions", &positions)?;
        let local_vocab_u32 = local_vocab as u32;
        let embed = descriptor(
            "head",
            "saber_embed_bf16",
            [16, 1, 1],
            256,
            abi::SIG_EMBED,
            vec![
                buffer("target_embed"),
                buffer("d.ids"),
                buffer("d.emb1"),
                scalar(h as i64),
                scalar((self.rank * local_vocab_u32) as i64),
                scalar(((self.rank + 1) * local_vocab_u32) as i64),
            ],
        )?;
        dispatch(
            stream,
            ctx,
            launcher,
            &embed,
            &self.buffers,
            Shape::new(0, 0, 0),
        )?;
        collective.reduce(
            stream,
            ctx,
            launcher,
            &self.buffers,
            "d.emb1",
            "d.emb0",
            16 * h,
        )?;
        self.custom(
            "ripple_rmsnorm_bf16",
            [16, 1, 1],
            256,
            abi::SIG_RIPPLE_RMSNORM,
            vec![
                buffer("d.emb0"),
                buffer("draft.layer.0.input_layernorm.weight"),
                buffer("d.h0"),
                scalar(16),
                scalar(h as i64),
                float(eps),
            ],
            stream,
            ctx,
            launcher,
        )?;
        self.mark(hip, stream, "embedding complete")?;
        for layer in 0..self.config.num_hidden_layers {
            let residual = if layer == 0 {
                "d.emb0".to_owned()
            } else {
                "d.res1".into()
            };
            let attention_residual = "d.res0";
            let next_residual = "d.res1";
            for (tag, is_attention) in [("attention_conv", true), ("mlp_conv", false)] {
                self.mark(hip, stream, &format!("layer {layer} {tag} begin"))?;
                let input = if is_attention { "d.h0" } else { "d.h1" };
                gemm_bf16(
                    bf16,
                    stream,
                    ctx,
                    launcher,
                    &mut self.buffers,
                    input,
                    &format!("draft.layer.{layer}.{tag}.kernel_projection.w"),
                    "d.coeff",
                    16,
                    (2 * 4 * (h / 16)) as u32,
                    h as u32,
                )?;
                if self.fp8 {
                    self.custom(
                        "ripple_draft_conv4_quant_bf16",
                        [(16 * h / 128).div_ceil(4) as u32, 1, 1],
                        128,
                        abi::SIG_DFLASH_CONV_QUANT,
                        vec![
                            buffer(input),
                            buffer("d.coeff"),
                            buffer(&format!("draft.layer.{layer}.{tag}.base")),
                            buffer(if is_attention { "d.q" } else { "d.act_q" }),
                            buffer(if is_attention { "d.q_s" } else { "d.act_s" }),
                            scalar(16),
                            scalar(h as i64),
                            scalar((h / 16) as i64),
                            scalar(16),
                            scalar(0),
                        ],
                        stream,
                        ctx,
                        launcher,
                    )?;
                } else {
                    self.custom(
                        "ripple_draft_conv4_bf16",
                        [(16 * h / 4).div_ceil(128) as u32, 1, 1],
                        128,
                        abi::SIG_DFLASH_CONV,
                        vec![
                            buffer(input),
                            buffer("d.coeff"),
                            buffer(&format!("draft.layer.{layer}.{tag}.base")),
                            buffer("d.prepared"),
                            scalar(16),
                            scalar(h as i64),
                            scalar((h / 16) as i64),
                            scalar(16),
                            scalar(0),
                        ],
                        stream,
                        ctx,
                        launcher,
                    )?;
                }
                let prepared = if self.fp8 {
                    if is_attention {
                        "d.q"
                    } else {
                        "d.act_q"
                    }
                } else {
                    "d.prepared"
                };
                if is_attention {
                    self.gemm(
                        rules,
                        prepared,
                        "d.q_s",
                        &format!("layer.{layer}.attn.qkv"),
                        "d.qkv",
                        16,
                        qkv_width as u32,
                        h as u32,
                        stream,
                        ctx,
                        launcher,
                    )?;
                    self.custom(
                        "ripple_draft_qk_rope_h128",
                        [16, (nq + 2 * nkv) as u32, 1],
                        32,
                        abi::SIG_RIPPLE_DRAFT_QK,
                        vec![
                            buffer("d.qkv"),
                            buffer(&format!("draft.layer.{layer}.attn.q_norm.weight")),
                            buffer(&format!("draft.layer.{layer}.attn.k_norm.weight")),
                            buffer("d.rope"),
                            buffer("d.positions"),
                            buffer("d.qn"),
                            buffer("d.kn"),
                            buffer("d.v"),
                            scalar(16),
                            scalar(nq as i64),
                            scalar(nkv as i64),
                            float(eps),
                            scalar(i64::from(layer != 4)),
                        ],
                        stream,
                        ctx,
                        launcher,
                    )?;
                    let window = if layer == 4 {
                        0
                    } else {
                        self.config.sliding_window as i64
                    };
                    let cache_capacity = if layer == 4 {
                        0
                    } else {
                        i64::from(self.cache_capacity)
                    };
                    self.custom(
                        "ripple_kv_append_h128_fp8",
                        [16, nkv as u32, 1],
                        32,
                        abi::SIG_RIPPLE_KV_APPEND_FP8,
                        vec![
                            buffer("d.kn"),
                            buffer("d.v"),
                            buffer(&format!("d.kv.{layer}.k")),
                            buffer(&format!("d.kv.{layer}.v")),
                            buffer(&format!("d.kv.{layer}.ks")),
                            buffer(&format!("d.kv.{layer}.vs")),
                            buffer("d.positions"),
                            scalar(16),
                            scalar(nkv as i64),
                            scalar(window),
                            scalar(cache_capacity),
                        ],
                        stream,
                        ctx,
                        launcher,
                    )?;
                    self.custom(
                        "ripple_draft_attn_h128_fp8",
                        [16, nq as u32, 1],
                        32,
                        abi::SIG_RIPPLE_DRAFT_ATTN,
                        vec![
                            buffer("d.qn"),
                            buffer(&format!("d.kv.{layer}.k")),
                            buffer(&format!("d.kv.{layer}.v")),
                            buffer(&format!("d.kv.{layer}.ks")),
                            buffer(&format!("d.kv.{layer}.vs")),
                            buffer("d.attn"),
                            scalar(16),
                            scalar(nq as i64),
                            scalar(nkv as i64),
                            scalar(position as i64),
                            scalar(window),
                            scalar(cache_capacity),
                        ],
                        stream,
                        ctx,
                        launcher,
                    )?;
                    if self.fp8 {
                        self.custom(
                            "ripple_act_quant_bf16",
                            [16, (qw / 128) as u32, 1],
                            256,
                            abi::SIG_DFLASH_ACT_QUANT,
                            vec![
                                buffer("d.attn"),
                                buffer("d.attn_q"),
                                buffer("d.attn_s"),
                                scalar(16),
                                scalar(qw as i64),
                            ],
                            stream,
                            ctx,
                            launcher,
                        )?;
                    }
                    let attention_input = if self.fp8 { "d.attn_q" } else { "d.attn" };
                    self.gemm(
                        rules,
                        attention_input,
                        "d.attn_s",
                        &format!("layer.{layer}.attn.o_proj"),
                        "d.o",
                        16,
                        h as u32,
                        qw as u32,
                        stream,
                        ctx,
                        launcher,
                    )?;
                    collective.reduce(
                        stream,
                        ctx,
                        launcher,
                        &self.buffers,
                        "d.o",
                        "d.o_sum",
                        16 * h,
                    )?;
                } else {
                    self.gemm(
                        rules,
                        prepared,
                        "d.act_s",
                        &format!("layer.{layer}.mlp.gate_up"),
                        "d.gate_up",
                        16,
                        (2 * intermediate) as u32,
                        h as u32,
                        stream,
                        ctx,
                        launcher,
                    )?;
                    if self.fp8 {
                        self.custom(
                            "ripple_draft_swiglu_quant_bf16",
                            [16, (intermediate / 128) as u32, 1],
                            256,
                            abi::SIG_DFLASH_SILU,
                            vec![
                                buffer("d.gate_up"),
                                buffer("d.act_q"),
                                buffer("d.act_s"),
                                scalar(16),
                                scalar(intermediate as i64),
                            ],
                            stream,
                            ctx,
                            launcher,
                        )?;
                    } else {
                        self.custom(
                            "ripple_draft_swiglu_bf16",
                            [(16 * intermediate).div_ceil(256) as u32, 1, 1],
                            256,
                            abi::SIG_RIPPLE_DRAFT_SWIGLU_BF16,
                            vec![
                                buffer("d.gate_up"),
                                buffer("d.act_bf16"),
                                scalar(16),
                                scalar(intermediate as i64),
                            ],
                            stream,
                            ctx,
                            launcher,
                        )?;
                    }
                    let activation = if self.fp8 { "d.act_q" } else { "d.act_bf16" };
                    self.gemm(
                        rules,
                        activation,
                        "d.act_s",
                        &format!("layer.{layer}.mlp.down"),
                        "d.down",
                        16,
                        h as u32,
                        intermediate as u32,
                        stream,
                        ctx,
                        launcher,
                    )?;
                    collective.reduce(
                        stream,
                        ctx,
                        launcher,
                        &self.buffers,
                        "d.down",
                        "d.down_sum",
                        16 * h,
                    )?;
                }
                self.custom(
                    "ripple_draft_conv4_bf16",
                    [(16 * h / 4).div_ceil(128) as u32, 1, 1],
                    128,
                    abi::SIG_DFLASH_CONV,
                    vec![
                        buffer(if is_attention {
                            "d.o_sum"
                        } else {
                            "d.down_sum"
                        }),
                        buffer("d.coeff"),
                        buffer(&format!("draft.layer.{layer}.{tag}.base")),
                        buffer("d.conv_out"),
                        scalar(16),
                        scalar(h as i64),
                        scalar((h / 16) as i64),
                        scalar(16),
                        scalar(1),
                    ],
                    stream,
                    ctx,
                    launcher,
                )?;
                let norm = if is_attention {
                    format!("draft.layer.{layer}.post_attention_layernorm.weight")
                } else if layer == 4 {
                    "draft.norm.weight".into()
                } else {
                    format!("draft.layer.{}.input_layernorm.weight", layer + 1)
                };
                let residual_input: &str = if is_attention {
                    residual.as_str()
                } else {
                    attention_residual
                };
                self.custom(
                    "ripple_add_rms_bf16",
                    [16, 1, 1],
                    256,
                    abi::SIG_RIPPLE_ADD_RMS,
                    vec![
                        buffer("d.conv_out"),
                        buffer(residual_input),
                        buffer(&norm),
                        buffer(if is_attention { "d.h1" } else { "d.h0" }),
                        buffer(if is_attention {
                            attention_residual
                        } else {
                            next_residual
                        }),
                        scalar(16),
                        scalar(h as i64),
                        float(eps),
                    ],
                    stream,
                    ctx,
                    launcher,
                )?;
            }
        }
        self.buffers.alias("a", &format!("d.h0@{}", h * 2))?;
        self.buffers.alias("w", "target_head")?;
        self.buffers.alias("c", "d.logits")?;
        let head = dflash::resolve_head_bgemm(15, local_vocab_u32, h as u32, bf16)?;
        dispatch(
            stream,
            ctx,
            launcher,
            &head,
            &self.buffers,
            Shape::new(15, local_vocab_u32, h as u32),
        )?;
        self.mark(hip, stream, "full head complete")?;
        self.custom(
            "ripple_draft_top32_bf16",
            [15, 1, 1],
            32,
            abi::SIG_DFLASH_TOPK,
            vec![
                buffer("d.logits"),
                buffer("d.top_vals"),
                buffer("d.top_ids"),
                scalar(15),
                scalar(local_vocab as i64),
                scalar(32),
            ],
            stream,
            ctx,
            launcher,
        )?;
        self.mark(hip, stream, "top32 complete")?;
        gemm_bf16(
            bf16,
            stream,
            ctx,
            launcher,
            &mut self.buffers,
            &format!("d.h0@{}", h * 2),
            "draft.selector.hidden_projection.w",
            "d.selector_hidden",
            15,
            256,
            h as u32,
        )?;
        hip.sync_stream(stream)?;
        let vals = self
            .buffers
            .download(hip, stream, "d.top_vals", 15 * 32 * 4)?;
        let local_ids = self
            .buffers
            .download(hip, stream, "d.top_ids", 15 * 32 * 8)?;
        let other_vals = peer
            .exchange()
            .swap_bytes(&format!("ripple.draft.vals.{position}"), &vals)?;
        let other_ids = peer
            .exchange()
            .swap_bytes(&format!("ripple.draft.ids.{position}"), &local_ids)?;
        let (vals0, ids0, vals1, ids1) = if self.rank == 0 {
            (vals, local_ids, other_vals, other_ids)
        } else {
            (other_vals, other_ids, vals, local_ids)
        };
        let mut candidates = Vec::with_capacity(15 * 32);
        let mut unary = Vec::with_capacity(15 * 32);
        for row in 0..15 {
            let mut entries = Vec::with_capacity(64);
            for (rank, (values, indices)) in [(&vals0, &ids0), (&vals1, &ids1)].iter().enumerate() {
                for j in 0..32 {
                    let offset = (row * 32 + j) * 4;
                    let value = f32::from_le_bytes(values[offset..offset + 4].try_into().unwrap());
                    let offset = (row * 32 + j) * 8;
                    let index = i64::from_le_bytes(indices[offset..offset + 8].try_into().unwrap());
                    entries.push((value, rank as i64 * local_vocab as i64 + index));
                }
            }
            entries.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            entries.truncate(32);
            for (value, id) in entries {
                unary.extend_from_slice(&value.to_le_bytes());
                candidates.extend_from_slice(&id.to_le_bytes());
            }
        }
        self.buffers.upload(hip, stream, "d.unary", &unary)?;
        self.buffers
            .upload(hip, stream, "d.candidates", &candidates)?;
        self.custom(
            "ripple_draft_score32_r256",
            [15 * 32 * 32, 1, 1],
            32,
            abi::SIG_DFLASH_SELECTOR,
            vec![
                buffer("draft.selector.score_edges.pred_cb"),
                buffer("draft.selector.score_edges.succ_cb"),
                buffer("d.candidates"),
                buffer("d.ids"),
                buffer("d.unary"),
                buffer("d.selector_hidden"),
                buffer("d.scores"),
                scalar(15),
                scalar(15),
                scalar(256),
                scalar(32),
            ],
            stream,
            ctx,
            launcher,
        )?;
        hip.sync_stream(stream)?;
        let score_bytes = self
            .buffers
            .download(hip, stream, "d.scores", 15 * 32 * 32 * 4)?;
        let scores: Vec<f32> = score_bytes
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        let candidate_ids: Vec<i64> = candidates
            .chunks_exact(8)
            .map(|x| i64::from_le_bytes(x.try_into().unwrap()))
            .collect();
        let mut result = Vec::with_capacity(15);
        let mut predecessor = 0usize;
        for row in 0..15 {
            let base = (row * 32 + if row == 0 { 0 } else { predecessor }) * 32;
            let mut best = 0usize;
            for candidate in 1..32 {
                if scores[base + candidate] > scores[base + best] {
                    best = candidate;
                }
            }
            result.push(candidate_ids[row * 32 + best] as u32);
            predecessor = best;
        }
        Ok(result)
    }
}
