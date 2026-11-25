#![allow(clippy::too_many_arguments)]

pub mod draft;

use std::collections::{BTreeMap, BTreeSet};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{ensure, Context};
use saber_exec::{
    ArgBindings, LaunchCounters, LaunchCtx, Launcher, LoadCounters, PluginSet, RankPeer,
    TcpExchange,
};
use saber_hip::{DeviceAddr, DevicePtr, HipThread, Job, MallocFlags, Stream, StreamFlags};
use saber_kernels::abi::{self, SigArg};
use saber_kernels::dflash::{self, DflashOp, DflashRules, DflashShape};
use saber_kernels::rules::{self, ALayout, ArgValue, LaunchDescriptor, LaunchRules, Shape, Slot};
use saber_mem::ledger::{Ledger, Pool};
use saber_mem::upload::{HipSink, PinnedRing, Upload, UploadConfig};
use saber_model::ripple::RippleConfig;
use saber_serve::engine::SampledLogprobs;
use saber_serve::sampler::{
    EntropyNoise, NoiseSource, PenaltyContext, RowParams, Sampler, SeededNoise,
};
use saber_serve::sampling::SamplingParams;

use super::reference::{bf16_to_f32, rope_table};

const AR_MB: u64 = 16 << 20;
const AR_BLOCKS: u64 = 512;
pub const LOCAL_CACHE_CAPACITY: u32 = 4096;

pub struct ExecArgs {
    pub pack: PathBuf,
    pub draft: Option<PathBuf>,
    pub checkpoint: PathBuf,
    pub prebuilt: PathBuf,
    pub device: i32,
    pub tokens: Vec<u32>,
    pub decode: u32,
    pub max_ctx: u32,
    pub max_rows: usize,
    pub rank: u32,
    pub peer_port: u16,
    pub trace_layers: bool,
}

fn buffer(name: &str) -> ArgValue {
    ArgValue::Buffer {
        buffer: name.into(),
    }
}

fn scalar(value: i64) -> ArgValue {
    ArgValue::I32 { ival: value }
}

fn float(value: f32) -> ArgValue {
    ArgValue::F32 { fval: value }
}

fn descriptor(
    plugin: &str,
    kernel: &str,
    grid: [u32; 3],
    block: u32,
    signature: &'static [SigArg],
    values: Vec<ArgValue>,
) -> anyhow::Result<LaunchDescriptor> {
    ensure!(signature.len() == values.len(), "{kernel}: argument count");
    let (offsets, kernarg_bytes) = abi::pack(signature);
    let args = signature
        .iter()
        .zip(offsets)
        .zip(values)
        .map(|((argument, (offset, size)), value)| Slot {
            name: argument.name.to_string(),
            offset,
            size,
            value,
        })
        .collect();
    Ok(LaunchDescriptor {
        op: kernel.into(),
        plugin: plugin.into(),
        kernel: kernel.into(),
        grid,
        block: [block, 1, 1],
        shared_bytes: 0,
        kernarg_bytes,
        args,
        workspaces: vec![],
        n_launches: 1,
    })
}

struct Buffers {
    device: BTreeMap<String, DevicePtr>,
    external: BTreeMap<String, DeviceAddr>,
}

impl Buffers {
    fn new() -> Self {
        Self {
            device: BTreeMap::new(),
            external: BTreeMap::new(),
        }
    }

    fn alloc(&mut self, hip: &HipThread, name: &str, bytes: usize) -> anyhow::Result<()> {
        let pointer = hip.malloc(bytes.max(4) as u64)?;
        self.device.insert(name.into(), pointer);
        Ok(())
    }

    fn insert_external(&mut self, name: &str, address: DeviceAddr) {
        self.external.insert(name.into(), address);
    }

    fn alias(&mut self, name: &str, target: &str) -> anyhow::Result<()> {
        let address = self.addr(target)?;
        self.external.insert(name.into(), address);
        Ok(())
    }

    fn addr(&self, name: &str) -> anyhow::Result<DeviceAddr> {
        let (base, offset) = name
            .split_once('@')
            .map_or(Ok((name, 0)), |(base, offset)| {
                offset
                    .parse::<u64>()
                    .map(|offset| (base, offset))
                    .map_err(|error| anyhow::anyhow!("{name}: invalid byte offset: {error}"))
            })?;
        let address = if let Some(pointer) = self.device.get(base) {
            pointer.addr()
        } else {
            *self
                .external
                .get(base)
                .ok_or_else(|| anyhow::anyhow!("missing device buffer {base}"))?
        };
        address
            .offset(offset)
            .ok_or_else(|| anyhow::anyhow!("invalid device offset in {name}"))
    }

    fn get(&self, name: &str) -> anyhow::Result<&DevicePtr> {
        self.device
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing device allocation {name}"))
    }

    fn upload(
        &self,
        hip: &HipThread,
        stream: &Stream,
        name: &str,
        bytes: &[u8],
    ) -> anyhow::Result<()> {
        hip.memcpy_h2d_blocking(self.get(name)?, bytes, stream)?;
        Ok(())
    }

    fn upload_bf16(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        name: &str,
        values: &[u16],
    ) -> anyhow::Result<()> {
        let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.alloc(hip, name, bytes.len())?;
        self.upload(hip, stream, name, &bytes)
    }

    fn copy(
        &self,
        hip: &HipThread,
        stream: &Stream,
        dst: &str,
        dst_offset: u64,
        src: &str,
        bytes: u64,
    ) -> anyhow::Result<()> {
        hip.run_batch(&[Job::MemcpyD2d {
            dst: self.get(dst)?,
            dst_offset,
            src: self.get(src)?,
            src_offset: 0,
            bytes,
            stream,
        }])?;
        Ok(())
    }

    fn download(
        &self,
        hip: &HipThread,
        stream: &Stream,
        name: &str,
        bytes: usize,
    ) -> anyhow::Result<Vec<u8>> {
        let mut output = vec![0u8; bytes];
        hip.memcpy_d2h_blocking(&mut output, self.get(name)?, stream)?;
        Ok(output)
    }
}

fn dispatch(
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    desc: &LaunchDescriptor,
    buffers: &Buffers,
    shape: Shape,
) -> anyhow::Result<()> {
    let workspaces: BTreeSet<&str> = desc.workspaces.iter().map(|w| w.name.as_str()).collect();
    let mut bindings = ArgBindings::new(shape);
    for argument in &desc.args {
        if let ArgValue::Buffer { buffer } = &argument.value {
            if !workspaces.contains(buffer.as_str()) {
                bindings.bind(buffer, buffers.addr(buffer)?);
            }
        }
    }
    launcher.dispatch_one(ctx, desc, &bindings, stream)?;
    Ok(())
}

fn custom(
    name: &str,
    grid: [u32; 3],
    block: u32,
    signature: &'static [SigArg],
    values: Vec<ArgValue>,
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    buffers: &Buffers,
) -> anyhow::Result<()> {
    let desc = descriptor("ripple_spike", name, grid, block, signature, values)?;
    dispatch(stream, ctx, launcher, &desc, buffers, Shape::new(0, 0, 0))
}

fn gemm_bf16(
    rules: &DflashRules,
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    buffers: &mut Buffers,
    input: &str,
    weight: &str,
    output: &str,
    rows: u32,
    n: u32,
    k: u32,
) -> anyhow::Result<()> {
    buffers.alias("a", input)?;
    buffers.alias("w", weight)?;
    buffers.alias("c", output)?;
    let op = if rows <= 16 {
        DflashOp::BgemmBf16
    } else {
        DflashOp::BgemmBf16Mt
    };
    let desc = dflash::resolve(op, DflashShape::new(rows, n, k), rules)?;
    dispatch(
        stream,
        ctx,
        launcher,
        &desc,
        buffers,
        Shape::new(rows, n, k),
    )
}

fn gemm_mxfp4(
    rules: &LaunchRules,
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    buffers: &mut Buffers,
    input: &str,
    scales: &str,
    weight_base: &str,
    output: &str,
    rows: u32,
    n: u32,
    k: u32,
) -> anyhow::Result<()> {
    buffers.alias("a", input)?;
    buffers.alias("as", scales)?;
    buffers.alias("w", &format!("{weight_base}.w"))?;
    buffers.alias("ws", &format!("{weight_base}.ws"))?;
    buffers.alias("wref", &format!("{weight_base}.wref"))?;
    buffers.alias("c", output)?;
    let desc = rules::resolve_with_layout(
        rules::Op::gemm(),
        Shape::new(rows, n, k),
        rules,
        ALayout::RowMajor,
    )?;
    dispatch(
        stream,
        ctx,
        launcher,
        &desc,
        buffers,
        Shape::new(rows, n, k),
    )
}

pub struct Collective {
    scratch: DevicePtr,
    flags: DevicePtr,
    seq: DevicePtr,
    peer_scratch: DeviceAddr,
    peer_flags: DeviceAddr,
}

impl Collective {
    fn new(hip: &HipThread, stream: &Stream, rank: u32, peer: &RankPeer) -> anyhow::Result<Self> {
        let scratch = hip.malloc_with_flags(2 * AR_MB, MallocFlags::FineGrained)?;
        let flags = hip.malloc_with_flags(AR_BLOCKS * 4, MallocFlags::FineGrained)?;
        let seq = hip.malloc(AR_BLOCKS * 4)?;
        for pointer in [&scratch, &flags, &seq] {
            hip.memset(pointer, 0, stream)?;
        }
        hip.sync_stream(stream)?;
        let peer_scratch =
            hip.ipc_import(peer.swap_handle("ripple.ar.scratch", hip.ipc_export(&scratch)?)?)?;
        let peer_flags =
            hip.ipc_import(peer.swap_handle("ripple.ar.flags", hip.ipc_export(&flags)?)?)?;
        ensure!(rank < 2, "Ripple TP2 rank must be 0 or 1");
        Ok(Self {
            scratch,
            flags,
            seq,
            peer_scratch,
            peer_flags,
        })
    }

    fn reduce(
        &self,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        buffers: &Buffers,
        input: &str,
        output: &str,
        elements: usize,
    ) -> anyhow::Result<()> {
        let bytes = elements * 2;
        ensure!(bytes as u64 <= AR_MB, "exact reduce exceeds the mailbox");
        let n16 = bytes / 16;
        let blocks = (n16 / 1400)
            .clamp(4, 24)
            .min(n16.max(1))
            .min(AR_BLOCKS as usize);
        let launch = descriptor(
            "saber_ar_exact",
            "saber_ar_oneshot_2rank_exact_bf16",
            [blocks as u32, 1, 1],
            1024,
            abi::SIG_SABER_AR_EXACT,
            vec![
                buffer(input),
                buffer("ar.peer_scratch"),
                buffer("ar.my_scratch"),
                buffer(output),
                buffer("ar.peer_flags"),
                buffer("ar.my_flags"),
                buffer("ar.seq"),
                scalar(n16 as i64),
                scalar((AR_MB / 16) as i64),
                scalar(3),
                scalar(0),
            ],
        )?;
        let mut bindings = ArgBindings::new(Shape::new(0, 0, 0));
        for argument in &launch.args {
            if let ArgValue::Buffer { buffer: name } = &argument.value {
                let address = match name.as_str() {
                    "ar.peer_scratch" => self.peer_scratch,
                    "ar.my_scratch" => self.scratch.addr(),
                    "ar.peer_flags" => self.peer_flags,
                    "ar.my_flags" => self.flags.addr(),
                    "ar.seq" => self.seq.addr(),
                    _ => buffers.addr(name)?,
                };
                bindings.bind(name, address);
            }
        }
        launcher.dispatch_one(ctx, &launch, &bindings, stream)?;
        Ok(())
    }
}

struct Model {
    rank: u32,
    config: RippleConfig,
    buffers: Buffers,
    _weight_arena: Rc<DevicePtr>,
    max_rows: usize,
    max_context: usize,
    trace_layers: bool,
    prefix_cache: PrefixCache,
}

impl Model {
    fn load(
        hip: &HipThread,
        stream: &Stream,
        args: &ExecArgs,
        config: RippleConfig,
    ) -> anyhow::Result<Self> {
        ensure!(
            args.max_rows > 0 && args.max_rows <= 64,
            "--max-rows must be 1..=64"
        );
        ensure!(
            !args.tokens.is_empty(),
            "at least one prompt token is required"
        );
        ensure!(
            args.max_ctx <= config.max_position_embeddings,
            "context exceeds checkpoint maximum"
        );
        let started = Instant::now();
        let mut buffers = Buffers::new();
        let upload = Upload::open(
            &args.pack,
            args.rank,
            UploadConfig {
                expect_tp: Some(2),
                ..UploadConfig::default()
            },
        )?;
        let plan = upload.plan();
        let mut ledger = Ledger::new();
        ledger.declare(Pool::Vram(args.rank), plan.arena_bytes.max(1))?;
        ledger.declare(Pool::HostPinned, plan.staging_bytes.max(1))?;
        let reserved = upload.reserve(&mut ledger)?;
        let mut staging = hip.host_malloc(reserved.staging_bytes())?;
        let ring = PinnedRing::new(&mut staging, &reserved)?;
        let weight_arena = Rc::new(hip.malloc(plan.arena_bytes)?);
        let mut sink = HipSink::over(hip, &reserved, Rc::clone(&weight_arena))?;
        let mut records = 0usize;
        let stats = upload.run(&mut sink, &ring, |record| {
            buffers.insert_external(record.name, record.addr);
            records += 1;
        })?;
        ensure!(records == plan.records, "not every pack record landed");
        println!(
            "ripple rank{}: {} records, {:.3} GiB weights, verified load {:.1}s ({:.2} GB/s)",
            args.rank,
            records,
            plan.payload_bytes as f64 / (1u64 << 30) as f64,
            started.elapsed().as_secs_f64(),
            stats.end_to_end_gbps()
        );
        let local_window = config.sliding_window as usize;
        let mut model = Self {
            rank: args.rank,
            config,
            buffers,
            _weight_arena: weight_arena,
            max_rows: args.max_rows,
            max_context: args.max_ctx as usize,
            trace_layers: args.trace_layers,
            prefix_cache: PrefixCache::new(args.max_ctx as usize, local_window),
        };
        model.allocate_scratch(hip, stream, args.max_ctx)?;
        hip.sync_stream(stream)?;
        Ok(model)
    }

    fn allocate_scratch(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        max_ctx: u32,
    ) -> anyhow::Result<()> {
        let cfg = &self.config;
        let rows = self.max_rows;
        let h = cfg.hidden_size as usize;
        let nq = cfg.num_attention_heads as usize / 2;
        let nkv = cfg.num_key_value_heads as usize / 2;
        let q_width = nq * 128;
        let kv_width = nkv * 128;
        let intermediate = cfg.intermediate_size as usize / 2;
        let local_vocab = cfg.vocab_size as usize / 2;
        for (name, elements) in [
            ("h0", rows * h),
            ("h1", rows * h),
            ("normed", rows * h),
            ("value_input", rows * h),
            ("o", rows * h),
            ("o_sum", rows * h),
            ("down", rows * h),
            ("down_sum", rows * h),
            ("aux", rows * 5 * h),
            ("final_norm", rows * h),
            ("q", rows * q_width),
            ("g", rows * q_width),
            ("qn", rows * q_width),
            ("attention", rows * q_width),
            ("gated", rows * q_width),
            ("k", rows * kv_width),
            ("v", rows * kv_width),
            ("kn", rows * kv_width),
            ("gate_up", rows * 2 * intermediate),
            ("logits", rows * local_vocab),
        ] {
            self.buffers.alloc(hip, name, elements * 2)?;
        }
        for (name, elements) in [("mlp_q", rows * h), ("act_q", rows * intermediate)] {
            self.buffers.alloc(hip, name, elements)?;
        }
        for name in ["mlp_scale", "act_scale"] {
            self.buffers.alloc(hip, name, rows * 4)?;
        }
        self.buffers.alloc(hip, "ids", rows * 4)?;
        self.buffers.alloc(hip, "positions", rows * 8)?;
        self.buffers.alloc(hip, "slots", rows * 8)?;
        self.buffers.alloc(hip, "value_seed", rows * h * 2)?;
        let table = rope_table(cfg, max_ctx as usize);
        self.buffers.upload_bf16(hip, stream, "rope", &table)?;
        let mut cache_bytes = 0usize;
        for layer in 0..cfg.num_hidden_layers {
            let slots = if cfg.uses_rope(layer) {
                max_ctx.min(LOCAL_CACHE_CAPACITY) as usize
            } else {
                max_ctx as usize
            };
            for name in [format!("kv.{layer}.k"), format!("kv.{layer}.v")] {
                self.buffers.alloc(hip, &name, slots * kv_width)?;
                cache_bytes += slots * kv_width;
            }
            for name in [format!("kv.{layer}.ks"), format!("kv.{layer}.vs")] {
                self.buffers.alloc(hip, &name, slots * nkv * 4)?;
                cache_bytes += slots * nkv * 4;
            }
        }
        println!(
            "ripple rank{}: FP8 KV {:.3} GiB, value seed {:.3} MiB, max context {max_ctx}",
            self.rank,
            cache_bytes as f64 / (1u64 << 30) as f64,
            (rows * h * 2) as f64 / (1u64 << 20) as f64
        );
        Ok(())
    }

    fn embed(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        collective: &Collective,
        tokens: &[u32],
        start: usize,
    ) -> anyhow::Result<()> {
        let rows = tokens.len();
        let h = self.config.hidden_size as usize;
        let local_vocab = self.config.vocab_size / 2;
        let ids: Vec<u8> = tokens
            .iter()
            .flat_map(|id| (*id as i32).to_le_bytes())
            .collect();
        self.buffers.upload(hip, stream, "ids", &ids)?;
        let positions: Vec<u8> = (start..start + rows)
            .flat_map(|position| (position as i64).to_le_bytes())
            .collect();
        self.buffers.upload(hip, stream, "positions", &positions)?;
        self.buffers.upload(hip, stream, "slots", &positions)?;
        let desc = descriptor(
            "head",
            "saber_embed_bf16",
            [rows as u32, 1, 1],
            256,
            abi::SIG_EMBED,
            vec![
                buffer("model.embed_tokens.weight"),
                buffer("ids"),
                buffer("h1"),
                scalar(h as i64),
                scalar((self.rank * local_vocab) as i64),
                scalar(((self.rank + 1) * local_vocab) as i64),
            ],
        )?;
        dispatch(
            stream,
            ctx,
            launcher,
            &desc,
            &self.buffers,
            Shape::new(0, 0, 0),
        )?;
        collective.reduce(stream, ctx, launcher, &self.buffers, "h1", "h0", rows * h)?;
        if let Some(scale) = self.config.embed_scale {
            custom(
                "ripple_scale_bf16",
                [(rows * h).div_ceil(256) as u32, 1, 1],
                256,
                abi::SIG_RIPPLE_SCALE,
                vec![buffer("h0"), scalar((rows * h) as i64), float(scale as f32)],
                stream,
                ctx,
                launcher,
                &self.buffers,
            )?;
        }
        self.buffers
            .copy(hip, stream, "value_seed", 0, "h0", (rows * h * 2) as u64)?;
        Ok(())
    }

    fn layer(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        collective: &Collective,
        bf16_rules: &DflashRules,
        mx_rules: &LaunchRules,
        layer: u32,
        rows: usize,
        start: usize,
    ) -> anyhow::Result<()> {
        let cfg = &self.config;
        let h = cfg.hidden_size as usize;
        let nq = cfg.num_attention_heads as usize / 2;
        let nkv = cfg.num_key_value_heads as usize / 2;
        let q_width = cfg.q_proj_rows() / 2;
        let kv_width = cfg.kv_proj_rows() / 2;
        let intermediate = cfg.intermediate_size / 2;
        let eps = cfg.rms_norm_eps as f32;
        let base = format!("model.layers.{layer}");
        custom(
            "ripple_rmsnorm_bf16",
            [rows as u32, 1, 1],
            256,
            abi::SIG_RIPPLE_RMSNORM,
            vec![
                buffer("h0"),
                buffer(&format!("{base}.input_layernorm.weight")),
                buffer("normed"),
                scalar(rows as i64),
                scalar(h as i64),
                float(eps),
            ],
            stream,
            ctx,
            launcher,
            &self.buffers,
        )?;
        custom(
            "ripple_value_lerp_bf16",
            [(rows * h).div_ceil(256) as u32, 1, 1],
            256,
            abi::SIG_RIPPLE_VALUE_LERP,
            vec![
                buffer("value_seed"),
                buffer("normed"),
                buffer(&format!("{base}.self_attn.value_res_lambda")),
                buffer("value_input"),
                scalar((rows * h) as i64),
            ],
            stream,
            ctx,
            launcher,
            &self.buffers,
        )?;
        for (input, weight, output, n) in [
            ("normed", "q_proj", "q", q_width),
            ("normed", "k_proj", "k", kv_width),
            ("normed", "g_proj", "g", q_width),
            ("value_input", "v_proj", "v", kv_width),
        ] {
            gemm_bf16(
                bf16_rules,
                stream,
                ctx,
                launcher,
                &mut self.buffers,
                input,
                &format!("{base}.self_attn.{weight}.weight"),
                output,
                rows as u32,
                n,
                cfg.hidden_size,
            )?;
        }
        custom(
            "ripple_qk_norm_rope_h128",
            [rows as u32, (nq + nkv) as u32, 1],
            32,
            abi::SIG_RIPPLE_QK_ROPE,
            vec![
                buffer("q"),
                buffer("k"),
                buffer(&format!("{base}.self_attn.q_norm.weight")),
                buffer(&format!("{base}.self_attn.k_norm.weight")),
                buffer("rope"),
                buffer("positions"),
                buffer("qn"),
                buffer("kn"),
                scalar(rows as i64),
                scalar(nq as i64),
                scalar(nkv as i64),
                float(eps),
                scalar(i64::from(cfg.uses_rope(layer))),
            ],
            stream,
            ctx,
            launcher,
            &self.buffers,
        )?;
        let local = cfg.uses_rope(layer);
        let window = if local { cfg.sliding_window as i64 } else { 0 };
        let cache_capacity = if local {
            i64::from((self.max_context as u32).min(LOCAL_CACHE_CAPACITY))
        } else {
            0
        };
        custom(
            "ripple_kv_append_h128_fp8",
            [rows as u32, nkv as u32, 1],
            32,
            abi::SIG_RIPPLE_KV_APPEND_FP8,
            vec![
                buffer("kn"),
                buffer("v"),
                buffer(&format!("kv.{layer}.k")),
                buffer(&format!("kv.{layer}.v")),
                buffer(&format!("kv.{layer}.ks")),
                buffer(&format!("kv.{layer}.vs")),
                buffer("slots"),
                scalar(rows as i64),
                scalar(nkv as i64),
                scalar(window),
                scalar(cache_capacity),
            ],
            stream,
            ctx,
            launcher,
            &self.buffers,
        )?;
        custom(
            "ripple_attn_h128_fp8",
            [rows as u32, nq as u32, 1],
            32,
            abi::SIG_RIPPLE_ATTN_FP8,
            vec![
                buffer("qn"),
                buffer(&format!("kv.{layer}.k")),
                buffer(&format!("kv.{layer}.v")),
                buffer(&format!("kv.{layer}.ks")),
                buffer(&format!("kv.{layer}.vs")),
                buffer("attention"),
                scalar(rows as i64),
                scalar(nq as i64),
                scalar(nkv as i64),
                scalar(start as i64),
                scalar(window),
                scalar(cache_capacity),
            ],
            stream,
            ctx,
            launcher,
            &self.buffers,
        )?;
        custom(
            "ripple_gate_mul_bf16",
            [(rows * q_width as usize).div_ceil(256) as u32, 1, 1],
            256,
            abi::SIG_RIPPLE_GATE_MUL,
            vec![
                buffer("attention"),
                buffer("g"),
                buffer("gated"),
                scalar((rows * q_width as usize) as i64),
            ],
            stream,
            ctx,
            launcher,
            &self.buffers,
        )?;
        gemm_bf16(
            bf16_rules,
            stream,
            ctx,
            launcher,
            &mut self.buffers,
            "gated",
            &format!("{base}.self_attn.o_proj.weight"),
            "o",
            rows as u32,
            cfg.hidden_size,
            q_width,
        )?;
        collective.reduce(stream, ctx, launcher, &self.buffers, "o", "o_sum", rows * h)?;
        custom(
            "ripple_postnorm_residual_bf16",
            [rows as u32, 1, 1],
            256,
            abi::SIG_RIPPLE_POSTNORM,
            vec![
                buffer("o_sum"),
                buffer("h0"),
                buffer(&format!("{base}.post_attention_layernorm.weight")),
                buffer("h1"),
                scalar(rows as i64),
                scalar(h as i64),
                float(eps),
            ],
            stream,
            ctx,
            launcher,
            &self.buffers,
        )?;
        custom(
            "ripple_rms_quant_bf16",
            [rows as u32, 1, 1],
            256,
            abi::SIG_RIPPLE_RMS_QUANT,
            vec![
                buffer("h1"),
                buffer(&format!("{base}.pre_feedforward_layernorm.weight")),
                buffer("mlp_q"),
                buffer("mlp_scale"),
                scalar(rows as i64),
                scalar(h as i64),
                float(eps),
            ],
            stream,
            ctx,
            launcher,
            &self.buffers,
        )?;
        gemm_mxfp4(
            mx_rules,
            stream,
            ctx,
            launcher,
            &mut self.buffers,
            "mlp_q",
            "mlp_scale",
            &format!("layer.{layer}.mlp.gate_up"),
            "gate_up",
            rows as u32,
            2 * intermediate,
            cfg.hidden_size,
        )?;
        custom(
            "ripple_swiglu_quant_bf16",
            [rows as u32, 1, 1],
            256,
            abi::SIG_RIPPLE_SWIGLU_QUANT,
            vec![
                buffer("gate_up"),
                buffer("act_q"),
                buffer("act_scale"),
                scalar(rows as i64),
                scalar(intermediate as i64),
                float(cfg.ffn_clamp_threshold.unwrap_or(0.0) as f32),
            ],
            stream,
            ctx,
            launcher,
            &self.buffers,
        )?;
        gemm_mxfp4(
            mx_rules,
            stream,
            ctx,
            launcher,
            &mut self.buffers,
            "act_q",
            "act_scale",
            &format!("layer.{layer}.mlp.down_proj"),
            "down",
            rows as u32,
            cfg.hidden_size,
            intermediate,
        )?;
        collective.reduce(
            stream,
            ctx,
            launcher,
            &self.buffers,
            "down",
            "down_sum",
            rows * h,
        )?;
        custom(
            "ripple_postnorm_residual_bf16",
            [rows as u32, 1, 1],
            256,
            abi::SIG_RIPPLE_POSTNORM,
            vec![
                buffer("down_sum"),
                buffer("h1"),
                buffer(&format!("{base}.post_feedforward_layernorm.weight")),
                buffer("h0"),
                scalar(rows as i64),
                scalar(h as i64),
                float(eps),
            ],
            stream,
            ctx,
            launcher,
            &self.buffers,
        )?;
        if self.trace_layers {
            hip.sync_stream(stream)?;
            if self.rank == 0 {
                println!("ripple layer {layer} complete");
            }
        }
        Ok(())
    }

    fn forward(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        collective: &Collective,
        bf16_rules: &DflashRules,
        mx_rules: &LaunchRules,
        tokens: &[u32],
        start: usize,
        logits: bool,
        capture: bool,
        all_logits: bool,
    ) -> anyhow::Result<()> {
        self.embed(hip, stream, ctx, launcher, collective, tokens, start)?;
        for layer in 0..self.config.num_hidden_layers {
            self.layer(
                hip,
                stream,
                ctx,
                launcher,
                collective,
                bf16_rules,
                mx_rules,
                layer,
                tokens.len(),
                start,
            )?;
            if capture {
                if let Some(tap) = [1, 20, 39, 58, 77].iter().position(|&id| id == layer) {
                    let rows = tokens.len();
                    let h = self.config.hidden_size as usize;
                    custom(
                        "ripple_aux_capture_bf16",
                        [(rows * h).div_ceil(256) as u32, 1, 1],
                        256,
                        abi::SIG_RIPPLE_AUX_CAPTURE,
                        vec![
                            buffer("h0"),
                            buffer("aux"),
                            scalar(rows as i64),
                            scalar(h as i64),
                            scalar(tap as i64),
                        ],
                        stream,
                        ctx,
                        launcher,
                        &self.buffers,
                    )?;
                }
            }
        }
        if !logits {
            return Ok(());
        }
        let h = self.config.hidden_size as usize;
        let head_rows = if all_logits { tokens.len() } else { 1 };
        let last = if all_logits {
            0
        } else {
            (tokens.len() - 1) * h * 2
        };
        custom(
            "ripple_rmsnorm_bf16",
            [head_rows as u32, 1, 1],
            256,
            abi::SIG_RIPPLE_RMSNORM,
            vec![
                buffer(&format!("h0@{last}")),
                buffer("model.norm.weight"),
                buffer("final_norm"),
                scalar(head_rows as i64),
                scalar(h as i64),
                float(self.config.rms_norm_eps as f32),
            ],
            stream,
            ctx,
            launcher,
            &self.buffers,
        )?;
        self.buffers.alias("a", "final_norm")?;
        self.buffers.alias("w", "lm_head.weight")?;
        self.buffers.alias("c", "logits")?;
        let local_vocab = self.config.vocab_size / 2;
        let desc = dflash::resolve_head_bgemm(
            head_rows as u32,
            local_vocab,
            self.config.hidden_size,
            bf16_rules,
        )?;
        dispatch(
            stream,
            ctx,
            launcher,
            &desc,
            &self.buffers,
            Shape::new(head_rows as u32, local_vocab, self.config.hidden_size),
        )?;
        Ok(())
    }

    fn pick_rows(
        &self,
        hip: &HipThread,
        stream: &Stream,
        peer: &RankPeer,
        position: usize,
        rows: usize,
    ) -> anyhow::Result<Vec<u32>> {
        hip.sync_stream(stream)?;
        let local_vocab = self.config.vocab_size as usize / 2;
        let local = self
            .buffers
            .download(hip, stream, "logits", rows * local_vocab * 2)?;
        let other = peer
            .exchange()
            .swap_bytes(&format!("ripple.logits.{position}.{rows}"), &local)?;
        let (first, second) = if self.rank == 0 {
            (local, other)
        } else {
            (other, local)
        };
        let mut output = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut best = f32::NEG_INFINITY;
            let mut best_id = 0u32;
            for (rank, bytes) in [first.clone(), second.clone()].iter().enumerate() {
                let start = row * local_vocab * 2;
                for (index, value) in bytes[start..start + local_vocab * 2]
                    .chunks_exact(2)
                    .enumerate()
                {
                    let logit = bf16_to_f32(u16::from_le_bytes([value[0], value[1]]));
                    ensure!(
                        logit.is_finite(),
                        "non-finite logit at rank {rank}, row {row}, index {index}"
                    );
                    if logit > best {
                        best = logit;
                        best_id = (rank * local_vocab + index) as u32;
                    }
                }
            }
            output.push(best_id);
        }
        Ok(output)
    }

    fn dense_rows(
        &self,
        hip: &HipThread,
        stream: &Stream,
        peer: &RankPeer,
        position: usize,
        rows: usize,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        hip.sync_stream(stream)?;
        let local_vocab = self.config.vocab_size as usize / 2;
        let local = self
            .buffers
            .download(hip, stream, "logits", rows * local_vocab * 2)?;
        let other = peer
            .exchange()
            .swap_bytes(&format!("ripple.logits.{position}.{rows}"), &local)?;
        let (first, second) = if self.rank == 0 {
            (local, other)
        } else {
            (other, local)
        };
        let vocab = self.config.vocab_size as usize;
        let mut output = vec![vec![0.0f32; vocab]; rows];
        for (rank, bytes) in [&first, &second].into_iter().enumerate() {
            for (row, output) in output.iter_mut().enumerate() {
                let start = row * local_vocab * 2;
                for (index, value) in bytes[start..start + local_vocab * 2]
                    .chunks_exact(2)
                    .enumerate()
                {
                    let logit = bf16_to_f32(u16::from_le_bytes([value[0], value[1]]));
                    ensure!(
                        logit.is_finite(),
                        "non-finite logit at rank {rank}, row {row}, index {index}"
                    );
                    output[rank * local_vocab + index] = logit;
                }
            }
        }
        Ok(output)
    }

    fn sample_dense_rows(
        &self,
        mut dense: Vec<Vec<f32>>,
        proposals: &[u32],
        request: &GenerateRequest,
        output: &[u32],
    ) -> anyhow::Result<TargetSample> {
        ensure!(dense.len() == proposals.len() + 1, "sample row count");
        let params = request.sampling.row_params(self.config.vocab_size as usize);
        let sampler = Sampler::new();
        let mut history = output.to_vec();
        let mut result = TargetSample::default();
        for (row, logits) in dense.iter_mut().enumerate() {
            let position = history.len() as u64;
            let mut noise: Box<dyn NoiseSource> = match request.sampling.seed {
                Some(seed) => Box::new(SeededNoise::new(seed, position, 0)),
                None => Box::new(EntropyNoise::new()),
            };
            let sampled = sampler
                .sample_dense(
                    logits,
                    &params,
                    &PenaltyContext {
                        prompt: &request.tokens,
                        output: &history,
                    },
                    noise.as_mut(),
                )
                .map_err(|error| anyhow::anyhow!("sampling target row {row}: {error}"))?;
            result.tokens.push(sampled.token_id);
            result.logprobs.extend(sampled.logprobs);
            history.push(sampled.token_id);
            match proposals.get(row) {
                Some(proposal) if *proposal == sampled.token_id => result.accepted += 1,
                Some(_) => break,
                None => break,
            }
        }
        Ok(result)
    }

    fn sample_rows(
        &self,
        hip: &HipThread,
        stream: &Stream,
        peer: &RankPeer,
        position: usize,
        proposals: &[u32],
        request: &GenerateRequest,
        output: &[u32],
    ) -> anyhow::Result<TargetSample> {
        let dense = self.dense_rows(hip, stream, peer, position, proposals.len() + 1)?;
        let mut result = if self.rank == 0 {
            self.sample_dense_rows(dense, proposals, request, output)?
        } else {
            TargetSample::default()
        };
        let mine = if self.rank == 0 {
            serde_json::to_vec(&SampleWire {
                tokens: result.tokens.clone(),
                accepted: result.accepted,
            })?
        } else {
            Vec::new()
        };
        let reply = peer.exchange().xchg_bytes(
            &format!("ripple.samples.{position}.{}", proposals.len() + 1),
            &mine,
        )?;
        if self.rank == 1 {
            let wire: SampleWire = serde_json::from_slice(&reply)?;
            result.tokens = wire.tokens;
            result.accepted = wire.accepted;
        }
        Ok(result)
    }

    fn pick(
        &self,
        hip: &HipThread,
        stream: &Stream,
        peer: &RankPeer,
        position: usize,
    ) -> anyhow::Result<u32> {
        Ok(self.pick_rows(hip, stream, peer, position, 1)?[0])
    }
}

struct Child(Option<std::process::Child>);

impl Drop for Child {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn forwarded_rank_args(args: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut forwarded = Vec::new();
    let mut skip_value = false;
    for arg in args {
        if skip_value {
            skip_value = false;
            continue;
        }
        if arg == "--device" {
            skip_value = true;
        } else if !arg.starts_with("--device=") {
            forwarded.push(arg);
        }
    }
    forwarded
}

pub fn config(checkpoint: &Path) -> anyhow::Result<RippleConfig> {
    let path = checkpoint.join("config.json");
    let text = std::fs::read_to_string(&path).with_context(|| path.display().to_string())?;
    RippleConfig::from_config_json(&text).map_err(|error| anyhow::anyhow!("{error}"))
}

#[derive(Debug)]
pub struct PrefixCache {
    history: Vec<u32>,
    capacity: usize,
    local_window: usize,
}

impl PrefixCache {
    fn new(capacity: usize, local_window: usize) -> Self {
        Self {
            history: Vec::new(),
            capacity,
            local_window,
        }
    }

    pub fn reuse_for(&self, prompt: &[u32]) -> usize {
        if self.history.len() > self.capacity {
            return 0;
        }
        let common = self
            .history
            .iter()
            .zip(prompt)
            .take_while(|(old, new)| old == new)
            .count();
        let reused = common.min(prompt.len().saturating_sub(1));
        let oldest_needed = reused.saturating_sub(self.local_window.saturating_sub(1));
        let oldest_resident = self
            .history
            .len()
            .saturating_sub(LOCAL_CACHE_CAPACITY as usize);
        if reused == 0 || oldest_needed < oldest_resident {
            0
        } else {
            reused
        }
    }

    fn record(&mut self, prompt: &[u32], output: &[u32]) {
        self.history.clear();
        self.history.extend_from_slice(prompt);
        self.history.extend_from_slice(output);
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SamplingSpec {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
    pub logprobs: Option<i32>,
}

impl Default for SamplingSpec {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            seed: None,
            logprobs: None,
        }
    }
}

impl SamplingSpec {
    fn params(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            repetition_penalty: self.repetition_penalty,
            seed: self.seed,
            logprobs: self.logprobs,
            ..SamplingParams::default()
        }
    }

    fn row_params(&self, vocab: usize) -> RowParams {
        RowParams::resolve(&self.params(), vocab)
    }

    fn fast_greedy(&self, vocab: usize) -> bool {
        let params = self.row_params(vocab);
        params.is_greedy() && params.no_penalties() && params.logprobs.is_none()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenerateRequest {
    pub id: u64,
    pub tokens: Vec<u32>,
    pub decode: usize,
    pub ignore_eos: bool,
    pub cached_tokens: usize,
    pub sampling: SamplingSpec,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationFinish {
    Stop,
    Length,
    Abort,
}

#[derive(Debug)]
pub struct GenerationStats {
    pub output: Vec<u32>,
    pub finish: GenerationFinish,
    pub prefill: Duration,
    pub decode: Duration,
    pub speculative_rounds: usize,
    pub accepted_drafts: usize,
    pub cached_tokens: usize,
}

#[derive(Debug, Default)]
struct TargetSample {
    tokens: Vec<u32>,
    accepted: usize,
    logprobs: Vec<SampledLogprobs>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct SampleWire {
    tokens: Vec<u32>,
    accepted: usize,
}

#[allow(clippy::too_many_arguments)]
fn generate(
    hip: &HipThread,
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    collective: &Collective,
    bf16_rules: &DflashRules,
    mx_rules: &LaunchRules,
    model: &mut Model,
    drafter: &mut Option<draft::Drafter>,
    peer: &RankPeer,
    max_rows: usize,
    trace_layers: bool,
    request: &GenerateRequest,
    mut on_step: impl FnMut(
        &[u32],
        &[SampledLogprobs],
        u32,
        Option<GenerationFinish>,
    ) -> anyhow::Result<bool>,
) -> anyhow::Result<GenerationStats> {
    ensure!(
        !request.tokens.is_empty(),
        "at least one prompt token is required"
    );
    ensure!(request.decode > 0, "decode must be positive");
    ensure!(
        request.tokens.len() + request.decode <= args_max_context(model),
        "prompt plus decode exceeds the allocated context"
    );
    peer.agree(
        &format!("ripple.tokens.{}", request.id),
        request.tokens.iter().fold(0u64, |hash, id| {
            hash.wrapping_mul(1_000_003).wrapping_add(u64::from(*id))
        }),
    )?;
    let started = Instant::now();
    let fast_greedy = request
        .sampling
        .fast_greedy(model.config.vocab_size as usize);
    let cached_tokens = if fast_greedy {
        model.prefix_cache.reuse_for(&request.tokens)
    } else {
        0
    };
    ensure!(
        request.id == 0 || request.cached_tokens == cached_tokens,
        "rank cache prefix disagreement"
    );
    let mut output = Vec::with_capacity(request.decode);
    let mut output_logprobs = Vec::new();
    let mut next = 0u32;
    for start in (cached_tokens..request.tokens.len()).step_by(max_rows) {
        let end = (start + max_rows).min(request.tokens.len());
        let last = end == request.tokens.len();
        let rows = end - start;
        model.forward(
            hip,
            stream,
            ctx,
            launcher,
            collective,
            bf16_rules,
            mx_rules,
            &request.tokens[start..end],
            start,
            last,
            drafter.is_some(),
            false,
        )?;
        if let Some(draft) = drafter.as_mut() {
            draft.context(
                hip,
                stream,
                ctx,
                launcher,
                bf16_rules,
                bf16_rules,
                &model.buffers,
                rows,
                start,
            )?;
        }
        if last {
            if fast_greedy {
                next = model.pick(hip, stream, peer, start)?;
            } else {
                let sampled = model.sample_rows(hip, stream, peer, start, &[], request, &output)?;
                next = sampled.tokens[0];
                output_logprobs.extend(sampled.logprobs);
            }
        }
    }
    let mut position = request.tokens.len();
    output.push(next);
    let prefill_elapsed = started.elapsed();
    let mut speculative_rounds = 0usize;
    let mut accepted_drafts = 0usize;
    let mut step_index = 0u32;
    let mut finish = if !request.ignore_eos && model.config.eos_token_id.contains(&next) {
        Some(GenerationFinish::Stop)
    } else if output.len() == request.decode {
        Some(GenerationFinish::Length)
    } else {
        None
    };
    let mut keep_going = on_step(&output, &output_logprobs, step_index, finish)?;
    keep_going = continue_on_both_ranks(peer, request.id, step_index, keep_going)?;
    if !keep_going && finish.is_none() {
        finish = Some(GenerationFinish::Abort);
    }
    while finish.is_none() {
        step_index += 1;
        let step_start = output.len();
        let remaining = request.decode - output.len();
        if drafter.is_some() && position + 16 <= args_max_context(model) {
            let draft = drafter.as_mut().unwrap();
            let proposals = draft.propose(
                hip, stream, ctx, launcher, collective, peer, bf16_rules, bf16_rules, next,
                position,
            )?;
            let mut verification = Vec::with_capacity(16);
            verification.push(next);
            verification.extend_from_slice(&proposals);
            model.forward(
                hip,
                stream,
                ctx,
                launcher,
                collective,
                bf16_rules,
                mx_rules,
                &verification,
                position,
                true,
                true,
                true,
            )?;
            draft.context(
                hip,
                stream,
                ctx,
                launcher,
                bf16_rules,
                bf16_rules,
                &model.buffers,
                16,
                position,
            )?;
            let mut step_logprobs = Vec::new();
            let (committed, accepted) = if fast_greedy {
                let decisions = model.pick_rows(hip, stream, peer, position, 16)?;
                if trace_layers && model.rank == 0 {
                    println!("ripple verification: proposals {proposals:?}; target {decisions:?}");
                }
                let mut accepted = 0usize;
                let mut committed = Vec::with_capacity(16);
                for (proposal, decision) in proposals.iter().zip(&decisions) {
                    if proposal == decision {
                        committed.push(*proposal);
                        accepted += 1;
                    } else {
                        committed.push(*decision);
                        break;
                    }
                }
                if accepted == proposals.len() {
                    committed.push(decisions[15]);
                }
                (committed, accepted)
            } else {
                let sampled =
                    model.sample_rows(hip, stream, peer, position, &proposals, request, &output)?;
                if trace_layers && model.rank == 0 {
                    println!(
                        "ripple sampled verification: proposals {proposals:?}; target {:?}",
                        sampled.tokens
                    );
                }
                step_logprobs = sampled.logprobs;
                (sampled.tokens, sampled.accepted)
            };
            let eos_at = (!request.ignore_eos)
                .then(|| {
                    committed
                        .iter()
                        .position(|id| model.config.eos_token_id.contains(id))
                })
                .flatten();
            let take = eos_at.map_or(committed.len(), |at| at + 1).min(remaining);
            position += take;
            next = committed[take - 1];
            output.extend_from_slice(&committed[..take]);
            output_logprobs.extend(step_logprobs.into_iter().take(take));
            speculative_rounds += 1;
            accepted_drafts += accepted;
            finish = if eos_at.is_some_and(|at| at < take) {
                Some(GenerationFinish::Stop)
            } else if output.len() == request.decode {
                Some(GenerationFinish::Length)
            } else {
                None
            };
        } else {
            model.forward(
                hip,
                stream,
                ctx,
                launcher,
                collective,
                bf16_rules,
                mx_rules,
                &[next],
                position,
                true,
                false,
                false,
            )?;
            if fast_greedy {
                next = model.pick(hip, stream, peer, position)?;
            } else {
                let sampled =
                    model.sample_rows(hip, stream, peer, position, &[], request, &output)?;
                next = sampled.tokens[0];
                output_logprobs.extend(sampled.logprobs);
            }
            position += 1;
            output.push(next);
            finish = if !request.ignore_eos && model.config.eos_token_id.contains(&next) {
                Some(GenerationFinish::Stop)
            } else if output.len() == request.decode {
                Some(GenerationFinish::Length)
            } else {
                None
            };
        }
        let step_logprobs = if output_logprobs.is_empty() {
            &[][..]
        } else {
            &output_logprobs[step_start..output.len()]
        };
        keep_going = on_step(&output[step_start..], step_logprobs, step_index, finish)?;
        keep_going = continue_on_both_ranks(peer, request.id, step_index, keep_going)?;
        if !keep_going {
            finish = Some(GenerationFinish::Abort);
        }
    }
    let decode_elapsed = started.elapsed() - prefill_elapsed;
    model.prefix_cache.record(&request.tokens, &output);
    Ok(GenerationStats {
        output,
        finish: finish.unwrap(),
        prefill: prefill_elapsed,
        decode: decode_elapsed,
        speculative_rounds,
        accepted_drafts,
        cached_tokens,
    })
}

fn args_max_context(model: &Model) -> usize {
    model.max_context
}

fn continue_on_both_ranks(
    peer: &RankPeer,
    request_id: u64,
    step: u32,
    mine: bool,
) -> anyhow::Result<bool> {
    if request_id == 0 {
        return Ok(mine);
    }
    let theirs = peer.exchange().xchg_bytes(
        &format!("ripple.serve.continue.{request_id}.{step}"),
        &[u8::from(mine)],
    )?;
    ensure!(
        theirs.len() == 1 && theirs[0] <= 1,
        "invalid continuation reply"
    );
    Ok(mine && theirs[0] != 0)
}

fn run_with(
    hip: &HipThread,
    args: &ExecArgs,
    driver: impl FnOnce(
        &Stream,
        &LaunchCtx<'_>,
        &mut Launcher,
        &mut Model,
        &Collective,
        &mut Option<draft::Drafter>,
        &RankPeer,
        &DflashRules,
        &LaunchRules,
    ) -> anyhow::Result<Vec<u32>>,
) -> anyhow::Result<Vec<u32>> {
    ensure!(args.rank < 2, "--rank must be 0 or 1");
    let mut child = Child(None);
    let exchange: Arc<TcpExchange> = if args.rank == 0 {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let exe = std::env::current_exe()?;
        let mut command = std::process::Command::new(exe);
        command
            .args(forwarded_rank_args(std::env::args().skip(1)))
            .arg("--rank")
            .arg("1")
            .arg("--device")
            .arg((args.device + 1).to_string())
            .arg("--peer-port")
            .arg(port.to_string());
        child.0 = Some(command.spawn().context("spawning Ripple rank 1")?);
        Arc::new(TcpExchange::accept(&listener, Duration::from_secs(300))?)
    } else {
        Arc::new(TcpExchange::connect(
            args.peer_port,
            Duration::from_secs(300),
        )?)
    };
    let peer = RankPeer::new(exchange);
    hip.set_device(args.device)?;
    let counters = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, &args.prebuilt, &counters)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;
    let launch_counters = LaunchCounters::new();
    let mut launcher = Launcher::new("ripple-exec", &launch_counters);
    let cfg = config(&args.checkpoint)?;
    cfg.validate_tp(2)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let bf16_rules = DflashRules::prod_dflash()?;
    let mx_rules = LaunchRules::prod_mxfp4()?;
    let mut model = Model::load(hip, &stream, args, cfg)?;
    let collective = Collective::new(hip, &stream, args.rank, &peer)?;
    let mut drafter = if let Some(pack) = &args.draft {
        let checkpoint = Path::new("/srv/models/saber-models/models/Ripple-65B-Preview-DFlash2");
        let text = std::fs::read_to_string(checkpoint.join("config.json"))?;
        let draft_cfg =
            saber_model::ripple::RippleDraftConfig::from_config_json(&text, &model.config)
                .map_err(|error| anyhow::anyhow!("{error}"))?;
        Some(draft::Drafter::load(
            hip,
            &stream,
            pack,
            &model.buffers,
            draft_cfg,
            args.rank,
            args.max_ctx,
            args.trace_layers,
        )?)
    } else {
        None
    };
    let result = driver(
        &stream,
        &ctx,
        &mut launcher,
        &mut model,
        &collective,
        &mut drafter,
        &peer,
        &bf16_rules,
        &mx_rules,
    );
    drop(child.0.take());
    result
}

pub fn run(hip: &HipThread, args: &ExecArgs) -> anyhow::Result<Vec<u32>> {
    ensure!(args.decode > 0, "--decode must be positive");
    run_with(
        hip,
        args,
        |stream, ctx, launcher, model, collective, drafter, peer, bf16_rules, mx_rules| {
            let request = GenerateRequest {
                id: 0,
                tokens: args.tokens.clone(),
                decode: args.decode as usize,
                ignore_eos: false,
                cached_tokens: 0,
                sampling: SamplingSpec::default(),
            };
            let stats = generate(
                hip,
                stream,
                ctx,
                launcher,
                collective,
                bf16_rules,
                mx_rules,
                model,
                drafter,
                peer,
                args.max_rows,
                args.trace_layers,
                &request,
                |_, _, _, _| Ok(true),
            )?;
            if drafter.is_some() {
                println!(
                    "ripple rank{}: DFlash2 {} rounds, {} accepted draft tokens",
                    args.rank, stats.speculative_rounds, stats.accepted_drafts
                );
            }
            println!(
                "ripple rank{}: prompt {} + {} generated tokens in {:.3}s; ids {:?}",
                args.rank,
                args.tokens.len(),
                stats.output.len(),
                (stats.prefill + stats.decode).as_secs_f64(),
                stats.output
            );
            println!(
                "ripple rank{}: prefill {:.3}s ({:.2} prompt tok/s); decode {} steps in {:.3}s ({:.2} tok/s)",
                args.rank,
                stats.prefill.as_secs_f64(),
                args.tokens.len() as f64 / stats.prefill.as_secs_f64(),
                stats.output.len() - 1,
                stats.decode.as_secs_f64(),
                if stats.output.len() > 1 {
                    (stats.output.len() - 1) as f64 / stats.decode.as_secs_f64()
                } else {
                    0.0
                }
            );
            Ok(if args.rank == 0 {
                stats.output
            } else {
                Vec::new()
            })
        },
    )
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
enum WireCommand {
    Run(GenerateRequest),
    Cancel,
    Noop,
    Stop,
}

fn exchange_command(peer: &RankPeer, command: &WireCommand) -> anyhow::Result<WireCommand> {
    let bytes = serde_json::to_vec(command)?;
    let reply = peer.exchange().xchg_bytes("ripple.serve.command", &bytes)?;
    if peer.rank() == 0 {
        Ok(WireCommand::Noop)
    } else {
        Ok(serde_json::from_slice(&reply).context("decoding rank 0 command")?)
    }
}

pub fn run_persistent(
    hip: &HipThread,
    args: &ExecArgs,
    host: &str,
    port: u16,
    served_name: &str,
) -> anyhow::Result<()> {
    run_with(
        hip,
        args,
        |stream, ctx, launcher, model, collective, drafter, peer, bf16_rules, mx_rules| {
            if args.rank == 0 {
                let server = super::serve::start(
                    host,
                    port,
                    served_name,
                    &args.checkpoint,
                    args.max_ctx as usize,
                    model.config.eos_token_id.clone(),
                    model.config.vocab_size as usize,
                )?;
                loop {
                    match server.jobs.recv_timeout(Duration::from_secs(240)) {
                        Ok(mut job) => {
                            job.begin();
                            if job.is_aborted() {
                                exchange_command(peer, &WireCommand::Cancel)?;
                                job.cancel_before_start();
                                continue;
                            }
                            job.request.cached_tokens = if job
                                .request
                                .sampling
                                .fast_greedy(model.config.vocab_size as usize)
                            {
                                model.prefix_cache.reuse_for(&job.request.tokens)
                            } else {
                                0
                            };
                            exchange_command(peer, &WireCommand::Run(job.request.clone()))?;
                            let stats = generate(
                                hip,
                                stream,
                                ctx,
                                launcher,
                                collective,
                                bf16_rules,
                                mx_rules,
                                model,
                                drafter,
                                peer,
                                args.max_rows,
                                args.trace_layers,
                                &job.request,
                                |tokens, logprobs, step, finish| {
                                    Ok(job.emit(tokens, logprobs, step, finish))
                                },
                            )?;
                            println!(
                                "ripple request {}: prompt {}, output {}, {:?}, cached {}, prefill {:.3}s, decode {:.3}s, {} draft rounds, {} accepted",
                                job.request.id,
                                job.request.tokens.len(),
                                stats.output.len(),
                                stats.finish,
                                stats.cached_tokens,
                                stats.prefill.as_secs_f64(),
                                stats.decode.as_secs_f64(),
                                stats.speculative_rounds,
                                stats.accepted_drafts
                            );
                            job.complete(&stats);
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            exchange_command(peer, &WireCommand::Noop)?;
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                            exchange_command(peer, &WireCommand::Stop)?;
                            break;
                        }
                    }
                }
                server
                    .join
                    .join()
                    .map_err(|_| anyhow::anyhow!("HTTP thread panicked"))??;
            } else {
                loop {
                    match exchange_command(peer, &WireCommand::Noop)? {
                        WireCommand::Run(request) => {
                            generate(
                                hip,
                                stream,
                                ctx,
                                launcher,
                                collective,
                                bf16_rules,
                                mx_rules,
                                model,
                                drafter,
                                peer,
                                args.max_rows,
                                args.trace_layers,
                                &request,
                                |_, _, _, _| Ok(true),
                            )?;
                        }
                        WireCommand::Cancel | WireCommand::Noop => {}
                        WireCommand::Stop => break,
                    }
                }
            }
            Ok(Vec::new())
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::PrefixCache;

    #[test]
    fn prefix_cache_reuses_only_the_exact_prompt_prefix() {
        let mut cache = PrefixCache::new(5, 1024);
        cache.record(&[1, 2, 3], &[8, 9]);
        assert_eq!(cache.reuse_for(&[1, 2, 3, 8, 9, 4]), 5);
        assert_eq!(cache.reuse_for(&[1, 2, 3]), 2);
        assert_eq!(cache.reuse_for(&[1, 2, 7, 8, 9]), 2);
        assert_eq!(cache.reuse_for(&[7, 2, 3]), 0);
        cache.record(&[1, 2, 3, 4], &[8, 9]);
        assert_eq!(cache.reuse_for(&[1, 2, 3, 4, 8, 9, 5]), 0);
    }

    #[test]
    fn prefix_cache_checks_local_ring_residency_beyond_4096_tokens() {
        let mut cache = PrefixCache::new(8192, 1024);
        let history = (0..5000).collect::<Vec<u32>>();
        cache.record(&history, &[]);

        let mut continuation = history.clone();
        continuation.push(5000);
        assert_eq!(cache.reuse_for(&continuation), 5000);
        assert_eq!(cache.reuse_for(&history), 4999);

        let mut recent_branch = history[..4500].to_vec();
        recent_branch.push(9000);
        assert_eq!(cache.reuse_for(&recent_branch), 4500);

        let mut expired_branch = history[..1926].to_vec();
        expired_branch.push(9000);
        assert_eq!(cache.reuse_for(&expired_branch), 0);
        let mut oldest_valid_branch = history[..1927].to_vec();
        oldest_valid_branch.push(9000);
        assert_eq!(cache.reuse_for(&oldest_valid_branch), 1927);
    }
}
