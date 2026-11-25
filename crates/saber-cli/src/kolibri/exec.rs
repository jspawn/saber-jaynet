#![allow(clippy::too_many_arguments)]

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
use saber_hip::{
    DeviceAddr, DevicePtr, Graph, GraphExec, HipThread, Job, MallocFlags, Stream,
    StreamCaptureMode, StreamFlags,
};
use saber_kernels::abi::{self, SigArg};
use saber_kernels::dflash::{self, DflashOp, DflashRules, DflashShape};
use saber_kernels::rules::{ArgValue, LaunchDescriptor, Shape, Slot};
use saber_mem::ledger::{Ledger, Pool};
use saber_mem::upload::{HipSink, PinnedRing, Upload, UploadConfig};
use saber_model::kolibri::KolibriConfig;
use saber_serve::engine::SampledLogprobs;
use saber_serve::sampler::{
    EntropyNoise, NoiseSource, PenaltyContext, RowParams, Sampler, SeededNoise,
};
use saber_serve::sampling::SamplingParams;

use super::super::ripple::reference::{bf16_to_f32, f32_to_bf16, rope_table_values};

const AR_MB: u64 = 16 << 20;
const AR_BLOCKS: u64 = 512;
const MAX_PREFILL_ROWS: usize = 256;
const TOP_K: usize = 6;
const EXPERTS: usize = 384;
const LOCAL_CACHE_RING: u32 = 528;
const MAX_DECODE_SPLITS: usize = 16;
const DECODE_KEYS_PER_SPLIT: usize = 4096;

pub struct ExecArgs {
    pub pack: PathBuf,
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
    pub layers: Option<u32>,
    pub layer_offset: u32,
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

    fn upload_new(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        name: &str,
        bytes: &[u8],
    ) -> anyhow::Result<()> {
        self.alloc(hip, name, bytes.len())?;
        self.upload(hip, stream, name, bytes)
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
    plugin: &str,
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
    let desc = descriptor(plugin, name, grid, block, signature, values)?;
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
            hip.ipc_import(peer.swap_handle("kolibri.ar.scratch", hip.ipc_export(&scratch)?)?)?;
        let peer_flags =
            hip.ipc_import(peer.swap_handle("kolibri.ar.flags", hip.ipc_export(&flags)?)?)?;
        ensure!(rank < 2, "Kolibri TP2 rank must be 0 or 1");
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

struct PrefillGraph {
    _graph: Graph,
    exec: GraphExec,
}

struct Model {
    rank: u32,
    config: KolibriConfig,
    buffers: Buffers,
    _weight_arena: Rc<DevicePtr>,
    max_rows: usize,
    max_context: usize,
    layers: u32,
    layer_offset: u32,
    trace_layers: bool,
    prefix_cache: PrefixCache,
    rows_cap: usize,
    tiles_cap: usize,
    graph_warm: BTreeSet<(usize, bool)>,
    graphs: BTreeMap<(usize, bool), PrefillGraph>,
}

impl Model {
    fn load(
        hip: &HipThread,
        stream: &Stream,
        args: &ExecArgs,
        config: KolibriConfig,
    ) -> anyhow::Result<Self> {
        ensure!(
            args.max_rows > 0 && args.max_rows <= MAX_PREFILL_ROWS,
            "--max-rows must be 1..=256"
        );
        ensure!(
            !args.tokens.is_empty(),
            "at least one prompt token is required"
        );
        ensure!(
            args.max_ctx <= config.max_position_embeddings,
            "context exceeds checkpoint maximum"
        );
        let layers = args.layers.unwrap_or(config.num_hidden_layers);
        ensure!(
            layers > 0 && args.layer_offset + layers <= config.num_hidden_layers,
            "invalid layer range"
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
            "kolibri rank{}: {} records, {:.3} GiB weights, verified load {:.1}s ({:.2} GB/s)",
            args.rank,
            records,
            plan.payload_bytes as f64 / (1u64 << 30) as f64,
            started.elapsed().as_secs_f64(),
            stats.end_to_end_gbps()
        );
        let rows_cap = args.max_rows * TOP_K * 16 + args.max_rows.div_ceil(16) * 16;
        let tiles_cap = args.max_rows * TOP_K + args.max_rows.div_ceil(16);
        let mut model = Self {
            rank: args.rank,
            config,
            buffers,
            _weight_arena: weight_arena,
            max_rows: args.max_rows,
            max_context: args.max_ctx as usize,
            layers,
            layer_offset: args.layer_offset,
            trace_layers: args.trace_layers,
            prefix_cache: PrefixCache::new(args.max_ctx as usize, 513),
            rows_cap,
            tiles_cap,
            graph_warm: BTreeSet::new(),
            graphs: BTreeMap::new(),
        };
        model.upload_expert_tables(hip, stream)?;
        model.allocate_scratch(hip, stream, args.max_ctx)?;
        hip.sync_stream(stream)?;
        Ok(model)
    }

    fn upload_expert_tables(&mut self, hip: &HipThread, stream: &Stream) -> anyhow::Result<()> {
        for layer in self.layer_offset..self.layer_offset + self.layers {
            for (table, projection) in [("xgu", "gate_up"), ("xdn", "down")] {
                let mut addresses = Vec::with_capacity((EXPERTS + 1) * 3 * 8);
                for expert in 0..=EXPERTS {
                    let base = if expert == EXPERTS {
                        format!("layer.{layer}.mlp.shared_expert.{projection}")
                    } else {
                        format!("layer.{layer}.expert.{expert}.{projection}")
                    };
                    for suffix in ["w", "ws", "wref"] {
                        let address = self.buffers.addr(&format!("{base}.{suffix}"))?.as_u64();
                        addresses.extend_from_slice(&address.to_le_bytes());
                    }
                }
                self.buffers
                    .upload_new(hip, stream, &format!("{table}.{layer}"), &addresses)?;
            }
        }
        Ok(())
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
        let nkv = cfg.num_key_value_heads as usize / 2;
        let q_width = cfg.q_proj_rows() as usize / 2;
        let kv_width = cfg.kv_proj_rows() as usize / 2;
        let intermediate = cfg.moe_intermediate_size as usize / 2;
        let local_vocab = cfg.vocab_size as usize / 2;
        for (name, elements) in [
            ("h0", rows * h),
            ("h1", rows * h),
            ("normed", rows * h),
            ("moe_input", rows * h),
            ("o", rows * h),
            ("o_sum", rows * h),
            ("moe", rows * h),
            ("down_sum", rows * h),
            ("final_norm", rows * h),
            ("q", rows * q_width),
            ("qn", rows * q_width),
            ("attention", rows * q_width),
            ("k", rows * kv_width),
            ("v", rows * kv_width),
            ("kn", rows * kv_width),
            ("gate_up", self.rows_cap * 2 * intermediate),
            ("y", self.rows_cap * h),
            ("logits", rows * local_vocab),
        ] {
            self.buffers.alloc(hip, name, elements * 2)?;
        }
        let decode_heads = q_width / 128;
        self.buffers.alloc(
            hip,
            "attn_partial",
            decode_heads * MAX_DECODE_SPLITS * 128 * 4,
        )?;
        self.buffers.alloc(
            hip,
            "attn_partial_ml",
            decode_heads * MAX_DECODE_SPLITS * 2 * 4,
        )?;
        for (name, elements) in [("mlp_q", rows * h), ("act_q", self.rows_cap * intermediate)] {
            self.buffers.alloc(hip, name, elements)?;
        }
        for (name, elements) in [("mlp_scale", rows), ("act_scale", self.rows_cap)] {
            self.buffers.alloc(hip, name, elements * 4)?;
        }
        self.buffers.alloc(hip, "ids", rows * 4)?;
        self.buffers.alloc(hip, "positions", rows * 8)?;
        self.buffers.alloc(hip, "slots", rows * 8)?;
        let block_count = (max_ctx as usize).div_ceil(16);
        let block_table: Vec<u8> = (0..block_count as i32)
            .flat_map(|block| block.to_le_bytes())
            .collect();
        self.buffers
            .upload_new(hip, stream, "attn_blocks", &block_table)?;
        for (name, elements) in [
            ("topk_ids", rows * TOP_K),
            ("topk_w", rows * TOP_K),
            ("route_pos", rows * TOP_K),
            ("row_tok", self.rows_cap),
            ("tile_expert", self.tiles_cap),
            ("tile_row0", self.tiles_cap),
            ("tile_rows", self.tiles_cap),
            ("expert_cnt", EXPERTS),
            ("meta", 6),
        ] {
            self.buffers.alloc(hip, name, elements * 4)?;
        }
        self.buffers
            .alloc(hip, "p_gu", 4 * self.rows_cap * 2 * intermediate * 4)?;
        self.buffers.alloc(hip, "p_dn", 2 * self.rows_cap * h * 4)?;
        self.buffers
            .alloc(hip, "cnt_gu", self.tiles_cap * (intermediate / 64) * 4)?;
        self.buffers
            .alloc(hip, "cnt_dn", self.tiles_cap * (h / 128) * 4)?;
        for name in ["cnt_gu", "cnt_dn"] {
            hip.memset(self.buffers.get(name)?, 0, stream)?;
        }
        let table = rope_table_values(cfg.rope_theta, cfg.head_dim, max_ctx as usize);
        self.buffers.upload_bf16(hip, stream, "rope", &table)?;
        let mut cache_bytes = 0usize;
        for layer in self.layer_offset..self.layer_offset + self.layers {
            let slots = if cfg.uses_rope(layer) {
                (max_ctx.min(LOCAL_CACHE_RING) as usize).div_ceil(16) * 16
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
            "kolibri rank{}: FP8 KV {:.3} GiB, max context {max_ctx}, layers {}",
            self.rank,
            cache_bytes as f64 / (1u64 << 30) as f64,
            self.layers
        );
        Ok(())
    }

    fn prepare_embed(
        &self,
        hip: &HipThread,
        stream: &Stream,
        tokens: &[u32],
        start: usize,
    ) -> anyhow::Result<()> {
        let ids: Vec<u8> = tokens
            .iter()
            .flat_map(|id| (*id as i32).to_le_bytes())
            .collect();
        self.buffers.upload(hip, stream, "ids", &ids)?;
        let positions: Vec<u8> = (start..start + tokens.len())
            .flat_map(|position| (position as i64).to_le_bytes())
            .collect();
        self.buffers.upload(hip, stream, "positions", &positions)?;
        self.buffers.upload(hip, stream, "slots", &positions)?;
        Ok(())
    }

    fn embed(
        &mut self,
        _hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        collective: &Collective,
        rows: usize,
    ) -> anyhow::Result<()> {
        let h = self.config.hidden_size as usize;
        let local_vocab = self.config.vocab_size / 2;
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
        layer: u32,
        rows: usize,
        start: usize,
    ) -> anyhow::Result<()> {
        let profile_started = Instant::now();
        let cfg = &self.config;
        let h = cfg.hidden_size as usize;
        let nq = cfg.num_attention_heads as usize / 2;
        let nkv = cfg.num_key_value_heads as usize / 2;
        let q_width = cfg.q_proj_rows() as usize / 2;
        let kv_width = cfg.kv_proj_rows() as usize / 2;
        let intermediate = cfg.moe_intermediate_size as usize / 2;
        let eps = cfg.rms_norm_eps as f32;
        let base = format!("model.layers.{layer}");
        custom(
            "ripple_spike",
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
        for (weight, output, n) in [
            ("q_proj", "q", q_width),
            ("k_proj", "k", kv_width),
            ("v_proj", "v", kv_width),
        ] {
            gemm_bf16(
                bf16_rules,
                stream,
                ctx,
                launcher,
                &mut self.buffers,
                "normed",
                &format!("{base}.self_attn.{weight}.weight"),
                output,
                rows as u32,
                n as u32,
                cfg.hidden_size,
            )?;
        }
        custom(
            "ripple_spike",
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
            i64::from((self.max_context as u32).min(LOCAL_CACHE_RING).div_ceil(16) * 16)
        } else {
            0
        };
        custom(
            "ripple_spike",
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
        if rows > 1 {
            custom(
                "kolibri_attn_t32",
                "kolibri_attn_prefill_h128_gqa12_fp8_t32",
                [rows.div_ceil(32) as u32, nkv as u32, 1],
                768,
                abi::SIG_KOLIBRI_ATTN_PREFILL,
                vec![
                    buffer("qn"),
                    buffer(&format!("kv.{layer}.k")),
                    buffer(&format!("kv.{layer}.v")),
                    buffer("attn_blocks"),
                    buffer(&format!("kv.{layer}.ks")),
                    buffer(&format!("kv.{layer}.vs")),
                    buffer("attention"),
                    buffer("positions"),
                    scalar(rows as i64),
                    scalar(nq as i64),
                    scalar(nkv as i64),
                    scalar(16),
                    scalar(if local { cache_capacity / 16 } else { 0 }),
                    scalar(window),
                    float(1.0 / 128f32.sqrt()),
                    scalar(0),
                ],
                stream,
                ctx,
                launcher,
                &self.buffers,
            )?;
        } else {
            let key_count = if local {
                (start + 1).min(cfg.sliding_window as usize)
            } else {
                start + 1
            };
            let splits = key_count
                .div_ceil(DECODE_KEYS_PER_SPLIT)
                .clamp(1, MAX_DECODE_SPLITS);
            if splits == 1 {
                custom(
                    "ripple_spike",
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
            } else {
                custom(
                    "kolibri_spike",
                    "kolibri_attn_decode_partial_h128_fp8",
                    [rows as u32, nq as u32, splits as u32],
                    32,
                    abi::SIG_KOLIBRI_ATTN_PARTIAL,
                    vec![
                        buffer("qn"),
                        buffer(&format!("kv.{layer}.k")),
                        buffer(&format!("kv.{layer}.v")),
                        buffer(&format!("kv.{layer}.ks")),
                        buffer(&format!("kv.{layer}.vs")),
                        buffer("attn_partial"),
                        buffer("attn_partial_ml"),
                        scalar(rows as i64),
                        scalar(nq as i64),
                        scalar(nkv as i64),
                        scalar(start as i64),
                        scalar(window),
                        scalar(cache_capacity),
                        scalar(splits as i64),
                    ],
                    stream,
                    ctx,
                    launcher,
                    &self.buffers,
                )?;
                custom(
                    "kolibri_spike",
                    "kolibri_attn_decode_reduce_h128",
                    [rows as u32, nq as u32, 1],
                    32,
                    abi::SIG_KOLIBRI_ATTN_REDUCE,
                    vec![
                        buffer("attn_partial"),
                        buffer("attn_partial_ml"),
                        buffer("attention"),
                        scalar(rows as i64),
                        scalar(nq as i64),
                        scalar(splits as i64),
                    ],
                    stream,
                    ctx,
                    launcher,
                    &self.buffers,
                )?;
            }
        }
        gemm_bf16(
            bf16_rules,
            stream,
            ctx,
            launcher,
            &mut self.buffers,
            "attention",
            &format!("{base}.self_attn.o_proj.weight"),
            "o",
            rows as u32,
            cfg.hidden_size,
            q_width as u32,
        )?;
        collective.reduce(stream, ctx, launcher, &self.buffers, "o", "o_sum", rows * h)?;
        custom(
            "ripple_spike",
            "ripple_postnorm_residual_bf16",
            [rows as u32, 1, 1],
            256,
            abi::SIG_RIPPLE_POSTNORM,
            vec![
                buffer("o_sum"),
                buffer("h0"),
                buffer(&format!("{base}.post_attn_norm.weight")),
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
        let mut attention_ms = 0.0;
        if self.trace_layers {
            hip.sync_stream(stream)?;
            attention_ms = profile_started.elapsed().as_secs_f64() * 1000.0;
            if self.rank == 0 {
                println!("kolibri layer {layer}: attention/projections {attention_ms:.3} ms");
            }
        }
        custom(
            "ripple_spike",
            "ripple_rmsnorm_bf16",
            [rows as u32, 1, 1],
            256,
            abi::SIG_RIPPLE_RMSNORM,
            vec![
                buffer("h1"),
                buffer(&format!("{base}.post_attention_layernorm.weight")),
                buffer("moe_input"),
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
            "kolibri_spike",
            "kolibri_quant_bf16",
            [rows as u32, 1, 1],
            256,
            abi::SIG_KOLIBRI_QUANT,
            vec![
                buffer("moe_input"),
                buffer("mlp_q"),
                buffer("mlp_scale"),
                scalar(rows as i64),
                scalar(h as i64),
            ],
            stream,
            ctx,
            launcher,
            &self.buffers,
        )?;
        if rows > 1 {
            custom(
                "kolibri_spike",
                "kolibri_router_bf16",
                [rows as u32, 1, 1],
                256,
                abi::SIG_KOLIBRI_ROUTER,
                vec![
                    buffer("moe_input"),
                    buffer(&format!("{base}.mlp.gate.weight")),
                    buffer(&format!("{base}.moe.router.expert_bias")),
                    buffer("topk_ids"),
                    buffer("topk_w"),
                    scalar(rows as i64),
                    scalar(h as i64),
                    scalar(EXPERTS as i64),
                    scalar(TOP_K as i64),
                ],
                stream,
                ctx,
                launcher,
                &self.buffers,
            )?;
            custom(
                "kolibri_spike",
                "kolibri_sort_top6",
                [1, 1, 1],
                256,
                abi::SIG_KOLIBRI_SORT,
                vec![
                    buffer("topk_ids"),
                    buffer("route_pos"),
                    buffer("row_tok"),
                    buffer("tile_expert"),
                    buffer("tile_row0"),
                    buffer("tile_rows"),
                    buffer("expert_cnt"),
                    buffer("meta"),
                    scalar(rows as i64),
                    scalar(EXPERTS as i64),
                    scalar(TOP_K as i64),
                ],
                stream,
                ctx,
                launcher,
                &self.buffers,
            )?;
        } else {
            custom(
                "kolibri_spike",
                "kolibri_route_bf16",
                [1, 1, 1],
                256,
                abi::SIG_KOLIBRI_ROUTE,
                vec![
                    buffer("moe_input"),
                    buffer(&format!("{base}.mlp.gate.weight")),
                    buffer(&format!("{base}.moe.router.expert_bias")),
                    buffer("topk_ids"),
                    buffer("topk_w"),
                    buffer("route_pos"),
                    buffer("row_tok"),
                    buffer("tile_expert"),
                    buffer("tile_row0"),
                    buffer("tile_rows"),
                    buffer("expert_cnt"),
                    buffer("meta"),
                    scalar(rows as i64),
                    scalar(h as i64),
                    scalar(EXPERTS as i64),
                    scalar(TOP_K as i64),
                ],
                stream,
                ctx,
                launcher,
                &self.buffers,
            )?;
        }
        let gate_ks = if rows == 1 { 4 } else { 1 };
        let down_ks = if rows == 1 { 2 } else { 1 };
        self.grouped_gemm(stream, ctx, launcher, layer, rows, true, gate_ks)?;
        custom(
            "moe_grouped",
            "moe_silu_mul_quant_t256",
            [
                if rows == 1 {
                    ((TOP_K + 1) * 16) as u32
                } else {
                    self.rows_cap as u32
                },
                1,
                1,
            ],
            256,
            abi::SIG_MOE_SILU,
            vec![
                buffer("gate_up"),
                buffer("act_q"),
                buffer("act_scale"),
                scalar(intermediate as i64),
                buffer("row_tok"),
                buffer("meta"),
            ],
            stream,
            ctx,
            launcher,
            &self.buffers,
        )?;
        self.grouped_gemm(stream, ctx, launcher, layer, rows, false, down_ks)?;
        custom(
            "kolibri_spike",
            "kolibri_combine_bf16",
            [rows as u32, 1, 1],
            256,
            abi::SIG_KOLIBRI_COMBINE,
            vec![
                buffer("y"),
                buffer("topk_w"),
                buffer("route_pos"),
                buffer("meta"),
                buffer("moe"),
                scalar(rows as i64),
                scalar(h as i64),
                scalar(TOP_K as i64),
            ],
            stream,
            ctx,
            launcher,
            &self.buffers,
        )?;
        collective.reduce(
            stream,
            ctx,
            launcher,
            &self.buffers,
            "moe",
            "down_sum",
            rows * h,
        )?;
        custom(
            "ripple_spike",
            "ripple_postnorm_residual_bf16",
            [rows as u32, 1, 1],
            256,
            abi::SIG_RIPPLE_POSTNORM,
            vec![
                buffer("down_sum"),
                buffer("h1"),
                buffer(&format!("{base}.post_ffn_norm.weight")),
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
                let total_ms = profile_started.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "kolibri layer {layer} complete: MoE/norms {:.3} ms, total {total_ms:.3} ms",
                    total_ms - attention_ms
                );
            }
        }
        Ok(())
    }

    fn check_attention_splits(
        &self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        layer: u32,
    ) -> anyhow::Result<()> {
        let cfg = &self.config;
        let nq = cfg.num_attention_heads as usize / 2;
        let nkv = cfg.num_key_value_heads as usize / 2;
        let local = cfg.uses_rope(layer);
        let window = if local { cfg.sliding_window as i64 } else { 0 };
        let cache_capacity = if local { LOCAL_CACHE_RING as i64 } else { 0 };
        let slots = if local {
            LOCAL_CACHE_RING as usize
        } else {
            self.max_context
        };
        let start = self.max_context - 1;
        let query: Vec<u8> = (0..nq * 128)
            .flat_map(|i| f32_to_bf16((((i * 37) % 257) as i32 - 128) as f32 / 128.0).to_le_bytes())
            .collect();
        self.buffers.upload(hip, stream, "qn", &query)?;
        for (name, bias) in [(format!("kv.{layer}.k"), 11), (format!("kv.{layer}.v"), 29)] {
            let bytes: Vec<u8> = (0..slots * nkv * 128)
                .map(|i| {
                    let magnitude = ((i * 37 + bias) % 96) as u8;
                    if i % 2 == 0 {
                        magnitude
                    } else {
                        magnitude | 0x80
                    }
                })
                .collect();
            self.buffers.upload(hip, stream, &name, &bytes)?;
        }
        for name in [format!("kv.{layer}.ks"), format!("kv.{layer}.vs")] {
            let bytes: Vec<u8> = (0..slots * nkv)
                .flat_map(|i| (0.01 + ((i * 13) % 17) as f32 / 1000.0).to_le_bytes())
                .collect();
            self.buffers.upload(hip, stream, &name, &bytes)?;
        }
        custom(
            "ripple_spike",
            "ripple_attn_h128_fp8",
            [1, nq as u32, 1],
            32,
            abi::SIG_RIPPLE_ATTN_FP8,
            vec![
                buffer("qn"),
                buffer(&format!("kv.{layer}.k")),
                buffer(&format!("kv.{layer}.v")),
                buffer(&format!("kv.{layer}.ks")),
                buffer(&format!("kv.{layer}.vs")),
                buffer("attention"),
                scalar(1),
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
        hip.sync_stream(stream)?;
        let expected = self
            .buffers
            .download(hip, stream, "attention", nq * 128 * 2)?;
        let splits = MAX_DECODE_SPLITS;
        custom(
            "kolibri_spike",
            "kolibri_attn_decode_partial_h128_fp8",
            [1, nq as u32, splits as u32],
            32,
            abi::SIG_KOLIBRI_ATTN_PARTIAL,
            vec![
                buffer("qn"),
                buffer(&format!("kv.{layer}.k")),
                buffer(&format!("kv.{layer}.v")),
                buffer(&format!("kv.{layer}.ks")),
                buffer(&format!("kv.{layer}.vs")),
                buffer("attn_partial"),
                buffer("attn_partial_ml"),
                scalar(1),
                scalar(nq as i64),
                scalar(nkv as i64),
                scalar(start as i64),
                scalar(window),
                scalar(cache_capacity),
                scalar(splits as i64),
            ],
            stream,
            ctx,
            launcher,
            &self.buffers,
        )?;
        custom(
            "kolibri_spike",
            "kolibri_attn_decode_reduce_h128",
            [1, nq as u32, 1],
            32,
            abi::SIG_KOLIBRI_ATTN_REDUCE,
            vec![
                buffer("attn_partial"),
                buffer("attn_partial_ml"),
                buffer("attention"),
                scalar(1),
                scalar(nq as i64),
                scalar(splits as i64),
            ],
            stream,
            ctx,
            launcher,
            &self.buffers,
        )?;
        hip.sync_stream(stream)?;
        let actual = self
            .buffers
            .download(hip, stream, "attention", nq * 128 * 2)?;
        let expected: Vec<u16> = expected
            .chunks_exact(2)
            .map(|x| u16::from_le_bytes([x[0], x[1]]))
            .collect();
        let actual: Vec<u16> = actual
            .chunks_exact(2)
            .map(|x| u16::from_le_bytes([x[0], x[1]]))
            .collect();
        let (rmse, maximum) = relative_rmse(&actual, &expected);
        ensure!(
            rmse < 0.0001,
            "split attention mismatch: relative RMSE {rmse}, max abs {maximum}"
        );
        if self.rank == 0 {
            println!(
                "split attention layer {layer}, position {start}, splits {splits}: relative RMSE {rmse:.9}, max abs {maximum:.9}"
            );
        }
        Ok(())
    }

    fn grouped_gemm(
        &mut self,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        layer: u32,
        rows: usize,
        gate: bool,
        ks: u32,
    ) -> anyhow::Result<()> {
        let h = self.config.hidden_size as usize;
        let intermediate = self.config.moe_intermediate_size as usize / 2;
        let (n, k) = if gate {
            (2 * intermediate, h)
        } else {
            (h, intermediate)
        };
        let suffix = if rows == 1 { "_p1" } else { "" };
        let name = format!(
            "moe_grouped_w4a8_bk128_ks{ks}_tm1_g{}{suffix}",
            u8::from(gate)
        );
        let (a, scales, xptr, partial, counters, output) = if gate {
            (
                "mlp_q",
                "mlp_scale",
                format!("xgu.{layer}"),
                "p_gu",
                "cnt_gu",
                "gate_up",
            )
        } else {
            (
                "act_q",
                "act_scale",
                format!("xdn.{layer}"),
                "p_dn",
                "cnt_dn",
                "y",
            )
        };
        custom(
            "moe_grouped",
            &name,
            [
                (n / 128) as u32,
                if rows == 1 {
                    (TOP_K + 1) as u32
                } else {
                    self.tiles_cap as u32
                },
                ks,
            ],
            256,
            abi::SIG_MOE_GROUPED,
            vec![
                buffer(a),
                buffer(&xptr),
                buffer("tile_expert"),
                buffer("tile_row0"),
                buffer("tile_rows"),
                buffer("row_tok"),
                buffer("meta"),
                buffer(scales),
                buffer(partial),
                buffer(counters),
                buffer(output),
                scalar(n as i64),
                scalar(k as i64),
                scalar(self.rows_cap as i64),
            ],
            stream,
            ctx,
            launcher,
            &self.buffers,
        )
    }

    fn forward(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        collective: &Collective,
        bf16_rules: &DflashRules,
        tokens: &[u32],
        start: usize,
        logits: bool,
    ) -> anyhow::Result<()> {
        self.prepare_embed(hip, stream, tokens, start)?;
        let key = (tokens.len(), logits);
        if tokens.len() == self.max_rows && !self.trace_layers {
            if let Some(graph) = self.graphs.get(&key) {
                hip.graph_launch(&graph.exec, stream)?;
                return Ok(());
            }
            if self.graph_warm.contains(&key) {
                hip.begin_capture(stream, StreamCaptureMode::ThreadLocal)?;
                let captured = self.forward_eager(
                    hip, stream, ctx, launcher, collective, bf16_rules, tokens, start, logits,
                );
                let graph = hip.end_capture(stream);
                captured?;
                let graph = graph?;
                let nodes = hip.graph_node_count(&graph)?;
                let exec = hip.instantiate(&graph)?;
                hip.graph_launch(&exec, stream)?;
                self.graphs.insert(
                    key,
                    PrefillGraph {
                        _graph: graph,
                        exec,
                    },
                );
                if self.rank == 0 {
                    println!("kolibri: captured prefill graph {key:?}, {nodes} nodes");
                }
                return Ok(());
            }
            self.graph_warm.insert(key);
        }
        self.forward_eager(
            hip, stream, ctx, launcher, collective, bf16_rules, tokens, start, logits,
        )
    }

    fn forward_eager(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        collective: &Collective,
        bf16_rules: &DflashRules,
        tokens: &[u32],
        start: usize,
        logits: bool,
    ) -> anyhow::Result<()> {
        self.embed(hip, stream, ctx, launcher, collective, tokens.len())?;
        for layer in self.layer_offset..self.layer_offset + self.layers {
            self.layer(
                hip,
                stream,
                ctx,
                launcher,
                collective,
                bf16_rules,
                layer,
                tokens.len(),
                start,
            )?;
        }
        if !logits {
            return Ok(());
        }
        let h = self.config.hidden_size as usize;
        let last = (tokens.len() - 1) * h * 2;
        custom(
            "ripple_spike",
            "ripple_rmsnorm_bf16",
            [1, 1, 1],
            256,
            abi::SIG_RIPPLE_RMSNORM,
            vec![
                buffer(&format!("h0@{last}")),
                buffer("model.norm.weight"),
                buffer("final_norm"),
                scalar(1),
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
        let desc = dflash::resolve_head_bgemm(1, local_vocab, self.config.hidden_size, bf16_rules)?;
        dispatch(
            stream,
            ctx,
            launcher,
            &desc,
            &self.buffers,
            Shape::new(1, local_vocab, self.config.hidden_size),
        )?;
        Ok(())
    }

    fn dense(
        &self,
        hip: &HipThread,
        stream: &Stream,
        peer: &RankPeer,
        position: usize,
    ) -> anyhow::Result<Vec<f32>> {
        hip.sync_stream(stream)?;
        let local_vocab = self.config.vocab_size as usize / 2;
        let local = self
            .buffers
            .download(hip, stream, "logits", local_vocab * 2)?;
        let other = peer
            .exchange()
            .swap_bytes(&format!("kolibri.logits.{position}"), &local)?;
        let (first, second) = if self.rank == 0 {
            (local, other)
        } else {
            (other, local)
        };
        let mut output = vec![0.0f32; self.config.vocab_size as usize];
        for (rank, bytes) in [&first, &second].into_iter().enumerate() {
            for (index, value) in bytes.chunks_exact(2).enumerate() {
                let logit = bf16_to_f32(u16::from_le_bytes([value[0], value[1]]));
                ensure!(
                    logit.is_finite(),
                    "non-finite logit at rank {rank}, index {index}"
                );
                output[rank * local_vocab + index] = logit;
            }
        }
        Ok(output)
    }

    fn pick(
        &self,
        hip: &HipThread,
        stream: &Stream,
        peer: &RankPeer,
        position: usize,
    ) -> anyhow::Result<u32> {
        let dense = self.dense(hip, stream, peer, position)?;
        let mut best = f32::NEG_INFINITY;
        let mut best_id = 0u32;
        for (index, logit) in dense.into_iter().enumerate() {
            if logit > best {
                best = logit;
                best_id = index as u32;
            }
        }
        Ok(best_id)
    }

    fn sample(
        &self,
        hip: &HipThread,
        stream: &Stream,
        peer: &RankPeer,
        position: usize,
        request: &GenerateRequest,
        output: &[u32],
    ) -> anyhow::Result<(u32, Vec<SampledLogprobs>)> {
        let mut dense = self.dense(hip, stream, peer, position)?;
        let sampled = if self.rank == 0 {
            let params = request.sampling.row_params(self.config.vocab_size as usize);
            let mut noise: Box<dyn NoiseSource> = match request.sampling.seed {
                Some(seed) => Box::new(SeededNoise::new(seed, output.len() as u64, 0)),
                None => Box::new(EntropyNoise::new()),
            };
            let sampled = Sampler::new()
                .sample_dense(
                    &mut dense,
                    &params,
                    &PenaltyContext {
                        prompt: &request.tokens,
                        output,
                    },
                    noise.as_mut(),
                )
                .map_err(|error| anyhow::anyhow!("sampling target: {error}"))?;
            Some(sampled)
        } else {
            None
        };
        let mine = if self.rank == 0 {
            vec![sampled.as_ref().unwrap().token_id]
        } else {
            Vec::new()
        };
        let reply = peer.exchange().xchg_bytes(
            &format!("kolibri.samples.{position}"),
            &serde_json::to_vec(&mine)?,
        )?;
        if self.rank == 0 {
            let sampled = sampled.unwrap();
            Ok((sampled.token_id, sampled.logprobs.into_iter().collect()))
        } else {
            let tokens: Vec<u32> = serde_json::from_slice(&reply)?;
            ensure!(tokens.len() == 1, "invalid sampled token reply");
            Ok((tokens[0], Vec::new()))
        }
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

pub fn config(checkpoint: &Path) -> anyhow::Result<KolibriConfig> {
    let path = checkpoint.join("config.json");
    let text = std::fs::read_to_string(&path).with_context(|| path.display().to_string())?;
    KolibriConfig::from_config_json(&text).map_err(|error| anyhow::anyhow!("{error}"))
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
        let oldest_resident = self.history.len().saturating_sub(LOCAL_CACHE_RING as usize);
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

fn generate(
    hip: &HipThread,
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    collective: &Collective,
    bf16_rules: &DflashRules,
    model: &mut Model,
    peer: &RankPeer,
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
        request.tokens.len() + request.decode <= model.max_context,
        "prompt plus decode exceeds the allocated context"
    );
    peer.agree(
        &format!("kolibri.tokens.{}", request.id),
        request.tokens.iter().fold(0u64, |hash, id| {
            hash.wrapping_mul(1_000_003).wrapping_add(u64::from(*id))
        }),
    )?;
    let started = Instant::now();
    let fast_greedy = request
        .sampling
        .fast_greedy(model.config.vocab_size as usize);
    let cached_tokens = model.prefix_cache.reuse_for(&request.tokens);
    ensure!(
        request.id == 0 || request.cached_tokens == cached_tokens,
        "rank cache prefix disagreement"
    );
    let mut output = Vec::with_capacity(request.decode);
    let mut output_logprobs = Vec::new();
    let mut next = 0u32;
    for start in (cached_tokens..request.tokens.len()).step_by(model.max_rows) {
        let end = (start + model.max_rows).min(request.tokens.len());
        let last = end == request.tokens.len();
        model.forward(
            hip,
            stream,
            ctx,
            launcher,
            collective,
            bf16_rules,
            &request.tokens[start..end],
            start,
            last,
        )?;
        if last {
            if fast_greedy {
                next = model.pick(hip, stream, peer, start)?;
            } else {
                let sampled = model.sample(hip, stream, peer, start, request, &output)?;
                next = sampled.0;
                output_logprobs.extend(sampled.1);
            }
        }
    }
    let mut position = request.tokens.len();
    output.push(next);
    let prefill_elapsed = started.elapsed();
    let mut step_index = 0u32;
    let mut finish = if !request.ignore_eos
        && [model.config.eos_token_id, model.config.pad_token_id].contains(&next)
    {
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
        model.forward(
            hip,
            stream,
            ctx,
            launcher,
            collective,
            bf16_rules,
            &[next],
            position,
            true,
        )?;
        if fast_greedy {
            next = model.pick(hip, stream, peer, position)?;
        } else {
            let sampled = model.sample(hip, stream, peer, position, request, &output)?;
            next = sampled.0;
            output_logprobs.extend(sampled.1);
        }
        position += 1;
        output.push(next);
        finish = if !request.ignore_eos
            && [model.config.eos_token_id, model.config.pad_token_id].contains(&next)
        {
            Some(GenerationFinish::Stop)
        } else if output.len() == request.decode {
            Some(GenerationFinish::Length)
        } else {
            None
        };
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
        speculative_rounds: 0,
        accepted_drafts: 0,
        cached_tokens,
    })
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
        &format!("kolibri.serve.continue.{request_id}.{step}"),
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
        &RankPeer,
        &DflashRules,
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
        child.0 = Some(command.spawn().context("spawning Kolibri rank 1")?);
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
    let mut launcher = Launcher::new("kolibri-exec", &launch_counters);
    let cfg = config(&args.checkpoint)?;
    cfg.validate_tp(2)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let bf16_rules = DflashRules::prod_dflash()?;
    let mut model = Model::load(hip, &stream, args, cfg)?;
    let collective = Collective::new(hip, &stream, args.rank, &peer)?;
    let result = driver(
        &stream,
        &ctx,
        &mut launcher,
        &mut model,
        &collective,
        &peer,
        &bf16_rules,
    );
    drop(child.0.take());
    result
}

pub fn run(hip: &HipThread, args: &ExecArgs) -> anyhow::Result<Vec<u32>> {
    ensure!(args.decode > 0, "--decode must be positive");
    run_with(
        hip,
        args,
        |stream, ctx, launcher, model, collective, peer, bf16_rules| {
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
                model,
                peer,
                &request,
                |_, _, _, _| Ok(true),
            )?;
            println!(
                "kolibri rank{}: prompt {} + {} generated tokens in {:.3}s; ids {:?}",
                args.rank,
                args.tokens.len(),
                stats.output.len(),
                (stats.prefill + stats.decode).as_secs_f64(),
                stats.output
            );
            println!(
                "kolibri rank{}: prefill {:.3}s ({:.2} prompt tok/s); decode {} steps in {:.3}s ({:.2} tok/s)",
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
    let reply = peer
        .exchange()
        .xchg_bytes("kolibri.serve.command", &bytes)?;
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
        |stream, ctx, launcher, model, collective, peer, bf16_rules| {
            if args.rank == 0 {
                let server = super::serve::start(
                    host,
                    port,
                    served_name,
                    &args.checkpoint,
                    args.max_ctx as usize,
                    vec![model.config.eos_token_id, model.config.pad_token_id],
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
                            job.request.cached_tokens =
                                model.prefix_cache.reuse_for(&job.request.tokens);
                            exchange_command(peer, &WireCommand::Run(job.request.clone()))?;
                            let stats = generate(
                                hip,
                                stream,
                                ctx,
                                launcher,
                                collective,
                                bf16_rules,
                                model,
                                peer,
                                &job.request,
                                |tokens, logprobs, step, finish| {
                                    Ok(job.emit(tokens, logprobs, step, finish))
                                },
                            )?;
                            println!(
                                "kolibri request {}: prompt {}, output {}, {:?}, cached {}, prefill {:.3}s, decode {:.3}s",
                                job.request.id,
                                job.request.tokens.len(),
                                stats.output.len(),
                                stats.finish,
                                stats.cached_tokens,
                                stats.prefill.as_secs_f64(),
                                stats.decode.as_secs_f64()
                            );
                            job.complete();
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
                                model,
                                peer,
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

fn relative_rmse(actual: &[u16], expected: &[u16]) -> (f64, f32) {
    let mut error = 0.0f64;
    let mut reference = 0.0f64;
    let mut maximum = 0.0f32;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let actual = bf16_to_f32(actual);
        let expected = bf16_to_f32(expected);
        error += f64::from(actual - expected).powi(2);
        reference += f64::from(expected).powi(2);
        maximum = maximum.max((actual - expected).abs());
    }
    ((error / reference.max(f64::MIN_POSITIVE)).sqrt(), maximum)
}

pub fn packed_spike(
    hip: &HipThread,
    args: &ExecArgs,
    artifact: &Path,
    iterations: u32,
    attention_split_check: bool,
) -> anyhow::Result<()> {
    use super::super::ripple::reference::TensorFile;

    ensure!(args.layers == Some(1), "packed parity requires --layers 1");
    ensure!(iterations > 0, "iterations must be positive");
    let file = TensorFile::open(&artifact.join("layer.safetensors"))?;
    let metadata: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(artifact.join("metadata.json"))?)?;
    let layer = metadata["layer"].as_u64().context("artifact layer")? as u32;
    ensure!(
        layer == args.layer_offset,
        "artifact layer does not match --layer-offset"
    );
    let input = file.read_bf16("input")?;
    let expected_after = file.read_bf16("after_attention")?;
    let expected_final = file.read_bf16("expected_w4a8")?;
    let expected_ids = file.read_i32("router_ids")?;
    let expected_weights = file.read_f32("router_weights")?;
    let h = config(&args.checkpoint)?.hidden_size as usize;
    let rows = input.len() / h;
    ensure!(rows <= MAX_PREFILL_ROWS, "parity artifact exceeds 256 rows");
    run_with(
        hip,
        args,
        |stream, ctx, launcher, model, collective, _, bf16_rules| {
            if attention_split_check {
                model.check_attention_splits(hip, stream, ctx, launcher, layer)?;
                return Ok(Vec::new());
            }
            let input_bytes: Vec<u8> = input.iter().flat_map(|x| x.to_le_bytes()).collect();
            model
                .buffers
                .upload_new(hip, stream, "parity_input", &input_bytes)?;
            let positions: Vec<u8> = (0..rows as i64).flat_map(|x| x.to_le_bytes()).collect();
            model.buffers.upload(hip, stream, "positions", &positions)?;
            model.buffers.upload(hip, stream, "slots", &positions)?;
            let mut forward = || -> anyhow::Result<()> {
                model
                    .buffers
                    .copy(hip, stream, "h0", 0, "parity_input", (rows * h * 2) as u64)?;
                model.layer(
                    hip, stream, ctx, launcher, collective, bf16_rules, layer, rows, 0,
                )
            };
            forward()?;
            hip.sync_stream(stream)?;
            let started = Instant::now();
            for _ in 0..iterations {
                forward()?;
            }
            hip.sync_stream(stream)?;
            let elapsed = started.elapsed().as_secs_f64();
            if model.rank == 0 {
                println!(
                    "packed TP2 layer: {:.3} ms/layer over {iterations} iterations (not model tok/s)",
                    elapsed * 1000.0 / f64::from(iterations)
                );
                let decode = |name: &str| -> anyhow::Result<Vec<u16>> {
                    Ok(model
                        .buffers
                        .download(hip, stream, name, rows * h * 2)?
                        .chunks_exact(2)
                        .map(|x| u16::from_le_bytes([x[0], x[1]]))
                        .collect())
                };
                let actual_after = decode("h1")?;
                let actual_final = decode("h0")?;
                let actual_ids: Vec<i32> = model
                    .buffers
                    .download(hip, stream, "topk_ids", rows * TOP_K * 4)?
                    .chunks_exact(4)
                    .map(|x| i32::from_le_bytes(x.try_into().unwrap()))
                    .collect();
                let actual_weights: Vec<f32> = model
                    .buffers
                    .download(hip, stream, "topk_w", rows * TOP_K * 4)?
                    .chunks_exact(4)
                    .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                    .collect();
                let actual_moe_input = decode("moe_input")?;
                let router_weight = file.read_bf16("router_weight")?;
                let router_bias = file.read_f32("router_bias")?;
                let mut reference_ids = vec![0i32; rows * TOP_K];
                let mut reference_weights = vec![0.0f32; rows * TOP_K];
                for row in 0..rows {
                    let mut logits: Vec<f32> = (0..EXPERTS)
                        .map(|expert| {
                            let mut sum = 0.0f32;
                            for column in 0..h {
                                sum += bf16_to_f32(actual_moe_input[row * h + column])
                                    * bf16_to_f32(router_weight[expert * h + column]);
                            }
                            sum
                        })
                        .collect();
                    for slot in 0..TOP_K {
                        let mut best = 0usize;
                        let mut best_score = f32::NEG_INFINITY;
                        for (expert, &logit) in logits.iter().enumerate() {
                            let score = logit + router_bias[expert];
                            if score > best_score {
                                best = expert;
                                best_score = score;
                            }
                        }
                        reference_ids[row * TOP_K + slot] = best as i32;
                        reference_weights[row * TOP_K + slot] = 1.0 / (1.0 + (-logits[best]).exp());
                        logits[best] = f32::NEG_INFINITY;
                    }
                }
                let (attention_error, attention_max) =
                    relative_rmse(&actual_after, &expected_after);
                let (final_error, final_max) = relative_rmse(&actual_final, &expected_final);
                let router_max = actual_weights
                    .iter()
                    .zip(&reference_weights)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0f32, f32::max);
                let upstream_drift = actual_weights
                    .iter()
                    .zip(&expected_weights)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0f32, f32::max);
                println!(
                    "attention: relative RMSE {attention_error:.6}, max abs {attention_max:.6}"
                );
                println!(
                    "router: original IDs exact {}, actual-input IDs exact {}, max weight abs {router_max:.9}, upstream-attention drift {upstream_drift:.9}",
                    actual_ids == expected_ids,
                    actual_ids == reference_ids
                );
                println!("W4A8 layer: relative RMSE {final_error:.6}, max abs {final_max:.6}");
                ensure!(attention_error < 0.02, "packed attention parity failed");
                if rows <= 2 {
                    ensure!(
                        actual_ids == expected_ids,
                        "packed router selected different experts"
                    );
                }
                ensure!(
                    actual_ids == reference_ids,
                    "router math disagrees at its actual input"
                );
                ensure!(router_max < 0.0005, "packed router weight parity failed");
                ensure!(final_error < 0.02, "packed W4A8 full-layer parity failed");
                println!("PASS: packed TP2 BF16 attention + MXFP4 W4A8 grouped MoE");
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
        let mut cache = PrefixCache::new(5, 513);
        cache.record(&[1, 2, 3], &[8, 9]);
        assert_eq!(cache.reuse_for(&[1, 2, 3, 8, 9, 4]), 5);
        assert_eq!(cache.reuse_for(&[1, 2, 3]), 2);
        assert_eq!(cache.reuse_for(&[1, 2, 7, 8, 9]), 2);
        assert_eq!(cache.reuse_for(&[7, 2, 3]), 0);
        cache.record(&[1, 2, 3, 4], &[8, 9]);
        assert_eq!(cache.reuse_for(&[1, 2, 3, 4, 8, 9, 5]), 0);
    }

    #[test]
    fn prefix_cache_checks_local_ring_residency() {
        let mut cache = PrefixCache::new(1024, 513);
        let history = (0..600).collect::<Vec<u32>>();
        cache.record(&history, &[]);
        let mut continuation = history.clone();
        continuation.push(600);
        assert_eq!(cache.reuse_for(&continuation), 600);
        let mut recent = history[..584].to_vec();
        recent.push(9000);
        assert_eq!(cache.reuse_for(&recent), 584);
        let mut expired = history[..583].to_vec();
        expired.push(9000);
        assert_eq!(cache.reuse_for(&expired), 0);
        let mut oldest = history[..584].to_vec();
        oldest.push(9000);
        assert_eq!(cache.reuse_for(&oldest), 584);
    }
}
