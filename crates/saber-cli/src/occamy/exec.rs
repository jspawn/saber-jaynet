#![allow(clippy::too_many_arguments)]

use std::collections::BTreeMap;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{ensure, Context};
use saber_exec::{
    ArgBindings, LaunchCounters, LaunchCtx, Launcher, LoadCounters, PluginSet, RankPeer,
    TcpExchange,
};
use saber_hip::{DeviceAddr, DevicePtr, HipThread, Job, MallocFlags, Stream, StreamFlags};
use saber_kernels::abi::{self, SigArg};
use saber_kernels::occamy;
use saber_kernels::rules::{ArgValue, LaunchDescriptor, Shape as BindingShape, Slot};
use saber_kernels::saber::{resolve, AttnShape, KvDtype, Op, SaberRules, Shape};
use saber_mem::pack::PackReader;
use saber_model::qwen3_5_moe::{LayerKind, Qwen35MoeConfig};

const H: usize = 2048;
const NQ: usize = 8;
const NKV: usize = 1;
const HD: usize = 256;
const Q: usize = NQ * 2 * HD;
const KV: usize = NKV * HD;
const QKV: usize = Q + 2 * KV;
const HK: usize = 8;
const HV: usize = 16;
const GHD: usize = 128;
const CONV: usize = (2 * HK + HV) * GHD;
const Z: usize = HV * GHD;
const GN: usize = CONV + Z + 2 * HV;
const LOCAL_EXPERTS: usize = 128;
const TOPK: usize = 8;
const INTERMEDIATE: usize = 512;
const AR_MB: u64 = 16 << 20;
const AR_BLOCKS: u64 = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MoeLayout {
    routes: usize,
    row_cap: usize,
    tile_cap: usize,
    route_blocks: usize,
}

fn moe_layout(rows: usize) -> MoeLayout {
    let routes = rows * TOPK;
    MoeLayout {
        routes,
        row_cap: routes * 16 + 16,
        tile_cap: routes + 1,
        route_blocks: rows.div_ceil(32),
    }
}

fn accepted_draft_count(target: &[u32], drafts: &[u32]) -> usize {
    assert!(target.len() > drafts.len());
    target
        .iter()
        .zip(drafts)
        .take_while(|(target, draft)| target == draft)
        .count()
}

#[derive(Debug, Clone)]
pub(super) struct SpeculativeToken {
    pub token: u32,
    pub logprobs: Option<saber_serve::engine::SampledLogprobs>,
}

pub(super) type TargetDecision<'a> =
    dyn FnMut(usize, &[u32], Vec<f32>) -> anyhow::Result<SpeculativeToken> + 'a;

struct StateSnapshot {
    tokens: Vec<u32>,
    slot: usize,
    mtp_ready: bool,
    bytes: u64,
    buffers: Vec<(String, DevicePtr)>,
}

struct CacheEntry {
    snapshot: StateSnapshot,
    used: u64,
}

struct PrefixCache {
    entries: Vec<CacheEntry>,
    budget: u64,
    bytes: u64,
    clock: u64,
}

impl PrefixCache {
    fn new(budget: u64) -> Self {
        Self {
            entries: Vec::new(),
            budget,
            bytes: 0,
            clock: 0,
        }
    }

    fn remove(&mut self, index: usize) {
        let entry = self.entries.remove(index);
        self.bytes -= entry.snapshot.bytes;
    }

    fn make_room(&mut self, bytes: u64) -> bool {
        if bytes > self.budget {
            return false;
        }
        while self.bytes + bytes > self.budget {
            let oldest = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.used)
                .map(|(index, _)| index);
            match oldest {
                Some(index) => self.remove(index),
                None => return false,
            }
        }
        true
    }

    fn insert(&mut self, snapshot: StateSnapshot) {
        if let Some(index) = self.entries.iter().position(|entry| {
            entry.snapshot.tokens == snapshot.tokens
                && entry.snapshot.mtp_ready == snapshot.mtp_ready
        }) {
            self.remove(index);
        }
        if !self.make_room(snapshot.bytes) {
            return;
        }
        self.clock += 1;
        self.bytes += snapshot.bytes;
        self.entries.push(CacheEntry {
            snapshot,
            used: self.clock,
        });
    }
}

fn cache_prefix_matches(cached: &[u32], prompt: &[u32], cached_mtp: bool, use_mtp: bool) -> bool {
    cached.len() < prompt.len() && (!use_mtp || cached_mtp) && prompt.starts_with(cached)
}

pub struct ExecArgs {
    pub layers: PathBuf,
    pub globals: PathBuf,
    pub fp8_experts: Option<PathBuf>,
    pub mtp: Option<PathBuf>,
    pub spec_depth: usize,
    pub checkpoint: PathBuf,
    pub prebuilt: PathBuf,
    pub device: i32,
    pub tokens: Vec<u32>,
    pub decode: u32,
    pub max_ctx: u32,
    pub max_rows: usize,
    pub max_requests: usize,
    pub prefix_cache_mib: u64,
    pub host: String,
    pub serve: Option<u16>,
    pub served_name: String,
    pub rank: u32,
    pub peer_port: u16,
    pub trace_layers: bool,
    pub quality_reference: Option<PathBuf>,
}

fn buffer(name: &str) -> ArgValue {
    ArgValue::Buffer {
        buffer: name.into(),
    }
}

fn null_buffer(name: &str) -> ArgValue {
    ArgValue::NullBuffer {
        buffer: name.into(),
    }
}

fn scalar(value: i64) -> ArgValue {
    ArgValue::I32 { ival: value }
}

fn wide(value: i64) -> ArgValue {
    ArgValue::I64 { ival: value }
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
    pinned: Vec<saber_hip::PinnedBuf>,
}

impl Buffers {
    fn new() -> Self {
        Self {
            device: BTreeMap::new(),
            external: BTreeMap::new(),
            pinned: Vec::new(),
        }
    }

    fn insert(&mut self, name: String, pointer: DevicePtr) {
        self.device.insert(name, pointer);
    }

    fn get(&self, name: &str) -> anyhow::Result<&DevicePtr> {
        let base = name.split_once('@').map_or(name, |(base, _)| base);
        self.device
            .get(base)
            .ok_or_else(|| anyhow::anyhow!("missing device buffer {base}"))
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

    fn pinned_bytes(&mut self, hip: &HipThread, name: String, bytes: &[u8]) -> anyhow::Result<()> {
        let mut pinned = hip.host_malloc(bytes.len().max(4) as u64)?;
        pinned.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        let address = hip.host_device_address(&pinned)?;
        self.external.insert(name, address);
        self.pinned.push(pinned);
        Ok(())
    }

    fn bytes(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        name: String,
        bytes: &[u8],
    ) -> anyhow::Result<()> {
        let pointer = hip.malloc(bytes.len().max(4) as u64)?;
        hip.memcpy_h2d_blocking(&pointer, bytes, stream)?;
        self.insert(name, pointer);
        Ok(())
    }

    fn zero(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        name: String,
        bytes: usize,
    ) -> anyhow::Result<()> {
        let pointer = hip.malloc(bytes.max(4) as u64)?;
        hip.memset(&pointer, 0, stream)?;
        self.insert(name, pointer);
        Ok(())
    }

    fn fill(&self, hip: &HipThread, stream: &Stream, name: &str, value: u8) -> anyhow::Result<()> {
        hip.memset(self.get(name)?, value, stream)?;
        Ok(())
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

    fn copy(
        &self,
        hip: &HipThread,
        stream: &Stream,
        dst: &str,
        dst_offset: u64,
        src: &str,
        src_offset: u64,
        bytes: u64,
    ) -> anyhow::Result<()> {
        hip.run_batch(&[Job::MemcpyD2d {
            dst: self.get(dst)?,
            dst_offset,
            src: self.get(src)?,
            src_offset,
            bytes,
            stream,
        }])?;
        Ok(())
    }

    fn bytes_total(&self) -> u64 {
        self.device.values().map(DevicePtr::len).sum()
    }
}

fn dispatch(
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    descriptor: &LaunchDescriptor,
    buffers: &Buffers,
) -> anyhow::Result<()> {
    let mut bindings = ArgBindings::new(BindingShape::new(0, 0, 0));
    for argument in &descriptor.args {
        if let ArgValue::Buffer { buffer } = &argument.value {
            bindings.bind(buffer, buffers.addr(buffer)?);
        }
    }
    launcher.dispatch_one(ctx, descriptor, &bindings, stream)?;
    Ok(())
}

pub(super) struct Collective {
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
            hip.ipc_import(peer.swap_handle("occamy.ar.scratch", hip.ipc_export(&scratch)?)?)?;
        let peer_flags =
            hip.ipc_import(peer.swap_handle("occamy.ar.flags", hip.ipc_export(&flags)?)?)?;
        ensure!(rank < 2, "Occamy TP2 rank must be 0 or 1");
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
        ensure!(
            bytes as u64 <= AR_MB,
            "exact reduce of {bytes} B exceeds the mailbox"
        );
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
                ArgValue::Buffer {
                    buffer: "ar.peer_scratch".into(),
                },
                ArgValue::Buffer {
                    buffer: "ar.my_scratch".into(),
                },
                buffer(output),
                ArgValue::Buffer {
                    buffer: "ar.peer_flags".into(),
                },
                ArgValue::Buffer {
                    buffer: "ar.my_flags".into(),
                },
                ArgValue::Buffer {
                    buffer: "ar.seq".into(),
                },
                scalar(n16 as i64),
                scalar((AR_MB / 16) as i64),
                scalar(3),
                scalar(0),
            ],
        )?;
        let mut bindings = ArgBindings::new(BindingShape::new(0, 0, 0));
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

struct QualityTrace {
    root: PathBuf,
    generated: Vec<u32>,
}

impl QualityTrace {
    fn load(path: &Path, layers: usize) -> anyhow::Result<Self> {
        let metadata: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path.join("reference.json")).context("reading quality metadata")?,
        )?;
        let generated = metadata
            .get("generated")
            .and_then(serde_json::Value::as_array)
            .context("quality metadata has no generated tokens")?
            .iter()
            .map(|token| {
                token
                    .as_u64()
                    .and_then(|token| u32::try_from(token).ok())
                    .context("invalid generated token")
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        for layer in 0..layers {
            let bytes = std::fs::read(path.join(format!("layer{layer:02}.bf16")))
                .with_context(|| format!("reading quality layer {layer}"))?;
            ensure!(bytes.len() == H * 2, "quality layer {layer} size");
        }
        Ok(Self {
            root: path.to_path_buf(),
            generated,
        })
    }
}

pub(super) struct Model {
    pub(super) rank: u32,
    pub(super) config: Qwen35MoeConfig,
    buffers: Buffers,
    pub(super) max_rows: usize,
    pub(super) max_requests: usize,
    max_ctx: usize,
    nbs: usize,
    conv_bytes: usize,
    state_bytes: usize,
    trace_layers: bool,
    quality: Option<QualityTrace>,
    fp8_experts: bool,
    pub(super) mtp: bool,
    spec_depth: usize,
    drafts: BTreeMap<usize, Vec<u32>>,
    prefix_cache: PrefixCache,
    cache_hits: u64,
    cache_misses: u64,
}

impl Model {
    fn load(
        hip: &HipThread,
        stream: &Stream,
        args: &ExecArgs,
        config: Qwen35MoeConfig,
    ) -> anyhow::Result<Self> {
        ensure!(
            args.max_rows > 0 && args.max_rows <= 64,
            "--max-rows must be 1..=64"
        );
        ensure!(
            args.max_requests > 0 && args.max_requests <= 4,
            "--max-requests must be 1..=4"
        );
        ensure!(
            args.serve.is_some() || !args.tokens.is_empty(),
            "at least one prompt token is required"
        );
        ensure!(
            args.serve.is_none() || args.quality_reference.is_none(),
            "--quality-reference is only valid with the token-ID harness"
        );
        ensure!(
            args.serve.is_some()
                || args.tokens.len() + args.decode as usize <= args.max_ctx as usize,
            "prompt plus decode exceeds --max-ctx"
        );
        if args.mtp.is_some() {
            ensure!(
                (1..=3).contains(&args.spec_depth) && args.spec_depth < args.max_rows,
                "--spec-depth must be 1..=3 and fit one verification chunk"
            );
            ensure!(
                args.max_requests == 1,
                "Occamy MTP currently requires --max-requests 1"
            );
        }
        let max_rows = args.max_rows;
        let nbs = args.max_ctx.div_ceil(16) as usize + 1;
        let quality = args
            .quality_reference
            .as_deref()
            .map(|path| QualityTrace::load(path, config.num_layers() as usize))
            .transpose()?;
        let fp8_experts = args.fp8_experts.is_some();
        let mut model = Self {
            rank: args.rank,
            config,
            buffers: Buffers::new(),
            max_rows,
            max_requests: args.max_requests,
            max_ctx: args.max_ctx as usize,
            nbs,
            conv_bytes: CONV * 3 * 2,
            state_bytes: HV * GHD * GHD * 4,
            trace_layers: args.trace_layers,
            quality,
            fp8_experts,
            mtp: args.mtp.is_some(),
            spec_depth: args.spec_depth,
            drafts: BTreeMap::new(),
            prefix_cache: PrefixCache::new(args.prefix_cache_mib << 20),
            cache_hits: 0,
            cache_misses: 0,
        };
        let started = Instant::now();
        let mut records = 0usize;
        for (root, globals) in [(&args.layers, false), (&args.globals, true)] {
            for shard in ["rank", "shared"] {
                let path = if shard == "rank" {
                    root.join(format!("rank{}/dense.pack", args.rank))
                } else {
                    root.join("shared/dense.pack")
                };
                let reader = PackReader::open(&path)?;
                for index in 0..reader.records().len() {
                    let name = reader.name(index).to_string();
                    let routed = name.contains(".mlp.experts.") || name.contains(".expert.");
                    if (fp8_experts && routed)
                        || (args.rank == 1 && name.contains(".mlp.shared_expert."))
                    {
                        continue;
                    }
                    let bytes = reader.read_record(index)?;
                    reader.check_hash(index, &bytes)?;
                    if !fp8_experts
                        && (matches!(
                            name.as_str(),
                            "model.language_model.embed_tokens.weight" | "lm_head.weight"
                        ) || name.contains(".mlp.shared_expert."))
                    {
                        model.buffers.pinned_bytes(hip, name, &bytes)?;
                    } else {
                        model.buffers.bytes(hip, stream, name, &bytes)?;
                    }
                    records += 1;
                }
            }
            if globals {
                ensure!(records > 0, "empty global pack");
            }
        }
        if let Some(root) = &args.fp8_experts {
            let path = root.join(format!("rank{}/dense.pack", args.rank));
            let reader = PackReader::open(&path)?;
            for index in 0..reader.records().len() {
                let name = reader.name(index).to_string();
                ensure!(
                    name.ends_with(".w8") || name.ends_with(".s8"),
                    "{name}: unexpected FP8 expert record"
                );
                let bytes = reader.read_record(index)?;
                reader.check_hash(index, &bytes)?;
                model.buffers.bytes(hip, stream, name, &bytes)?;
                records += 1;
            }
        }
        if let Some(path) = &args.mtp {
            model.load_mtp(hip, stream, path)?;
        }
        model.make_expert_tables(hip, stream)?;
        model.allocate_scratch(hip, stream, args.max_ctx)?;
        hip.sync_stream(stream)?;
        println!(
            "occamy rank{}: {} records, {:.3} GiB device allocations, loaded in {:.1}s ({})",
            args.rank,
            records,
            model.buffers.bytes_total() as f64 / (1u64 << 30) as f64,
            started.elapsed().as_secs_f64(),
            if fp8_experts {
                "FP8 routed W8A8; BF16 dense/shared/KV; fully VRAM resident"
            } else {
                "BF16"
            }
        );
        Ok(model)
    }

    fn load_mtp(&mut self, hip: &HipThread, stream: &Stream, path: &Path) -> anyhow::Result<()> {
        let mtp = super::mtp::MtpFile::open(path)?;
        let rank = self.rank as u64;
        for name in [
            "mtp.norm.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
        ] {
            let bytes = mtp.whole(name, &[H as u64])?;
            self.buffers.bytes(hip, stream, name.into(), &bytes)?;
        }
        for (source, destination) in [
            ("input_layernorm", "input_layernorm"),
            ("post_attention_layernorm", "post_attention_layernorm"),
        ] {
            let name = format!("mtp.layers.0.{source}.weight");
            let bytes = mtp.whole(&name, &[H as u64])?;
            self.buffers.bytes(
                hip,
                stream,
                format!("model.language_model.layers.40.{destination}.weight"),
                &bytes,
            )?;
        }
        for (projection, rows, local) in [
            ("q_proj", 8192u64, 4096u64),
            ("k_proj", 512, 256),
            ("v_proj", 512, 256),
        ] {
            let source = format!("mtp.layers.0.self_attn.{projection}.weight");
            let bytes = mtp.rows(&source, &[rows, H as u64], rank * local, local)?;
            self.buffers.bytes(
                hip,
                stream,
                format!("model.language_model.layers.40.self_attn.{projection}.weight"),
                &bytes,
            )?;
        }
        for (source, destination, rows, columns) in [
            (
                "mtp.layers.0.self_attn.o_proj.weight",
                "model.language_model.layers.40.self_attn.o_proj.weight",
                2048u64,
                4096u64,
            ),
            ("mtp.fc.weight", "mtp.fc.weight", 2048, 4096),
        ] {
            let bytes = mtp.columns(source, rows, columns, rank * 2048, 2048)?;
            self.buffers
                .bytes(hip, stream, destination.into(), &bytes)?;
        }
        for name in ["q_norm", "k_norm"] {
            let source = format!("mtp.layers.0.self_attn.{name}.weight");
            let bytes = mtp.whole(&source, &[HD as u64])?;
            self.buffers.bytes(
                hip,
                stream,
                format!("model.language_model.layers.40.self_attn.{name}.weight"),
                &bytes,
            )?;
        }
        for (source, suffix, shape) in [
            (
                "mtp.layers.0.mlp.gate.weight",
                "gate.weight",
                vec![256, H as u64],
            ),
            (
                "mtp.layers.0.mlp.shared_expert_gate.weight",
                "shared_expert_gate.weight",
                vec![1, H as u64],
            ),
            (
                "mtp.layers.0.mlp.shared_expert.down_proj.weight",
                "shared_expert.down_proj.weight",
                vec![H as u64, INTERMEDIATE as u64],
            ),
        ] {
            let bytes = mtp.whole(source, &shape)?;
            self.buffers.bytes(
                hip,
                stream,
                format!("model.language_model.layers.40.mlp.{suffix}"),
                &bytes,
            )?;
        }
        let mut shared = mtp.whole(
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
            &[INTERMEDIATE as u64, H as u64],
        )?;
        shared.extend(mtp.whole(
            "mtp.layers.0.mlp.shared_expert.up_proj.weight",
            &[INTERMEDIATE as u64, H as u64],
        )?);
        self.buffers.bytes(
            hip,
            stream,
            "layer.40.mlp.shared_expert.gate_up.w".into(),
            &shared,
        )?;
        for local in 0..LOCAL_EXPERTS {
            let global = self.rank as usize * LOCAL_EXPERTS + local;
            let gate_up = mtp.rows(
                "mtp.layers.0.mlp.experts.gate_up_proj",
                &[256, (2 * INTERMEDIATE) as u64, H as u64],
                global as u64,
                1,
            )?;
            self.buffers.bytes(
                hip,
                stream,
                format!("layer.40.expert.{global}.gate_up.w"),
                &gate_up,
            )?;
            let down = mtp.rows(
                "mtp.layers.0.mlp.experts.down_proj",
                &[256, H as u64, INTERMEDIATE as u64],
                global as u64,
                1,
            )?;
            self.buffers.bytes(
                hip,
                stream,
                format!("model.language_model.layers.40.mlp.experts.{global}.down_proj.weight"),
                &down,
            )?;
        }
        if self.rank == 0 {
            println!(
                "occamy MTP: loaded separate BF16 drafter, depth {}",
                self.spec_depth
            );
        }
        Ok(())
    }

    fn make_expert_tables(&mut self, hip: &HipThread, stream: &Stream) -> anyhow::Result<()> {
        for layer in 0..self.config.num_layers() + u32::from(self.mtp) {
            for down in [false, true] {
                let mut addresses = Vec::new();
                for local in 0..LOCAL_EXPERTS + usize::from(self.rank == 0) {
                    let global = self.rank as usize * LOCAL_EXPERTS + local;
                    let (weight, scales) = if local == LOCAL_EXPERTS {
                        if down {
                            (
                                format!(
                                    "model.language_model.layers.{layer}.mlp.shared_expert.down_proj.weight"
                                ),
                                None,
                            )
                        } else {
                            (format!("layer.{layer}.mlp.shared_expert.gate_up.w"), None)
                        }
                    } else if down {
                        let prefix = format!(
                            "model.language_model.layers.{layer}.mlp.experts.{global}.down_proj"
                        );
                        if self.fp8_experts && layer < self.config.num_layers() {
                            (format!("{prefix}.w8"), Some(format!("{prefix}.s8")))
                        } else {
                            (format!("{prefix}.weight"), None)
                        }
                    } else {
                        let prefix = format!("layer.{layer}.expert.{global}.gate_up");
                        if self.fp8_experts && layer < self.config.num_layers() {
                            (format!("{prefix}.w8"), Some(format!("{prefix}.s8")))
                        } else {
                            (format!("{prefix}.w"), None)
                        }
                    };
                    addresses.push(self.buffers.addr(&weight)?.as_u64());
                    addresses.push(match scales {
                        Some(name) => self.buffers.addr(&name)?.as_u64(),
                        None => 0,
                    });
                }
                let bytes: Vec<u8> = addresses.into_iter().flat_map(u64::to_le_bytes).collect();
                let table = format!("layer.{layer}.{}", if down { "xdn" } else { "xgu" });
                self.buffers.bytes(hip, stream, table, &bytes)?;
            }
        }
        Ok(())
    }

    fn convolution_state_bytes(&self) -> usize {
        2 * self.conv_bytes
    }

    fn allocate_scratch(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        max_ctx: u32,
    ) -> anyhow::Result<()> {
        let m = self.max_rows;
        let MoeLayout {
            routes: route_rows,
            row_cap,
            tile_cap,
            route_blocks,
        } = moe_layout(m);
        let bf16 = |rows: usize, cols: usize| rows * cols * 2;
        for (name, bytes) in [
            ("ids".into(), m * 4),
            ("embed".into(), bf16(m, H)),
            ("res0".into(), bf16(m, H)),
            ("res1".into(), bf16(m, H)),
            ("mid".into(), bf16(m, H)),
            ("zeros".into(), bf16(m, H)),
            ("normed".into(), bf16(m, H)),
            ("moe_in".into(), bf16(m, H)),
            ("mixer".into(), bf16(m, H)),
            ("reduced".into(), bf16(m, H)),
            ("moe".into(), bf16(m, H)),
            ("proj".into(), bf16(m, GN)),
            ("q".into(), bf16(m, NQ * HD)),
            ("k".into(), bf16(m, NQ * HD)),
            ("v".into(), bf16(m, NQ * HD)),
            ("g".into(), m * HV * 4),
            ("beta".into(), m * HV * 4),
            ("a".into(), bf16(m, HV * 64)),
            ("core".into(), bf16(m, NQ * HD)),
            ("gated".into(), bf16(m, NQ * HD)),
            (
                "part".into(),
                (4 * m * GN * 4).max(4 * self.config.vocab_size as usize / 2 * 4),
            ),
            (
                "cnt".into(),
                (GN.div_ceil(64) * m.div_ceil(16) * 4)
                    .max(self.config.vocab_size as usize / 2 / 64 * 4),
            ),
            ("router_logits".into(), bf16(m, 256)),
            ("shared_gate".into(), bf16(m, 1)),
            ("route_ids".into(), route_rows * 4),
            ("route_weights".into(), route_rows * 4),
            ("route_pos".into(), route_rows * 4),
            ("row_tok".into(), row_cap * 4),
            ("row_route".into(), row_cap * 4),
            ("tile_expert".into(), tile_cap * 4),
            ("tile_row0".into(), tile_cap * 4),
            ("tile_rows".into(), tile_cap * 4),
            ("expert_cnt".into(), LOCAL_EXPERTS * 4),
            ("meta".into(), 6 * 4),
            ("blk_cnt".into(), route_blocks * LOCAL_EXPERTS * 4),
            ("route_rank".into(), route_rows * 4),
            ("arrive".into(), 4),
            ("gu".into(), bf16(row_cap, 2 * INTERMEDIATE)),
            ("act".into(), bf16(row_cap, INTERMEDIATE)),
            ("y".into(), bf16(row_cap, H)),
            (
                "head_logits".into(),
                (self.config.vocab_size as usize / 2) * 2,
            ),
            ("local_pick".into(), 4),
            ("pos3".into(), 3 * m * 8),
            ("positions".into(), m * 8),
            ("slots".into(), m * 8),
            ("ring_slots".into(), m * 8),
            ("block_table".into(), self.nbs * 4),
            ("seqused_k".into(), 4),
            ("cu".into(), 8),
            ("cache_idx".into(), 4),
            ("sidx".into(), m * 4),
            ("naccept".into(), 4),
            ("has_init".into(), 1),
            ("bar".into(), 4),
        ] {
            self.buffers.zero(hip, stream, name, bytes)?;
        }
        if self.fp8_experts {
            for (name, bytes) in [
                ("moe_q".into(), m * H),
                ("moe_s".into(), m * (H / 128) * 4),
                ("act_q".into(), row_cap * INTERMEDIATE),
                ("act_s".into(), row_cap * (INTERMEDIATE / 128) * 4),
            ] {
                self.buffers.zero(hip, stream, name, bytes)?;
            }
        }
        if self.mtp {
            for name in [
                "target_hidden",
                "mtp_ne",
                "mtp_nh",
                "mtp_fc",
                "mtp_res",
                "mtp_mid",
            ] {
                self.buffers.zero(hip, stream, name.into(), m * H * 2)?;
            }
        }
        for slot in 0..self.max_requests {
            if self.mtp {
                self.buffers
                    .zero(hip, stream, format!("mtp.{slot}.hidden"), m * H * 2)?;
                self.buffers.zero(
                    hip,
                    stream,
                    format!("state.{slot}.40.kv"),
                    self.nbs * NKV * 16 * 2 * HD * 2,
                )?;
            }
            for layer in 0..self.config.num_layers() {
                match self.config.layers[layer as usize] {
                    LayerKind::Gdn => {
                        self.buffers.zero(
                            hip,
                            stream,
                            format!("state.{slot}.{layer}.conv"),
                            self.convolution_state_bytes(),
                        )?;
                        self.buffers.zero(
                            hip,
                            stream,
                            format!("state.{slot}.{layer}.h0"),
                            2 * self.state_bytes,
                        )?;
                        self.buffers.zero(
                            hip,
                            stream,
                            format!("state.{slot}.{layer}.h1"),
                            self.state_bytes,
                        )?;
                        if self.mtp {
                            for step in 0..=self.spec_depth {
                                for (suffix, bytes) in [
                                    ("conv", self.convolution_state_bytes()),
                                    ("h0", 2 * self.state_bytes),
                                    ("h1", self.state_bytes),
                                ] {
                                    self.buffers.zero(
                                        hip,
                                        stream,
                                        format!("verify.{slot}.{layer}.{step}.{suffix}"),
                                        bytes,
                                    )?;
                                }
                            }
                        }
                    }
                    LayerKind::FullAttention => {
                        self.buffers.zero(
                            hip,
                            stream,
                            format!("state.{slot}.{layer}.kv"),
                            self.nbs * NKV * 16 * 2 * HD * 2,
                        )?;
                    }
                }
            }
        }
        let rope = crate::flashnext::run::rope_table(max_ctx.max(16));
        self.buffers.bytes(hip, stream, "cos_sin".into(), &rope)?;
        let rules = SaberRules::occamy().map_err(|error| anyhow::anyhow!(error))?;
        let workspace = rules
            .workspace
            .get("attn_decode_scratch")
            .context("attention workspace")?;
        self.buffers.zero(
            hip,
            stream,
            "attn_decode_scratch".into(),
            workspace.capacity_bytes as usize,
        )?;
        Ok(())
    }

    fn launch(
        &self,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        launch: &LaunchDescriptor,
    ) -> anyhow::Result<()> {
        if self.trace_layers {
            println!(
                "occamy rank{}: dispatch {} {}",
                self.rank, launch.plugin, launch.kernel
            );
        }
        dispatch(stream, ctx, launcher, launch, &self.buffers)
            .with_context(|| format!("dispatch {} {}", launch.plugin, launch.kernel))
    }

    fn dense(
        &self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        input: &str,
        weight: &str,
        output: &str,
        m: usize,
        n: usize,
        k: usize,
        ldc: usize,
    ) -> anyhow::Result<()> {
        if ldc != n {
            ensure!(k == H, "strided projection K {k}");
            let launch = descriptor(
                "flashnext_trunk8",
                "fn_t8_bgemm_m16_ldc",
                [(n as u32).div_ceil(64), (m as u32).div_ceil(16), 1],
                128,
                abi::SIG_FN_T8_BGEMM_LDC,
                vec![
                    buffer(input),
                    buffer(weight),
                    buffer(output),
                    scalar(m as i64),
                    scalar(n as i64),
                    scalar(k as i64),
                    scalar(ldc as i64),
                ],
            )?;
            return self.launch(stream, ctx, launcher, &launch);
        }
        let cell = occamy::dense_projection(m as u32, n as u32, k as u32)
            .map_err(|error| anyhow::anyhow!(error))?;
        if cell.splits > 1 {
            self.buffers.fill(hip, stream, "part", 0)?;
            self.buffers.fill(hip, stream, "cnt", 0)?;
        }
        let mut arguments = vec![
            buffer(input),
            buffer(weight),
            if cell.splits > 1 {
                buffer("part")
            } else {
                null_buffer("part")
            },
            if cell.splits > 1 {
                buffer("cnt")
            } else {
                null_buffer("cnt")
            },
            buffer(output),
            scalar(m as i64),
            scalar(n as i64),
            scalar(k as i64),
        ];
        let signature = if cell.symbol.starts_with("fn_bgemm_sk_") {
            abi::SIG_FN_BGEMM_SK
        } else {
            arguments = vec![
                buffer(input),
                buffer(weight),
                buffer(output),
                scalar(m as i64),
                scalar(n as i64),
                scalar(k as i64),
            ];
            abi::SIG_FN_BGEMM
        };
        let launch = descriptor(
            "flashnext_hc",
            cell.symbol,
            cell.grid,
            cell.threads,
            signature,
            arguments,
        )?;
        self.launch(stream, ctx, launcher, &launch)
    }

    fn add_norm(
        &self,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        y: &str,
        residual: &str,
        weight: &str,
        output: &str,
        residual_out: &str,
        rows: usize,
    ) -> anyhow::Result<()> {
        let launch = descriptor(
            "head",
            "saber_add_rms_bf16",
            [rows as u32, 1, 1],
            256,
            abi::SIG_ADD_RMS_BF16,
            vec![
                buffer(y),
                buffer(residual),
                buffer(weight),
                buffer(output),
                buffer(residual_out),
                scalar(H as i64),
                ArgValue::F32 {
                    fval: self.config.rms_norm_eps as f32,
                },
            ],
        )?;
        self.launch(stream, ctx, launcher, &launch)
    }

    fn embed(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        collective: &Collective,
        tokens: &[u32],
    ) -> anyhow::Result<()> {
        let ids: Vec<u8> = tokens.iter().flat_map(|id| id.to_le_bytes()).collect();
        self.buffers.upload(hip, stream, "ids", &ids)?;
        let launch = descriptor(
            "head",
            "saber_embed_bf16",
            [tokens.len() as u32, 1, 1],
            256,
            abi::SIG_EMBED,
            vec![
                buffer("model.language_model.embed_tokens.weight"),
                buffer("ids"),
                buffer("embed"),
                scalar(H as i64),
                scalar((self.rank * self.config.vocab_size / 2) as i64),
                scalar(((self.rank + 1) * self.config.vocab_size / 2) as i64),
            ],
        )?;
        self.launch(stream, ctx, launcher, &launch)?;
        collective.reduce(
            stream,
            ctx,
            launcher,
            &self.buffers,
            "embed",
            "res0",
            tokens.len() * H,
        )
    }

    fn update_tables(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        rows: usize,
        position: usize,
        decode: bool,
    ) -> anyhow::Result<()> {
        let pos: Vec<i64> = (0..rows).map(|index| (position + index) as i64).collect();
        let pos3: Vec<u8> = (0..3)
            .flat_map(|_| pos.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .collect();
        let positions: Vec<u8> = pos.iter().flat_map(|v| v.to_le_bytes()).collect();
        let slots: Vec<u8> = pos.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ring_slots = vec![0xffu8; rows * 8];
        for (name, bytes) in [
            ("pos3", pos3),
            ("positions", positions),
            ("slots", slots),
            ("ring_slots", ring_slots),
        ] {
            self.buffers.upload(hip, stream, name, &bytes)?;
        }
        self.buffers.upload(
            hip,
            stream,
            "block_table",
            &(0..self.nbs as i32)
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        self.buffers.upload(
            hip,
            stream,
            "seqused_k",
            &((position + rows) as i32).to_le_bytes(),
        )?;
        self.buffers
            .upload(hip, stream, "cu", &[0, 0, 0, 0, rows as u8, 0, 0, 0])?;
        self.buffers
            .upload(hip, stream, "cache_idx", &(i32::from(decode)).to_le_bytes())?;
        self.buffers
            .upload(hip, stream, "sidx", &(i32::from(decode)).to_le_bytes())?;
        self.buffers
            .upload(hip, stream, "naccept", &1i32.to_le_bytes())?;
        self.buffers
            .upload(hip, stream, "has_init", &[u8::from(position > 0)])?;
        Ok(())
    }

    fn attention(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        slot: usize,
        layer: u32,
        rows: usize,
        position: usize,
        decode: bool,
    ) -> anyhow::Result<()> {
        let prefix = format!("model.language_model.layers.{layer}.self_attn");
        for (projection, n, offset) in [("q_proj", Q, 0), ("k_proj", KV, Q), ("v_proj", KV, Q + KV)]
        {
            self.dense(
                hip,
                stream,
                ctx,
                launcher,
                "normed",
                &format!("{prefix}.{projection}.weight"),
                &format!("proj@{}", offset * 2),
                rows,
                n,
                H,
                QKV,
            )?;
        }
        if self.trace_layers {
            hip.sync_stream(stream)?;
            println!("occamy rank{}: layer {layer} qkv done", self.rank);
        }
        let rope = descriptor(
            "flashnext_glue",
            "fn_qk_norm_rope_h256_rot64_mi1_v2",
            [((rows * (NQ + NKV)) as u32).div_ceil(8), 1, 1],
            256,
            abi::SIG_QK_NORM_ROPE,
            vec![
                buffer("proj"),
                buffer(&format!("{prefix}.q_norm.weight")),
                buffer(&format!("{prefix}.k_norm.weight")),
                buffer("cos_sin"),
                buffer("pos3"),
                buffer("q"),
                buffer("k"),
                scalar(rows as i64),
                scalar(NQ as i64),
                scalar(NKV as i64),
                scalar(QKV as i64),
                ArgValue::F32 {
                    fval: self.config.rms_norm_eps as f32,
                },
                scalar(11),
                scalar(11),
                scalar(10),
            ],
        )?;
        self.launch(stream, ctx, launcher, &rope)?;
        if self.trace_layers {
            hip.sync_stream(stream)?;
            println!("occamy rank{}: layer {layer} rope done", self.rank);
        }
        let kv_name = format!("state.{slot}.{layer}.kv");
        let block_elements = NKV * 16 * 2 * HD;
        let append = descriptor(
            "flashnext_glue",
            occamy::KV_APPEND_SYMBOL,
            [rows as u32, NKV as u32, 1],
            128,
            abi::SIG_FN_KV_APPEND_RING,
            vec![
                buffer("k"),
                wide((NKV * HD) as i64),
                buffer(&format!("proj@{}", (Q + KV) * 2)),
                wide(QKV as i64),
                buffer(&kv_name),
                buffer("slots"),
                wide(block_elements as i64),
                wide((2 * HD) as i64),
                wide(block_elements as i64),
                null_buffer("ikey"),
                wide(0),
                null_buffer("ring"),
                null_buffer("ring_pos"),
                buffer("ring_slots"),
                null_buffer("positions"),
                scalar(rows as i64),
            ],
        )?;
        self.launch(stream, ctx, launcher, &append)?;
        if self.trace_layers {
            hip.sync_stream(stream)?;
            println!(
                "occamy rank{}: layer {layer} attention core start",
                self.rank
            );
        }
        let canonical_verify = decode && rows > 1;
        ensure!(
            !canonical_verify || self.mtp,
            "multi-row decode is only valid for MTP verification"
        );
        let rules = SaberRules::occamy().map_err(|error| anyhow::anyhow!(error))?;
        for row in 0..if canonical_verify { rows } else { 1 } {
            let active_rows = if canonical_verify { 1 } else { rows };
            let active_position = position + row;
            if canonical_verify {
                self.buffers.upload(
                    hip,
                    stream,
                    "seqused_k",
                    &((active_position + 1) as i32).to_le_bytes(),
                )?;
            }
            let shape = AttnShape {
                num_seqs: 1,
                q_len: active_rows as u32,
                live_rows: active_rows as u32,
                q_heads: NQ as u32,
                kv_heads: NKV as u32,
                head_dim: HD as u32,
                block_size: 16,
                max_blocks: self.nbs as u32,
                max_ctx: (active_position + active_rows) as u32,
                splits: u32::from(decode),
                kv_dtype: KvDtype::Bf16,
                kv_block_stride: block_elements as i64,
                kv_head_stride: (16 * 2 * HD) as i64,
                scale_bits: (1.0f32 / 16.0).to_bits(),
                has_descales: false,
                ragged_rows: 0,
            };
            let op = if decode {
                Op::AttnDecode
            } else {
                Op::AttnPrefill { fp8_mode: 0 }
            };
            let attention =
                resolve(op, Shape::Attn(shape), &rules).map_err(|error| anyhow::anyhow!(error))?;
            let q = if canonical_verify {
                format!("q@{}", row * NQ * HD * 2)
            } else {
                "q".into()
            };
            let core = if canonical_verify {
                format!("core@{}", row * NQ * HD * 2)
            } else {
                "core".into()
            };
            let rebound = self.rebind_attention(attention, &kv_name, &q, &core)?;
            self.launch(stream, ctx, launcher, &rebound)?;
            if decode {
                let combine = saber_kernels::saber::attn_decode_combine(shape, &rules)
                    .map_err(|error| anyhow::anyhow!(error))?;
                let rebound = self.rebind_attention(combine, &kv_name, &q, &core)?;
                self.launch(stream, ctx, launcher, &rebound)?;
            }
        }
        if self.trace_layers {
            hip.sync_stream(stream)?;
            println!(
                "occamy rank{}: layer {layer} attention core done",
                self.rank
            );
        }
        let gate = descriptor(
            "flashnext_hc",
            "fn_attn_gate_bf16_t256",
            [rows as u32, 1, 1],
            256,
            abi::SIG_FN_ATTN_GATE_BF16,
            vec![
                buffer("core"),
                buffer("proj"),
                buffer("gated"),
                scalar((NQ * HD) as i64),
                scalar(HD as i64),
                scalar(QKV as i64),
            ],
        )?;
        self.launch(stream, ctx, launcher, &gate)?;
        self.dense(
            hip,
            stream,
            ctx,
            launcher,
            "gated",
            &format!("{prefix}.o_proj.weight"),
            "mixer",
            rows,
            H,
            NQ * HD,
            H,
        )
    }

    fn rebind_attention(
        &self,
        mut launch: LaunchDescriptor,
        kv: &str,
        q: &str,
        output: &str,
    ) -> anyhow::Result<LaunchDescriptor> {
        for argument in &mut launch.args {
            if let ArgValue::Buffer { buffer: name } = &mut argument.value {
                match name.as_str() {
                    "q" => *name = q.into(),
                    "kv" => *name = kv.into(),
                    "block_table" => *name = "block_table".into(),
                    "seqused_k" => *name = "seqused_k".into(),
                    "out" => *name = output.into(),
                    _ => {}
                }
            }
        }
        Ok(launch)
    }

    fn gdn(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        slot: usize,
        layer: u32,
        rows: usize,
        decode: bool,
    ) -> anyhow::Result<()> {
        let prefix = format!("model.language_model.layers.{layer}.linear_attn");
        for (weight, n, offset) in [
            (format!("layer.{layer}.gdn.in_proj_qkvz.w"), CONV + Z, 0),
            (format!("layer.{layer}.gdn.in_proj_ba.w"), 2 * HV, CONV + Z),
        ] {
            self.dense(
                hip,
                stream,
                ctx,
                launcher,
                "normed",
                &weight,
                &format!("proj@{}", offset * 2),
                rows,
                n,
                H,
                GN,
            )?;
        }
        let conv_state = if decode {
            format!("state.{slot}.{layer}.conv")
        } else {
            format!("state.{slot}.{layer}.conv@{}", self.conv_bytes)
        };
        let state = format!("state.{slot}.{layer}.h0");
        if decode {
            let canonical_verify = rows > 1;
            ensure!(
                !canonical_verify || self.mtp,
                "multi-row decode is only valid for MTP verification"
            );
            for row in 0..if canonical_verify { rows } else { 1 } {
                if canonical_verify {
                    self.buffers
                        .upload(hip, stream, "cu", &[0, 0, 0, 0, 1, 0, 0, 0])?;
                }
                let projection = row * GN * 2;
                let activations = row * NQ * HD * 2;
                let fused = descriptor(
                    "saber_gdn_fused_update",
                    "saber_gdn_fused_update_w4k128v128_ab1",
                    [HV as u32, 1, 1],
                    256,
                    abi::SIG_SABER_GDN_FUSED_UPDATE,
                    vec![
                        buffer(&format!("proj@{projection}")),
                        wide(GN as i64),
                        buffer(&format!("{prefix}.conv1d.weight")),
                        null_buffer("bias"),
                        buffer(&conv_state),
                        wide((3 * CONV) as i64),
                        wide(3),
                        wide(1),
                        scalar(3),
                        buffer("cache_idx"),
                        wide(1),
                        buffer("naccept"),
                        buffer("cu"),
                        scalar(1),
                        scalar(HV as i64),
                        scalar(HK as i64),
                        scalar(1),
                        buffer(&format!("q@{activations}")),
                        buffer(&format!("k@{activations}")),
                        buffer(&format!("v@{activations}")),
                        buffer(&format!("proj@{}", projection + (CONV + Z + HV) * 2)),
                        buffer(&format!("proj@{}", projection + (CONV + Z) * 2)),
                        wide(GN as i64),
                        buffer(&format!("{prefix}.A_log")),
                        buffer(&format!("{prefix}.dt_bias")),
                        buffer(&state),
                        wide((HV * GHD * GHD) as i64),
                        wide((GHD * GHD) as i64),
                        buffer(&format!("core@{activations}")),
                        buffer("sidx"),
                        wide(1),
                        ArgValue::F32 {
                            fval: 1.0 / (GHD as f32).sqrt(),
                        },
                        ArgValue::F32 { fval: 20.0 },
                        buffer("bar"),
                        scalar(HV as i64),
                        scalar(1),
                    ],
                )?;
                self.launch(stream, ctx, launcher, &fused)?;
                if canonical_verify {
                    self.verification_layer_state(hip, stream, slot, layer, row, false)?;
                }
            }
        } else {
            let prep = descriptor(
                "saber_gdn_conv",
                "saber_gdn_conv_prep_w4_h128_hpb8_ab1",
                [(rows.div_ceil(64) + 1) as u32, (CONV / 1024) as u32, 1],
                256,
                abi::SIG_SABER_GDN_CONV_PREP,
                vec![
                    buffer("proj"),
                    wide(GN as i64),
                    buffer(&format!("{prefix}.conv1d.weight")),
                    null_buffer("bias"),
                    buffer(&conv_state),
                    wide((3 * CONV) as i64),
                    wide(3),
                    wide(1),
                    buffer("cache_idx"),
                    wide(1),
                    buffer("has_init"),
                    buffer(&format!("proj@{}", (CONV + Z + HV) * 2)),
                    buffer(&format!("proj@{}", (CONV + Z) * 2)),
                    wide(GN as i64),
                    buffer(&format!("{prefix}.A_log")),
                    buffer(&format!("{prefix}.dt_bias")),
                    buffer("q"),
                    buffer("k"),
                    buffer("v"),
                    buffer("g"),
                    buffer("beta"),
                    buffer("cu"),
                    scalar(1),
                    scalar(HV as i64),
                    scalar(HK as i64),
                    ArgValue::F32 { fval: 20.0 },
                ],
            )?;
            self.launch(stream, ctx, launcher, &prep)?;
            let solve = descriptor(
                "saber_gdn_kkt_solve",
                occamy::GDN_KKT_SYMBOL,
                [(rows.div_ceil(64) + 1) as u32, HK as u32, 1],
                128,
                abi::SIG_SABER_GDN_KKT,
                vec![
                    buffer("k"),
                    buffer("beta"),
                    buffer("g"),
                    buffer("a"),
                    buffer("cu"),
                    scalar(1),
                    scalar(HV as i64),
                    scalar(HK as i64),
                ],
            )?;
            self.launch(stream, ctx, launcher, &solve)?;
            let next_state = format!("state.{slot}.{layer}.h1");
            let scan = descriptor(
                "flashnext_gdn",
                "fn_gdn_chunk_scan_xg_k128_v128_c64",
                [2, HV as u32, 1],
                256,
                abi::SIG_SABER_GDN_CHUNK_SCAN,
                vec![
                    buffer("q"),
                    buffer("k"),
                    buffer("v"),
                    buffer("a"),
                    buffer("g"),
                    buffer("beta"),
                    buffer(&format!("{state}@{}", self.state_bytes)),
                    buffer("core"),
                    buffer(&next_state),
                    buffer("cu"),
                    scalar(HV as i64),
                    scalar(HK as i64),
                    ArgValue::F32 {
                        fval: 1.0 / (GHD as f32).sqrt(),
                    },
                ],
            )?;
            self.launch(stream, ctx, launcher, &scan)?;
            self.buffers.copy(
                hip,
                stream,
                &state,
                self.state_bytes as u64,
                &next_state,
                0,
                self.state_bytes as u64,
            )?;
        }
        let norm = descriptor(
            "flashnext_hc",
            "fn_gdn_norm_bf16_t256",
            [rows as u32, 1, 1],
            256,
            abi::SIG_FN_GDN_NORM_BF16,
            vec![
                buffer("core"),
                buffer(&format!("proj@{}", CONV * 2)),
                wide(GN as i64),
                buffer(&format!("{prefix}.norm.weight")),
                buffer("gated"),
                scalar(Z as i64),
                ArgValue::F32 {
                    fval: self.config.rms_norm_eps as f32,
                },
            ],
        )?;
        self.launch(stream, ctx, launcher, &norm)?;
        self.dense(
            hip,
            stream,
            ctx,
            launcher,
            "gated",
            &format!("{prefix}.out_proj.weight"),
            "mixer",
            rows,
            H,
            Z,
            H,
        )
    }

    fn moe(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        layer: u32,
        rows: usize,
    ) -> anyhow::Result<()> {
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        self.dense(
            hip,
            stream,
            ctx,
            launcher,
            "moe_in",
            &format!("{prefix}.gate.weight"),
            "router_logits",
            rows,
            256,
            H,
            256,
        )?;
        self.dense(
            hip,
            stream,
            ctx,
            launcher,
            "moe_in",
            &format!("{prefix}.shared_expert_gate.weight"),
            "shared_gate",
            rows,
            1,
            H,
            1,
        )?;
        self.buffers.fill(hip, stream, "row_tok", 0xff)?;
        for name in ["expert_cnt", "meta", "blk_cnt", "arrive"] {
            self.buffers.fill(hip, stream, name, 0)?;
        }
        let route = descriptor(
            "occamy_moe",
            occamy::ROUTE_SYMBOL,
            occamy::route_grid(rows as u32),
            256,
            abi::SIG_MOE_ROUTE,
            vec![
                buffer("router_logits"),
                scalar(rows as i64),
                scalar((self.rank as usize * LOCAL_EXPERTS) as i64),
                scalar(LOCAL_EXPERTS as i64),
                scalar(1),
                scalar(i64::from(self.rank == 0)),
                buffer("route_ids"),
                buffer("route_weights"),
                buffer("route_pos"),
                buffer("row_tok"),
                buffer("row_route"),
                buffer("tile_expert"),
                buffer("tile_row0"),
                buffer("tile_rows"),
                buffer("expert_cnt"),
                buffer("meta"),
                buffer("blk_cnt"),
                buffer("route_rank"),
                buffer("arrive"),
            ],
        )?;
        self.launch(stream, ctx, launcher, &route)?;
        if self.fp8_experts && layer < self.config.num_layers() {
            return self.moe_fp8_after_route(stream, ctx, launcher, layer, rows);
        }
        let MoeLayout {
            row_cap, tile_cap, ..
        } = moe_layout(rows);
        let gate_up = descriptor(
            "flashnext_moe",
            occamy::GATE_UP_SYMBOL,
            occamy::grouped_grid((2 * INTERMEDIATE) as u32, tile_cap as u32),
            128,
            abi::SIG_FN_MOE_BGEMM,
            vec![
                buffer("moe_in"),
                buffer(&format!("layer.{layer}.xgu")),
                buffer("tile_expert"),
                buffer("tile_row0"),
                buffer("tile_rows"),
                buffer("row_tok"),
                buffer("meta"),
                null_buffer("part"),
                null_buffer("cnt"),
                buffer("gu"),
                scalar((2 * INTERMEDIATE) as i64),
                scalar(H as i64),
                scalar(row_cap as i64),
            ],
        )?;
        let silu = descriptor(
            "flashnext_glue",
            occamy::SILU_SYMBOL,
            [row_cap as u32, 1, 1],
            256,
            abi::SIG_FN_MOE_SILU_BF16,
            vec![
                buffer("gu"),
                buffer("act"),
                scalar(INTERMEDIATE as i64),
                buffer("row_tok"),
                buffer("meta"),
            ],
        )?;
        let down = descriptor(
            "flashnext_moe",
            occamy::DOWN_SYMBOL,
            occamy::grouped_grid(H as u32, tile_cap as u32),
            128,
            abi::SIG_FN_MOE_BGEMM,
            vec![
                buffer("act"),
                buffer(&format!("layer.{layer}.xdn")),
                buffer("tile_expert"),
                buffer("tile_row0"),
                buffer("tile_rows"),
                buffer("row_tok"),
                buffer("meta"),
                null_buffer("part"),
                null_buffer("cnt"),
                buffer("y"),
                scalar(H as i64),
                scalar(INTERMEDIATE as i64),
                scalar(row_cap as i64),
            ],
        )?;
        let combine = descriptor(
            "occamy_moe",
            occamy::COMBINE_SYMBOL,
            [rows as u32, 1, 1],
            256,
            abi::SIG_MOE_COMBINE,
            vec![
                buffer("y"),
                buffer("route_weights"),
                buffer("route_pos"),
                buffer("shared_gate"),
                buffer("meta"),
                buffer("moe"),
                scalar(H as i64),
            ],
        )?;
        for launch in [&gate_up, &silu, &down, &combine] {
            self.launch(stream, ctx, launcher, launch)?;
        }
        Ok(())
    }

    fn quant_activations(
        &self,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        input: &str,
        output: &str,
        scales: &str,
        rows: usize,
        columns: usize,
    ) -> anyhow::Result<()> {
        ensure!(columns % 128 == 0, "FP8 activation width {columns}");
        let launch = descriptor(
            saber_kernels::fn50::QUANT_PLUGIN,
            saber_kernels::fn50::ACT_QUANT_SYMBOL,
            [
                saber_kernels::fn50::act_quant_blocks(rows as u64, columns as u64) as u32,
                1,
                1,
            ],
            saber_kernels::fn50::ACT_QUANT_THREADS as u32,
            abi::SIG_DFLASH_ACT_QUANT,
            vec![
                buffer(input),
                buffer(output),
                buffer(scales),
                scalar(rows as i64),
                scalar(columns as i64),
            ],
        )?;
        self.launch(stream, ctx, launcher, &launch)
    }

    fn moe_fp8_after_route(
        &self,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        layer: u32,
        rows: usize,
    ) -> anyhow::Result<()> {
        let MoeLayout {
            row_cap, tile_cap, ..
        } = moe_layout(rows);
        self.quant_activations(stream, ctx, launcher, "moe_in", "moe_q", "moe_s", rows, H)?;
        let gate_up_fp8 = descriptor(
            saber_kernels::fn50::PLUGIN,
            saber_kernels::fn50::fp8_symbol(true),
            [(2 * INTERMEDIATE / 128) as u32, tile_cap as u32, 1],
            128,
            abi::SIG_FN_MOE_A8W8,
            vec![
                buffer("moe_q"),
                buffer(&format!("layer.{layer}.xgu")),
                buffer("tile_expert"),
                buffer("tile_row0"),
                buffer("tile_rows"),
                buffer("row_tok"),
                buffer("meta"),
                buffer("moe_s"),
                buffer("gu"),
                scalar((2 * INTERMEDIATE) as i64),
                scalar(H as i64),
            ],
        )?;
        let gate_up_shared = descriptor(
            saber_kernels::fn50::PLUGIN,
            saber_kernels::fn50::bf16_symbol(true),
            occamy::grouped_grid((2 * INTERMEDIATE) as u32, tile_cap as u32),
            128,
            abi::SIG_FN_MOE_BGEMM,
            vec![
                buffer("moe_in"),
                buffer(&format!("layer.{layer}.xgu")),
                buffer("tile_expert"),
                buffer("tile_row0"),
                buffer("tile_rows"),
                buffer("row_tok"),
                buffer("meta"),
                null_buffer("part"),
                null_buffer("cnt"),
                buffer("gu"),
                scalar((2 * INTERMEDIATE) as i64),
                scalar(H as i64),
                scalar(row_cap as i64),
            ],
        )?;
        let silu = descriptor(
            "flashnext_glue",
            occamy::SILU_SYMBOL,
            [row_cap as u32, 1, 1],
            256,
            abi::SIG_FN_MOE_SILU_BF16,
            vec![
                buffer("gu"),
                buffer("act"),
                scalar(INTERMEDIATE as i64),
                buffer("row_tok"),
                buffer("meta"),
            ],
        )?;
        for launch in [&gate_up_fp8, &gate_up_shared, &silu] {
            self.launch(stream, ctx, launcher, launch)?;
        }
        self.quant_activations(
            stream,
            ctx,
            launcher,
            "act",
            "act_q",
            "act_s",
            row_cap,
            INTERMEDIATE,
        )?;
        let down_fp8 = descriptor(
            saber_kernels::fn50::PLUGIN,
            saber_kernels::fn50::fp8_symbol(false),
            [(H / 128) as u32, tile_cap as u32, 1],
            128,
            abi::SIG_FN_MOE_A8W8,
            vec![
                buffer("act_q"),
                buffer(&format!("layer.{layer}.xdn")),
                buffer("tile_expert"),
                buffer("tile_row0"),
                buffer("tile_rows"),
                buffer("row_tok"),
                buffer("meta"),
                buffer("act_s"),
                buffer("y"),
                scalar(H as i64),
                scalar(INTERMEDIATE as i64),
            ],
        )?;
        let down_shared = descriptor(
            saber_kernels::fn50::PLUGIN,
            saber_kernels::fn50::bf16_symbol(false),
            occamy::grouped_grid(H as u32, tile_cap as u32),
            128,
            abi::SIG_FN_MOE_BGEMM,
            vec![
                buffer("act"),
                buffer(&format!("layer.{layer}.xdn")),
                buffer("tile_expert"),
                buffer("tile_row0"),
                buffer("tile_rows"),
                buffer("row_tok"),
                buffer("meta"),
                null_buffer("part"),
                null_buffer("cnt"),
                buffer("y"),
                scalar(H as i64),
                scalar(INTERMEDIATE as i64),
                scalar(row_cap as i64),
            ],
        )?;
        let combine = descriptor(
            "occamy_moe",
            occamy::COMBINE_SYMBOL,
            [rows as u32, 1, 1],
            256,
            abi::SIG_MOE_COMBINE,
            vec![
                buffer("y"),
                buffer("route_weights"),
                buffer("route_pos"),
                buffer("shared_gate"),
                buffer("meta"),
                buffer("moe"),
                scalar(H as i64),
            ],
        )?;
        for launch in [&down_fp8, &down_shared, &combine] {
            self.launch(stream, ctx, launcher, launch)?;
        }
        Ok(())
    }

    fn head(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        rows: usize,
    ) -> anyhow::Result<()> {
        self.head_row(hip, stream, ctx, launcher, rows - 1)
    }

    pub(super) fn head_row(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        row: usize,
    ) -> anyhow::Result<()> {
        self.dense(
            hip,
            stream,
            ctx,
            launcher,
            &format!("normed@{}", row * H * 2),
            "lm_head.weight",
            "head_logits",
            1,
            self.config.vocab_size as usize / 2,
            H,
            self.config.vocab_size as usize / 2,
        )?;
        let launch = descriptor(
            "head",
            "saber_argmax_bf16",
            [1, 1, 1],
            256,
            abi::SIG_ARGMAX,
            vec![
                buffer("head_logits"),
                buffer("local_pick"),
                scalar((self.config.vocab_size / 2) as i64),
                wide((self.config.vocab_size / 2) as i64),
            ],
        )?;
        self.launch(stream, ctx, launcher, &launch)
    }

    #[allow(clippy::too_many_arguments)]
    fn mtp_forward(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        collective: &Collective,
        slot: usize,
        tokens: &[u32],
        position: usize,
        decode: bool,
        hidden: &str,
        with_head: bool,
    ) -> anyhow::Result<()> {
        let rows = tokens.len();
        ensure!(self.mtp && rows > 0 && rows <= self.max_rows);
        self.update_tables(hip, stream, rows, position, decode)?;
        self.embed(hip, stream, ctx, launcher, collective, tokens)?;
        self.add_norm(
            stream,
            ctx,
            launcher,
            "zeros",
            "res0",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp_ne",
            "mid",
            rows,
        )?;
        self.add_norm(
            stream,
            ctx,
            launcher,
            "zeros",
            hidden,
            "mtp.pre_fc_norm_hidden.weight",
            "mtp_nh",
            "mid",
            rows,
        )?;
        self.dense(
            hip,
            stream,
            ctx,
            launcher,
            if self.rank == 0 { "mtp_ne" } else { "mtp_nh" },
            "mtp.fc.weight",
            "mtp_fc",
            rows,
            H,
            H,
            H,
        )?;
        collective.reduce(
            stream,
            ctx,
            launcher,
            &self.buffers,
            "mtp_fc",
            "mtp_res",
            rows * H,
        )?;
        self.add_norm(
            stream,
            ctx,
            launcher,
            "zeros",
            "mtp_res",
            "model.language_model.layers.40.input_layernorm.weight",
            "normed",
            "mid",
            rows,
        )?;
        self.attention(hip, stream, ctx, launcher, slot, 40, rows, position, decode)?;
        collective.reduce(
            stream,
            ctx,
            launcher,
            &self.buffers,
            "mixer",
            "reduced",
            rows * H,
        )?;
        self.add_norm(
            stream,
            ctx,
            launcher,
            "reduced",
            "mtp_res",
            "model.language_model.layers.40.post_attention_layernorm.weight",
            "moe_in",
            "mtp_mid",
            rows,
        )?;
        self.moe(hip, stream, ctx, launcher, 40, rows)?;
        collective.reduce(
            stream,
            ctx,
            launcher,
            &self.buffers,
            "moe",
            "reduced",
            rows * H,
        )?;
        self.add_norm(
            stream,
            ctx,
            launcher,
            "reduced",
            "mtp_mid",
            "mtp.norm.weight",
            "normed",
            &format!("mtp.{slot}.hidden"),
            rows,
        )?;
        if with_head {
            self.head(hip, stream, ctx, launcher, rows)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn extend_drafts(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        collective: &Collective,
        peer: &RankPeer,
        slot: usize,
        position: usize,
        hidden_rows: usize,
        step: u32,
    ) -> anyhow::Result<()> {
        for index in 1..self.spec_depth {
            let token = self.drafts[&slot][index - 1];
            let row = if index == 1 { hidden_rows - 1 } else { 0 };
            let hidden = format!("mtp.{slot}.hidden@{}", row * H * 2);
            self.mtp_forward(
                hip,
                stream,
                ctx,
                launcher,
                collective,
                slot,
                &[token],
                position + index,
                true,
                &hidden,
                true,
            )?;
            let draft = self.pick(hip, stream, peer, step + index as u32)?;
            self.drafts.get_mut(&slot).unwrap().push(draft);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn mtp_prefill_chunk(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        collective: &Collective,
        peer: &RankPeer,
        slot: usize,
        shifted_tokens: &[u32],
        position: usize,
        make_drafts: bool,
        step: u32,
    ) -> anyhow::Result<()> {
        self.mtp_forward(
            hip,
            stream,
            ctx,
            launcher,
            collective,
            slot,
            shifted_tokens,
            position,
            false,
            "target_hidden",
            make_drafts,
        )?;
        if make_drafts {
            let draft = self.pick(hip, stream, peer, step)?;
            self.drafts.insert(slot, vec![draft]);
            self.extend_drafts(
                hip,
                stream,
                ctx,
                launcher,
                collective,
                peer,
                slot,
                position + shifted_tokens.len() - 1,
                shifted_tokens.len(),
                step + 16,
            )?;
        }
        Ok(())
    }

    fn verification_layer_state(
        &self,
        hip: &HipThread,
        stream: &Stream,
        slot: usize,
        layer: u32,
        step: usize,
        commit: bool,
    ) -> anyhow::Result<()> {
        ensure!(step <= self.spec_depth, "invalid verification state step");
        ensure!(self.config.layers[layer as usize] == LayerKind::Gdn);
        for suffix in ["conv", "h0", "h1"] {
            let state = format!("state.{slot}.{layer}.{suffix}");
            let verify = format!("verify.{slot}.{layer}.{step}.{suffix}");
            let (destination, source) = if commit {
                (state, verify)
            } else {
                (verify, state)
            };
            let bytes = match suffix {
                "conv" => self.convolution_state_bytes(),
                "h0" => 2 * self.state_bytes,
                _ => self.state_bytes,
            };
            self.buffers
                .copy(hip, stream, &destination, 0, &source, 0, bytes as u64)?;
        }
        Ok(())
    }

    fn verification_state(
        &self,
        hip: &HipThread,
        stream: &Stream,
        slot: usize,
        step: usize,
    ) -> anyhow::Result<()> {
        for layer in 0..self.config.num_layers() {
            if self.config.layers[layer as usize] == LayerKind::Gdn {
                self.verification_layer_state(hip, stream, slot, layer, step, true)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn speculative_round(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        collective: &Collective,
        peer: &RankPeer,
        slot: usize,
        next: u32,
        position: usize,
        step: u32,
        sample_target: bool,
        decide: &mut TargetDecision<'_>,
    ) -> anyhow::Result<Vec<SpeculativeToken>> {
        let drafts = self
            .drafts
            .get(&slot)
            .filter(|drafts| !drafts.is_empty())
            .context("MTP has no drafts; prefill has not finished")?
            .clone();
        ensure!(
            drafts.len() == self.spec_depth,
            "incomplete MTP draft chain"
        );
        let mut verification = Vec::with_capacity(drafts.len() + 1);
        verification.push(next);
        verification.extend(drafts.iter().copied());
        ensure!(
            position + verification.len() <= self.max_ctx,
            "MTP verification exceeds --max-ctx"
        );
        self.forward_slot(
            hip,
            stream,
            ctx,
            launcher,
            collective,
            slot,
            &verification,
            position,
            true,
            false,
        )?;
        let mut target = Vec::with_capacity(verification.len());
        for row in 0..verification.len() {
            self.head_row(hip, stream, ctx, launcher, row)?;
            if sample_target {
                let logits = self.logits(hip, stream, peer, step + 64 + row as u32)?;
                let local = if self.rank == 0 {
                    decide(row, &drafts, logits)?
                } else {
                    SpeculativeToken {
                        token: 0,
                        logprobs: None,
                    }
                };
                let theirs = peer.exchange().xchg_bytes(
                    &format!("occamy.target.{step}", step = step + 96 + row as u32),
                    &local.token.to_le_bytes(),
                )?;
                ensure!(theirs.len() == 4, "peer target frame");
                let token = if self.rank == 0 {
                    local.token
                } else {
                    u32::from_le_bytes(theirs.try_into().unwrap())
                };
                target.push(SpeculativeToken {
                    token,
                    logprobs: local.logprobs,
                });
            } else {
                target.push(SpeculativeToken {
                    token: self.pick(hip, stream, peer, step + 64 + row as u32)?,
                    logprobs: None,
                });
            }
        }
        let target_ids: Vec<u32> = target.iter().map(|decision| decision.token).collect();
        let accepted = accepted_draft_count(&target_ids, &drafts);
        let committed = target[..=accepted].to_vec();
        if self.trace_layers && self.rank == 0 {
            println!(
                "occamy MTP verify: position={position} input={verification:?} drafts={drafts:?} target={target_ids:?} accepted={accepted}"
            );
        }
        self.verification_state(hip, stream, slot, accepted)?;
        let next = committed.last().unwrap().token;
        let next_position = position + committed.len();
        let hidden = format!("target_hidden@{}", accepted * H * 2);
        self.mtp_forward(
            hip,
            stream,
            ctx,
            launcher,
            collective,
            slot,
            &[next],
            next_position - 1,
            true,
            &hidden,
            true,
        )?;
        let draft = self.pick(hip, stream, peer, step + 128)?;
        self.drafts.insert(slot, vec![draft]);
        self.extend_drafts(
            hip,
            stream,
            ctx,
            launcher,
            collective,
            peer,
            slot,
            next_position - 1,
            1,
            step + 144,
        )?;
        Ok(committed)
    }

    fn snapshot_layout(&self, slot: usize, tokens: usize, mtp_ready: bool) -> Vec<(String, u64)> {
        let mut layout = Vec::new();
        let kv_bytes = (tokens.div_ceil(16) * NKV * 16 * 2 * HD * 2) as u64;
        for layer in 0..self.config.num_layers() {
            match self.config.layers[layer as usize] {
                LayerKind::Gdn => {
                    for (suffix, bytes) in [
                        ("conv", self.convolution_state_bytes() as u64),
                        ("h0", (2 * self.state_bytes) as u64),
                        ("h1", self.state_bytes as u64),
                    ] {
                        layout.push((format!("state.{slot}.{layer}.{suffix}"), bytes));
                    }
                }
                LayerKind::FullAttention => {
                    layout.push((format!("state.{slot}.{layer}.kv"), kv_bytes));
                }
            }
        }
        if mtp_ready {
            layout.push((format!("state.{slot}.40.kv"), kv_bytes));
        }
        layout
    }

    pub(super) fn capture_prefix(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        slot: usize,
        tokens: &[u32],
        mtp_ready: bool,
    ) -> anyhow::Result<()> {
        ensure!(slot < self.max_requests, "invalid request state slot");
        if tokens.len() < 2 || self.prefix_cache.budget == 0 {
            return Ok(());
        }
        let layout = self.snapshot_layout(slot, tokens.len(), mtp_ready);
        let bytes = layout.iter().map(|(_, bytes)| *bytes).sum();
        if let Some(index) = self.prefix_cache.entries.iter().position(|entry| {
            entry.snapshot.tokens == tokens && entry.snapshot.mtp_ready == mtp_ready
        }) {
            self.prefix_cache.remove(index);
        }
        if !self.prefix_cache.make_room(bytes) {
            if self.rank == 0 {
                println!(
                    "occamy prefix cache: skip {} tokens ({:.2} GiB exceeds free budget)",
                    tokens.len(),
                    bytes as f64 / (1u64 << 30) as f64
                );
            }
            return Ok(());
        }
        let mut saved = Vec::with_capacity(layout.len());
        for (name, bytes) in &layout {
            saved.push((name.clone(), hip.malloc((*bytes).max(4))?));
        }
        let mut jobs = Vec::with_capacity(layout.len());
        for ((name, destination), (_, bytes)) in saved.iter().zip(&layout) {
            jobs.push(Job::MemcpyD2d {
                dst: destination,
                dst_offset: 0,
                src: self.buffers.get(name)?,
                src_offset: 0,
                bytes: *bytes,
                stream,
            });
        }
        hip.run_batch(&jobs)?;
        drop(jobs);
        self.prefix_cache.insert(StateSnapshot {
            tokens: tokens.to_vec(),
            slot,
            mtp_ready,
            bytes,
            buffers: saved,
        });
        Ok(())
    }

    pub(super) fn restore_prefix(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        slot: usize,
        prompt: &[u32],
        use_mtp: bool,
    ) -> anyhow::Result<usize> {
        ensure!(slot < self.max_requests, "invalid request state slot");
        let best = self
            .prefix_cache
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                cache_prefix_matches(
                    &entry.snapshot.tokens,
                    prompt,
                    entry.snapshot.mtp_ready,
                    use_mtp,
                )
            })
            .max_by_key(|(_, entry)| entry.snapshot.tokens.len())
            .map(|(index, _)| index);
        let Some(index) = best else {
            if self.prefix_cache.budget > 0 && prompt.len() > 1 {
                self.cache_misses += 1;
                if self.rank == 0 {
                    println!(
                        "occamy prefix cache: miss ({} hits, {} misses)",
                        self.cache_hits, self.cache_misses
                    );
                }
            }
            return Ok(0);
        };
        let entry = &self.prefix_cache.entries[index];
        let prefix = entry.snapshot.tokens.len();
        let source_prefix = format!("state.{}.", entry.snapshot.slot);
        let destination_prefix = format!("state.{slot}.");
        let mut jobs = Vec::with_capacity(entry.snapshot.buffers.len());
        for (name, source) in &entry.snapshot.buffers {
            let destination = name.replacen(&source_prefix, &destination_prefix, 1);
            jobs.push(Job::MemcpyD2d {
                dst: self.buffers.get(&destination)?,
                dst_offset: 0,
                src: source,
                src_offset: 0,
                bytes: source.len(),
                stream,
            });
        }
        hip.run_batch(&jobs)?;
        drop(jobs);
        self.drafts.remove(&slot);
        self.prefix_cache.clock += 1;
        self.prefix_cache.entries[index].used = self.prefix_cache.clock;
        self.cache_hits += 1;
        if self.rank == 0 {
            println!(
                "occamy prefix cache: hit {prefix}/{} tokens ({} hits, {} misses, {:.2}/{:.2} GiB)",
                prompt.len(),
                self.cache_hits,
                self.cache_misses,
                self.prefix_cache.bytes as f64 / (1u64 << 30) as f64,
                self.prefix_cache.budget as f64 / (1u64 << 30) as f64
            );
        }
        Ok(prefix)
    }

    pub(super) fn reset_slot(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        slot: usize,
    ) -> anyhow::Result<()> {
        ensure!(slot < self.max_requests, "invalid request state slot");
        for layer in 0..self.config.num_layers() {
            match self.config.layers[layer as usize] {
                LayerKind::Gdn => {
                    for suffix in ["conv", "h0", "h1"] {
                        self.buffers.fill(
                            hip,
                            stream,
                            &format!("state.{slot}.{layer}.{suffix}"),
                            0,
                        )?;
                    }
                }
                LayerKind::FullAttention => {
                    self.buffers
                        .fill(hip, stream, &format!("state.{slot}.{layer}.kv"), 0)?;
                }
            }
        }
        if self.mtp {
            self.buffers
                .fill(hip, stream, &format!("state.{slot}.40.kv"), 0)?;
            self.buffers
                .fill(hip, stream, &format!("mtp.{slot}.hidden"), 0)?;
            self.drafts.remove(&slot);
        }
        self.buffers.fill(hip, stream, "bar", 0)?;
        hip.sync_stream(stream)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        collective: &Collective,
        tokens: &[u32],
        position: usize,
        decode: bool,
        with_head: bool,
    ) -> anyhow::Result<()> {
        self.forward_slot(
            hip, stream, ctx, launcher, collective, 0, tokens, position, decode, with_head,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_slot(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        collective: &Collective,
        slot: usize,
        tokens: &[u32],
        position: usize,
        decode: bool,
        with_head: bool,
    ) -> anyhow::Result<()> {
        let rows = tokens.len();
        ensure!(slot < self.max_requests, "invalid request state slot");
        ensure!(
            rows <= self.max_rows,
            "{rows} rows exceeds the resident scratch"
        );
        self.update_tables(hip, stream, rows, position, decode)?;
        self.forward_body(
            hip, stream, ctx, launcher, collective, slot, tokens, position, decode, with_head,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_body(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        ctx: &LaunchCtx<'_>,
        launcher: &mut Launcher,
        collective: &Collective,
        slot: usize,
        tokens: &[u32],
        position: usize,
        decode: bool,
        with_head: bool,
    ) -> anyhow::Result<()> {
        let rows = tokens.len();
        self.embed(hip, stream, ctx, launcher, collective, tokens)?;
        self.add_norm(
            stream,
            ctx,
            launcher,
            "zeros",
            "res0",
            "model.language_model.layers.0.input_layernorm.weight",
            "normed",
            "mid",
            rows,
        )?;
        if with_head && !decode && self.quality.is_some() {
            self.compare_quality(hip, stream, "embed", "res0", rows)?;
            self.compare_quality(hip, stream, "layer00.input_norm", "normed", rows)?;
        }
        let mut current = "res0";
        let mut next = "res1";
        for layer in 0..self.config.num_layers() {
            if self.trace_layers {
                hip.sync_stream(stream)?;
                println!("occamy rank{}: layer {layer} start", self.rank);
            }
            match self.config.layers[layer as usize] {
                LayerKind::Gdn => {
                    self.gdn(hip, stream, ctx, launcher, slot, layer, rows, decode)?
                }
                LayerKind::FullAttention => self.attention(
                    hip, stream, ctx, launcher, slot, layer, rows, position, decode,
                )?,
            }
            collective.reduce(
                stream,
                ctx,
                launcher,
                &self.buffers,
                "mixer",
                "reduced",
                rows * H,
            )?;
            if layer == 0 && with_head && !decode && self.quality.is_some() {
                self.compare_quality(hip, stream, "layer00.mixer", "reduced", rows)?;
            }
            self.add_norm(
                stream,
                ctx,
                launcher,
                "reduced",
                current,
                &format!("model.language_model.layers.{layer}.post_attention_layernorm.weight"),
                "moe_in",
                "mid",
                rows,
            )?;
            if layer == 0 && with_head && !decode && self.quality.is_some() {
                self.compare_quality(hip, stream, "layer00.mixer_residual", "mid", rows)?;
                self.compare_quality(hip, stream, "layer00.moe_input", "moe_in", rows)?;
            }
            self.moe(hip, stream, ctx, launcher, layer, rows)?;
            collective.reduce(
                stream,
                ctx,
                launcher,
                &self.buffers,
                "moe",
                "reduced",
                rows * H,
            )?;
            if layer == 0 && with_head && !decode && self.quality.is_some() {
                self.compare_quality(hip, stream, "layer00.moe", "reduced", rows)?;
            }
            let next_weight = if layer + 1 < self.config.num_layers() {
                format!(
                    "model.language_model.layers.{}.input_layernorm.weight",
                    layer + 1
                )
            } else {
                "model.language_model.norm.weight".into()
            };
            self.add_norm(
                stream,
                ctx,
                launcher,
                "reduced",
                "mid",
                &next_weight,
                "normed",
                next,
                rows,
            )?;
            std::mem::swap(&mut current, &mut next);
            if with_head && !decode && self.quality.is_some() {
                self.compare_quality(hip, stream, &format!("layer{layer:02}"), current, rows)?;
            }
        }
        if self.mtp {
            self.buffers.copy(
                hip,
                stream,
                "target_hidden",
                0,
                "normed",
                0,
                (rows * H * 2) as u64,
            )?;
        }
        if with_head {
            self.head(hip, stream, ctx, launcher, rows)?;
        }
        Ok(())
    }

    fn compare_quality(
        &self,
        hip: &HipThread,
        stream: &Stream,
        name: &str,
        residual: &str,
        rows: usize,
    ) -> anyhow::Result<()> {
        let expected = std::fs::read(
            self.quality
                .as_ref()
                .unwrap()
                .root
                .join(format!("{name}.bf16")),
        )?;
        ensure!(expected.len() == H * 2, "quality {name} size");
        let got = self.buffers.download(hip, stream, residual, rows * H * 2)?;
        let got = &got[(rows - 1) * H * 2..];
        let mut maximum = 0.0f32;
        let mut squared = 0.0f64;
        for (got, expected) in got.chunks_exact(2).zip(expected.chunks_exact(2)) {
            let got = u16::from_le_bytes([got[0], got[1]]);
            let expected = u16::from_le_bytes([expected[0], expected[1]]);
            let difference = (f32::from_bits(u32::from(got) << 16)
                - f32::from_bits(u32::from(expected) << 16))
            .abs();
            maximum = maximum.max(difference);
            squared += f64::from(difference) * f64::from(difference);
        }
        if self.rank == 0 {
            println!(
                "occamy quality {name}: max_abs={maximum:.6}, rms={:.6}",
                (squared / H as f64).sqrt()
            );
        }
        Ok(())
    }

    pub(super) fn pick(
        &self,
        hip: &HipThread,
        stream: &Stream,
        peer: &RankPeer,
        step: u32,
    ) -> anyhow::Result<u32> {
        hip.sync_stream(stream)?;
        let local = self.buffers.download(hip, stream, "local_pick", 4)?;
        let local = u32::from_le_bytes(local.try_into().unwrap());
        let logits =
            self.buffers
                .download(hip, stream, "head_logits", self.config.vocab_size as usize)?;
        let offset = local as usize * 2;
        let logit = u16::from_le_bytes([logits[offset], logits[offset + 1]]);
        let id = local + self.rank * (self.config.vocab_size / 2);
        let mine = [
            id.to_le_bytes().as_slice(),
            logit.to_le_bytes().as_slice(),
            &[0, 0],
        ]
        .concat();
        let theirs = peer
            .exchange()
            .xchg_bytes(&format!("occamy.pick.{step}"), &mine)?;
        ensure!(theirs.len() == 8, "peer pick frame");
        let other_id = u32::from_le_bytes(theirs[0..4].try_into().unwrap());
        let other_logit = u16::from_le_bytes([theirs[4], theirs[5]]);
        let mine_better = logit > other_logit || (logit == other_logit && id < other_id);
        Ok(if mine_better { id } else { other_id })
    }

    pub(super) fn logits(
        &self,
        hip: &HipThread,
        stream: &Stream,
        peer: &RankPeer,
        step: u32,
    ) -> anyhow::Result<Vec<f32>> {
        hip.sync_stream(stream)?;
        let mine =
            self.buffers
                .download(hip, stream, "head_logits", self.config.vocab_size as usize)?;
        let theirs = peer
            .exchange()
            .xchg_bytes(&format!("occamy.logits.{step}"), &mine)?;
        ensure!(theirs.len() == mine.len(), "peer logits frame");
        let (first, second) = if self.rank == 0 {
            (mine, theirs)
        } else {
            (theirs, mine)
        };
        Ok(first
            .into_iter()
            .chain(second)
            .collect::<Vec<_>>()
            .chunks_exact(2)
            .map(|value| f32::from_bits(u32::from(u16::from_le_bytes([value[0], value[1]])) << 16))
            .collect())
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

fn config(checkpoint: &Path) -> anyhow::Result<Qwen35MoeConfig> {
    let path = checkpoint.join("config.json");
    let text = std::fs::read_to_string(&path).with_context(|| path.display().to_string())?;
    Qwen35MoeConfig::from_config_json(&text).map_err(|error| anyhow::anyhow!("{error}"))
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

pub fn run(hip: &HipThread, args: &ExecArgs) -> anyhow::Result<Vec<u32>> {
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
        child.0 = Some(command.spawn().context("spawning Occamy rank 1")?);
        Arc::new(TcpExchange::accept(&listener, Duration::from_secs(300))?)
    } else {
        Arc::new(TcpExchange::connect(
            args.peer_port,
            Duration::from_secs(300),
        )?)
    };
    let peer = RankPeer::new(exchange.clone());
    hip.set_device(args.device)?;
    let counters = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, &args.prebuilt, &counters)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;
    let launch_counters = LaunchCounters::new();
    let mut launcher = Launcher::new("occamy-exec", &launch_counters);
    let cfg = config(&args.checkpoint)?;
    ensure!(
        cfg.hidden_size as usize == H && cfg.num_layers() == 40,
        "unexpected Occamy geometry"
    );
    let mut model = Model::load(hip, &stream, args, cfg)?;
    let collective = Collective::new(hip, &stream, args.rank, &peer)?;
    peer.agree(
        "occamy.tokens",
        args.tokens.iter().fold(0u64, |hash, id| {
            hash.wrapping_mul(1_000_003).wrapping_add(u64::from(*id))
        }),
    )?;
    if args.serve.is_some() {
        // Startup waits stay bounded; a serving follower may legitimately idle for hours.
        exchange.set_read_timeout(None)?;
        super::serve::run(
            hip,
            &mut model,
            &collective,
            &ctx,
            &mut launcher,
            &stream,
            &peer,
            args,
        )?;
        return Ok(Vec::new());
    }
    ensure!(args.decode > 0, "--decode must be positive");
    let started = Instant::now();
    let mut output = Vec::with_capacity(args.decode as usize);
    let mut position = 0usize;
    let mut next = 0u32;
    for start in (0..args.tokens.len()).step_by(args.max_rows) {
        let end = (start + args.max_rows).min(args.tokens.len());
        let last = end == args.tokens.len();
        if last {
            next = model
                .forward(
                    hip,
                    &stream,
                    &ctx,
                    &mut launcher,
                    &collective,
                    &args.tokens[start..end],
                    start,
                    false,
                    true,
                )
                .and_then(|_| model.pick(hip, &stream, &peer, 0))?;
        } else {
            model.forward(
                hip,
                &stream,
                &ctx,
                &mut launcher,
                &collective,
                &args.tokens[start..end],
                start,
                false,
                false,
            )?;
        }
        if model.mtp {
            let mut shifted = args.tokens[start + 1..end].to_vec();
            shifted.push(if last { next } else { args.tokens[end] });
            model.mtp_prefill_chunk(
                hip,
                &stream,
                &ctx,
                &mut launcher,
                &collective,
                &peer,
                0,
                &shifted,
                start,
                last,
                (start * 16) as u32,
            )?;
        }
    }
    position += args.tokens.len();
    output.push(next);
    let prefill_elapsed = started.elapsed();
    let mut step = 1u32;
    let mut accepted_rounds = 0usize;
    while output.len() < args.decode as usize {
        if model.mtp {
            let mut decide = |_: usize, _: &[u32], _: Vec<f32>| {
                unreachable!("the token harness uses the plain greedy verifier")
            };
            let committed = model.speculative_round(
                hip,
                &stream,
                &ctx,
                &mut launcher,
                &collective,
                &peer,
                0,
                next,
                position,
                step * 256,
                false,
                &mut decide,
            )?;
            accepted_rounds += usize::from(committed.len() > 1);
            position += committed.len();
            next = committed.last().unwrap().token;
            output.extend(committed.into_iter().map(|decision| decision.token));
            output.truncate(args.decode as usize);
        } else {
            next = model
                .forward(
                    hip,
                    &stream,
                    &ctx,
                    &mut launcher,
                    &collective,
                    &[next],
                    position,
                    true,
                    true,
                )
                .and_then(|_| model.pick(hip, &stream, &peer, step))?;
            position += 1;
            output.push(next);
        }
        step += 1;
    }
    let decode_elapsed = started.elapsed() - prefill_elapsed;
    if model.mtp && args.rank == 0 {
        println!(
            "occamy MTP: {} rounds, {} fully accepted, {:.3}s speculative decode",
            step - 1,
            accepted_rounds,
            decode_elapsed.as_secs_f64()
        );
    }
    if let Some(quality) = &model.quality {
        if model.fp8_experts {
            if args.rank == 0 {
                let prefix = output
                    .iter()
                    .zip(&quality.generated)
                    .take_while(|(got, expected)| got == expected)
                    .count();
                println!(
                    "occamy FP8 quality: got {output:?}, BF16 reference {:?}; exact prefix {prefix}/{}",
                    quality.generated,
                    output.len()
                );
            }
        } else {
            ensure!(
                output == quality.generated,
                "generated token mismatch: got {output:?}, reference {:?}",
                quality.generated
            );
            if args.rank == 0 {
                println!("occamy quality: generated tokens match the independent reference");
            }
        }
    }
    println!(
        "occamy rank{}: prompt {} + {} generated tokens in {:.3}s; ids {:?}",
        args.rank,
        args.tokens.len(),
        output.len(),
        started.elapsed().as_secs_f64(),
        output
    );
    println!(
        "occamy rank{}: prefill {:.3}s ({:.2} prompt tok/s); decode {} steps in {:.3}s ({:.2} tok/s)",
        args.rank,
        prefill_elapsed.as_secs_f64(),
        args.tokens.len() as f64 / prefill_elapsed.as_secs_f64(),
        output.len() - 1,
        decode_elapsed.as_secs_f64(),
        if output.len() > 1 {
            (output.len() - 1) as f64 / decode_elapsed.as_secs_f64()
        } else {
            0.0
        }
    );
    Ok(if args.rank == 0 { output } else { Vec::new() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_layout_covers_padded_routes_shared_rows_and_route_blocks() {
        for rows in [1, 3, 32, 33, 64] {
            let layout = moe_layout(rows);
            assert_eq!(layout.routes, rows * TOPK);
            let padded = layout.routes + 15 * layout.routes.min(LOCAL_EXPERTS) + 16;
            assert!(layout.row_cap >= padded);
            assert!(layout.tile_cap >= layout.routes + 1);
            assert_eq!(layout.route_blocks, rows.div_ceil(32));
        }
        assert_eq!(moe_layout(3).row_cap, 400);
        assert_eq!(moe_layout(33).route_blocks, 2);
    }

    #[test]
    fn mtp_accepts_only_the_matching_target_draft_prefix() {
        assert_eq!(accepted_draft_count(&[15, 197, 1], &[15, 197]), 2);
        assert_eq!(accepted_draft_count(&[15, 197, 1], &[15, 15]), 1);
        assert_eq!(accepted_draft_count(&[197, 15, 1], &[15, 15]), 0);
    }

    #[test]
    fn prefix_cache_requires_an_exact_prefix_and_a_fresh_logit_token() {
        assert!(cache_prefix_matches(
            &[1, 2, 3],
            &[1, 2, 3, 4],
            false,
            false
        ));
        assert!(cache_prefix_matches(&[1, 2, 3], &[1, 2, 3, 4], true, true));
        assert!(!cache_prefix_matches(&[1, 2, 3], &[1, 2, 3], false, false));
        assert!(!cache_prefix_matches(
            &[1, 2, 4],
            &[1, 2, 3, 4],
            false,
            false
        ));
        assert!(!cache_prefix_matches(
            &[1, 2, 3],
            &[1, 2, 3, 4],
            false,
            true
        ));
    }

    #[test]
    fn rank_forwarding_replaces_explicit_device_arguments() {
        let args = [
            "occamy-exec",
            "--device",
            "0",
            "--host",
            "127.0.0.1",
            "--device=3",
            "--serve",
            "8090",
        ]
        .map(str::to_owned);
        assert_eq!(
            forwarded_rank_args(args),
            ["occamy-exec", "--host", "127.0.0.1", "--serve", "8090"]
        );
    }
}
