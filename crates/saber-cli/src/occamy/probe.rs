use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use anyhow::ensure;
use saber_exec::{ArgBindings, LaunchCounters, LaunchCtx, Launcher, LoadCounters, PluginSet};
use saber_hip::{DeviceAddr, DevicePtr, HipThread, Stream, StreamFlags};
use saber_kernels::abi::{self, SigArg};
use saber_kernels::occamy;
use saber_kernels::rules::{ArgValue, LaunchDescriptor, Shape as BindingShape, Slot};
use saber_kernels::saber::{resolve, AttnShape, KvDtype, Op, SaberRules, Shape};

use crate::npy;

pub struct ProbeArgs {
    pub data: std::path::PathBuf,
    pub prebuilt: std::path::PathBuf,
    pub device: i32,
    pub mode: String,
    pub iters: u32,
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

struct Buffers(BTreeMap<String, DevicePtr>);

impl Buffers {
    fn new() -> Self {
        Self(BTreeMap::new())
    }

    fn addr(&self, name: &str) -> anyhow::Result<DeviceAddr> {
        self.0
            .get(name)
            .map(DevicePtr::addr)
            .ok_or_else(|| anyhow::anyhow!("missing device buffer {name}"))
    }

    fn upload(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        name: &str,
        path: &Path,
    ) -> anyhow::Result<()> {
        let bytes = npy::read_u8(path)?;
        let pointer = hip.malloc(bytes.len().max(1) as u64)?;
        hip.memcpy_h2d_blocking(&pointer, &bytes, stream)?;
        self.0.insert(name.into(), pointer);
        Ok(())
    }

    fn bytes(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        name: &str,
        bytes: &[u8],
    ) -> anyhow::Result<()> {
        let pointer = hip.malloc(bytes.len().max(1) as u64)?;
        hip.memcpy_h2d_blocking(&pointer, bytes, stream)?;
        self.0.insert(name.into(), pointer);
        Ok(())
    }

    fn zero(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        name: &str,
        bytes: usize,
    ) -> anyhow::Result<()> {
        let pointer = hip.malloc(bytes.max(4) as u64)?;
        hip.memset(&pointer, 0, stream)?;
        self.0.insert(name.into(), pointer);
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
        hip.memcpy_d2h_blocking(&mut output, &self.0[name], stream)?;
        Ok(output)
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

fn bf16_diff(got: &[u8], expected: &[u8], tolerance: f32) -> (usize, f32, f32) {
    assert_eq!(got.len(), expected.len());
    let mut differing = 0;
    let mut maximum = 0.0f32;
    let mut squared = 0.0f64;
    for (a, b) in got.chunks_exact(2).zip(expected.chunks_exact(2)) {
        let a = f32::from_bits(u32::from(u16::from_le_bytes([a[0], a[1]])) << 16);
        let b = f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16);
        let difference = (a - b).abs();
        if difference > tolerance {
            differing += 1;
        }
        maximum = maximum.max(difference);
        squared += f64::from(difference) * f64::from(difference);
    }
    let count = got.len() / 2;
    (differing, maximum, (squared / count as f64).sqrt() as f32)
}

fn bf16_max_ulp(got: &[u8], expected: &[u8]) -> u32 {
    got.chunks_exact(2)
        .zip(expected.chunks_exact(2))
        .map(|(a, b)| {
            let a = u16::from_le_bytes([a[0], a[1]]);
            let b = u16::from_le_bytes([b[0], b[1]]);
            let ordered = |bits: u16| {
                if bits & 0x8000 == 0 {
                    i32::from(bits)
                } else {
                    -i32::from(bits & 0x7fff)
                }
            };
            ordered(a).abs_diff(ordered(b))
        })
        .max()
        .unwrap_or(0)
}

fn f32_diff(got: &[u8], expected: &[u8], tolerance: f32) -> usize {
    got.chunks_exact(4)
        .zip(expected.chunks_exact(4))
        .filter(|(a, b)| {
            let a = f32::from_le_bytes([a[0], a[1], a[2], a[3]]);
            let b = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            (a - b).abs() > tolerance
        })
        .count()
}

fn attention(
    hip: &HipThread,
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    args: &ProbeArgs,
) -> anyhow::Result<bool> {
    let mut buffers = Buffers::new();
    for name in ["q", "kv", "block_table", "seqused_k", "expected"] {
        buffers.upload(
            hip,
            stream,
            name,
            &args.data.join(format!("attn.{name}.npy")),
        )?;
    }
    buffers.zero(hip, stream, "out", 5 * 8 * 256 * 2)?;
    let shape = AttnShape {
        num_seqs: 1,
        q_len: 5,
        live_rows: 5,
        q_heads: 8,
        kv_heads: 1,
        head_dim: 256,
        block_size: 16,
        max_blocks: 2,
        max_ctx: 32,
        splits: 0,
        kv_dtype: KvDtype::Bf16,
        kv_block_stride: 16 * 2 * 256,
        kv_head_stride: 16 * 2 * 256,
        scale_bits: (1.0f32 / 16.0).to_bits(),
        has_descales: false,
        ragged_rows: 0,
    };
    let rules = SaberRules::occamy()?;
    let descriptor = resolve(Op::AttnPrefill { fp8_mode: 0 }, Shape::Attn(shape), &rules)?;
    dispatch(stream, ctx, launcher, &descriptor, &buffers)?;
    hip.sync_stream(stream)?;
    let prefill_got = buffers.download(hip, stream, "out", 5 * 8 * 256 * 2)?;
    let expected = npy::read_u8(&args.data.join("attn.expected.npy"))?;
    let (different, maximum, rms) = bf16_diff(&prefill_got, &expected, 0.03);
    println!("occamy attention: differing={different}, max_abs={maximum:.6}, rms={rms:.6}");
    let start = Instant::now();
    for _ in 0..args.iters {
        dispatch(stream, ctx, launcher, &descriptor, &buffers)?;
    }
    hip.sync_stream(stream)?;
    println!(
        "occamy attention prefill: {:.2} us/launch (host wall, {} iterations)",
        start.elapsed().as_secs_f64() * 1e6 / f64::from(args.iters),
        args.iters
    );

    buffers.upload(hip, stream, "dq", &args.data.join("attn.decode.q.npy"))?;
    buffers.zero(hip, stream, "dout", 8 * 256 * 2)?;
    buffers.zero(hip, stream, "attn_decode_scratch", 8 * 520)?;
    let decode_shape = AttnShape {
        q_len: 1,
        live_rows: 1,
        splits: 1,
        ..shape
    };
    let decode = resolve(Op::AttnDecode, Shape::Attn(decode_shape), &rules)?;
    let combine = saber_kernels::saber::attn_decode_combine(decode_shape, &rules)?;
    dispatch(stream, ctx, launcher, &decode, &buffers)?;
    dispatch(stream, ctx, launcher, &combine, &buffers)?;
    hip.sync_stream(stream)?;
    let got = buffers.download(hip, stream, "dout", 8 * 256 * 2)?;
    let expected = npy::read_u8(&args.data.join("attn.decode.expected.npy"))?;
    let (decode_different, decode_maximum, decode_rms) = bf16_diff(&got, &expected, 0.05);
    let decode_ulp = bf16_max_ulp(&got, &expected);
    let (cross_different, cross_maximum, cross_rms) =
        bf16_diff(&got, &prefill_got[4 * 8 * 256 * 2..], 0.05);
    let cross_ulp = bf16_max_ulp(&got, &prefill_got[4 * 8 * 256 * 2..]);
    println!(
        "occamy attention decode: differing={decode_different}, max_abs={decode_maximum:.6}, rms={decode_rms:.6}, max_ulp={decode_ulp}; vs_prefill max_abs={cross_maximum:.6}, rms={cross_rms:.6}, max_ulp={cross_ulp}"
    );
    let start = Instant::now();
    for _ in 0..args.iters {
        dispatch(stream, ctx, launcher, &decode, &buffers)?;
        dispatch(stream, ctx, launcher, &combine, &buffers)?;
    }
    hip.sync_stream(stream)?;
    println!(
        "occamy attention decode+combine: {:.2} us (host wall, {} iterations)",
        start.elapsed().as_secs_f64() * 1e6 / f64::from(args.iters),
        args.iters
    );
    Ok(different == 0 && decode_different == 0 && cross_different == 0)
}

#[allow(clippy::too_many_lines)]
fn moe_rank(
    hip: &HipThread,
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    args: &ProbeArgs,
    rank: u32,
) -> anyhow::Result<bool> {
    const T: u32 = 3;
    const TOPK: u32 = 8;
    const LOCAL: u32 = 128;
    const ROW_CAP: u32 = 128;
    const TILE_CAP: u32 = 16;
    let mut buffers = Buffers::new();
    for name in ["x", "logits", "shared_gate"] {
        buffers.upload(
            hip,
            stream,
            name,
            &args.data.join(format!("moe.{name}.npy")),
        )?;
    }
    for expert in 0..4u32 {
        let global = rank * LOCAL + expert;
        for projection in ["gate_up", "down"] {
            buffers.upload(
                hip,
                stream,
                &format!("e{expert}.{projection}"),
                &args
                    .data
                    .join(format!("moe.expert.{global}.{projection}.npy")),
            )?;
        }
    }
    if rank == 0 {
        for projection in ["gate_up", "down"] {
            buffers.upload(
                hip,
                stream,
                &format!("shared.{projection}"),
                &args.data.join(format!("moe.shared.{projection}.npy")),
            )?;
        }
    }
    for (name, bytes) in [
        ("ids", T * TOPK * 4),
        ("weights", T * TOPK * 4),
        ("pos", T * TOPK * 4),
        ("row_tok", ROW_CAP * 4),
        ("row_route", ROW_CAP * 4),
        ("tile_expert", TILE_CAP * 4),
        ("tile_row0", TILE_CAP * 4),
        ("tile_rows", TILE_CAP * 4),
        ("expert_cnt", LOCAL * 4),
        ("meta", 6 * 4),
        ("blk_cnt", LOCAL * 4),
        ("rank", T * TOPK * 4),
        ("arrive", 4),
        ("gu", ROW_CAP * 1024 * 2),
        ("act", ROW_CAP * 512 * 2),
        ("y", ROW_CAP * 2048 * 2),
        ("out", T * 2048 * 2),
    ] {
        buffers.zero(hip, stream, name, bytes as usize)?;
    }
    for (table, down) in [("xgu", false), ("xdn", true)] {
        let mut addresses = Vec::new();
        for expert in 0..(LOCAL + u32::from(rank == 0)) {
            if expert == LOCAL || expert < 4 {
                let name = if expert == LOCAL {
                    format!("shared.{}", if down { "down" } else { "gate_up" })
                } else {
                    format!("e{expert}.{}", if down { "down" } else { "gate_up" })
                };
                addresses.push(buffers.addr(&name)?.as_u64());
            } else {
                addresses.push(0);
            }
            addresses.push(0);
        }
        let bytes: Vec<u8> = addresses.into_iter().flat_map(u64::to_le_bytes).collect();
        buffers.bytes(hip, stream, table, &bytes)?;
    }
    let route = descriptor(
        "occamy_moe",
        occamy::ROUTE_SYMBOL,
        occamy::route_grid(T),
        256,
        abi::SIG_MOE_ROUTE,
        vec![
            buffer("logits"),
            scalar(T.into()),
            scalar((rank * LOCAL).into()),
            scalar(LOCAL.into()),
            scalar(1),
            scalar(u32::from(rank == 0).into()),
            buffer("ids"),
            buffer("weights"),
            buffer("pos"),
            buffer("row_tok"),
            buffer("row_route"),
            buffer("tile_expert"),
            buffer("tile_row0"),
            buffer("tile_rows"),
            buffer("expert_cnt"),
            buffer("meta"),
            buffer("blk_cnt"),
            buffer("rank"),
            buffer("arrive"),
        ],
    )?;
    let gate_up = descriptor(
        "flashnext_moe",
        occamy::GATE_UP_SYMBOL,
        occamy::grouped_grid(1024, TILE_CAP),
        128,
        abi::SIG_FN_MOE_BGEMM,
        vec![
            buffer("x"),
            buffer("xgu"),
            buffer("tile_expert"),
            buffer("tile_row0"),
            buffer("tile_rows"),
            buffer("row_tok"),
            buffer("meta"),
            null_buffer("part"),
            null_buffer("cnt"),
            buffer("gu"),
            scalar(1024),
            scalar(2048),
            scalar(ROW_CAP.into()),
        ],
    )?;
    let silu = descriptor(
        "flashnext_glue",
        occamy::SILU_SYMBOL,
        [ROW_CAP, 1, 1],
        256,
        abi::SIG_FN_MOE_SILU_BF16,
        vec![
            buffer("gu"),
            buffer("act"),
            scalar(512),
            buffer("row_tok"),
            buffer("meta"),
        ],
    )?;
    let down = descriptor(
        "flashnext_moe",
        occamy::DOWN_SYMBOL,
        occamy::grouped_grid(2048, TILE_CAP),
        128,
        abi::SIG_FN_MOE_BGEMM,
        vec![
            buffer("act"),
            buffer("xdn"),
            buffer("tile_expert"),
            buffer("tile_row0"),
            buffer("tile_rows"),
            buffer("row_tok"),
            buffer("meta"),
            null_buffer("part"),
            null_buffer("cnt"),
            buffer("y"),
            scalar(2048),
            scalar(512),
            scalar(ROW_CAP.into()),
        ],
    )?;
    let combine = descriptor(
        "occamy_moe",
        occamy::COMBINE_SYMBOL,
        [T, 1, 1],
        256,
        abi::SIG_MOE_COMBINE,
        vec![
            buffer("y"),
            buffer("weights"),
            buffer("pos"),
            buffer("shared_gate"),
            buffer("meta"),
            buffer("out"),
            scalar(2048),
        ],
    )?;
    let launches = [&route, &gate_up, &silu, &down, &combine];
    for launch in launches {
        dispatch(stream, ctx, launcher, launch, &buffers)?;
    }
    hip.sync_stream(stream)?;

    let got_ids = buffers.download(hip, stream, "ids", (T * TOPK * 4) as usize)?;
    let want_ids = npy::read_u8(&args.data.join("moe.ids.npy"))?;
    let ids_ok = got_ids == want_ids;
    let got_weights = buffers.download(hip, stream, "weights", (T * TOPK * 4) as usize)?;
    let want_weights = npy::read_u8(&args.data.join("moe.weights.npy"))?;
    let weight_errors = f32_diff(&got_weights, &want_weights, 0.0001);
    let meta: Vec<i32> = buffers
        .download(hip, stream, "meta", 24)?
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let expected_meta = if rank == 0 {
        [5, 80, 64, 12, 4, 64]
    } else {
        [4, 64, -1, 12, 4, 64]
    };
    let got = buffers.download(hip, stream, "out", (T * 2048 * 2) as usize)?;
    let expected = npy::read_u8(&args.data.join(format!("moe.rank{rank}.expected.npy")))?;
    let (different, maximum, rms) = bf16_diff(&got, &expected, 0.03);
    println!(
        "occamy MoE rank{rank}: ids_exact={ids_ok}, weight_errors={weight_errors}, meta={meta:?}, differing={different}, max_abs={maximum:.6}, rms={rms:.6}"
    );
    let start = Instant::now();
    for _ in 0..args.iters {
        for launch in [&route, &gate_up, &silu, &down, &combine] {
            dispatch(stream, ctx, launcher, launch, &buffers)?;
        }
    }
    hip.sync_stream(stream)?;
    println!(
        "occamy MoE rank{rank}: {:.2} us/block (host wall, {} iterations)",
        start.elapsed().as_secs_f64() * 1e6 / f64::from(args.iters),
        args.iters
    );
    Ok(ids_ok && weight_errors == 0 && meta == expected_meta && different == 0)
}

fn dense(
    hip: &HipThread,
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    args: &ProbeArgs,
) -> anyhow::Result<bool> {
    let mut ok = true;
    for (m, n) in [
        (3, 6144),
        (3, 32),
        (3, 2048),
        (3, 4096),
        (3, 256),
        (3, 1),
        (17, 256),
        (33, 256),
        (65, 256),
    ] {
        let prefix = format!("dense.m{m}.n{n}");
        let mut buffers = Buffers::new();
        for name in ["x", "w", "expected"] {
            buffers.upload(
                hip,
                stream,
                name,
                &args.data.join(format!("{prefix}.{name}.npy")),
            )?;
        }
        buffers.zero(hip, stream, "out", m * n * 2)?;
        let launch = occamy::dense_projection(m as u32, n as u32, 2048)
            .map_err(|error| anyhow::anyhow!(error))?;
        if launch.splits > 1 {
            buffers.zero(hip, stream, "part", launch.partial_bytes as usize)?;
            buffers.zero(hip, stream, "cnt", launch.counter_bytes as usize)?;
        }
        let mut values = vec![
            buffer("x"),
            buffer("w"),
            if launch.splits > 1 {
                buffer("part")
            } else {
                null_buffer("part")
            },
            if launch.splits > 1 {
                buffer("cnt")
            } else {
                null_buffer("cnt")
            },
            buffer("out"),
            scalar(m as i64),
            scalar(n as i64),
            scalar(2048),
        ];
        let signature = if launch.symbol.starts_with("fn_bgemm_sk_") {
            abi::SIG_FN_BGEMM_SK
        } else {
            values = vec![
                buffer("x"),
                buffer("w"),
                buffer("out"),
                scalar(m as i64),
                scalar(n as i64),
                scalar(2048),
            ];
            abi::SIG_FN_BGEMM
        };
        let descriptor = descriptor(
            "flashnext_hc",
            launch.symbol,
            launch.grid,
            launch.threads,
            signature,
            values,
        )?;
        dispatch(stream, ctx, launcher, &descriptor, &buffers)?;
        hip.sync_stream(stream)?;
        let got = buffers.download(hip, stream, "out", m * n * 2)?;
        let expected = npy::read_u8(&args.data.join(format!("{prefix}.expected.npy")))?;
        let (different, maximum, rms) = bf16_diff(&got, &expected, 0.03);
        println!(
            "occamy dense M{m} N{n} K2048 {}: differing={different}, max_abs={maximum:.6}, rms={rms:.6}",
            launch.symbol
        );
        ok &= different == 0;
    }
    Ok(ok)
}

fn kv_append(
    hip: &HipThread,
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    args: &ProbeArgs,
) -> anyhow::Result<bool> {
    const ROWS: usize = 3;
    const CACHE_BYTES: usize = 2 * 16 * 2 * 256 * 2;
    let mut buffers = Buffers::new();
    for name in ["key", "value", "slot", "ring_slot", "expected"] {
        buffers.upload(hip, stream, name, &args.data.join(format!("kv.{name}.npy")))?;
    }
    buffers.zero(hip, stream, "kv", CACHE_BYTES)?;
    let wide = |value| ArgValue::I64 { ival: value };
    let descriptor = descriptor(
        "flashnext_glue",
        occamy::KV_APPEND_SYMBOL,
        [ROWS as u32, 1, 1],
        128,
        abi::SIG_FN_KV_APPEND_RING,
        vec![
            buffer("key"),
            wide(256),
            buffer("value"),
            wide(256),
            buffer("kv"),
            buffer("slot"),
            wide(16 * 512),
            wide(512),
            wide(16 * 512),
            null_buffer("ikey"),
            wide(0),
            null_buffer("ring"),
            null_buffer("ring_pos"),
            buffer("ring_slot"),
            null_buffer("pos"),
            scalar(ROWS as i64),
        ],
    )?;
    dispatch(stream, ctx, launcher, &descriptor, &buffers)?;
    hip.sync_stream(stream)?;
    let got = buffers.download(hip, stream, "kv", CACHE_BYTES)?;
    let expected = npy::read_u8(&args.data.join("kv.expected.npy"))?;
    let exact = got == expected;
    let (different, maximum, rms) = bf16_diff(&got, &expected, 0.0);
    println!(
        "occamy BF16 KV append: exact={exact}, differing={different}, max_abs={maximum:.6}, rms={rms:.6}"
    );
    Ok(exact)
}

pub fn run(hip: &HipThread, args: &ProbeArgs) -> anyhow::Result<usize> {
    ensure!(args.iters > 0, "--iters must be positive");
    hip.set_device(args.device)?;
    let counters = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, &args.prebuilt, &counters)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;
    let launch_counters = LaunchCounters::new();
    let mut launcher = Launcher::new("occamy-probe", &launch_counters);
    let mut failed = 0;
    if args.mode == "attention" || args.mode == "all" {
        failed += usize::from(!attention(hip, &stream, &ctx, &mut launcher, args)?);
    }
    if args.mode == "moe" || args.mode == "all" {
        for rank in 0..2 {
            failed += usize::from(!moe_rank(hip, &stream, &ctx, &mut launcher, args, rank)?);
        }
    }
    if args.mode == "dense" || args.mode == "all" {
        failed += usize::from(!dense(hip, &stream, &ctx, &mut launcher, args)?);
    }
    if args.mode == "kv" || args.mode == "all" {
        failed += usize::from(!kv_append(hip, &stream, &ctx, &mut launcher, args)?);
    }
    ensure!(
        matches!(
            args.mode.as_str(),
            "attention" | "moe" | "dense" | "kv" | "all"
        ),
        "unknown mode"
    );
    Ok(failed)
}
