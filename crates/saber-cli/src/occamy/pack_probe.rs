#![allow(clippy::too_many_arguments)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use anyhow::{ensure, Context};
use saber_exec::{ArgBindings, LaunchCounters, LaunchCtx, Launcher, LoadCounters, PluginSet};
use saber_hip::{DeviceAddr, DevicePtr, HipThread, Stream, StreamFlags};
use saber_kernels::abi::{self, SigArg};
use saber_kernels::occamy;
use saber_kernels::rules::{ArgValue, LaunchDescriptor, Shape as BindingShape, Slot};
use saber_kernels::saber::{resolve, AttnShape, KvDtype, Op, SaberRules, Shape};
use saber_mem::pack::{Dtype, PackReader};

pub struct PackProbeArgs {
    pub pack: PathBuf,
    pub fp8_experts: Option<PathBuf>,
    pub prebuilt: PathBuf,
    pub device: i32,
    pub mode: String,
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
        let (base, offset) = name
            .split_once('@')
            .map_or(Ok((name, 0)), |(base, offset)| {
                offset
                    .parse::<u64>()
                    .map(|offset| (base, offset))
                    .map_err(|error| anyhow::anyhow!("{name}: invalid byte offset: {error}"))
            })?;
        self.0
            .get(base)
            .map(DevicePtr::addr)
            .and_then(|address| address.offset(offset))
            .ok_or_else(|| anyhow::anyhow!("missing device buffer {base} or invalid offset"))
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

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 40) as f32 / (1u32 << 24) as f32 - 0.5
    }
}

fn bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn from_bf16(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

fn from_fp8(value: u8) -> f32 {
    let exponent = (value >> 3) & 15;
    let fraction = value & 7;
    let magnitude = if exponent == 0 {
        f32::from(fraction) * 2.0f32.powi(-9)
    } else {
        (1.0 + f32::from(fraction) / 8.0) * 2.0f32.powi(i32::from(exponent) - 7)
    };
    if value & 0x80 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn values(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|value| u16::from_le_bytes([value[0], value[1]]))
        .collect()
}

fn bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn random(count: usize, seed: u64) -> Vec<u16> {
    let mut rng = Rng(seed);
    (0..count).map(|_| bf16(rng.next() * 0.125)).collect()
}

fn matmul(x: &[u16], rows: usize, k: usize, w: &[u16], n: usize) -> Vec<u16> {
    let mut output = vec![0u16; rows * n];
    for row in 0..rows {
        for col in 0..n {
            let mut sum = 0.0f32;
            for inner in 0..k {
                sum += from_bf16(x[row * k + inner]) * from_bf16(w[col * k + inner]);
            }
            output[row * n + col] = bf16(sum);
        }
    }
    output
}

fn diff(got: &[u8], expected: &[u16], tolerance: f32) -> (usize, f32, f32) {
    let expected = bytes(expected);
    assert_eq!(got.len(), expected.len());
    let mut differing = 0;
    let mut maximum = 0.0f32;
    let mut squared = 0.0f64;
    for (a, b) in got.chunks_exact(2).zip(expected.chunks_exact(2)) {
        let a = from_bf16(u16::from_le_bytes([a[0], a[1]]));
        let b = from_bf16(u16::from_le_bytes([b[0], b[1]]));
        let difference = (a - b).abs();
        differing += usize::from(difference > tolerance);
        maximum = maximum.max(difference);
        squared += f64::from(difference) * f64::from(difference);
    }
    (
        differing,
        maximum,
        (squared / (got.len() / 2) as f64).sqrt() as f32,
    )
}

fn record(reader: &PackReader, name: &str, n: usize, k: usize) -> anyhow::Result<Vec<u8>> {
    let index = reader
        .index_of(name)
        .ok_or_else(|| anyhow::anyhow!("{}: missing {name}", reader.path().display()))?;
    let record = &reader.records()[index];
    ensure!(
        record.dtype == Dtype::BF16 && record.shape == [n as u64, k as u64],
        "{name}: {:?} {:?}, expected BF16 [{n}, {k}]",
        record.dtype,
        record.shape
    );
    let bytes = reader.read_record(index)?;
    reader.check_hash(index, &bytes)?;
    Ok(bytes.to_vec())
}

fn vector_record(reader: &PackReader, name: &str, len: usize) -> anyhow::Result<Vec<u8>> {
    let index = reader
        .index_of(name)
        .ok_or_else(|| anyhow::anyhow!("{}: missing {name}", reader.path().display()))?;
    let record = &reader.records()[index];
    ensure!(
        record.dtype == Dtype::BF16 && record.shape == [len as u64],
        "{name}: {:?} {:?}, expected BF16 [{len}]",
        record.dtype,
        record.shape
    );
    let bytes = reader.read_record(index)?;
    reader.check_hash(index, &bytes)?;
    Ok(bytes.to_vec())
}

fn raw_record(
    reader: &PackReader,
    name: &str,
    dtype: Dtype,
    shape: &[u64],
) -> anyhow::Result<Vec<u8>> {
    let index = reader
        .index_of(name)
        .ok_or_else(|| anyhow::anyhow!("{}: missing {name}", reader.path().display()))?;
    let record = &reader.records()[index];
    ensure!(
        record.dtype == dtype && record.shape == shape,
        "{name}: {:?} {:?}, expected {dtype:?} {shape:?}",
        record.dtype,
        record.shape
    );
    let bytes = reader.read_record(index)?;
    reader.check_hash(index, &bytes)?;
    Ok(bytes.to_vec())
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
        .collect()
}

fn report_f32(label: &str, got: &[u8], expected: &[f32], tolerance: f32) -> bool {
    let got = f32_values(got);
    let mut different = 0;
    let mut maximum = 0.0f32;
    let mut worst = (0.0f32, 0.0f32);
    let mut squared = 0.0f64;
    for (&got, &expected) in got.iter().zip(expected) {
        let difference = (got - expected).abs();
        different += usize::from(difference > tolerance);
        if difference > maximum {
            maximum = difference;
            worst = (got, expected);
        }
        squared += f64::from(difference) * f64::from(difference);
    }
    println!(
        "{label}: differing={different}, max_abs={maximum:.6}, rms={:.6}, worst_got={:.6}, worst_expected={:.6}",
        (squared / expected.len() as f64).sqrt(),
        worst.0,
        worst.1,
    );
    different == 0
}

fn rms_norm(x: &[u16], weight: &[u16], width: usize) -> Vec<u16> {
    x.chunks_exact(width)
        .flat_map(|row| {
            let sum: f32 = row
                .iter()
                .map(|&value| {
                    let value = from_bf16(value);
                    value * value
                })
                .sum();
            let inverse = 1.0 / (sum / width as f32 + 1e-6).sqrt();
            row.iter()
                .zip(weight)
                .map(|(&value, &weight)| {
                    bf16(from_bf16(value) * inverse * (from_bf16(weight) + 1.0))
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn report(label: &str, got: &[u8], expected: &[u16], tolerance: f32) -> bool {
    let (different, maximum, rms) = diff(got, expected, tolerance);
    println!("{label}: differing={different}, max_abs={maximum:.6}, rms={rms:.6}");
    different == 0
}

fn launch_dense(
    hip: &HipThread,
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    buffers: &mut Buffers,
    input: &str,
    weight: &str,
    output: &str,
    m: usize,
    n: usize,
) -> anyhow::Result<()> {
    let launch = occamy::dense_projection(m as u32, n as u32, 2048)
        .map_err(|error| anyhow::anyhow!(error))?;
    if launch.splits > 1 {
        buffers.zero(hip, stream, "part", launch.partial_bytes as usize)?;
        buffers.zero(hip, stream, "cnt", launch.counter_bytes as usize)?;
    }
    let mut arguments = vec![
        buffer(input),
        buffer(weight),
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
        buffer(output),
        scalar(m as i64),
        scalar(n as i64),
        scalar(2048),
    ];
    let signature = if launch.symbol.starts_with("fn_bgemm_sk_") {
        abi::SIG_FN_BGEMM_SK
    } else {
        arguments = vec![
            buffer(input),
            buffer(weight),
            buffer(output),
            scalar(m as i64),
            scalar(n as i64),
            scalar(2048),
        ];
        abi::SIG_FN_BGEMM
    };
    let launch = descriptor(
        "flashnext_hc",
        launch.symbol,
        launch.grid,
        launch.threads,
        signature,
        arguments,
    )?;
    dispatch(stream, ctx, launcher, &launch, buffers)
}

#[allow(clippy::too_many_arguments)]
fn launch_dense_strided(
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    buffers: &Buffers,
    input: &str,
    weight: &str,
    output: &str,
    m: usize,
    n: usize,
    ldc: usize,
) -> anyhow::Result<()> {
    ensure!(m <= 16, "strided dense probe only uses the 16-row cell");
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
            scalar(2048),
            scalar(ldc as i64),
        ],
    )?;
    dispatch(stream, ctx, launcher, &launch, buffers)
}

fn dense_rank(
    hip: &HipThread,
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    rank: u32,
    shared: &PackReader,
    args: &PackProbeArgs,
) -> anyhow::Result<bool> {
    let path = args.pack.join(format!("rank{rank}/dense.pack"));
    let reader = PackReader::open(&path)?;
    let cases = [
        ("layer.0.gdn.in_proj_qkvz.w", 6144),
        ("layer.0.gdn.in_proj_ba.w", 32),
        (
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            2048,
        ),
        (
            "model.language_model.layers.3.self_attn.q_proj.weight",
            4096,
        ),
        ("model.language_model.layers.3.self_attn.k_proj.weight", 256),
        ("model.language_model.layers.3.self_attn.v_proj.weight", 256),
        (
            "model.language_model.layers.3.self_attn.o_proj.weight",
            2048,
        ),
    ];
    let mut ok = true;
    for (case, (name, n)) in cases.iter().enumerate() {
        let weight = record(&reader, name, *n, 2048)?;
        let input = random(3 * 2048, 900 + case as u64);
        let expected = matmul(&input, 3, 2048, &values(&weight), *n);
        let mut buffers = Buffers::new();
        buffers.bytes(hip, stream, "x", &bytes(&input))?;
        buffers.bytes(hip, stream, "w", &weight)?;
        buffers.zero(hip, stream, "out", 3 * n * 2)?;
        launch_dense(
            hip,
            stream,
            ctx,
            launcher,
            &mut buffers,
            "x",
            "w",
            "out",
            3,
            *n,
        )?;
        hip.sync_stream(stream)?;
        let got = buffers.download(hip, stream, "out", 3 * n * 2)?;
        let (different, maximum, rms) = diff(&got, &expected, 0.03);
        println!(
            "occamy pack rank{rank} dense {name}: differing={different}, max_abs={maximum:.6}, rms={rms:.6}"
        );
        ok &= different == 0;
    }
    for (case, name) in [
        "model.language_model.layers.0.mlp.gate.weight",
        "model.language_model.layers.0.mlp.shared_expert_gate.weight",
    ]
    .iter()
    .enumerate()
    {
        let n = if case == 0 { 256 } else { 1 };
        let weight = record(shared, name, n, 2048)?;
        let input = random(3 * 2048, 920 + case as u64);
        let expected = matmul(&input, 3, 2048, &values(&weight), n);
        let mut buffers = Buffers::new();
        buffers.bytes(hip, stream, "x", &bytes(&input))?;
        buffers.bytes(hip, stream, "w", &weight)?;
        buffers.zero(hip, stream, "out", 3 * n * 2)?;
        launch_dense(
            hip,
            stream,
            ctx,
            launcher,
            &mut buffers,
            "x",
            "w",
            "out",
            3,
            n,
        )?;
        hip.sync_stream(stream)?;
        let got = buffers.download(hip, stream, "out", 3 * n * 2)?;
        let (different, maximum, rms) = diff(&got, &expected, 0.03);
        println!(
            "occamy pack shared dense {name}: differing={different}, max_abs={maximum:.6}, rms={rms:.6}"
        );
        ok &= different == 0;
    }
    Ok(ok)
}

struct AttentionRank {
    partial: Vec<u8>,
    expected: Vec<u16>,
    input: Vec<u16>,
}

fn truncate_bf16(value: f32) -> f32 {
    f32::from_bits(value.to_bits() & 0xffff_0000)
}

fn norm_rope(
    projected: &[u16],
    weight: &[u16],
    rows: usize,
    heads: usize,
    stride: usize,
    offset: usize,
    gated: bool,
) -> Vec<u16> {
    const HD: usize = 256;
    const RD: usize = 64;
    let mut output = vec![0u16; rows * heads * HD];
    for token in 0..rows {
        for head in 0..heads {
            let source = token * stride + offset + head * if gated { 2 * HD } else { HD };
            let sum: f32 = projected[source..source + HD]
                .iter()
                .map(|&value| {
                    let value = from_bf16(value);
                    value * value
                })
                .sum();
            let inverse = 1.0 / (sum / HD as f32 + 1e-6).sqrt();
            let mut normed = [0u16; HD];
            for dim in 0..HD {
                normed[dim] = bf16(
                    from_bf16(projected[source + dim]) * inverse * (from_bf16(weight[dim]) + 1.0),
                );
            }
            for dim in 0..HD {
                let value = if dim < RD {
                    let pair = if dim < RD / 2 {
                        dim + RD / 2
                    } else {
                        dim - RD / 2
                    };
                    let index = dim % (RD / 2);
                    let angle = token as f32 * 10_000_000f32.powf(-(index as f32) / 32.0);
                    let cos = from_bf16(bf16(angle.cos()));
                    let sin = from_bf16(bf16(angle.sin()));
                    let mine = from_bf16(normed[dim]);
                    let other = from_bf16(normed[pair]);
                    if dim < RD / 2 {
                        truncate_bf16(mine * cos) - truncate_bf16(other * sin)
                    } else {
                        truncate_bf16(mine * cos) + truncate_bf16(other * sin)
                    }
                } else {
                    from_bf16(normed[dim])
                };
                output[(token * heads + head) * HD + dim] = bf16(value);
            }
        }
    }
    output
}

fn attention_values(q: &[u16], k: &[u16], v: &[u16], rows: usize) -> Vec<u16> {
    const HEADS: usize = 8;
    const HD: usize = 256;
    let mut output = vec![0u16; rows * HEADS * HD];
    for token in 0..rows {
        for head in 0..HEADS {
            let mut scores = vec![0.0f32; token + 1];
            for (key, score) in scores.iter_mut().enumerate() {
                for dim in 0..HD {
                    *score += from_bf16(q[(token * HEADS + head) * HD + dim])
                        * from_bf16(k[key * HD + dim]);
                }
                *score /= 16.0;
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = scores.iter().map(|score| (*score - maximum).exp()).sum();
            for dim in 0..HD {
                let mut value = 0.0f32;
                for (key, score) in scores.iter().enumerate() {
                    value += (*score - maximum).exp() / denominator * from_bf16(v[key * HD + dim]);
                }
                output[(token * HEADS + head) * HD + dim] = bf16(value);
            }
        }
    }
    output
}

#[allow(clippy::too_many_lines)]
fn attention_rank(
    hip: &HipThread,
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    rank: u32,
    shared: &PackReader,
    args: &PackProbeArgs,
) -> anyhow::Result<AttentionRank> {
    const T: usize = 3;
    const H: usize = 2048;
    const NQ: usize = 8;
    const HD: usize = 256;
    const Q: usize = NQ * 2 * HD;
    const KV: usize = HD;
    const QKV: usize = Q + 2 * KV;
    let reader = PackReader::open(&args.pack.join(format!("rank{rank}/dense.pack")))?;
    let x = random(T * H, 1100);
    let norm_weight = vector_record(
        shared,
        "model.language_model.layers.3.input_layernorm.weight",
        H,
    )?;
    let expected_norm = rms_norm(&x, &values(&norm_weight), H);
    let mut buffers = Buffers::new();
    buffers.bytes(hip, stream, "x", &bytes(&x))?;
    buffers.bytes(hip, stream, "norm_w", &norm_weight)?;
    buffers.zero(hip, stream, "zero", T * H * 2)?;
    buffers.zero(hip, stream, "normed", T * H * 2)?;
    buffers.zero(hip, stream, "res_out", T * H * 2)?;
    let norm = descriptor(
        "head",
        "saber_add_rms_bf16",
        [T as u32, 1, 1],
        256,
        abi::SIG_ADD_RMS_BF16,
        vec![
            buffer("zero"),
            buffer("x"),
            buffer("norm_w"),
            buffer("normed"),
            buffer("res_out"),
            scalar(H as i64),
            ArgValue::F32 { fval: 1e-6 },
        ],
    )?;
    dispatch(stream, ctx, launcher, &norm, &buffers)?;
    hip.sync_stream(stream)?;
    let got_norm = buffers.download(hip, stream, "normed", T * H * 2)?;
    ensure!(
        report(
            &format!("occamy layer3 rank{rank} input norm"),
            &got_norm,
            &expected_norm,
            0.03,
        ),
        "layer3 rank{rank} input norm failed"
    );

    let mut expected_qkv = Vec::with_capacity(T * QKV);
    buffers.zero(hip, stream, "qkv", T * QKV * 2)?;
    for (projection, n, offset) in [("q_proj", Q, 0), ("k_proj", KV, Q), ("v_proj", KV, Q + KV)] {
        let name = format!("model.language_model.layers.3.self_attn.{projection}.weight");
        let weight = record(&reader, &name, n, H)?;
        buffers.bytes(hip, stream, projection, &weight)?;
        launch_dense_strided(
            stream,
            ctx,
            launcher,
            &buffers,
            "normed",
            projection,
            &format!("qkv@{}", offset * 2),
            T,
            n,
            QKV,
        )?;
        let projected = matmul(&expected_norm, T, H, &values(&weight), n);
        if expected_qkv.is_empty() {
            expected_qkv.resize(T * QKV, 0);
        }
        for token in 0..T {
            expected_qkv[token * QKV + offset..token * QKV + offset + n]
                .copy_from_slice(&projected[token * n..(token + 1) * n]);
        }
    }
    hip.sync_stream(stream)?;
    let got_qkv = buffers.download(hip, stream, "qkv", T * QKV * 2)?;
    ensure!(
        report(
            &format!("occamy layer3 rank{rank} qkv"),
            &got_qkv,
            &expected_qkv,
            0.03,
        ),
        "layer3 rank{rank} qkv failed"
    );

    let qw = vector_record(
        shared,
        "model.language_model.layers.3.self_attn.q_norm.weight",
        HD,
    )?;
    let kw = vector_record(
        shared,
        "model.language_model.layers.3.self_attn.k_norm.weight",
        HD,
    )?;
    let expected_q = norm_rope(&expected_qkv, &values(&qw), T, NQ, QKV, 0, true);
    let expected_k = norm_rope(&expected_qkv, &values(&kw), T, 1, QKV, Q, false);
    let expected_v: Vec<u16> = (0..T)
        .flat_map(|token| expected_qkv[token * QKV + Q + KV..token * QKV + Q + 2 * KV].to_vec())
        .collect();
    let mut cos_sin = Vec::with_capacity(16 * 64);
    for position in 0..16 {
        for index in 0..32 {
            let angle = position as f32 * 10_000_000f32.powf(-(index as f32) / 32.0);
            cos_sin.push(bf16(angle.cos()));
        }
        for index in 0..32 {
            let angle = position as f32 * 10_000_000f32.powf(-(index as f32) / 32.0);
            cos_sin.push(bf16(angle.sin()));
        }
    }
    let positions: Vec<u8> = (0..3)
        .flat_map(|_| (0..T as i64).flat_map(i64::to_le_bytes).collect::<Vec<_>>())
        .collect();
    for (name, payload) in [
        ("qw", qw),
        ("kw", kw),
        ("cos_sin", bytes(&cos_sin)),
        ("pos", positions),
    ] {
        buffers.bytes(hip, stream, name, &payload)?;
    }
    buffers.zero(hip, stream, "q_out", T * NQ * HD * 2)?;
    buffers.zero(hip, stream, "k_out", T * HD * 2)?;
    let rope = descriptor(
        "flashnext_glue",
        "fn_qk_norm_rope_h256_rot64_mi1_v2",
        [((T * (NQ + 1)) as u32).div_ceil(8), 1, 1],
        256,
        abi::SIG_QK_NORM_ROPE,
        vec![
            buffer("qkv"),
            buffer("qw"),
            buffer("kw"),
            buffer("cos_sin"),
            buffer("pos"),
            buffer("q_out"),
            buffer("k_out"),
            scalar(T as i64),
            scalar(NQ as i64),
            scalar(1),
            scalar(QKV as i64),
            ArgValue::F32 { fval: 1e-6 },
            scalar(11),
            scalar(11),
            scalar(10),
        ],
    )?;
    dispatch(stream, ctx, launcher, &rope, &buffers)?;
    hip.sync_stream(stream)?;
    let got_q = buffers.download(hip, stream, "q_out", T * NQ * HD * 2)?;
    let got_k = buffers.download(hip, stream, "k_out", T * HD * 2)?;
    ensure!(
        report(
            &format!("occamy layer3 rank{rank} q norm+rope"),
            &got_q,
            &expected_q,
            0.03,
        ) && report(
            &format!("occamy layer3 rank{rank} k norm+rope"),
            &got_k,
            &expected_k,
            0.03,
        ),
        "layer3 rank{rank} qk norm/rope failed"
    );

    let slots: Vec<u8> = (0..T as i64).flat_map(i64::to_le_bytes).collect();
    let ring_slots: Vec<u8> = (0..T).flat_map(|_| (-1i64).to_le_bytes()).collect();
    buffers.bytes(hip, stream, "slot", &slots)?;
    buffers.bytes(hip, stream, "ring_slot", &ring_slots)?;
    buffers.zero(hip, stream, "kv", 16 * 2 * HD * 2)?;
    let append = descriptor(
        "flashnext_glue",
        occamy::KV_APPEND_SYMBOL,
        [T as u32, 1, 1],
        128,
        abi::SIG_FN_KV_APPEND_RING,
        vec![
            buffer("k_out"),
            ArgValue::I64 { ival: HD as i64 },
            buffer(&format!("qkv@{}", (Q + KV) * 2)),
            ArgValue::I64 { ival: QKV as i64 },
            buffer("kv"),
            buffer("slot"),
            ArgValue::I64 {
                ival: (16 * 2 * HD) as i64,
            },
            ArgValue::I64 {
                ival: (2 * HD) as i64,
            },
            ArgValue::I64 {
                ival: (16 * 2 * HD) as i64,
            },
            null_buffer("ikey"),
            ArgValue::I64 { ival: 0 },
            null_buffer("ring"),
            null_buffer("ring_pos"),
            buffer("ring_slot"),
            null_buffer("pos"),
            scalar(T as i64),
        ],
    )?;
    dispatch(stream, ctx, launcher, &append, &buffers)?;
    hip.sync_stream(stream)?;
    let mut expected_kv = vec![0u16; 16 * 2 * HD];
    let got_k_values = values(&got_k);
    let got_qkv_values = values(&got_qkv);
    for token in 0..T {
        expected_kv[token * 2 * HD..token * 2 * HD + HD]
            .copy_from_slice(&got_k_values[token * HD..(token + 1) * HD]);
        expected_kv[token * 2 * HD + HD..(token + 1) * 2 * HD]
            .copy_from_slice(&got_qkv_values[token * QKV + Q + KV..token * QKV + Q + 2 * KV]);
    }
    let got_kv = buffers.download(hip, stream, "kv", 16 * 2 * HD * 2)?;
    ensure!(
        report(
            &format!("occamy layer3 rank{rank} KV cache"),
            &got_kv,
            &expected_kv,
            0.0,
        ),
        "layer3 rank{rank} KV cache failed"
    );

    buffers.bytes(hip, stream, "block_table", &[0, 0, 0, 0])?;
    buffers.bytes(hip, stream, "seqused_k", &(T as i32).to_le_bytes())?;
    buffers.zero(hip, stream, "core", T * NQ * HD * 2)?;
    let shape = AttnShape {
        num_seqs: 1,
        q_len: T as u32,
        live_rows: T as u32,
        q_heads: NQ as u32,
        kv_heads: 1,
        head_dim: HD as u32,
        block_size: 16,
        max_blocks: 1,
        max_ctx: 16,
        splits: 0,
        kv_dtype: KvDtype::Bf16,
        kv_block_stride: (16 * 2 * HD) as i64,
        kv_head_stride: (16 * 2 * HD) as i64,
        scale_bits: (1.0f32 / 16.0).to_bits(),
        has_descales: false,
        ragged_rows: 0,
    };
    let rules = SaberRules::occamy()?;
    let attention = resolve(Op::AttnPrefill { fp8_mode: 0 }, Shape::Attn(shape), &rules)?;
    let mut attention_buffers = Buffers::new();
    attention_buffers.bytes(hip, stream, "q", &got_q)?;
    attention_buffers.bytes(hip, stream, "kv", &got_kv)?;
    attention_buffers.bytes(hip, stream, "block_table", &[0, 0, 0, 0])?;
    attention_buffers.bytes(hip, stream, "seqused_k", &(T as i32).to_le_bytes())?;
    attention_buffers.zero(hip, stream, "out", T * NQ * HD * 2)?;
    dispatch(stream, ctx, launcher, &attention, &attention_buffers)?;
    hip.sync_stream(stream)?;
    let got_core = attention_buffers.download(hip, stream, "out", T * NQ * HD * 2)?;
    buffers.bytes(hip, stream, "core", &got_core)?;
    let expected_core = attention_values(&expected_q, &expected_k, &expected_v, T);
    ensure!(
        report(
            &format!("occamy layer3 rank{rank} attention core"),
            &got_core,
            &expected_core,
            0.05,
        ),
        "layer3 rank{rank} attention core failed"
    );

    buffers.zero(hip, stream, "gated", T * NQ * HD * 2)?;
    let gate = descriptor(
        "flashnext_hc",
        "fn_attn_gate_bf16_t256",
        [T as u32, 1, 1],
        256,
        abi::SIG_FN_ATTN_GATE_BF16,
        vec![
            buffer("core"),
            buffer("qkv"),
            buffer("gated"),
            scalar((NQ * HD) as i64),
            scalar(HD as i64),
            scalar(QKV as i64),
        ],
    )?;
    dispatch(stream, ctx, launcher, &gate, &buffers)?;
    hip.sync_stream(stream)?;
    let got_gated = buffers.download(hip, stream, "gated", T * NQ * HD * 2)?;
    let mut expected_gated = vec![0u16; T * NQ * HD];
    for token in 0..T {
        for head in 0..NQ {
            for dim in 0..HD {
                let index = (token * NQ + head) * HD + dim;
                let gate = from_bf16(expected_qkv[token * QKV + head * 2 * HD + HD + dim]);
                let sigmoid = bf16(1.0 / (1.0 + (-gate).exp()));
                expected_gated[index] = bf16(from_bf16(expected_core[index]) * from_bf16(sigmoid));
            }
        }
    }
    ensure!(
        report(
            &format!("occamy layer3 rank{rank} attention gate"),
            &got_gated,
            &expected_gated,
            0.03,
        ),
        "layer3 rank{rank} attention gate failed"
    );

    let output_weight = record(
        &reader,
        "model.language_model.layers.3.self_attn.o_proj.weight",
        H,
        NQ * HD,
    )?;
    buffers.bytes(hip, stream, "o_w", &output_weight)?;
    buffers.zero(hip, stream, "attn_partial", T * H * 2)?;
    launch_dense(
        hip,
        stream,
        ctx,
        launcher,
        &mut buffers,
        "gated",
        "o_w",
        "attn_partial",
        T,
        H,
    )?;
    hip.sync_stream(stream)?;
    let partial = buffers.download(hip, stream, "attn_partial", T * H * 2)?;
    let expected = matmul(&expected_gated, T, NQ * HD, &values(&output_weight), H);
    ensure!(
        report(
            &format!("occamy layer3 rank{rank} attention output"),
            &partial,
            &expected,
            0.03,
        ),
        "layer3 rank{rank} attention output failed"
    );
    Ok(AttentionRank {
        partial,
        expected,
        input: x,
    })
}

fn silu_mul(projected: &[u16], inter: usize) -> Vec<u16> {
    projected[..inter]
        .iter()
        .zip(&projected[inter..])
        .map(|(&gate, &up)| {
            let gate = from_bf16(gate);
            bf16(gate / (1.0 + (-gate).exp()) * from_bf16(up))
        })
        .collect()
}

struct GdnFront {
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    g: Vec<f32>,
    beta: Vec<f32>,
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

fn gdn_front_reference(
    projected: &[u16],
    convolution: &[u16],
    a_log: &[f32],
    dt_bias: &[f32],
) -> GdnFront {
    const T: usize = 3;
    const HK: usize = 8;
    const HV: usize = 16;
    const HD: usize = 128;
    const CONV: usize = 4096;
    const GN: usize = 6176;
    let mut convolved = vec![0u16; T * CONV];
    for token in 0..T {
        for channel in 0..CONV {
            let mut sum = 0.0f32;
            for tap in 0..4 {
                let source = token as isize + tap as isize - 3;
                if source >= 0 {
                    sum += from_bf16(convolution[channel * 4 + tap])
                        * from_bf16(projected[source as usize * GN + channel]);
                }
            }
            let value = sum / (1.0 + (-sum).exp());
            convolved[token * CONV + channel] = bf16(value);
        }
    }
    let mut q = vec![0u16; T * HK * HD];
    let mut k = vec![0u16; T * HK * HD];
    let mut v = vec![0u16; T * HV * HD];
    for token in 0..T {
        for head in 0..HK {
            for (region, output) in [(0, &mut q), (1, &mut k)] {
                let source = token * CONV + (region * HK + head) * HD;
                let sum: f32 = convolved[source..source + HD]
                    .iter()
                    .map(|&value| {
                        let value = from_bf16(value);
                        value * value
                    })
                    .sum();
                let inverse = 1.0 / (sum + 1e-6).sqrt();
                for dim in 0..HD {
                    output[(token * HK + head) * HD + dim] =
                        bf16(from_bf16(convolved[source + dim]) * inverse);
                }
            }
        }
        v[token * HV * HD..(token + 1) * HV * HD]
            .copy_from_slice(&convolved[token * CONV + 2 * HK * HD..(token + 1) * CONV]);
    }
    let mut g = vec![0.0f32; T * HV];
    let mut beta = vec![0.0f32; T * HV];
    for token in 0..T {
        for head in 0..HV {
            let a = from_bf16(projected[token * GN + CONV + HV * HD + HV + head]);
            let b = from_bf16(projected[token * GN + CONV + HV * HD + head]);
            let gate = -a_log[head].exp() * softplus(a + dt_bias[head]);
            g[token * HV + head] = gate
                + if token == 0 {
                    0.0
                } else {
                    g[(token - 1) * HV + head]
                };
            beta[token * HV + head] = 1.0 / (1.0 + (-b).exp());
        }
    }
    GdnFront { q, k, v, g, beta }
}

fn gdn_recurrence_reference(front: &GdnFront) -> (Vec<u16>, Vec<f32>) {
    const T: usize = 3;
    const HK: usize = 8;
    const HV: usize = 16;
    const HD: usize = 128;
    let scale = 1.0 / (HD as f32).sqrt();
    let mut state = vec![0.0f32; HV * HD * HD];
    let mut output = vec![0u16; T * HV * HD];
    for token in 0..T {
        for head in 0..HV {
            let key_head = head / (HV / HK);
            let decay = if token == 0 {
                front.g[head].exp()
            } else {
                (front.g[token * HV + head] - front.g[(token - 1) * HV + head]).exp()
            };
            for value_dim in 0..HD {
                let mut dot = 0.0f32;
                for key_dim in 0..HD {
                    let index = (head * HD + value_dim) * HD + key_dim;
                    state[index] *= decay;
                    dot +=
                        state[index] * from_bf16(front.k[(token * HK + key_head) * HD + key_dim]);
                }
                let update = (from_bf16(front.v[(token * HV + head) * HD + value_dim]) - dot)
                    * front.beta[token * HV + head];
                let mut value = 0.0f32;
                for key_dim in 0..HD {
                    let index = (head * HD + value_dim) * HD + key_dim;
                    state[index] +=
                        update * from_bf16(front.k[(token * HK + key_head) * HD + key_dim]);
                    value += state[index]
                        * from_bf16(front.q[(token * HK + key_head) * HD + key_dim])
                        * scale;
                }
                output[(token * HV + head) * HD + value_dim] = bf16(value);
            }
        }
    }
    (output, state)
}

fn gdn_norm_reference(core: &[u16], z: &[u16], weight: &[u16]) -> Vec<u16> {
    const T: usize = 3;
    const HV: usize = 16;
    const HD: usize = 128;
    let mut output = vec![0u16; T * HV * HD];
    for token in 0..T {
        for head in 0..HV {
            let base = (token * HV + head) * HD;
            let sum: f32 = core[base..base + HD]
                .iter()
                .map(|&value| {
                    let value = from_bf16(value);
                    value * value
                })
                .sum();
            let inverse = 1.0 / (sum / HD as f32 + 1e-6).sqrt();
            for dim in 0..HD {
                let gate = from_bf16(z[base + dim]);
                let silu = gate / (1.0 + (-gate).exp());
                output[base + dim] =
                    bf16(from_bf16(core[base + dim]) * inverse * from_bf16(weight[dim]) * silu);
            }
        }
    }
    output
}

#[allow(clippy::too_many_lines)]
fn gdn_rank(
    hip: &HipThread,
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    rank: u32,
    shared: &PackReader,
    args: &PackProbeArgs,
) -> anyhow::Result<AttentionRank> {
    const T: usize = 3;
    const H: usize = 2048;
    const HK: usize = 8;
    const HV: usize = 16;
    const HD: usize = 128;
    const CONV: usize = (2 * HK + HV) * HD;
    const Z: usize = HV * HD;
    const GN: usize = CONV + Z + 2 * HV;
    let reader = PackReader::open(&args.pack.join(format!("rank{rank}/dense.pack")))?;
    let x = random(T * H, 1100);
    let input_weight = vector_record(
        shared,
        "model.language_model.layers.0.input_layernorm.weight",
        H,
    )?;
    let (normed, expected_norm) = norm_only(hip, stream, ctx, launcher, &x, &input_weight)?;
    ensure!(
        report(
            &format!("occamy layer0 rank{rank} input norm"),
            &normed,
            &expected_norm,
            0.03,
        ),
        "layer0 rank{rank} input norm failed"
    );
    let qkvz = record(&reader, "layer.0.gdn.in_proj_qkvz.w", CONV + Z, H)?;
    let ba = record(&reader, "layer.0.gdn.in_proj_ba.w", 2 * HV, H)?;
    let mut buffers = Buffers::new();
    buffers.bytes(hip, stream, "normed", &normed)?;
    buffers.bytes(hip, stream, "qkvz_w", &qkvz)?;
    buffers.bytes(hip, stream, "ba_w", &ba)?;
    buffers.zero(hip, stream, "proj", T * GN * 2)?;
    for (weight, n, offset) in [("qkvz_w", CONV + Z, 0), ("ba_w", 2 * HV, CONV + Z)] {
        launch_dense_strided(
            stream,
            ctx,
            launcher,
            &buffers,
            "normed",
            weight,
            &format!("proj@{}", offset * 2),
            T,
            n,
            GN,
        )?;
    }
    hip.sync_stream(stream)?;
    let got_proj = buffers.download(hip, stream, "proj", T * GN * 2)?;
    let expected_qkvz = matmul(&expected_norm, T, H, &values(&qkvz), CONV + Z);
    let expected_ba = matmul(&expected_norm, T, H, &values(&ba), 2 * HV);
    let mut expected_proj = vec![0u16; T * GN];
    for token in 0..T {
        expected_proj[token * GN..token * GN + CONV + Z]
            .copy_from_slice(&expected_qkvz[token * (CONV + Z)..(token + 1) * (CONV + Z)]);
        expected_proj[token * GN + CONV + Z..(token + 1) * GN]
            .copy_from_slice(&expected_ba[token * 2 * HV..(token + 1) * 2 * HV]);
    }
    ensure!(
        report(
            &format!("occamy layer0 rank{rank} merged projection"),
            &got_proj,
            &expected_proj,
            0.03,
        ),
        "layer0 rank{rank} merged projection failed"
    );

    let convolution = raw_record(
        &reader,
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        Dtype::BF16,
        &[CONV as u64, 1, 4],
    )?;
    let a_log = raw_record(
        &reader,
        "model.language_model.layers.0.linear_attn.A_log",
        Dtype::F32,
        &[HV as u64],
    )?;
    let dt_bias = raw_record(
        &reader,
        "model.language_model.layers.0.linear_attn.dt_bias",
        Dtype::F32,
        &[HV as u64],
    )?;
    let expected_front = gdn_front_reference(
        &expected_proj,
        &values(&convolution),
        &f32_values(&a_log),
        &f32_values(&dt_bias),
    );
    for (name, payload) in [
        ("conv_w", convolution),
        ("a_log", a_log),
        ("dt_bias", dt_bias),
        ("cu", vec![0, 0, 0, 0, 3, 0, 0, 0]),
        ("cache_idx", vec![0; 8]),
        ("has_init", vec![0; 4]),
    ] {
        buffers.bytes(hip, stream, name, &payload)?;
    }
    for (name, size) in [
        ("cstate", CONV * 3 * 2),
        ("q", T * HK * HD * 2),
        ("k", T * HK * HD * 2),
        ("v", T * HV * HD * 2),
        ("g", T * HV * 4),
        ("beta", T * HV * 4),
        ("a", T * HV * 64 * 2),
        ("core", T * HV * HD * 2),
        ("h0", HV * HD * HD * 4),
        ("ht", HV * HD * HD * 4),
    ] {
        buffers.zero(hip, stream, name, size)?;
    }
    let prep = descriptor(
        "saber_gdn_conv",
        "saber_gdn_conv_prep_w4_h128_hpb8_ab1",
        [2, (CONV / 1024) as u32, 1],
        256,
        abi::SIG_SABER_GDN_CONV_PREP,
        vec![
            buffer("proj"),
            ArgValue::I64 { ival: GN as i64 },
            buffer("conv_w"),
            null_buffer("bias"),
            buffer("cstate"),
            ArgValue::I64 {
                ival: (3 * CONV) as i64,
            },
            ArgValue::I64 { ival: 3 },
            ArgValue::I64 { ival: 1 },
            buffer("cache_idx"),
            ArgValue::I64 { ival: 1 },
            buffer("has_init"),
            buffer(&format!("proj@{}", (CONV + Z + HV) * 2)),
            buffer(&format!("proj@{}", (CONV + Z) * 2)),
            ArgValue::I64 { ival: GN as i64 },
            buffer("a_log"),
            buffer("dt_bias"),
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
    dispatch(stream, ctx, launcher, &prep, &buffers)?;
    hip.sync_stream(stream)?;
    for (name, expected) in [
        ("q", &expected_front.q),
        ("k", &expected_front.k),
        ("v", &expected_front.v),
    ] {
        let got = buffers.download(hip, stream, name, expected.len() * 2)?;
        ensure!(
            report(
                &format!("occamy layer0 rank{rank} conv {name}"),
                &got,
                expected,
                0.03,
            ),
            "layer0 rank{rank} conv {name} failed"
        );
    }
    let got_g = buffers.download(hip, stream, "g", T * HV * 4)?;
    let got_beta = buffers.download(hip, stream, "beta", T * HV * 4)?;
    ensure!(
        report_f32(
            &format!("occamy layer0 rank{rank} gates"),
            &got_g,
            &expected_front.g,
            0.0005,
        ) && report_f32(
            &format!("occamy layer0 rank{rank} beta"),
            &got_beta,
            &expected_front.beta,
            0.0002,
        ),
        "layer0 rank{rank} gates failed"
    );

    let solve = descriptor(
        "saber_gdn_kkt_solve",
        occamy::GDN_KKT_SYMBOL,
        [2, HK as u32, 1],
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
    dispatch(stream, ctx, launcher, &solve, &buffers)?;
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
            buffer("h0"),
            buffer("core"),
            buffer("ht"),
            buffer("cu"),
            scalar(HV as i64),
            scalar(HK as i64),
            ArgValue::F32 {
                fval: 1.0 / (HD as f32).sqrt(),
            },
        ],
    )?;
    dispatch(stream, ctx, launcher, &scan, &buffers)?;
    hip.sync_stream(stream)?;
    let got_core = buffers.download(hip, stream, "core", T * HV * HD * 2)?;
    let got_state = buffers.download(hip, stream, "ht", HV * HD * HD * 4)?;
    let (expected_core, expected_state) = gdn_recurrence_reference(&expected_front);
    ensure!(
        report(
            &format!("occamy layer0 rank{rank} GDN core"),
            &got_core,
            &expected_core,
            0.05,
        ) && report_f32(
            &format!("occamy layer0 rank{rank} GDN FP32 state"),
            &got_state,
            &expected_state,
            0.05,
        ),
        "layer0 rank{rank} GDN recurrence failed"
    );

    let norm_weight = vector_record(
        shared,
        "model.language_model.layers.0.linear_attn.norm.weight",
        HD,
    )?;
    let expected_z: Vec<u16> = (0..T)
        .flat_map(|token| expected_proj[token * GN + CONV..token * GN + CONV + Z].to_vec())
        .collect();
    let expected_gdn_norm = gdn_norm_reference(&expected_core, &expected_z, &values(&norm_weight));
    buffers.bytes(hip, stream, "gdn_norm_w", &norm_weight)?;
    buffers.zero(hip, stream, "gdn_norm", T * Z * 2)?;
    let norm = descriptor(
        "flashnext_hc",
        "fn_gdn_norm_bf16_t256",
        [T as u32, 1, 1],
        256,
        abi::SIG_FN_GDN_NORM_BF16,
        vec![
            buffer("core"),
            buffer(&format!("proj@{}", CONV * 2)),
            ArgValue::I64 { ival: GN as i64 },
            buffer("gdn_norm_w"),
            buffer("gdn_norm"),
            scalar(Z as i64),
            ArgValue::F32 { fval: 1e-6 },
        ],
    )?;
    dispatch(stream, ctx, launcher, &norm, &buffers)?;
    hip.sync_stream(stream)?;
    let got_gdn_norm = buffers.download(hip, stream, "gdn_norm", T * Z * 2)?;
    ensure!(
        report(
            &format!("occamy layer0 rank{rank} GDN gated norm"),
            &got_gdn_norm,
            &expected_gdn_norm,
            0.05,
        ),
        "layer0 rank{rank} GDN norm failed"
    );
    let output_weight = record(
        &reader,
        "model.language_model.layers.0.linear_attn.out_proj.weight",
        H,
        Z,
    )?;
    buffers.bytes(hip, stream, "out_w", &output_weight)?;
    buffers.zero(hip, stream, "gdn_partial", T * H * 2)?;
    launch_dense(
        hip,
        stream,
        ctx,
        launcher,
        &mut buffers,
        "gdn_norm",
        "out_w",
        "gdn_partial",
        T,
        H,
    )?;
    hip.sync_stream(stream)?;
    let partial = buffers.download(hip, stream, "gdn_partial", T * H * 2)?;
    let expected = matmul(&expected_gdn_norm, T, Z, &values(&output_weight), H);
    ensure!(
        report(
            &format!("occamy layer0 rank{rank} GDN output"),
            &partial,
            &expected,
            0.05,
        ),
        "layer0 rank{rank} GDN output failed"
    );
    Ok(AttentionRank {
        partial,
        expected,
        input: x,
    })
}

struct MoeOutcome {
    partial: Vec<u8>,
    expected: Vec<u16>,
}

#[allow(clippy::too_many_arguments)]
fn moe_rank(
    hip: &HipThread,
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    rank: u32,
    layer: u32,
    x: Vec<u16>,
    shared: &PackReader,
    args: &PackProbeArgs,
) -> anyhow::Result<MoeOutcome> {
    const T: usize = 3;
    const TOPK: usize = 8;
    const LOCAL: usize = 128;
    const H: usize = 2048;
    const I: usize = 512;
    const ROW_CAP: usize = T * TOPK * 16 + 16;
    const TILE_CAP: usize = T * TOPK + 1;
    ensure!(x.len() == T * H, "MoE input has {} elements", x.len());
    let reader = PackReader::open(&args.pack.join(format!("rank{rank}/dense.pack")))?;
    let fp8_reader = args
        .fp8_experts
        .as_ref()
        .map(|root| PackReader::open(&root.join(format!("rank{rank}/dense.pack"))))
        .transpose()?;
    let mut buffers = Buffers::new();
    buffers.bytes(hip, stream, "x", &bytes(&x))?;
    let router = format!("model.language_model.layers.{layer}.mlp.gate.weight");
    let shared_router =
        format!("model.language_model.layers.{layer}.mlp.shared_expert_gate.weight");
    for (name, n, output) in [(router, 256, "logits"), (shared_router, 1, "shared_gate")] {
        let weight = record(shared, &name, n, H)?;
        let weight_name = format!("{output}_w");
        buffers.bytes(hip, stream, &weight_name, &weight)?;
        buffers.zero(hip, stream, output, T * n * 2)?;
        launch_dense(
            hip,
            stream,
            ctx,
            launcher,
            &mut buffers,
            "x",
            &weight_name,
            output,
            T,
            n,
        )?;
        hip.sync_stream(stream)?;
    }
    for (name, size) in [
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
    ] {
        buffers.zero(hip, stream, name, size)?;
    }
    let route = descriptor(
        "occamy_moe",
        occamy::ROUTE_SYMBOL,
        occamy::route_grid(T as u32),
        256,
        abi::SIG_MOE_ROUTE,
        vec![
            buffer("logits"),
            scalar(T as i64),
            scalar((rank as usize * LOCAL) as i64),
            scalar(LOCAL as i64),
            scalar(1),
            scalar(i64::from(rank == 0)),
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
    dispatch(stream, ctx, launcher, &route, &buffers)?;
    hip.sync_stream(stream)?;
    let ids: Vec<i32> = buffers
        .download(hip, stream, "ids", T * TOPK * 4)?
        .chunks_exact(4)
        .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let selected: BTreeSet<usize> = ids
        .iter()
        .map(|&id| id as usize)
        .filter(|id| (rank as usize * LOCAL..(rank as usize + 1) * LOCAL).contains(id))
        .collect();
    ensure!(!selected.is_empty(), "rank{rank}: no local expert selected");
    for &global in &selected {
        for (projection, n, k) in [("gate_up", 2 * I, H), ("down", H, I)] {
            let prefix = match projection {
                "gate_up" => format!("layer.{layer}.expert.{global}.gate_up"),
                _ => format!("model.language_model.layers.{layer}.mlp.experts.{global}.down_proj"),
            };
            if let Some(fp8) = &fp8_reader {
                let weight = raw_record(
                    fp8,
                    &format!("{prefix}.w8"),
                    Dtype::F8E4M3,
                    &[n as u64, k as u64],
                )?;
                let scales = raw_record(
                    fp8,
                    &format!("{prefix}.s8"),
                    Dtype::F32,
                    &[(n / 128) as u64, (k / 128) as u64],
                )?;
                buffers.bytes(hip, stream, &format!("e{global}.{projection}"), &weight)?;
                buffers.bytes(hip, stream, &format!("e{global}.{projection}.s"), &scales)?;
            } else {
                let suffix = if projection == "gate_up" {
                    "w"
                } else {
                    "weight"
                };
                let weight = record(&reader, &format!("{prefix}.{suffix}"), n, k)
                    .with_context(|| format!("rank{rank} expert {global} {projection}"))?;
                buffers.bytes(hip, stream, &format!("e{global}.{projection}"), &weight)?;
            }
        }
    }
    if rank == 0 {
        let shared_gu = format!("layer.{layer}.mlp.shared_expert.gate_up.w");
        let shared_down =
            format!("model.language_model.layers.{layer}.mlp.shared_expert.down_proj.weight");
        for (name, projection, n, k) in [
            (shared_gu, "gate_up", 2 * I, H),
            (shared_down, "down", H, I),
        ] {
            let weight = record(shared, &name, n, k)?;
            buffers.bytes(hip, stream, &format!("shared.{projection}"), &weight)?;
        }
    }
    for (table, down) in [("xgu", false), ("xdn", true)] {
        let mut addresses = Vec::new();
        for local in 0..LOCAL + usize::from(rank == 0) {
            let global = rank as usize * LOCAL + local;
            let name = if local == LOCAL {
                format!("shared.{}", if down { "down" } else { "gate_up" })
            } else {
                format!("e{global}.{}", if down { "down" } else { "gate_up" })
            };
            addresses.push(if local == LOCAL || selected.contains(&global) {
                buffers.addr(&name)?.as_u64()
            } else {
                0
            });
            addresses.push(
                if fp8_reader.is_some() && local != LOCAL && selected.contains(&global) {
                    buffers.addr(&format!("{name}.s"))?.as_u64()
                } else {
                    0
                },
            );
        }
        let table_bytes: Vec<u8> = addresses.into_iter().flat_map(u64::to_le_bytes).collect();
        buffers.bytes(hip, stream, table, &table_bytes)?;
    }
    for (name, size) in [
        ("gu", ROW_CAP * 2 * I * 2),
        ("act", ROW_CAP * I * 2),
        ("y", ROW_CAP * H * 2),
        ("out", T * H * 2),
    ] {
        buffers.zero(hip, stream, name, size)?;
    }
    let gate_up = descriptor(
        if fp8_reader.is_some() {
            saber_kernels::fn50::PLUGIN
        } else {
            "flashnext_moe"
        },
        if fp8_reader.is_some() {
            saber_kernels::fn50::bf16_symbol(true)
        } else {
            occamy::GATE_UP_SYMBOL
        },
        occamy::grouped_grid((2 * I) as u32, TILE_CAP as u32),
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
            scalar((2 * I) as i64),
            scalar(H as i64),
            scalar(ROW_CAP as i64),
        ],
    )?;
    let silu = descriptor(
        "flashnext_glue",
        occamy::SILU_SYMBOL,
        [ROW_CAP as u32, 1, 1],
        256,
        abi::SIG_FN_MOE_SILU_BF16,
        vec![
            buffer("gu"),
            buffer("act"),
            scalar(I as i64),
            buffer("row_tok"),
            buffer("meta"),
        ],
    )?;
    let down = descriptor(
        if fp8_reader.is_some() {
            saber_kernels::fn50::PLUGIN
        } else {
            "flashnext_moe"
        },
        if fp8_reader.is_some() {
            saber_kernels::fn50::bf16_symbol(false)
        } else {
            occamy::DOWN_SYMBOL
        },
        occamy::grouped_grid(H as u32, TILE_CAP as u32),
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
            scalar(H as i64),
            scalar(I as i64),
            scalar(ROW_CAP as i64),
        ],
    )?;
    let combine = descriptor(
        "occamy_moe",
        occamy::COMBINE_SYMBOL,
        [T as u32, 1, 1],
        256,
        abi::SIG_MOE_COMBINE,
        vec![
            buffer("y"),
            buffer("weights"),
            buffer("pos"),
            buffer("shared_gate"),
            buffer("meta"),
            buffer("out"),
            scalar(H as i64),
        ],
    )?;
    if fp8_reader.is_some() {
        for (name, size) in [
            ("x_q", T * H),
            ("x_s", T * (H / 128) * 4),
            ("act_q", ROW_CAP * I),
            ("act_s", ROW_CAP * (I / 128) * 4),
        ] {
            buffers.zero(hip, stream, name, size)?;
        }
        let quant = |input: &str, output: &str, scales: &str, rows: usize, columns: usize| {
            descriptor(
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
            )
        };
        let gate_up_fp8 = descriptor(
            saber_kernels::fn50::PLUGIN,
            saber_kernels::fn50::fp8_symbol(true),
            [(2 * I / 128) as u32, TILE_CAP as u32, 1],
            128,
            abi::SIG_FN_MOE_A8W8,
            vec![
                buffer("x_q"),
                buffer("xgu"),
                buffer("tile_expert"),
                buffer("tile_row0"),
                buffer("tile_rows"),
                buffer("row_tok"),
                buffer("meta"),
                buffer("x_s"),
                buffer("gu"),
                scalar((2 * I) as i64),
                scalar(H as i64),
            ],
        )?;
        let down_fp8 = descriptor(
            saber_kernels::fn50::PLUGIN,
            saber_kernels::fn50::fp8_symbol(false),
            [(H / 128) as u32, TILE_CAP as u32, 1],
            128,
            abi::SIG_FN_MOE_A8W8,
            vec![
                buffer("act_q"),
                buffer("xdn"),
                buffer("tile_expert"),
                buffer("tile_row0"),
                buffer("tile_rows"),
                buffer("row_tok"),
                buffer("meta"),
                buffer("act_s"),
                buffer("y"),
                scalar(H as i64),
                scalar(I as i64),
            ],
        )?;
        for launch in [
            quant("x", "x_q", "x_s", T, H)?,
            gate_up_fp8,
            gate_up,
            silu,
            quant("act", "act_q", "act_s", ROW_CAP, I)?,
            down_fp8,
            down,
            combine,
        ]
        .iter()
        {
            dispatch(stream, ctx, launcher, launch, &buffers)?;
        }
    } else {
        for launch in [&gate_up, &silu, &down, &combine] {
            dispatch(stream, ctx, launcher, launch, &buffers)?;
        }
    }
    hip.sync_stream(stream)?;
    let got = buffers.download(hip, stream, "out", T * H * 2)?;
    if fp8_reader.is_some() {
        let quantized = buffers.download(hip, stream, "x_q", T * H)?;
        let scales = f32_values(&buffers.download(hip, stream, "x_s", T * (H / 128) * 4)?);
        let mut squared = 0.0f64;
        let mut changed = 0usize;
        for (index, &code) in quantized.iter().enumerate() {
            let value = from_fp8(code) * scales[index / H * (H / 128) + (index % H) / 128];
            let difference = value - from_bf16(x[index]);
            changed += usize::from(difference != 0.0);
            squared += f64::from(difference) * f64::from(difference);
        }
        println!(
            "occamy FP8 activation rank{rank}: changed={changed}/{}, rms={:.6}, first scales={:?}",
            T * H,
            (squared / (T * H) as f64).sqrt(),
            &scales[..3.min(scales.len())]
        );
    }
    let route_weights: Vec<f32> = buffers
        .download(hip, stream, "weights", T * TOPK * 4)?
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let shared_gate = values(&buffers.download(hip, stream, "shared_gate", T * 2)?);
    let mut expected = vec![0.0f32; T * H];
    for (slot, &global) in ids.iter().enumerate() {
        let token = slot / TOPK;
        let global = global as usize;
        if global / LOCAL != rank as usize {
            continue;
        }
        let gu_name = format!("layer.{layer}.expert.{global}.gate_up.w");
        let down_name =
            format!("model.language_model.layers.{layer}.mlp.experts.{global}.down_proj.weight");
        let gu = record(&reader, &gu_name, 2 * I, H)?;
        let projected = matmul(&x[token * H..(token + 1) * H], 1, H, &values(&gu), 2 * I);
        let act = silu_mul(&projected, I);
        let down = record(&reader, &down_name, H, I)?;
        let y = matmul(&act, 1, I, &values(&down), H);
        for col in 0..H {
            expected[token * H + col] += route_weights[slot] * from_bf16(y[col]);
        }
    }
    if rank == 0 {
        let gu_name = format!("layer.{layer}.mlp.shared_expert.gate_up.w");
        let down_name =
            format!("model.language_model.layers.{layer}.mlp.shared_expert.down_proj.weight");
        let gu = record(shared, &gu_name, 2 * I, H)?;
        let down = record(shared, &down_name, H, I)?;
        for token in 0..T {
            let projected = matmul(&x[token * H..(token + 1) * H], 1, H, &values(&gu), 2 * I);
            let act = silu_mul(&projected, I);
            let y = matmul(&act, 1, I, &values(&down), H);
            let gate = from_bf16(shared_gate[token]);
            let gate = 1.0 / (1.0 + (-gate).exp());
            for col in 0..H {
                expected[token * H + col] += gate * from_bf16(y[col]);
            }
        }
    }
    let expected: Vec<u16> = expected.into_iter().map(bf16).collect();
    let (different, maximum, rms) = diff(&got, &expected, 0.03);
    println!(
        "occamy pack layer{layer} MoE rank{rank} ({}): selected={selected:?}, differing={different}, max_abs={maximum:.6}, rms={rms:.6}",
        if fp8_reader.is_some() { "FP8 W8A8 vs BF16" } else { "BF16" }
    );
    if fp8_reader.is_some() {
        ensure!(rms < 0.05, "layer{layer} rank{rank} FP8 MoE RMS {rms}");
    } else {
        ensure!(different == 0, "layer{layer} rank{rank} MoE failed");
    }
    Ok(MoeOutcome {
        partial: got,
        expected,
    })
}

fn sum_bf16(base: &[u16], parts: &[&[u8]]) -> Vec<u16> {
    base.iter()
        .enumerate()
        .map(|(index, &base)| {
            let mut value = from_bf16(base);
            for part in parts {
                value += from_bf16(u16::from_le_bytes([part[2 * index], part[2 * index + 1]]));
            }
            bf16(value)
        })
        .collect()
}

fn norm_only(
    hip: &HipThread,
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    input: &[u16],
    weight: &[u8],
) -> anyhow::Result<(Vec<u8>, Vec<u16>)> {
    let width = weight.len() / 2;
    let rows = input.len() / width;
    let expected = rms_norm(input, &values(weight), width);
    let mut buffers = Buffers::new();
    buffers.bytes(hip, stream, "x", &bytes(input))?;
    buffers.bytes(hip, stream, "w", weight)?;
    buffers.zero(hip, stream, "zero", input.len() * 2)?;
    buffers.zero(hip, stream, "out", input.len() * 2)?;
    buffers.zero(hip, stream, "res_out", input.len() * 2)?;
    let norm = descriptor(
        "head",
        "saber_add_rms_bf16",
        [rows as u32, 1, 1],
        256,
        abi::SIG_ADD_RMS_BF16,
        vec![
            buffer("zero"),
            buffer("x"),
            buffer("w"),
            buffer("out"),
            buffer("res_out"),
            scalar(width as i64),
            ArgValue::F32 { fval: 1e-6 },
        ],
    )?;
    dispatch(stream, ctx, launcher, &norm, &buffers)?;
    hip.sync_stream(stream)?;
    Ok((
        buffers.download(hip, stream, "out", input.len() * 2)?,
        expected,
    ))
}

fn finish_layer(
    layer: u32,
    hip: &HipThread,
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    rank0: AttentionRank,
    rank1: AttentionRank,
    shared: &PackReader,
    args: &PackProbeArgs,
) -> anyhow::Result<()> {
    ensure!(rank0.input == rank1.input, "layer{layer} input mismatch");
    let residual = sum_bf16(&rank0.input, &[&rank0.partial, &rank1.partial]);
    let expected_residual = sum_bf16(
        &rank0.input,
        &[&bytes(&rank0.expected), &bytes(&rank1.expected)],
    );
    ensure!(
        report(
            &format!("occamy layer{layer} mixer residual"),
            &bytes(&residual),
            &expected_residual,
            0.05,
        ),
        "layer{layer} mixer residual failed"
    );
    let post_weight = vector_record(
        shared,
        &format!("model.language_model.layers.{layer}.post_attention_layernorm.weight"),
        2048,
    )?;
    let (normed, expected_normed) = norm_only(hip, stream, ctx, launcher, &residual, &post_weight)?;
    ensure!(
        report(
            &format!("occamy layer{layer} post-attention norm"),
            &normed,
            &expected_normed,
            0.03,
        ),
        "layer{layer} post-attention norm failed"
    );
    let moe0 = moe_rank(
        hip,
        stream,
        ctx,
        launcher,
        0,
        layer,
        values(&normed),
        shared,
        args,
    )?;
    let moe1 = moe_rank(
        hip,
        stream,
        ctx,
        launcher,
        1,
        layer,
        values(&normed),
        shared,
        args,
    )?;
    let output = sum_bf16(&residual, &[&moe0.partial, &moe1.partial]);
    let expected = sum_bf16(
        &expected_residual,
        &[&bytes(&moe0.expected), &bytes(&moe1.expected)],
    );
    ensure!(
        report(
            &format!("occamy layer{layer} final output"),
            &bytes(&output),
            &expected,
            0.05,
        ),
        "layer{layer} final output failed"
    );
    Ok(())
}

fn layer3(
    hip: &HipThread,
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    shared: &PackReader,
    args: &PackProbeArgs,
) -> anyhow::Result<()> {
    let rank0 = attention_rank(hip, stream, ctx, launcher, 0, shared, args)?;
    let rank1 = attention_rank(hip, stream, ctx, launcher, 1, shared, args)?;
    finish_layer(3, hip, stream, ctx, launcher, rank0, rank1, shared, args)
}

fn layer0(
    hip: &HipThread,
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    shared: &PackReader,
    args: &PackProbeArgs,
) -> anyhow::Result<()> {
    let rank0 = gdn_rank(hip, stream, ctx, launcher, 0, shared, args)?;
    let rank1 = gdn_rank(hip, stream, ctx, launcher, 1, shared, args)?;
    finish_layer(0, hip, stream, ctx, launcher, rank0, rank1, shared, args)
}

pub fn run(hip: &HipThread, args: &PackProbeArgs) -> anyhow::Result<usize> {
    ensure!(
        matches!(
            args.mode.as_str(),
            "dense" | "moe" | "layer0" | "layer3" | "all"
        ),
        "unknown mode"
    );
    hip.set_device(args.device)?;
    let counters = LoadCounters::new();
    let plugins = PluginSet::load_dir(hip, &args.prebuilt, &counters)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;
    let launch_counters = LaunchCounters::new();
    let mut launcher = Launcher::new("occamy-pack-probe", &launch_counters);
    let shared = PackReader::open(&args.pack.join("shared/dense.pack"))?;
    if args.mode == "layer0" {
        layer0(hip, &stream, &ctx, &mut launcher, &shared, args)?;
        return Ok(0);
    }
    if args.mode == "layer3" {
        layer3(hip, &stream, &ctx, &mut launcher, &shared, args)?;
        return Ok(0);
    }
    let mut failed = 0;
    for rank in 0..2 {
        if args.mode == "dense" || args.mode == "all" {
            failed += usize::from(!dense_rank(
                hip,
                &stream,
                &ctx,
                &mut launcher,
                rank,
                &shared,
                args,
            )?);
        }
        if args.mode == "moe" || args.mode == "all" {
            moe_rank(
                hip,
                &stream,
                &ctx,
                &mut launcher,
                rank,
                0,
                random(3 * 2048, 1000),
                &shared,
                args,
            )?;
        }
    }
    Ok(failed)
}
