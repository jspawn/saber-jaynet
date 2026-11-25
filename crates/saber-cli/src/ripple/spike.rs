#![allow(clippy::too_many_arguments)]

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{ensure, Context};
use saber_exec::{ArgBindings, LaunchCounters, LaunchCtx, Launcher, LoadCounters, PluginSet};
use saber_hip::{DeviceAddr, DevicePtr, HipThread, Stream, StreamFlags};
use saber_kernels::abi::{self, SigArg};
use saber_kernels::dflash::{self, DflashOp, DflashRules, DflashShape};
use saber_kernels::rules::{ArgValue, LaunchDescriptor, Shape, Slot};
use saber_model::ripple::RippleConfig;

use super::reference::{
    bf16_to_f32, f32_to_bf16, forward_attention, rope_table, AttentionWeights, TensorFile,
};

#[derive(Default)]
struct Buffers {
    device: BTreeMap<String, DevicePtr>,
    external: BTreeMap<String, DeviceAddr>,
}

impl Buffers {
    fn alloc(&mut self, hip: &HipThread, name: &str, bytes: usize) -> anyhow::Result<()> {
        let pointer = hip.malloc(bytes.max(4) as u64)?;
        self.device.insert(name.into(), pointer);
        Ok(())
    }

    fn upload(
        &mut self,
        hip: &HipThread,
        stream: &Stream,
        name: &str,
        bytes: &[u8],
    ) -> anyhow::Result<()> {
        self.alloc(hip, name, bytes.len())?;
        hip.memcpy_h2d_blocking(self.device.get(name).unwrap(), bytes, stream)?;
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
        self.upload(hip, stream, name, &bytes)
    }

    fn alias(&mut self, name: &str, target: &str) -> anyhow::Result<()> {
        let addr = self.addr(target)?;
        self.external.insert(name.into(), addr);
        Ok(())
    }

    fn addr(&self, name: &str) -> anyhow::Result<DeviceAddr> {
        if let Some(pointer) = self.device.get(name) {
            Ok(pointer.addr())
        } else {
            self.external
                .get(name)
                .copied()
                .with_context(|| format!("missing buffer {name}"))
        }
    }

    fn download_bf16(
        &self,
        hip: &HipThread,
        stream: &Stream,
        name: &str,
    ) -> anyhow::Result<Vec<u16>> {
        let pointer = self.device.get(name).unwrap();
        let mut bytes = vec![0u8; pointer.len() as usize];
        hip.memcpy_d2h_blocking(&mut bytes, pointer, stream)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|x| u16::from_le_bytes([x[0], x[1]]))
            .collect())
    }
}

fn buffer(name: &str) -> ArgValue {
    ArgValue::Buffer {
        buffer: name.into(),
    }
}

fn i32_value(value: i32) -> ArgValue {
    ArgValue::I32 {
        ival: i64::from(value),
    }
}

fn f32_value(value: f32) -> ArgValue {
    ArgValue::F32 { fval: value }
}

fn descriptor(
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
        plugin: "ripple_spike".into(),
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

fn dispatch(
    stream: &Stream,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    descriptor: &LaunchDescriptor,
    buffers: &Buffers,
) -> anyhow::Result<()> {
    let mut bindings = ArgBindings::new(Shape::new(0, 0, 0));
    for argument in &descriptor.args {
        if let ArgValue::Buffer { buffer } = &argument.value {
            bindings.bind(buffer, buffers.addr(buffer)?);
        }
    }
    launcher.dispatch_one(ctx, descriptor, &bindings, stream)?;
    Ok(())
}

fn gemm(
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
    dispatch(stream, ctx, launcher, &desc, buffers)
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
    let desc = descriptor(name, grid, block, signature, values)?;
    dispatch(stream, ctx, launcher, &desc, buffers)
}

pub fn run(
    hip: &HipThread,
    checkpoint: &Path,
    prebuilt: &Path,
    device: i32,
    host_only: bool,
) -> anyhow::Result<()> {
    let cfg =
        RippleConfig::from_config_json(&std::fs::read_to_string(checkpoint.join("config.json"))?)?;
    let shard = TensorFile::open(&checkpoint.join("model-00001-of-00006.safetensors"))?;
    let weights = AttentionWeights::load(&shard, 0)?;
    let rows = 2usize;
    let h = cfg.hidden_size as usize;
    let nq = cfg.num_attention_heads as usize;
    let nkv = cfg.num_key_value_heads as usize;
    let positions = vec![0i64, 1];
    let table = rope_table(&cfg, rows);
    let hidden: Vec<_> = (0..rows * h)
        .map(|i| f32_to_bf16((((i * 37) % 257) as i32 - 128) as f32 / 128.0))
        .collect();
    let initial: Vec<_> = (0..rows * h)
        .map(|i| f32_to_bf16((((i * 73 + 11) % 251) as i32 - 125) as f32 / 96.0))
        .collect();
    let started = Instant::now();
    let expected = forward_attention(&cfg, 0, &weights, &hidden, &initial, &positions, &table);
    println!("host reference: {:.3} s", started.elapsed().as_secs_f64());
    ensure!(
        expected.iter().all(|&x| bf16_to_f32(x).is_finite()),
        "non-finite host reference"
    );
    if host_only {
        println!("PASS: host reference completed without GPU access");
        return Ok(());
    }
    hip.set_device(device)?;

    let counters = LoadCounters::new();
    let plugins =
        PluginSet::load_named(hip, prebuilt, &["dflash_bgemm", "ripple_spike"], &counters)?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;
    let launch_counters = LaunchCounters::new();
    let mut launcher = Launcher::new("ripple-spike", &launch_counters);
    let draft = DflashRules::prod_dflash()?;
    let mut b = Buffers::default();

    b.upload_bf16(hip, &stream, "hidden", &hidden)?;
    b.upload_bf16(hip, &stream, "initial", &initial)?;
    b.upload_bf16(hip, &stream, "rope", &table)?;
    b.upload(
        hip,
        &stream,
        "positions",
        &[0i64.to_le_bytes(), 1i64.to_le_bytes()].concat(),
    )?;
    b.upload(
        hip,
        &stream,
        "slots",
        &[0i64.to_le_bytes(), 1i64.to_le_bytes()].concat(),
    )?;
    for (name, values) in [
        ("input_norm", &weights.input_norm),
        ("post_norm", &weights.post_norm),
        ("q_norm", &weights.q_norm),
        ("k_norm", &weights.k_norm),
        ("q_w", &weights.q),
        ("k_w", &weights.k),
        ("v_w", &weights.v),
        ("g_w", &weights.g),
        ("o_w", &weights.o),
    ] {
        b.upload_bf16(hip, &stream, name, values)?;
    }
    b.upload_bf16(hip, &stream, "lambda", &[weights.lambda])?;
    for (name, elements) in [
        ("normed", rows * h),
        ("value_input", rows * h),
        ("q", rows * cfg.q_proj_rows() as usize),
        ("g", rows * cfg.q_proj_rows() as usize),
        ("qn", rows * cfg.q_proj_rows() as usize),
        ("k", rows * cfg.kv_proj_rows() as usize),
        ("kn", rows * cfg.kv_proj_rows() as usize),
        ("v", rows * cfg.kv_proj_rows() as usize),
        ("attention", rows * cfg.q_proj_rows() as usize),
        ("gated", rows * cfg.q_proj_rows() as usize),
        ("o", rows * h),
        ("output", rows * h),
    ] {
        b.alloc(hip, name, elements * 2)?;
    }
    b.alloc(hip, "kc", rows * nkv * 128 * 2)?;
    b.alloc(hip, "vc", rows * nkv * 128 * 2)?;
    hip.sync_stream(&stream)?;

    let started = Instant::now();
    custom(
        "ripple_rmsnorm_bf16",
        [rows as u32, 1, 1],
        256,
        abi::SIG_RIPPLE_RMSNORM,
        vec![
            buffer("hidden"),
            buffer("input_norm"),
            buffer("normed"),
            i32_value(rows as i32),
            i32_value(h as i32),
            f32_value(cfg.rms_norm_eps as f32),
        ],
        &stream,
        &ctx,
        &mut launcher,
        &b,
    )?;
    custom(
        "ripple_value_lerp_bf16",
        [(rows * h).div_ceil(256) as u32, 1, 1],
        256,
        abi::SIG_RIPPLE_VALUE_LERP,
        vec![
            buffer("initial"),
            buffer("normed"),
            buffer("lambda"),
            buffer("value_input"),
            i32_value((rows * h) as i32),
        ],
        &stream,
        &ctx,
        &mut launcher,
        &b,
    )?;
    for (input, weight, output, n) in [
        ("normed", "q_w", "q", cfg.q_proj_rows()),
        ("normed", "k_w", "k", cfg.kv_proj_rows()),
        ("normed", "g_w", "g", cfg.q_proj_rows()),
        ("value_input", "v_w", "v", cfg.kv_proj_rows()),
    ] {
        gemm(
            &draft,
            &stream,
            &ctx,
            &mut launcher,
            &mut b,
            input,
            weight,
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
            buffer("q_norm"),
            buffer("k_norm"),
            buffer("rope"),
            buffer("positions"),
            buffer("qn"),
            buffer("kn"),
            i32_value(rows as i32),
            i32_value(nq as i32),
            i32_value(nkv as i32),
            f32_value(cfg.rms_norm_eps as f32),
            i32_value(1),
        ],
        &stream,
        &ctx,
        &mut launcher,
        &b,
    )?;
    custom(
        "ripple_kv_append_h128_bf16",
        [rows as u32, 1, 1],
        256,
        abi::SIG_RIPPLE_KV_APPEND,
        vec![
            buffer("kn"),
            buffer("v"),
            buffer("kc"),
            buffer("vc"),
            buffer("slots"),
            i32_value(rows as i32),
            i32_value(nkv as i32),
        ],
        &stream,
        &ctx,
        &mut launcher,
        &b,
    )?;
    custom(
        "ripple_attn_h128_bf16",
        [rows as u32, nq as u32, 1],
        32,
        abi::SIG_RIPPLE_ATTN,
        vec![
            buffer("qn"),
            buffer("kc"),
            buffer("vc"),
            buffer("attention"),
            i32_value(rows as i32),
            i32_value(nq as i32),
            i32_value(nkv as i32),
            i32_value(0),
            i32_value(cfg.sliding_window as i32),
        ],
        &stream,
        &ctx,
        &mut launcher,
        &b,
    )?;
    custom(
        "ripple_gate_mul_bf16",
        [
            (rows * cfg.q_proj_rows() as usize).div_ceil(256) as u32,
            1,
            1,
        ],
        256,
        abi::SIG_RIPPLE_GATE_MUL,
        vec![
            buffer("attention"),
            buffer("g"),
            buffer("gated"),
            i32_value((rows * cfg.q_proj_rows() as usize) as i32),
        ],
        &stream,
        &ctx,
        &mut launcher,
        &b,
    )?;
    gemm(
        &draft,
        &stream,
        &ctx,
        &mut launcher,
        &mut b,
        "gated",
        "o_w",
        "o",
        rows as u32,
        cfg.hidden_size,
        cfg.q_proj_rows(),
    )?;
    custom(
        "ripple_postnorm_residual_bf16",
        [rows as u32, 1, 1],
        256,
        abi::SIG_RIPPLE_POSTNORM,
        vec![
            buffer("o"),
            buffer("hidden"),
            buffer("post_norm"),
            buffer("output"),
            i32_value(rows as i32),
            i32_value(h as i32),
            f32_value(cfg.rms_norm_eps as f32),
        ],
        &stream,
        &ctx,
        &mut launcher,
        &b,
    )?;
    hip.sync_stream(&stream)?;
    let gpu_time = started.elapsed().as_secs_f64();
    let actual = b.download_bf16(hip, &stream, "output")?;
    hip.sync_stream(&stream)?;

    let mut different = 0usize;
    let mut max_abs = 0.0f32;
    let mut error2 = 0.0f64;
    let mut expected2 = 0.0f64;
    for (&got, &want) in actual.iter().zip(&expected) {
        let got = bf16_to_f32(got);
        let want = bf16_to_f32(want);
        different += usize::from(got.to_bits() != want.to_bits());
        max_abs = max_abs.max((got - want).abs());
        error2 += f64::from(got - want).powi(2);
        expected2 += f64::from(want).powi(2);
    }
    let relative = (error2 / expected2.max(f64::MIN_POSITIVE)).sqrt();
    println!(
        "GPU attention block: {gpu_time:.3} s; {} / {} BF16 values differ; max abs {max_abs:.6}; relative RMSE {relative:.6}",
        different,
        actual.len()
    );
    ensure!(
        actual.iter().all(|&x| bf16_to_f32(x).is_finite()),
        "non-finite GPU output"
    );
    ensure!(
        relative < 0.02,
        "Ripple attention parity failed: relative RMSE {relative}"
    );
    println!("PASS: real layer-0 Ripple attention block");
    Ok(())
}
