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
use saber_model::kolibri::KolibriConfig;

use super::super::ripple::reference::{bf16_to_f32, rope_table_values, TensorFile};

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
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        self.upload(hip, stream, name, &bytes)
    }

    fn alias(&mut self, name: &str, target: &str) -> anyhow::Result<()> {
        let address = self.addr(target)?;
        self.external.insert(name.into(), address);
        Ok(())
    }

    fn alias_offset(&mut self, name: &str, target: &str, bytes: u64) -> anyhow::Result<()> {
        let address = self
            .addr(target)?
            .offset(bytes)
            .context("device address overflow")?;
        self.external.insert(name.into(), address);
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

    fn download(&self, hip: &HipThread, stream: &Stream, name: &str) -> anyhow::Result<Vec<u8>> {
        let pointer = self.device.get(name).unwrap();
        let mut bytes = vec![0u8; pointer.len() as usize];
        hip.memcpy_d2h_blocking(&mut bytes, pointer, stream)?;
        Ok(bytes)
    }

    fn download_bf16(
        &self,
        hip: &HipThread,
        stream: &Stream,
        name: &str,
    ) -> anyhow::Result<Vec<u16>> {
        Ok(self
            .download(hip, stream, name)?
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
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
    let descriptor = dflash::resolve(op, DflashShape::new(rows, n, k), rules)?;
    dispatch(stream, ctx, launcher, &descriptor, buffers)
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
    let descriptor = descriptor(plugin, name, grid, block, signature, values)?;
    dispatch(stream, ctx, launcher, &descriptor, buffers)
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

pub fn run(
    hip: &HipThread,
    checkpoint: &Path,
    artifact: &Path,
    prebuilt: &Path,
    device: i32,
    iterations: u32,
    host_only: bool,
) -> anyhow::Result<()> {
    ensure!(iterations > 0, "--iterations must be positive");
    let cfg =
        KolibriConfig::from_config_json(&std::fs::read_to_string(checkpoint.join("config.json"))?)?;
    let metadata: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(artifact.join("metadata.json"))?)?;
    let layer = metadata["layer"].as_u64().unwrap() as u32;
    ensure!(
        layer < cfg.num_hidden_layers,
        "artifact layer outside model"
    );
    let selected: Vec<u32> = metadata["selected_experts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as u32)
        .collect();
    let file = TensorFile::open(&artifact.join("layer.safetensors"))?;
    let hidden = file.read_bf16("input")?;
    let expected_after = file.read_bf16("after_attention")?;
    let expected_final = file.read_bf16("expected")?;
    let expected_ids = file.read_i32("router_ids")?;
    let expected_router_weights = file.read_f32("router_weights")?;
    let rows = hidden.len() / cfg.hidden_size as usize;
    let h = cfg.hidden_size as usize;
    let nq = cfg.num_attention_heads as usize;
    let nkv = cfg.num_key_value_heads as usize;
    let q_width = cfg.q_proj_rows() as usize;
    let kv_width = cfg.kv_proj_rows() as usize;
    let intermediate = cfg.moe_intermediate_size as usize;
    ensure!(
        expected_after.len() == rows * h && expected_final.len() == rows * h,
        "artifact shapes"
    );
    ensure!(
        expected_ids.len() == rows * cfg.num_experts_per_tok as usize,
        "router shape"
    );
    println!(
        "Kolibri layer {layer}: {rows} rows, {} selected experts, {:.1} MiB artifact",
        selected.len(),
        std::fs::metadata(artifact.join("layer.safetensors"))?.len() as f64 / 2f64.powi(20)
    );
    if host_only {
        println!("PASS: artifact and configuration validated without GPU access");
        return Ok(());
    }
    hip.set_device(device)?;
    let counters = LoadCounters::new();
    let plugins = PluginSet::load_named(
        hip,
        prebuilt,
        &["dflash_bgemm", "ripple_spike", "kolibri_spike"],
        &counters,
    )?;
    let functions = plugins.functions(hip)?;
    let ctx = LaunchCtx::new(hip, &functions);
    let stream = hip.create_stream(StreamFlags::NonBlocking)?;
    let launch_counters = LaunchCounters::new();
    let mut launcher = Launcher::new("kolibri-spike", &launch_counters);
    let rules = DflashRules::prod_dflash()?;
    let mut buffers = Buffers::default();
    buffers.upload_bf16(hip, &stream, "hidden", &hidden)?;
    let mut tensor_names = vec![
        "input_norm",
        "post_attn_norm",
        "post_attention_norm",
        "post_ffn_norm",
        "q_norm",
        "k_norm",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "router_weight",
    ];
    for projection in ["gate_proj", "up_proj", "down_proj"] {
        tensor_names.push(match projection {
            "gate_proj" => "shared.gate_proj",
            "up_proj" => "shared.up_proj",
            _ => "shared.down_proj",
        });
    }
    for name in tensor_names {
        buffers.upload_bf16(hip, &stream, name, &file.read_bf16(name)?)?;
    }
    for &expert in &selected {
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            let source = format!("expert.{expert}.{projection}");
            let target = format!("e{expert}.{projection}");
            buffers.upload_bf16(hip, &stream, &target, &file.read_bf16(&source)?)?;
        }
    }
    let bias: Vec<u8> = file
        .read_f32("router_bias")?
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    buffers.upload(hip, &stream, "router_bias", &bias)?;
    let table = rope_table_values(cfg.rope_theta, cfg.head_dim, rows);
    buffers.upload_bf16(hip, &stream, "rope", &table)?;
    let positions: Vec<u8> = (0..rows as i64)
        .flat_map(|value| value.to_le_bytes())
        .collect();
    buffers.upload(hip, &stream, "positions", &positions)?;
    buffers.upload(hip, &stream, "slots", &positions)?;
    for (name, elements) in [
        ("normed", rows * h),
        ("q", rows * q_width),
        ("k", rows * kv_width),
        ("v", rows * kv_width),
        ("qn", rows * q_width),
        ("kn", rows * kv_width),
        ("attention", rows * q_width),
        ("o", rows * h),
        ("after", rows * h),
        ("moe_input", rows * h),
        ("routed", rows * h),
        ("gate", rows * intermediate),
        ("up", rows * intermediate),
        ("intermediate", rows * intermediate),
        ("down", rows * h),
        ("output", rows * h),
    ] {
        buffers.alloc(hip, name, elements * 2)?;
    }
    buffers.alloc(hip, "kc", rows * nkv * 128 * 2)?;
    buffers.alloc(hip, "vc", rows * nkv * 128 * 2)?;
    buffers.alloc(
        hip,
        "router_ids",
        rows * cfg.num_experts_per_tok as usize * 4,
    )?;
    buffers.alloc(
        hip,
        "router_weights",
        rows * cfg.num_experts_per_tok as usize * 4,
    )?;
    hip.sync_stream(&stream)?;

    let mut forward = || -> anyhow::Result<()> {
        custom(
            "ripple_spike",
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
            &buffers,
        )?;
        for (input, weight, output, n) in [
            ("normed", "q_proj", "q", cfg.q_proj_rows()),
            ("normed", "k_proj", "k", cfg.kv_proj_rows()),
            ("normed", "v_proj", "v", cfg.kv_proj_rows()),
        ] {
            gemm(
                &rules,
                &stream,
                &ctx,
                &mut launcher,
                &mut buffers,
                input,
                weight,
                output,
                rows as u32,
                n,
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
                i32_value(i32::from(cfg.uses_rope(layer))),
            ],
            &stream,
            &ctx,
            &mut launcher,
            &buffers,
        )?;
        custom(
            "ripple_spike",
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
            &buffers,
        )?;
        custom(
            "ripple_spike",
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
                i32_value(if cfg.uses_rope(layer) {
                    cfg.sliding_window as i32
                } else {
                    0
                }),
            ],
            &stream,
            &ctx,
            &mut launcher,
            &buffers,
        )?;
        gemm(
            &rules,
            &stream,
            &ctx,
            &mut launcher,
            &mut buffers,
            "attention",
            "o_proj",
            "o",
            rows as u32,
            cfg.hidden_size,
            cfg.q_proj_rows(),
        )?;
        custom(
            "ripple_spike",
            "ripple_postnorm_residual_bf16",
            [rows as u32, 1, 1],
            256,
            abi::SIG_RIPPLE_POSTNORM,
            vec![
                buffer("o"),
                buffer("hidden"),
                buffer("post_attn_norm"),
                buffer("after"),
                i32_value(rows as i32),
                i32_value(h as i32),
                f32_value(cfg.rms_norm_eps as f32),
            ],
            &stream,
            &ctx,
            &mut launcher,
            &buffers,
        )?;
        custom(
            "ripple_spike",
            "ripple_rmsnorm_bf16",
            [rows as u32, 1, 1],
            256,
            abi::SIG_RIPPLE_RMSNORM,
            vec![
                buffer("after"),
                buffer("post_attention_norm"),
                buffer("moe_input"),
                i32_value(rows as i32),
                i32_value(h as i32),
                f32_value(cfg.rms_norm_eps as f32),
            ],
            &stream,
            &ctx,
            &mut launcher,
            &buffers,
        )?;
        custom(
            "kolibri_spike",
            "kolibri_router_bf16",
            [rows as u32, 1, 1],
            256,
            abi::SIG_KOLIBRI_ROUTER,
            vec![
                buffer("moe_input"),
                buffer("router_weight"),
                buffer("router_bias"),
                buffer("router_ids"),
                buffer("router_weights"),
                i32_value(rows as i32),
                i32_value(h as i32),
                i32_value(cfg.num_experts as i32),
                i32_value(cfg.num_experts_per_tok as i32),
            ],
            &stream,
            &ctx,
            &mut launcher,
            &buffers,
        )?;
        custom(
            "ripple_spike",
            "ripple_scale_bf16",
            [(rows * h).div_ceil(256) as u32, 1, 1],
            256,
            abi::SIG_RIPPLE_SCALE,
            vec![
                buffer("routed"),
                i32_value((rows * h) as i32),
                f32_value(0.0),
            ],
            &stream,
            &ctx,
            &mut launcher,
            &buffers,
        )?;
        for row in 0..rows {
            let row_name = format!("moe_row_{row}");
            let accumulated = format!("routed_row_{row}");
            buffers.alias_offset(&row_name, "moe_input", (row * h * 2) as u64)?;
            buffers.alias_offset(&accumulated, "routed", (row * h * 2) as u64)?;
            for slot in 0..cfg.num_experts_per_tok as usize {
                let expert = expected_ids[row * cfg.num_experts_per_tok as usize + slot] as u32;
                let gate = format!("e{expert}.gate_proj");
                let up = format!("e{expert}.up_proj");
                let down = format!("e{expert}.down_proj");
                gemm(
                    &rules,
                    &stream,
                    &ctx,
                    &mut launcher,
                    &mut buffers,
                    &row_name,
                    &gate,
                    "gate",
                    1,
                    cfg.moe_intermediate_size,
                    cfg.hidden_size,
                )?;
                gemm(
                    &rules,
                    &stream,
                    &ctx,
                    &mut launcher,
                    &mut buffers,
                    &row_name,
                    &up,
                    "up",
                    1,
                    cfg.moe_intermediate_size,
                    cfg.hidden_size,
                )?;
                custom(
                    "kolibri_spike",
                    "kolibri_swiglu_bf16",
                    [intermediate.div_ceil(256) as u32, 1, 1],
                    256,
                    abi::SIG_KOLIBRI_SWIGLU,
                    vec![
                        buffer("gate"),
                        buffer("up"),
                        buffer("intermediate"),
                        i32_value(intermediate as i32),
                    ],
                    &stream,
                    &ctx,
                    &mut launcher,
                    &buffers,
                )?;
                gemm(
                    &rules,
                    &stream,
                    &ctx,
                    &mut launcher,
                    &mut buffers,
                    "intermediate",
                    &down,
                    "down",
                    1,
                    cfg.hidden_size,
                    cfg.moe_intermediate_size,
                )?;
                custom(
                    "kolibri_spike",
                    "kolibri_accumulate_bf16",
                    [h.div_ceil(256) as u32, 1, 1],
                    256,
                    abi::SIG_KOLIBRI_ACCUMULATE,
                    vec![
                        buffer(&accumulated),
                        buffer("down"),
                        f32_value(
                            expected_router_weights[row * cfg.num_experts_per_tok as usize + slot],
                        ),
                        i32_value(h as i32),
                    ],
                    &stream,
                    &ctx,
                    &mut launcher,
                    &buffers,
                )?;
            }
        }
        gemm(
            &rules,
            &stream,
            &ctx,
            &mut launcher,
            &mut buffers,
            "moe_input",
            "shared.gate_proj",
            "gate",
            rows as u32,
            cfg.shared_expert_intermediate_size,
            cfg.hidden_size,
        )?;
        gemm(
            &rules,
            &stream,
            &ctx,
            &mut launcher,
            &mut buffers,
            "moe_input",
            "shared.up_proj",
            "up",
            rows as u32,
            cfg.shared_expert_intermediate_size,
            cfg.hidden_size,
        )?;
        custom(
            "kolibri_spike",
            "kolibri_swiglu_bf16",
            [(rows * intermediate).div_ceil(256) as u32, 1, 1],
            256,
            abi::SIG_KOLIBRI_SWIGLU,
            vec![
                buffer("gate"),
                buffer("up"),
                buffer("intermediate"),
                i32_value((rows * intermediate) as i32),
            ],
            &stream,
            &ctx,
            &mut launcher,
            &buffers,
        )?;
        gemm(
            &rules,
            &stream,
            &ctx,
            &mut launcher,
            &mut buffers,
            "intermediate",
            "shared.down_proj",
            "down",
            rows as u32,
            cfg.hidden_size,
            cfg.shared_expert_intermediate_size,
        )?;
        custom(
            "kolibri_spike",
            "kolibri_accumulate_bf16",
            [(rows * h).div_ceil(256) as u32, 1, 1],
            256,
            abi::SIG_KOLIBRI_ACCUMULATE,
            vec![
                buffer("routed"),
                buffer("down"),
                f32_value(1.0),
                i32_value((rows * h) as i32),
            ],
            &stream,
            &ctx,
            &mut launcher,
            &buffers,
        )?;
        custom(
            "ripple_spike",
            "ripple_postnorm_residual_bf16",
            [rows as u32, 1, 1],
            256,
            abi::SIG_RIPPLE_POSTNORM,
            vec![
                buffer("routed"),
                buffer("after"),
                buffer("post_ffn_norm"),
                buffer("output"),
                i32_value(rows as i32),
                i32_value(h as i32),
                f32_value(cfg.rms_norm_eps as f32),
            ],
            &stream,
            &ctx,
            &mut launcher,
            &buffers,
        )?;
        Ok(())
    };

    let warmup = Instant::now();
    forward()?;
    hip.sync_stream(&stream)?;
    println!("warmup forward: {:.3} s", warmup.elapsed().as_secs_f64());
    let started = Instant::now();
    for _ in 0..iterations {
        forward()?;
    }
    hip.sync_stream(&stream)?;
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "unoptimized dispatch: {:.3} ms/layer over {iterations} iterations (not model tok/s)",
        elapsed * 1000.0 / f64::from(iterations)
    );
    drop(forward);
    let actual_after = buffers.download_bf16(hip, &stream, "after")?;
    let actual_final = buffers.download_bf16(hip, &stream, "output")?;
    let actual_ids: Vec<i32> = buffers
        .download(hip, &stream, "router_ids")?
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let actual_router_weights: Vec<f32> = buffers
        .download(hip, &stream, "router_weights")?
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    hip.sync_stream(&stream)?;
    let (attention_error, attention_max) = relative_rmse(&actual_after, &expected_after);
    let (final_error, final_max) = relative_rmse(&actual_final, &expected_final);
    let router_max = actual_router_weights
        .iter()
        .zip(&expected_router_weights)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    println!("attention: relative RMSE {attention_error:.6}, max abs {attention_max:.6}");
    println!(
        "router: IDs exact {}, max weight abs {router_max:.9}",
        actual_ids == expected_ids
    );
    println!("full layer: relative RMSE {final_error:.6}, max abs {final_max:.6}");
    ensure!(attention_error < 0.02, "attention parity failed");
    ensure!(
        actual_ids == expected_ids,
        "router selected different experts"
    );
    ensure!(
        router_max < 0.0005,
        "router weight parity failed: FP32 GEMM accumulation difference exceeds 0.0005"
    );
    ensure!(final_error < 0.02, "full-layer parity failed");
    println!("PASS: real-weight Kolibri layer-{layer} attention, router, MoE and norms");
    Ok(())
}
