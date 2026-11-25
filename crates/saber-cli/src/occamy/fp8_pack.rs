use std::path::Path;

use anyhow::{ensure, Context};
use saber_mem::pack::writer::{write_pack, PackItem};
use saber_mem::pack::{Dtype, PackReader, ShardKind, LAYOUT_RAW};
use saber_model::qwen3_5_moe::pack_plan::SPIKE_LAYERS;
use saber_model::qwen3_5_moe::{QuantConfig, Qwen35MoeConfig};
use saber_model::qwen4_exp::st::ShardSet;
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::flashnext::mtp8::preshuffle;

const H: usize = 2048;
const I: usize = 512;
const LOCAL_EXPERTS: u32 = 128;

fn read(shards: &mut ShardSet, name: &str, dtype: &str, shape: &[u64]) -> anyhow::Result<Vec<u8>> {
    let (info, bytes) = shards.read(name).map_err(|error| anyhow::anyhow!(error))?;
    ensure!(
        info.dtype == dtype && info.shape == shape,
        "{name}: checkpoint has {} {:?}, expected {dtype} {shape:?}",
        info.dtype,
        info.shape
    );
    Ok(bytes)
}

fn check_headers(shards: &mut ShardSet, layers: &[u32], experts: u32) -> anyhow::Result<()> {
    for &layer in layers {
        for expert in 0..experts {
            let prefix = format!("model.language_model.layers.{layer}.mlp.experts.{expert}");
            for (projection, n, k) in [("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I)]
            {
                let weight = format!("{prefix}.{projection}.weight");
                let scales = format!("{weight}_scale_inv");
                for (name, dtype, shape) in [
                    (&weight, "F8_E4M3", vec![n as u64, k as u64]),
                    (&scales, "F32", vec![(n / 128) as u64, (k / 128) as u64]),
                ] {
                    let info = shards.info(name).ok_or_else(|| {
                        anyhow::anyhow!("{name}: not in a complete downloaded shard")
                    })?;
                    ensure!(
                        info.dtype == dtype && info.shape == shape,
                        "{name}: checkpoint has {} {:?}, expected {dtype} {shape:?}",
                        info.dtype,
                        info.shape
                    );
                }
            }
        }
    }
    Ok(())
}

fn expert(shards: &mut ShardSet, layer: u32, global: u32) -> anyhow::Result<[PackItem; 4]> {
    let prefix = format!("model.language_model.layers.{layer}.mlp.experts.{global}");
    let mut gate_up = Vec::with_capacity(2 * I * H);
    let mut gate_scales = Vec::with_capacity(2 * (I / 128) * (H / 128) * 4);
    for projection in ["gate_proj", "up_proj"] {
        gate_up.extend(read(
            shards,
            &format!("{prefix}.{projection}.weight"),
            "F8_E4M3",
            &[I as u64, H as u64],
        )?);
        gate_scales.extend(read(
            shards,
            &format!("{prefix}.{projection}.weight_scale_inv"),
            "F32",
            &[(I / 128) as u64, (H / 128) as u64],
        )?);
    }
    let down = read(
        shards,
        &format!("{prefix}.down_proj.weight"),
        "F8_E4M3",
        &[H as u64, I as u64],
    )?;
    let down_scales = read(
        shards,
        &format!("{prefix}.down_proj.weight_scale_inv"),
        "F32",
        &[(H / 128) as u64, (I / 128) as u64],
    )?;
    let item = |name: String, dtype, shape: Vec<u64>, data| PackItem {
        name,
        dtype,
        shape,
        shard_kind: ShardKind::Dim0,
        layout_id: LAYOUT_RAW,
        data,
    };
    Ok([
        item(
            format!("layer.{layer}.expert.{global}.gate_up.w8"),
            Dtype::F8E4M3,
            vec![(2 * I) as u64, H as u64],
            preshuffle(&gate_up, 2 * I, H),
        ),
        item(
            format!("layer.{layer}.expert.{global}.gate_up.s8"),
            Dtype::F32,
            vec![(2 * I / 128) as u64, (H / 128) as u64],
            gate_scales,
        ),
        item(
            format!("{prefix}.down_proj.w8"),
            Dtype::F8E4M3,
            vec![H as u64, I as u64],
            preshuffle(&down, H, I),
        ),
        item(
            format!("{prefix}.down_proj.s8"),
            Dtype::F32,
            vec![(H / 128) as u64, (I / 128) as u64],
            down_scales,
        ),
    ])
}

pub fn run(checkpoint: &Path, out: &Path, dry_run: bool, all_layers: bool) -> anyhow::Result<()> {
    let config_path = checkpoint.join("config.json");
    let config_text =
        std::fs::read_to_string(&config_path).with_context(|| config_path.display().to_string())?;
    let config = Qwen35MoeConfig::from_config_json(&config_text)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    ensure!(
        config.quant == QuantConfig::Fp8Block128,
        "Occamy FP8 packing requires the official block-128 E4M3 checkpoint"
    );
    let layers = if all_layers {
        (0..config.num_layers()).collect::<Vec<_>>()
    } else {
        SPIKE_LAYERS.to_vec()
    };
    let mut shards = ShardSet::open(checkpoint).map_err(|error| anyhow::anyhow!(error))?;
    check_headers(&mut shards, &layers, config.moe.num_experts)?;
    let per_rank = layers.len() as u64 * u64::from(LOCAL_EXPERTS) * 4;
    let bytes_per_rank = layers.len() as u64
        * u64::from(LOCAL_EXPERTS)
        * ((2 * I * H + 2 * (I / 128) * (H / 128) * 4 + H * I + (H / 128) * (I / 128) * 4) as u64);
    println!(
        "occamy-fp8-pack: {} layers, {per_rank} records and {:.3} GiB per rank; official F8_E4M3/F32 block-128 tensors verified",
        layers.len(),
        bytes_per_rank as f64 / (1u64 << 30) as f64
    );
    if dry_run {
        println!("occamy-fp8-pack: dry run; no files written");
        return Ok(());
    }
    ensure!(
        !out.exists(),
        "{}: output exists; refusing to overwrite",
        out.display()
    );
    let plan_json = serde_json::to_vec_pretty(&json!({
        "model": "qwen3_5_moe",
        "source": checkpoint,
        "layers": layers,
        "layout": {"tp": 2, "ep": 2, "routed_experts_per_rank": LOCAL_EXPERTS},
        "quantization": "official F8_E4M3 weights and F32 [128,128] block scales; WMMA preshuffle",
    }))?;
    let plan_hash: [u8; 32] = Sha256::digest(&plan_json).into();
    let tier_id: [u8; 32] = Sha256::new()
        .chain_update(b"occamy-fp8-experts")
        .chain_update([0])
        .chain_update(&plan_json)
        .finalize()
        .into();
    std::fs::create_dir_all(out)?;
    let mut members = Vec::new();
    for rank in 0..2u32 {
        let directory = out.join(format!("rank{rank}"));
        std::fs::create_dir(&directory)?;
        let path = directory.join("dense.pack");
        let temporary = directory.join("dense.pack.tmp");
        let mut items = Vec::with_capacity(per_rank as usize);
        for &layer in &layers {
            for global in rank * LOCAL_EXPERTS..(rank + 1) * LOCAL_EXPERTS {
                items.extend(expert(&mut shards, layer, global)?);
            }
        }
        let count = items.len();
        let bytes = write_pack(&temporary, 2, 0, plan_hash, tier_id, &items)?;
        drop(items);
        std::fs::rename(&temporary, &path)?;
        let reader = PackReader::open(&path)?;
        ensure!(
            reader.records().len() == count,
            "{}: record count",
            path.display()
        );
        reader.verify_all()?;
        println!(
            "occamy-fp8-pack: verified {} ({bytes} bytes)",
            path.display()
        );
        members.push(json!({"rank": rank, "file": path, "records": count, "bytes": bytes}));
    }
    std::fs::write(out.join("plan.json"), &plan_json)?;
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&json!({
            "model": "qwen3_5_moe", "source": checkpoint, "layers": layers,
            "tp": 2, "ep": 2, "quantization": "FP8_E4M3_BLOCK128_W8A8",
            "dense_and_shared": "use the existing BF16 packs", "kv": "BF16",
            "plan_sha256": hex::encode(plan_hash), "tier_id": hex::encode(tier_id),
            "members": members,
        }))?,
    )?;
    println!("occamy-fp8-pack: complete -> {}", out.display());
    Ok(())
}
