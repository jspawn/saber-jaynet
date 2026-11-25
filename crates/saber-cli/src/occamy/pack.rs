use std::collections::HashSet;
use std::path::Path;

use anyhow::{ensure, Context};
use saber_mem::pack::writer::{write_pack, PackItem};
use saber_mem::pack::{Dtype, PackReader, ShardKind, LAYOUT_RAW};
use saber_model::qwen3_5_moe::pack_plan::{PackPlan, Placement, Record, Shard, Slice, SourceKind};
use saber_model::qwen3_5_moe::Qwen35MoeConfig;
use saber_model::qwen4_exp::st::ShardSet;
use serde_json::json;
use sha2::{Digest, Sha256};

fn config(checkpoint: &Path) -> anyhow::Result<Qwen35MoeConfig> {
    let path = checkpoint.join("config.json");
    let text = std::fs::read_to_string(&path).with_context(|| path.display().to_string())?;
    Qwen35MoeConfig::from_config_json(&text).map_err(|error| anyhow::anyhow!("{error}"))
}

fn validate_headers(plan: &PackPlan, shards: &mut ShardSet) -> anyhow::Result<usize> {
    let mut seen = HashSet::new();
    for record in &plan.records {
        for source in &record.sources {
            if !seen.insert(source.tensor.as_str()) {
                continue;
            }
            let info = shards
                .info(&source.tensor)
                .ok_or_else(|| anyhow::anyhow!("{}: not in a complete shard", source.tensor))?;
            ensure!(
                info.dtype == "BF16" && info.shape == source.shape,
                "{}: checkpoint has {} {:?}, plan expects BF16 {:?}",
                source.tensor,
                info.dtype,
                info.shape,
                source.shape
            );
        }
    }
    Ok(seen.len())
}

fn extract(source: &Slice, raw: &[u8]) -> anyhow::Result<Vec<u8>> {
    let columns = source.shape[1..].iter().product::<u64>().max(1);
    ensure!(
        source.rows.0 <= source.rows.1
            && source.rows.1 <= source.shape[0]
            && source.cols.0 <= source.cols.1
            && source.cols.1 <= columns,
        "{}: invalid rows {:?} or columns {:?} for {:?}",
        source.tensor,
        source.rows,
        source.cols,
        source.shape
    );
    ensure!(
        raw.len() as u64 == source.shape.iter().product::<u64>() * 2,
        "{}: {} payload bytes for {:?}",
        source.tensor,
        raw.len(),
        source.shape
    );
    let width = (source.cols.1 - source.cols.0) as usize * 2;
    let mut output = Vec::with_capacity((source.rows.1 - source.rows.0) as usize * width);
    for row in source.rows.0..source.rows.1 {
        let start = (row * columns + source.cols.0) as usize * 2;
        output.extend_from_slice(&raw[start..start + width]);
    }
    Ok(output)
}

fn payload(record: &Record, shards: &mut ShardSet) -> anyhow::Result<Vec<u8>> {
    let mut output = Vec::with_capacity(record.bytes as usize);
    for source in &record.sources {
        let (info, raw) = shards
            .read(&source.tensor)
            .map_err(|error| anyhow::anyhow!(error))?;
        ensure!(info.dtype == "BF16", "{}: not BF16", source.tensor);
        let bytes = extract(source, &raw)?;
        if record.kind == SourceKind::F32FromBf16 {
            output.extend(bytes.chunks_exact(2).flat_map(|value| {
                (u32::from(u16::from_le_bytes([value[0], value[1]])) << 16).to_le_bytes()
            }));
        } else {
            output.extend_from_slice(&bytes);
        }
    }
    ensure!(
        output.len() as u64 == record.bytes,
        "{}: produced {} bytes, expected {}",
        record.name,
        output.len(),
        record.bytes
    );
    Ok(output)
}

fn member(
    plan: &PackPlan,
    placement: Placement,
    shards: &mut ShardSet,
) -> anyhow::Result<Vec<PackItem>> {
    plan.records
        .iter()
        .filter(|record| record.placement == placement)
        .map(|record| {
            Ok(PackItem {
                name: record.name.clone(),
                dtype: match record.dtype {
                    "BF16" => Dtype::BF16,
                    "F32" => Dtype::F32,
                    other => anyhow::bail!("unsupported record dtype {other}"),
                },
                shape: record.shape.clone(),
                shard_kind: match record.shard {
                    Shard::Replicated => ShardKind::Replicated,
                    Shard::Dim0 => ShardKind::Dim0,
                    Shard::Dim1 => ShardKind::Dim1,
                    Shard::Dim0Segments => ShardKind::Dim0Segments,
                },
                layout_id: LAYOUT_RAW,
                data: payload(record, shards)?,
            })
        })
        .collect()
}

pub fn run(
    checkpoint: &Path,
    out: &Path,
    dry_run: bool,
    all_layers: bool,
    globals: bool,
) -> anyhow::Result<()> {
    ensure!(
        !(all_layers && globals),
        "--all-layers and --globals are mutually exclusive"
    );
    let config = config(checkpoint)?;
    let plan = if all_layers {
        PackPlan::all_layers(&config)
    } else if globals {
        PackPlan::globals(&config)
    } else {
        PackPlan::spike(&config)
    }
    .map_err(|error| anyhow::anyhow!(error))?;
    let mut shards = ShardSet::open(checkpoint).map_err(|error| anyhow::anyhow!(error))?;
    let checked = validate_headers(&plan, &mut shards)?;
    for placement in [Placement::Rank(0), Placement::Rank(1), Placement::Shared] {
        let records: Vec<_> = plan
            .records
            .iter()
            .filter(|record| record.placement == placement)
            .collect();
        let bytes: u64 = records.iter().map(|record| record.bytes).sum();
        println!(
            "occamy-pack {}: {} records, {:.3} GiB",
            placement.dir(),
            records.len(),
            bytes as f64 / (1u64 << 30) as f64
        );
    }
    println!("occamy-pack: {checked} source tensors match the BF16 checkpoint headers");
    if dry_run {
        println!("occamy-pack: dry run; no files written");
        return Ok(());
    }

    ensure!(
        !out.exists(),
        "{}: output exists; refusing to overwrite",
        out.display()
    );
    let plan_json = serde_json::to_vec_pretty(&plan.to_json())?;
    let plan_hash: [u8; 32] = Sha256::digest(&plan_json).into();
    let tier_name = if all_layers {
        "occamy-bf16-all-layers"
    } else if globals {
        "occamy-bf16-globals"
    } else {
        "occamy-bf16-l0-l3"
    };
    let tier_id: [u8; 32] = Sha256::new()
        .chain_update(tier_name.as_bytes())
        .chain_update([0])
        .chain_update(&plan_json)
        .finalize()
        .into();
    std::fs::create_dir_all(out)?;

    let mut members = Vec::new();
    for placement in [Placement::Rank(0), Placement::Rank(1), Placement::Shared] {
        let directory = out.join(placement.dir());
        std::fs::create_dir(&directory)?;
        let path = directory.join("dense.pack");
        let temporary = directory.join("dense.pack.tmp");
        let items = member(&plan, placement, &mut shards)?;
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
        println!("occamy-pack: verified {} ({bytes} bytes)", path.display());
        members.push(
            json!({"placement": placement.dir(), "file": path, "records": count,
                            "bytes": bytes}),
        );
    }
    std::fs::write(out.join("plan.json"), &plan_json)?;
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&json!({
            "model": "qwen3_5_moe", "source": checkpoint, "layers": plan.layers,
            "tp": 2, "ep": 2, "quantization": "BF16", "mtp": false, "vision": false,
            "plan_sha256": hex::encode(plan_hash), "tier_id": hex::encode(tier_id),
            "members": members,
        }))?,
    )?;
    println!("occamy-pack: complete -> {}", out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_copies_row_and_column_slices() {
        let raw: Vec<u8> = (0..24).collect();
        let slice = Slice {
            tensor: "tiny".into(),
            shape: vec![3, 4],
            rows: (1, 3),
            cols: (1, 3),
        };
        assert_eq!(
            extract(&slice, &raw).unwrap(),
            vec![10, 11, 12, 13, 18, 19, 20, 21]
        );
    }

    #[test]
    fn extract_rejects_an_out_of_range_slice() {
        let slice = Slice {
            tensor: "tiny".into(),
            shape: vec![2, 2],
            rows: (0, 2),
            cols: (1, 3),
        };
        assert!(extract(&slice, &[0; 8]).is_err());
    }
}
