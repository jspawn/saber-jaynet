use std::process::ExitCode;

use saber_model::checkpoint_head::read_header;
use saber_model::dflash_graph::{DraftGraph, DraftKvCache};
use saber_model::ripple::{
    draft_runtime_pack_records, validate_draft_tensors, RippleConfig, RippleDraftConfig,
};

fn main() -> ExitCode {
    let mut tp = 2u32;
    let mut checkpoint = None;
    let mut target = None;
    let mut max_positions = 65536u32;
    let mut kv_block = 16u32;
    let mut bf16 = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--bf16" {
            bf16 = true;
            continue;
        }
        let Some(value) = args.next() else {
            return fail(&format!("{arg} needs a value"));
        };
        match arg.as_str() {
            "--tp" => match value.parse() {
                Ok(parsed) => tp = parsed,
                Err(_) => return fail("--tp needs a number"),
            },
            "--checkpoint" => checkpoint = Some(value),
            "--target" => target = Some(value),
            "--rope-max-pos" => match value.parse() {
                Ok(parsed) => max_positions = parsed,
                Err(_) => return fail("--rope-max-pos needs a number"),
            },
            "--kv-block" => match value.parse() {
                Ok(parsed) => kv_block = parsed,
                Err(_) => return fail("--kv-block needs a number"),
            },
            other => return fail(&format!("unknown argument {other}")),
        }
    }
    let (Some(checkpoint), Some(target)) = (checkpoint, target) else {
        return fail("--checkpoint DIR and --target DIR are required");
    };
    let target_root = std::path::Path::new(&target);
    let target_cfg = match std::fs::read_to_string(target_root.join("config.json"))
        .map_err(|e| e.to_string())
        .and_then(|json| RippleConfig::from_config_json(&json).map_err(|e| e.to_string()))
    {
        Ok(cfg) => cfg,
        Err(e) => return fail(&e),
    };
    let root = std::path::Path::new(&checkpoint);
    let cfg = match std::fs::read_to_string(root.join("config.json"))
        .map_err(|e| e.to_string())
        .and_then(|json| {
            RippleDraftConfig::from_config_json(&json, &target_cfg).map_err(|e| e.to_string())
        }) {
        Ok(cfg) => cfg,
        Err(e) => return fail(&e),
    };
    let header = match read_header(&root.join("model.safetensors")) {
        Ok(header) => header,
        Err(e) => return fail(&format!("reading drafter header: {e}")),
    };
    let names: Vec<&str> = header
        .as_object()
        .into_iter()
        .flat_map(|map| map.keys())
        .filter(|name| name.as_str() != "__metadata__")
        .map(String::as_str)
        .collect();
    if let Err(e) = validate_draft_tensors(&cfg, tp, max_positions, names) {
        return fail(&e.to_string());
    }
    let records = draft_runtime_pack_records(&cfg, tp, max_positions, bf16);
    let rows: Vec<_> = records
        .iter()
        .map(|record| {
            let params: serde_json::Map<String, serde_json::Value> = record
                .params
                .iter()
                .map(|(key, value)| ((*key).to_owned(), serde_json::json!(value)))
                .collect();
            serde_json::json!({
                "name": record.name,
                "layout_id": record.layout_id,
                "role": record.role.as_str(),
                "op": record.op,
                "dtype": record.dtype.manifest_name(),
                "shard_kind": record.shard.manifest_name(),
                "shape": record.shape,
                "transformed": record.layout_id != 0,
                "params": params,
                "sources": record.sources.iter().map(|source| serde_json::json!({
                    "checkpoint": source.checkpoint.as_str(),
                    "weight": source.weight,
                    "quant": source.quant.as_str(),
                    "scale": source.scale,
                    "global_scale": null,
                    "input_scale": null,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    let graph = DraftGraph::build(cfg.dflash_config(tp));
    let kv: DraftKvCache = graph.kv_cache(kv_block);
    let out = serde_json::json!({
        "producer": "saber-packplan",
        "model": "ricursiveAI/Ripple-65B-Preview-DFlash2",
        "architecture": "DFlash2DraftModel",
        "target_model": "ricursiveAI/Ripple-65B-Preview-NVFP4",
        "tp_size": tp,
        "block_size": cfg.dflash_config.block_size,
        "target_layer_ids": cfg.dflash_config.target_layer_ids,
        "kv_cache": {
            "layers": kv.layers,
            "block_size": kv.block_size,
            "kv_heads": kv.kv_heads,
            "head_dim": kv.head_dim,
            "elem_bytes": kv.elem_bytes,
        },
        "records": rows,
    });
    println!("{}", serde_json::to_string(&out).expect("serialize"));
    ExitCode::SUCCESS
}

fn fail(message: &str) -> ExitCode {
    eprintln!("saber-ripple-draft-packplan: {message}");
    ExitCode::FAILURE
}
