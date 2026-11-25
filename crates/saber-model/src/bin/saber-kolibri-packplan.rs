use std::process::ExitCode;

use saber_model::kolibri::{KolibriConfig, KolibriPackPlan};

fn main() -> ExitCode {
    let mut tp = 2u32;
    let mut checkpoint = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tp" => match args.next().and_then(|v| v.parse().ok()) {
                Some(value) => tp = value,
                None => return fail("--tp needs a number"),
            },
            "--checkpoint" => match args.next() {
                Some(value) => checkpoint = Some(value),
                None => return fail("--checkpoint needs a directory"),
            },
            "-h" | "--help" => {
                eprintln!("usage: saber-kolibri-packplan --checkpoint DIR [--tp 2]");
                return ExitCode::SUCCESS;
            }
            other => return fail(&format!("unknown argument {other}")),
        }
    }
    let Some(checkpoint) = checkpoint else {
        return fail("--checkpoint DIR is required");
    };
    let root = std::path::Path::new(&checkpoint);
    let cfg = match std::fs::read_to_string(root.join("config.json"))
        .map_err(|e| e.to_string())
        .and_then(|json| KolibriConfig::from_config_json(&json).map_err(|e| e.to_string()))
    {
        Ok(cfg) => cfg,
        Err(e) => return fail(&e),
    };
    let plan = match KolibriPackPlan::new(&cfg, tp) {
        Ok(plan) => plan,
        Err(e) => return fail(&e.to_string()),
    };
    let index: serde_json::Value =
        match std::fs::read_to_string(root.join("model.safetensors.index.json"))
            .map_err(|e| e.to_string())
            .and_then(|json| serde_json::from_str(&json).map_err(|e| e.to_string()))
        {
            Ok(index) => index,
            Err(e) => return fail(&format!("reading model.safetensors.index.json: {e}")),
        };
    let names = match index["weight_map"].as_object() {
        Some(map) => map.keys().map(String::as_str).collect::<Vec<_>>(),
        None => return fail("model.safetensors.index.json has no weight_map"),
    };
    if let Err(e) = plan.validate_tensor_names(names) {
        return fail(&e.to_string());
    }
    let records: Vec<_> = plan
        .records
        .iter()
        .map(|record| {
            serde_json::json!({
                "name": record.name,
                "layout_id": record.layout_id,
                "role": record.role.as_str(),
                "op": record.op,
                "dtype": record.dtype.manifest_name(),
                "shard_kind": record.shard.manifest_name(),
                "shape": record.shape,
                "transformed": record.is_transformed(),
                "sources": record.sources.iter().map(|source| serde_json::json!({
                    "weight": source.weight,
                    "quant": source.quant.as_str(),
                    "scale": source.scale,
                    "global_scale": source.global_scale,
                    "input_scale": source.input_scale,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    let out = serde_json::json!({
        "producer": "saber-packplan",
        "model": "Aleph-Alpha/Kolibri-1",
        "architecture": "Kolibri1ForCausalLM",
        "tp_size": tp,
        "head": "bf16",
        "layers": cfg.num_hidden_layers,
        "full_attention_layers": cfg.full_attention_layers(),
        "sliding_window": cfg.sliding_window,
        "bytes_per_rank": plan.bytes_per_rank(),
        "records": records,
    });
    println!("{}", serde_json::to_string(&out).expect("serialize"));
    ExitCode::SUCCESS
}

fn fail(message: &str) -> ExitCode {
    eprintln!("saber-kolibri-packplan: {message}");
    ExitCode::FAILURE
}
