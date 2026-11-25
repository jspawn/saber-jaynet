use std::process::ExitCode;

use saber_model::kolibri::{spike::kernel_coverage, KolibriConfig, KolibriLayerPackPlan};

fn main() -> ExitCode {
    let mut checkpoint = None;
    let mut tp = 2u32;
    let mut plan_layer = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next();
        match arg.as_str() {
            "--checkpoint" => checkpoint = value(),
            "--tp" => tp = value().and_then(|value| value.parse().ok()).unwrap_or(0),
            "--plan-layer" => {
                plan_layer = value().and_then(|value| value.parse::<u32>().ok());
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: saber-kolibri-spike --checkpoint DIR [--tp 1|2] [--plan-layer N]"
                );
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
        .map_err(|error| error.to_string())
        .and_then(|json| KolibriConfig::from_config_json(&json).map_err(|error| error.to_string()))
    {
        Ok(cfg) => cfg,
        Err(error) => return fail(&error),
    };
    let layer = plan_layer.unwrap_or(0);
    let mut plan = match KolibriLayerPackPlan::new(&cfg, tp, layer) {
        Ok(plan) => plan,
        Err(error) => return fail(&error.to_string()),
    };
    if plan_layer.is_some() {
        plan.add_global_records(&cfg);
        let index: serde_json::Value =
            match std::fs::read_to_string(root.join("model.safetensors.index.json"))
                .map_err(|error| error.to_string())
                .and_then(|json| serde_json::from_str(&json).map_err(|error| error.to_string()))
            {
                Ok(index) => index,
                Err(error) => return fail(&error),
            };
        let prefix = format!("model.layers.{layer}.");
        let names: Vec<_> = match index["weight_map"].as_object() {
            Some(map) => map
                .keys()
                .map(String::as_str)
                .filter(|name| {
                    name.starts_with(&prefix)
                        || matches!(
                            *name,
                            "model.embed_tokens.weight" | "model.norm.weight" | "lm_head.weight"
                        )
                })
                .collect(),
            None => return fail("model.safetensors.index.json has no weight_map"),
        };
        if let Err(error) = plan.validate_tensor_names(names) {
            return fail(&error.to_string());
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
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "producer": "saber-packplan",
                "model": "Aleph-Alpha/Kolibri-1",
                "architecture": "Kolibri1ForCausalLM",
                "tp_size": tp,
                "head": "bf16",
                "layers": 1,
                "records": records,
            }))
            .expect("serialize")
        );
        return ExitCode::SUCCESS;
    }
    println!(
        "Kolibri TP{tp}: {:.2} GiB weights/rank; {:.2} GiB FP8 KV total at 256K; {} records in one layer",
        plan.estimated_model_bytes_per_rank(&cfg) as f64 / 2f64.powi(30),
        cfg.kv_cache_bytes_total(262_144, 1) as f64 / 2f64.powi(30),
        plan.records.len(),
    );
    match kernel_coverage(&cfg, tp) {
        Ok(probes) => {
            for probe in probes {
                match probe.coverage {
                    Ok(kernel) => println!("OK      {}: {kernel}", probe.operation),
                    Err(reason) => println!("NEW     {}: {reason}", probe.operation),
                }
            }
            ExitCode::SUCCESS
        }
        Err(error) => fail(&error),
    }
}

fn fail(message: &str) -> ExitCode {
    eprintln!("saber-kolibri-spike: {message}");
    ExitCode::FAILURE
}
