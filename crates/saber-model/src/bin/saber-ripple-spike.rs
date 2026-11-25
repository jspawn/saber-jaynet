use std::process::ExitCode;

use saber_model::ripple::{spike::kernel_coverage, RippleConfig, RipplePackPlan};

fn main() -> ExitCode {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        eprintln!("usage: saber-ripple-spike CHECKPOINT_DIR");
        return ExitCode::FAILURE;
    }
    let json = match std::fs::read_to_string(format!("{}/config.json", args[0])) {
        Ok(json) => json,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let cfg = match RippleConfig::from_config_json(&json) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let plan = match RipplePackPlan::new(&cfg, 2) {
        Ok(plan) => plan,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "Ripple TP2 pack: {:.2} GiB across both ranks; 64K FP8 target KV: {:.2} GiB total",
        2.0 * plan.bytes_per_rank() as f64 / 2f64.powi(30),
        cfg.kv_cache_bytes_total(65536, 1) as f64 / 2f64.powi(30)
    );
    match kernel_coverage(&cfg, 2) {
        Ok(probes) => {
            for probe in probes {
                match probe.coverage {
                    Ok(kernel) => println!("OK      {}: {kernel}", probe.operation),
                    Err(reason) => println!("MISSING {}: {reason}", probe.operation),
                }
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
