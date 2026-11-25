#[path = "../kolibri/mod.rs"]
mod kolibri;
#[path = "../ripple/mod.rs"]
mod ripple;

use std::path::PathBuf;

use kolibri::exec::ExecArgs;
use saber_hip::HipThread;

fn main() -> anyhow::Result<()> {
    let mut args = ExecArgs {
        pack: PathBuf::from("/srv/models/saber-models/spike/kolibri-layer0/pack-tp2"),
        checkpoint: PathBuf::from("/srv/models/saber-models/models/Kolibri-1"),
        prebuilt: PathBuf::from("prebuilt/gfx1201"),
        device: 0,
        tokens: vec![1],
        decode: 1,
        max_ctx: 1024,
        max_rows: 16,
        rank: 0,
        peer_port: 0,
        trace_layers: false,
        layers: Some(1),
        layer_offset: 0,
    };
    let mut artifact = PathBuf::from("/srv/models/saber-models/spike/kolibri-layer0");
    let mut hip_lib = "/opt/rocm/lib/libamdhip64.so.7".to_owned();
    let mut iterations = 20u32;
    let mut attention_split_check = false;
    let mut cli = std::env::args().skip(1);
    while let Some(arg) = cli.next() {
        if arg == "--trace-layers" {
            args.trace_layers = true;
            continue;
        }
        if arg == "--attention-split-check" {
            attention_split_check = true;
            continue;
        }
        let value = cli.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--pack" => args.pack = value.into(),
            "--artifact" => artifact = value.into(),
            "--checkpoint" => args.checkpoint = value.into(),
            "--prebuilt" => args.prebuilt = value.into(),
            "--hip-lib" => hip_lib = value,
            "--device" => args.device = value.parse()?,
            "--max-ctx" => args.max_ctx = value.parse()?,
            "--max-rows" => args.max_rows = value.parse()?,
            "--rank" => args.rank = value.parse()?,
            "--peer-port" => args.peer_port = value.parse()?,
            "--layer-offset" => args.layer_offset = value.parse()?,
            "--iterations" => iterations = value.parse()?,
            other => panic!("unknown argument {other}"),
        }
    }
    let hip = HipThread::spawn(&hip_lib)?;
    kolibri::exec::packed_spike(&hip, &args, &artifact, iterations, attention_split_check)
}
