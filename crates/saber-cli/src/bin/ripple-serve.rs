#[path = "../ripple/mod.rs"]
mod ripple;

use std::path::PathBuf;

use ripple::exec::ExecArgs;
use saber_hip::HipThread;

fn main() -> anyhow::Result<()> {
    let mut args = ExecArgs {
        pack: PathBuf::from("/srv/models/saber-models/saber-packs/ripple-65b-preview-mxfp4-tp2"),
        draft: None,
        checkpoint: PathBuf::from("/srv/models/saber-models/models/Ripple-65B-Preview-NVFP4"),
        prebuilt: PathBuf::from("prebuilt/gfx1201"),
        device: 0,
        tokens: vec![1],
        decode: 1,
        max_ctx: 131072,
        max_rows: 64,
        rank: 0,
        peer_port: 0,
        trace_layers: false,
    };
    let mut hip_lib = "/opt/rocm/lib/libamdhip64.so.7".to_owned();
    let mut host = "127.0.0.1".to_owned();
    let mut port = 8000u16;
    let mut served_name = "Ripple-65B-Preview".to_owned();
    let mut cli = std::env::args().skip(1);
    while let Some(arg) = cli.next() {
        if arg == "--trace-layers" {
            args.trace_layers = true;
            continue;
        }
        let value = cli.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--pack" => args.pack = value.into(),
            "--draft" => args.draft = Some(value.into()),
            "--checkpoint" => args.checkpoint = value.into(),
            "--prebuilt" => args.prebuilt = value.into(),
            "--hip-lib" => hip_lib = value,
            "--device" => args.device = value.parse()?,
            "--max-ctx" => args.max_ctx = value.parse()?,
            "--max-rows" => args.max_rows = value.parse()?,
            "--rank" => args.rank = value.parse()?,
            "--peer-port" => args.peer_port = value.parse()?,
            "--host" => host = value,
            "--port" => port = value.parse()?,
            "--served-name" => served_name = value,
            other => panic!("unknown argument {other}"),
        }
    }
    let hip = HipThread::spawn(&hip_lib)?;
    ripple::exec::run_persistent(&hip, &args, &host, port, &served_name)
}
