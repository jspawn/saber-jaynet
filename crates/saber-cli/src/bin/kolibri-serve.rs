#[path = "../kolibri/mod.rs"]
mod kolibri;
#[path = "../ripple/mod.rs"]
mod ripple;

use std::path::PathBuf;

use kolibri::exec::ExecArgs;
use saber_hip::HipThread;

fn main() -> anyhow::Result<()> {
    let mut args = ExecArgs {
        pack: PathBuf::from("/srv/models/saber-models/saber-packs/kolibri-1-mxfp4-tp2"),
        checkpoint: PathBuf::from("/srv/models/saber-models/models/Kolibri-1-BF16"),
        prebuilt: PathBuf::from("prebuilt/gfx1201"),
        device: 0,
        tokens: vec![1],
        decode: 1,
        max_ctx: 262144,
        max_rows: 256,
        rank: 0,
        peer_port: 0,
        trace_layers: false,
        layers: None,
        layer_offset: 0,
    };
    let mut hip_lib = "/opt/rocm/lib/libamdhip64.so.7".to_owned();
    let mut host = "127.0.0.1".to_owned();
    let mut port = 8002u16;
    let mut served_name = "Kolibri-1".to_owned();
    let mut cli = std::env::args().skip(1);
    while let Some(arg) = cli.next() {
        if arg == "--trace-layers" {
            args.trace_layers = true;
            continue;
        }
        let value = cli.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--pack" => args.pack = value.into(),
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
    kolibri::exec::run_persistent(&hip, &args, &host, port, &served_name)
}
