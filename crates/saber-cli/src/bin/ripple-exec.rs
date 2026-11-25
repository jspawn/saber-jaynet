#[path = "../ripple/mod.rs"]
mod ripple;

use std::path::PathBuf;

use ripple::exec::ExecArgs;
use saber_hip::HipThread;
use saber_model::Tokenizer;

fn main() -> anyhow::Result<()> {
    let mut args = ExecArgs {
        pack: PathBuf::from("/srv/models/saber-models/saber-packs/ripple-65b-preview-mxfp4-tp2"),
        draft: None,
        checkpoint: PathBuf::from("/srv/models/saber-models/models/Ripple-65B-Preview-NVFP4"),
        prebuilt: PathBuf::from("prebuilt/gfx1201"),
        device: 0,
        tokens: vec![1, 2, 3],
        decode: 8,
        max_ctx: 65536,
        max_rows: 64,
        rank: 0,
        peer_port: 0,
        trace_layers: false,
    };
    let mut hip_lib = "/opt/rocm/lib/libamdhip64.so.7".to_owned();
    let mut text: Option<String> = None;
    let mut chat = false;
    let mut cli = std::env::args().skip(1);
    while let Some(arg) = cli.next() {
        match arg.as_str() {
            "--chat" => {
                chat = true;
                continue;
            }
            "--trace-layers" => {
                args.trace_layers = true;
                continue;
            }
            _ => {}
        }
        let value = cli.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--pack" => args.pack = value.into(),
            "--draft" => args.draft = Some(value.into()),
            "--checkpoint" => args.checkpoint = value.into(),
            "--prebuilt" => args.prebuilt = value.into(),
            "--hip-lib" => hip_lib = value,
            "--device" => args.device = value.parse()?,
            "--tokens" => {
                args.tokens = value.split(',').map(str::parse).collect::<Result<_, _>>()?
            }
            "--text" => text = Some(value),
            "--decode" => args.decode = value.parse()?,
            "--max-ctx" => args.max_ctx = value.parse()?,
            "--max-rows" => args.max_rows = value.parse()?,
            "--rank" => args.rank = value.parse()?,
            "--peer-port" => args.peer_port = value.parse()?,
            other => panic!("unknown argument {other}"),
        }
    }
    let tokenizer = Tokenizer::from_model_dir(&args.checkpoint)
        .map_err(|error| anyhow::anyhow!("loading checkpoint tokenizer: {error}"))?;
    if let Some(text) = text {
        let rendered = if chat {
            format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
        } else {
            text
        };
        args.tokens = tokenizer
            .encode(&rendered)
            .map_err(|error| anyhow::anyhow!("encoding prompt: {error}"))?;
    }
    let hip = HipThread::spawn(&hip_lib)?;
    let output = ripple::exec::run(&hip, &args)?;
    if args.rank == 0 {
        let decoded = tokenizer
            .decode(&output, false)
            .map_err(|error| anyhow::anyhow!("decoding output: {error}"))?;
        println!("ripple output text: {decoded:?}");
    }
    Ok(())
}
