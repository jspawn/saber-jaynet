#[path = "../kolibri/mod.rs"]
mod kolibri;
#[path = "../ripple/mod.rs"]
mod ripple;

use std::path::PathBuf;

use saber_hip::HipThread;

fn main() -> anyhow::Result<()> {
    let mut checkpoint = PathBuf::from("/srv/models/saber-models/models/Kolibri-1");
    let mut artifact = PathBuf::from("/srv/models/saber-models/spike/kolibri-layer0");
    let mut prebuilt = PathBuf::from("prebuilt/gfx1201");
    let mut hip_lib = "/opt/rocm/lib/libamdhip64.so.7".to_owned();
    let mut device = 0;
    let mut iterations = 20u32;
    let mut host_only = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--host-only" {
            host_only = true;
            continue;
        }
        let value = args.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--checkpoint" => checkpoint = value.into(),
            "--artifact" => artifact = value.into(),
            "--prebuilt" => prebuilt = value.into(),
            "--hip-lib" => hip_lib = value,
            "--device" => device = value.parse()?,
            "--iterations" => iterations = value.parse()?,
            other => panic!("unknown argument {other}"),
        }
    }
    let hip = HipThread::spawn(&hip_lib)?;
    kolibri::spike::run(
        &hip,
        &checkpoint,
        &artifact,
        &prebuilt,
        device,
        iterations,
        host_only,
    )
}
