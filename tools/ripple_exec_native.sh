#!/usr/bin/env bash
# Run the experimental Ripple TP2 token/chat harness directly on the host.
set -euo pipefail
REPO=${SABER_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PACK=${RIPPLE_PACK:-/srv/models/saber-models/saber-packs/ripple-65b-preview-mxfp4-tp2}
DRAFT=${RIPPLE_DRAFT:-/srv/models/saber-models/saber-packs/ripple-65b-dflash2-fp8-runtime-tp2}
CHECKPOINT=${RIPPLE_CHECKPOINT:-/srv/models/saber-models/models/Ripple-65B-Preview-NVFP4}
PREBUILT=${PREBUILT:-$REPO/prebuilt/gfx1201}
SABER=${RIPPLE_BIN:-$REPO/target/release/ripple-exec}
HIP_DEVICES=${HIP_DEVICES:-0,1}
ROCM=${ROCM_PATH:-/opt/rocm}
MAX_CTX=${MAX_CTX:-131072}
MAX_ROWS=${MAX_ROWS:-64}
BUILD=1
USE_DRAFT=1
while [ $# -gt 0 ]; do
  case "$1" in
    --no-build) BUILD=0; shift ;;
    --target-only) USE_DRAFT=0; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: tools/ripple_exec_native.sh [--no-build] [--target-only] [ripple-exec options]

Examples:
  tools/ripple_exec_native.sh --text 'Reply with exactly OK.' --chat --decode 32
  tools/ripple_exec_native.sh --target-only --tokens 1,2,3 --decode 8

The harness uses both GPUs and runs in the foreground; stop it with Ctrl-C.
Environment overrides: RIPPLE_PACK, RIPPLE_DRAFT, RIPPLE_CHECKPOINT, RIPPLE_BIN,
PREBUILT, HIP_DEVICES, ROCM_PATH, MAX_CTX, MAX_ROWS.
EOF
      exit 0 ;;
    *) break ;;
  esac
done
[[ "$HIP_DEVICES" =~ ^[0-9]+,[0-9]+$ ]] || { echo "HIP_DEVICES must contain two GPU indices" >&2; exit 2; }
for path in "$PACK/manifest.json" "$CHECKPOINT/config.json" "$PREBUILT"; do
  [ -e "$path" ] || { echo "missing $path" >&2; exit 1; }
done
if [ "$USE_DRAFT" = 1 ] && [ ! -f "$DRAFT/manifest.json" ]; then
  echo "missing $DRAFT/manifest.json (build it or use --target-only)" >&2
  exit 1
fi
if [ "$BUILD" = 1 ]; then
  (cd "$REPO" && cargo build --release -p saber-cli --bin ripple-exec)
fi
[ -x "$SABER" ] || { echo "no executable $SABER" >&2; exit 1; }
"$ROCM/bin/rocm-smi" --showpids
CMD=("$SABER" --pack "$PACK" --checkpoint "$CHECKPOINT" --prebuilt "$PREBUILT"
     --max-ctx "$MAX_CTX" --max-rows "$MAX_ROWS")
if [ "$USE_DRAFT" = 1 ]; then CMD+=(--draft "$DRAFT"); fi
CMD+=("$@")
exec env HIP_VISIBLE_DEVICES="$HIP_DEVICES" LD_LIBRARY_PATH="$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  TOKENIZERS_PARALLELISM=false "${CMD[@]}"
