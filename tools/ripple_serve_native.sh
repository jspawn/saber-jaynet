#!/usr/bin/env bash
# Run the experimental persistent Ripple TP2 OpenAI server directly on the host.
set -euo pipefail
REPO=${SABER_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PACK=${RIPPLE_PACK:-/srv/models/saber-models/saber-packs/ripple-65b-preview-mxfp4-tp2}
DRAFT=${RIPPLE_DRAFT:-/srv/models/saber-models/saber-packs/ripple-65b-dflash2-fp8-runtime-tp2}
CHECKPOINT=${RIPPLE_CHECKPOINT:-/srv/models/saber-models/models/Ripple-65B-Preview-NVFP4}
PREBUILT=${PREBUILT:-$REPO/prebuilt/gfx1201}
SABER=${RIPPLE_SERVE_BIN:-$REPO/target/release/ripple-serve}
HIP_DEVICES=${HIP_DEVICES:-0,1}
ROCM=${ROCM_PATH:-/opt/rocm}
MAX_CTX=${MAX_CTX:-131072}
MAX_ROWS=${MAX_ROWS:-64}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
SERVED_NAME=${SERVED_NAME:-Ripple-65B-Preview}
BUILD=1
USE_DRAFT=1
while [ $# -gt 0 ]; do
  case "$1" in
    --no-build) BUILD=0; shift ;;
    --target-only) USE_DRAFT=0; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: tools/ripple_serve_native.sh [--no-build] [--target-only] [ripple-serve options]

The server listens on $HOST:$PORT (default 127.0.0.1:8000) and uses both GPUs.
Only one text request runs at a time; concurrent requests queue.
Stop the foreground server with Ctrl-C, or send SIGTERM to rank 0's saved PID.

Environment overrides: RIPPLE_PACK, RIPPLE_DRAFT, RIPPLE_CHECKPOINT,
RIPPLE_SERVE_BIN, PREBUILT, HIP_DEVICES, ROCM_PATH, MAX_CTX, MAX_ROWS,
HOST, PORT, SERVED_NAME.
EOF
      exit 0 ;;
    *) break ;;
  esac
done
[[ "$HIP_DEVICES" =~ ^[0-9]+,[0-9]+$ ]] || { echo "HIP_DEVICES must contain two GPU indices" >&2; exit 2; }
for path in "$PACK/manifest.json" "$CHECKPOINT/config.json" "$CHECKPOINT/chat_template.jinja" "$PREBUILT"; do
  [ -e "$path" ] || { echo "missing $path" >&2; exit 1; }
done
if [ "$USE_DRAFT" = 1 ] && [ ! -f "$DRAFT/manifest.json" ]; then
  echo "missing $DRAFT/manifest.json (build it or use --target-only)" >&2
  exit 1
fi
if [ "$BUILD" = 1 ]; then
  (cd "$REPO" && cargo build --release -p saber-cli --bin ripple-serve)
fi
[ -x "$SABER" ] || { echo "no executable $SABER" >&2; exit 1; }
"$ROCM/bin/rocm-smi" --showpids
CMD=("$SABER" --pack "$PACK" --checkpoint "$CHECKPOINT" --prebuilt "$PREBUILT"
     --max-ctx "$MAX_CTX" --max-rows "$MAX_ROWS" --host "$HOST" --port "$PORT" --served-name "$SERVED_NAME")
if [ "$USE_DRAFT" = 1 ]; then CMD+=(--draft "$DRAFT"); fi
CMD+=("$@")
exec env HIP_VISIBLE_DEVICES="$HIP_DEVICES" LD_LIBRARY_PATH="$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  TOKENIZERS_PARALLELISM=false "${CMD[@]}"
