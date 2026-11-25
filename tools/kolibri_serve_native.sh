#!/usr/bin/env bash
set -euo pipefail
REPO=${SABER_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PACK=${KOLIBRI_PACK:-/srv/models/saber-models/saber-packs/kolibri-1-mxfp4-tp2}
CHECKPOINT=${KOLIBRI_CHECKPOINT:-/srv/models/saber-models/models/Kolibri-1-BF16}
PREBUILT=${PREBUILT:-$REPO/prebuilt/gfx1201}
SABER=${KOLIBRI_SERVE_BIN:-$REPO/target/release/kolibri-serve}
HIP_DEVICES=${HIP_DEVICES:-0,1}
ROCM=${ROCM_PATH:-/opt/rocm}
MAX_CTX=${MAX_CTX:-262144}
MAX_ROWS=${MAX_ROWS:-256}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8002}
SERVED_NAME=${SERVED_NAME:-Kolibri-1}
BUILD=1
while [ $# -gt 0 ]; do
  case "$1" in
    --no-build) BUILD=0; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: tools/kolibri_serve_native.sh [--no-build] [kolibri-serve options]

The TP2 server listens on $HOST:$PORT (default 127.0.0.1:8002), uses both
GPUs, and runs in the foreground. Stop it with Ctrl-C or SIGTERM.
Only one text request runs at a time; concurrent requests queue.

Environment overrides: KOLIBRI_PACK, KOLIBRI_CHECKPOINT, KOLIBRI_SERVE_BIN,
PREBUILT, HIP_DEVICES, ROCM_PATH, MAX_CTX, MAX_ROWS, HOST, PORT, SERVED_NAME.
EOF
      exit 0 ;;
    *) break ;;
  esac
done
[[ "$HIP_DEVICES" =~ ^[0-9]+,[0-9]+$ ]] || { echo "HIP_DEVICES must contain two GPU indices" >&2; exit 2; }
for path in "$PACK/manifest.json" "$CHECKPOINT/config.json" "$CHECKPOINT/chat_template.jinja" "$PREBUILT"; do
  [ -e "$path" ] || { echo "missing $path" >&2; exit 1; }
done
if [ "$BUILD" = 1 ]; then
  (cd "$REPO" && cargo build --release -p saber-cli --bin kolibri-serve)
fi
[ -x "$SABER" ] || { echo "no executable $SABER" >&2; exit 1; }
"$ROCM/bin/rocm-smi" --showpids
CMD=("$SABER" --pack "$PACK" --checkpoint "$CHECKPOINT" --prebuilt "$PREBUILT"
     --max-ctx "$MAX_CTX" --max-rows "$MAX_ROWS" --host "$HOST" --port "$PORT" --served-name "$SERVED_NAME")
CMD+=("$@")
exec env HIP_VISIBLE_DEVICES="$HIP_DEVICES" LD_LIBRARY_PATH="$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  TOKENIZERS_PARALLELISM=false "${CMD[@]}"
