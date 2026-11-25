#!/usr/bin/env bash
set -euo pipefail

REPO=${SABER_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CHECKPOINT=${DAISY_CHECKPOINT:-/srv/models/saber-models/models/DaisyRose-27B-BF16}
PACK=${DAISY_PACK:-/srv/models/saber-models/saber-packs/daisyrose-27b-mxfp4-tp2-bf16head}
DRAFT=${DAISY_DRAFT:-/srv/models/saber-models/saber-packs/daisyrose-27b-dflash2-tp2-bf16head}
PREBUILT=${PREBUILT:-$REPO/prebuilt/gfx1201}
SABER=${SABER_BIN:-$REPO/target/release/saber}
HIP_DEVICES=${HIP_DEVICES:-0,1}
ROCM=${ROCM_PATH:-/opt/rocm}
MAX_CTX=${MAX_CTX:-}
MAX_SEQS=${MAX_SEQS:-1}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
SERVED_NAME=${SERVED_NAME:-DaisyRose-27B}
TOKENIZER=${TOKENIZER:-$CHECKPOINT/tokenizer.json}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-$REPO/tools/templates/daisyrose_trailing_system.jinja}
BUILD=1
DFLASH=0

while [ $# -gt 0 ]; do
  case "$1" in
    --no-build) BUILD=0; shift ;;
    --dflash) DFLASH=1; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: tools/serve_daisyrose_native.sh [--dflash] [--no-build] [saber serve options]

Runs DaisyRose-27B TP2 in the foreground. The default is target-only with 262144
context. --dflash enables the Qwen3.8 DFlash2 drafter and defaults to 131072
context with chain verification. The FP16 recurrent carry fix makes greedy and
sampled output match target-only in the validated configurations. DFlash remains
experimental pending the independent model-quality and JayNet acceptance gates.
Both modes use 127.0.0.1:8080, GPUs 0,1 and one sequence.

Environment overrides: DAISY_CHECKPOINT, DAISY_PACK, DAISY_DRAFT, TOKENIZER,
CHAT_TEMPLATE, SABER_REPO, SABER_BIN, PREBUILT, ROCM_PATH, HIP_DEVICES, MAX_CTX,
MAX_SEQS, WIDTHS, DRAFT_BLOCK, SPEC_TREE, SPEC_TREE_SAMPLED, BUDGET_MIB, HOST,
PORT and SERVED_NAME.
EOF
      exit 0 ;;
    *) break ;;
  esac
done

if [ "$DFLASH" = 1 ]; then
  MAX_CTX=${MAX_CTX:-131072}
  DFLASH_ARGS=(
    --engine-decode-widths "${WIDTHS:-1,8}"
    --engine-draft-block "${DRAFT_BLOCK:-8}"
    --engine-draft-pack "$DRAFT"
    --engine-spec-tree "${SPEC_TREE:-off}"
    --engine-spec-tree-sampled "${SPEC_TREE_SAMPLED:-off}"
  )
  MODE="DFlash2 block ${DRAFT_BLOCK:-8}, tree ${SPEC_TREE:-off}"
else
  MAX_CTX=${MAX_CTX:-262144}
  DFLASH_ARGS=(--engine-decode-widths "${WIDTHS:-1}")
  MODE="target-only"
fi

[[ "$HIP_DEVICES" =~ ^[0-9]+,[0-9]+$ ]] || { echo "HIP_DEVICES must contain two GPU indices" >&2; exit 2; }
REQUIRED=("$PACK/manifest.json" "$PACK/rank0.pack" "$PACK/rank1.pack"
          "$TOKENIZER" "$CHAT_TEMPLATE" "$PREBUILT")
if [ "$DFLASH" = 1 ]; then
  REQUIRED+=("$DRAFT/manifest.json" "$DRAFT/rank0.pack" "$DRAFT/rank1.pack")
fi
for path in "${REQUIRED[@]}"; do
  [ -e "$path" ] || { echo "missing $path" >&2; exit 1; }
done
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":$PORT\$"; then
  echo "port $PORT is already in use" >&2
  exit 1
fi
if [ "$BUILD" = 1 ]; then
  (cd "$REPO" && cargo build --release -p saber-cli)
fi
[ -x "$SABER" ] || { echo "no executable $SABER" >&2; exit 1; }

CMD=(
  "$SABER" serve --engine real
  --engine-tp 2 --engine-device 0 --engine-peer-device 1
  --hip-lib "$ROCM/lib/libamdhip64.so.7"
  --pack "$PACK"
  --prebuilt "$PREBUILT"
  --tokenizer "$TOKENIZER"
  --chat-template "$CHAT_TEMPLATE"
  --engine-max-ctx "$MAX_CTX"
  --engine-prefill-rows '96,128,192,256,320,384,512,768,1024,1536,2048'
  --engine-decode-batches '1'
  "${DFLASH_ARGS[@]}"
  --engine-max-seqs "$MAX_SEQS"
  --engine-rectangular-widths
  --engine-budget-mib "${BUDGET_MIB:-30500}"
  --served-model-name "$SERVED_NAME"
  --host "$HOST"
  --port "$PORT"
  "$@"
)

ulimit -c 0
echo "== starting DaisyRose on $HOST:$PORT (TP=2, HIP_VISIBLE_DEVICES=$HIP_DEVICES, max ctx $MAX_CTX, $MODE)" >&2
exec env \
  HIP_VISIBLE_DEVICES="$HIP_DEVICES" \
  GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-3}" \
  CUDA_VISIBLE_DEVICES= \
  TOKENIZERS_PARALLELISM=false \
  LD_LIBRARY_PATH="$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "${CMD[@]}"
