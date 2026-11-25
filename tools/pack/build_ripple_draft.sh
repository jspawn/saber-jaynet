#!/usr/bin/env bash
# Build Ripple's DFlash2 TP pack directly on the host from the official BF16 drafter.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CHECKPOINT=${RIPPLE_DRAFT_CHECKPOINT:-/srv/models/saber-models/models/Ripple-65B-Preview-DFlash2}
TARGET=${RIPPLE_CHECKPOINT:-/srv/models/saber-models/models/Ripple-65B-Preview-NVFP4}
OUT=""
TP=2
WORKERS=${SABER_PACK_WORKERS:-8}
BF16=0
PLAN_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) CHECKPOINT=${2:?}; shift 2 ;;
    --target) TARGET=${2:?}; shift 2 ;;
    --out) OUT=${2:?}; shift 2 ;;
    --tp) TP=${2:?}; shift 2 ;;
    --workers) WORKERS=${2:?}; shift 2 ;;
    --bf16) BF16=1; shift ;;
    --plan-only) PLAN_ONLY=1; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: tools/pack/build_ripple_draft.sh [--checkpoint DIR] [--target DIR] [--out DIR]
                                      [--tp 1|2] [--workers N] [--bf16] [--plan-only]

The default writes the runtime FP8 pack. --bf16 writes the larger BF16 control.
Both forms share the target pack's embedding and LM head at runtime.
EOF
      exit 0 ;;
    *) echo "build_ripple_draft.sh: unknown argument '$1'" >&2; exit 2 ;;
  esac
done
case "$TP" in 1|2) ;; *) echo "--tp must be 1 or 2" >&2; exit 2 ;; esac
PYTHON=${SABER_PYTHON:-$REPO/../.venv/bin/python}
[ -x "$PYTHON" ] || PYTHON=python3
"$PYTHON" -c 'import numpy' >/dev/null 2>&1 || { echo "$PYTHON: NumPy is required" >&2; exit 1; }
[ -f "$CHECKPOINT/model.safetensors" ] || { echo "missing $CHECKPOINT/model.safetensors" >&2; exit 1; }
[ -f "$TARGET/config.json" ] || { echo "missing $TARGET/config.json" >&2; exit 1; }
if [ "$BF16" = 1 ]; then
  KIND=bf16
  FLAG=--bf16
else
  KIND=fp8-runtime
  FLAG=""
fi
OUT=${OUT:-${SABER_PACKS:-/mnt/nvme/saber-packs}/ripple-65b-dflash2-$KIND-tp$TP}
PLANS=${SABER_PACK_PLANS:-$REPO/.cache/pack}
mkdir -p "$PLANS"
PLAN=$PLANS/ripple-65b-dflash2-$KIND-tp$TP.json
cargo run -q --release --manifest-path "$REPO/Cargo.toml" -p saber-model \
  --bin saber-ripple-draft-packplan -- --checkpoint "$CHECKPOINT" --target "$TARGET" \
  --tp "$TP" --rope-max-pos 65536 $FLAG > "$PLAN"
echo "Ripple DFlash2 plan: $PLAN"
if [ "$PLAN_ONLY" = 1 ]; then exit 0; fi
[ ! -e "$OUT/manifest.json" ] || { echo "$OUT already holds a pack; refusing to overwrite" >&2; exit 1; }
CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  nice -n 19 "$PYTHON" "$REPO/tools/pack/build_pack.py" --model "$CHECKPOINT" --out "$OUT" \
  --tp "$TP" --layout device-final --plan-json "$PLAN" --workers "$WORKERS"
