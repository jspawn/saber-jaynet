#!/usr/bin/env bash
# Build Ripple-65B Preview's TP pack directly on the host from the official NVFP4 checkpoint.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CHECKPOINT=${RIPPLE_CHECKPOINT:-/srv/models/saber-models/models/Ripple-65B-Preview-NVFP4}
OUT=""
TP=2
WORKERS=${SABER_PACK_WORKERS:-8}
PLAN_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) CHECKPOINT=${2:?}; shift 2 ;;
    --out) OUT=${2:?}; shift 2 ;;
    --tp) TP=${2:?}; shift 2 ;;
    --workers) WORKERS=${2:?}; shift 2 ;;
    --plan-only) PLAN_ONLY=1; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: tools/pack/build_ripple.sh [--checkpoint DIR] [--out DIR] [--tp 1|2] [--workers N] [--plan-only]

Uses the official NVFP4-MLP/BF16-attention checkpoint and produces Saber MXFP4 packs.
Runs on the host with Cargo and Python NumPy; no GPU or container is used.
EOF
      exit 0 ;;
    *) echo "build_ripple.sh: unknown argument '$1'" >&2; exit 2 ;;
  esac
done
case "$TP" in 1|2) ;; *) echo "--tp must be 1 or 2" >&2; exit 2 ;; esac
PYTHON=${SABER_PYTHON:-$REPO/../.venv/bin/python}
[ -x "$PYTHON" ] || PYTHON=python3
"$PYTHON" -c 'import numpy' >/dev/null 2>&1 || { echo "$PYTHON: NumPy is required" >&2; exit 1; }
[ -f "$CHECKPOINT/model.safetensors.index.json" ] || { echo "missing checkpoint index under $CHECKPOINT" >&2; exit 1; }
OUT=${OUT:-${SABER_PACKS:-/mnt/nvme/saber-packs}/ripple-65b-preview-mxfp4-tp$TP}
PLANS=${SABER_PACK_PLANS:-$REPO/.cache/pack}
mkdir -p "$PLANS"
PLAN=$PLANS/ripple-65b-tp$TP.json
cargo run -q --release --manifest-path "$REPO/Cargo.toml" -p saber-model \
  --bin saber-ripple-packplan -- --checkpoint "$CHECKPOINT" --tp "$TP" > "$PLAN"
echo "Ripple plan: $PLAN"
if [ "$PLAN_ONLY" = 1 ]; then exit 0; fi
[ ! -e "$OUT/manifest.json" ] || { echo "$OUT already holds a pack; refusing to overwrite" >&2; exit 1; }
CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  nice -n 19 "$PYTHON" "$REPO/tools/pack/build_pack.py" --model "$CHECKPOINT" --out "$OUT" \
  --tp "$TP" --layout device-final --plan-json "$PLAN" --workers "$WORKERS"
