#!/usr/bin/env bash
# Compile one HIP source to a gfx1201 code object directly on the host.
set -euo pipefail
src=$1
out=$2
shift 2
ROCM=${ROCM_PATH:-/opt/rocm}
exec "$ROCM/bin/hipcc" --genco --offload-arch=gfx1201 -O3 "$@" "$src" -o "$out"
