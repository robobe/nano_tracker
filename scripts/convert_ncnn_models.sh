#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
models="$root/models/nanotrackv3"
output="$root/models/nanotrackv3_ncnn"
pnnx_bin="${PNNX_BIN:-pnnx}"

if ! command -v "$pnnx_bin" >/dev/null; then
  echo "PNNX is required. Install it with: uv tool install pnnx" >&2
  exit 1
fi

mkdir -p "$output"
workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT
cd "$workdir"

"$pnnx_bin" "$models/nanotrack_backbone.onnx" \
  inputshape=[1,3,127,127] inputshape2=[1,3,255,255] fp16=0 \
  pnnxparam="$workdir/backbone.pnnx.param" pnnxbin="$workdir/backbone.pnnx.bin" \
  pnnxpy="$workdir/backbone_pnnx.py" pnnxonnx="$workdir/backbone.pnnx.onnx" ncnnpy="$workdir/backbone_ncnn.py" \
  ncnnparam="$output/nanotrack_backbone.param" ncnnbin="$output/nanotrack_backbone.bin"
"$pnnx_bin" "$models/nanotrack_head.onnx" \
  inputshape=[1,96,8,8],[1,96,16,16] fp16=0 \
  pnnxparam="$workdir/head.pnnx.param" pnnxbin="$workdir/head.pnnx.bin" \
  pnnxpy="$workdir/head_pnnx.py" pnnxonnx="$workdir/head.pnnx.onnx" ncnnpy="$workdir/head_ncnn.py" \
  ncnnparam="$output/nanotrack_head.param" ncnnbin="$output/nanotrack_head.bin"

for intermediate in "$models/nanotrack_backbone.pnnxsim.onnx" "$models/nanotrack_head.pnnxsim.onnx"; do
  [[ ! -e "$intermediate" ]] || mv -- "$intermediate" "$workdir/"
done
