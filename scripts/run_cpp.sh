#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
binary="$root/build/debug/cpp/nanotracker_cpp"

if [[ ! -x "$binary" ]]; then
  echo "C++ binary not found. Run: cmake --build --preset debug" >&2
  exit 1
fi

shopt -s nullglob
nvidia_libs=("$root"/.venv/lib/python*/site-packages/nvidia/*/lib)
if (( ${#nvidia_libs[@]} == 0 )); then
  echo "CUDA runtime libraries not found. Run: uv sync" >&2
  exit 1
fi

library_path=$(IFS=:; echo "${nvidia_libs[*]}")
export LD_LIBRARY_PATH="$library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$binary" "$@"
