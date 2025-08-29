#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-v0.1.0}"
BASE="https://github.com/namas191297/rtmo-ort/releases/download/${TAG}"
OUT="${RTMO_MODELS_DIR:-models}"

declare -a MODELS=(
  "rtmo_t_416x416_body7"
  "rtmo_s_640x640_coco"
  "rtmo_s_640x640_crowdpose"
  "rtmo_s_640x640_body7"
  "rtmo_m_640x640_coco"
  "rtmo_m_640x640_body7"
  "rtmo_l_640x640_coco"
  "rtmo_l_640x640_crowdpose"
  "rtmo_l_640x640_body7"
)

mkdir -p "${OUT}"
for M in "${MODELS[@]}"; do
  mkdir -p "${OUT}/${M}"
  echo "-> ${M}"
  curl -fL "${BASE}/${M}.onnx" -o "${OUT}/${M}/${M}.onnx"
done

echo "Done. Models in: ${OUT}"
