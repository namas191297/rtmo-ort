# RTMO-ORT — RTMO pose on pure ONNX Runtime

Minimal, fast RTMO (person detection + 2D pose) inference with **no heavy frameworks**. One tiny Python class, three simple CLIs, and ready-to-download ONNX models.

If this saves you time, please consider **starring the repo** — it really helps.

---

## Install (two ways)

### A) pip (recommended)
```bash
# CPU
pip install "rtmo-ort[cpu]"

# GPU (uses onnxruntime-gpu if present)
pip install "rtmo-ort[gpu]"
```

### B) From source
```bash
git clone https://github.com/namas191297/rtmo-ort.git
cd rtmo-ort
python -m venv .venv && source .venv/bin/activate   # optional
pip install -e ".[cpu]"                             # or ".[gpu]"
```

Python ≥ 3.8. Works on Linux/macOS/Windows.

---

## Get models

This repo ships a helper to fetch ONNX files into `models/…`.

```bash
# fetch a specific release tag (e.g., v0.1.0)
./get_models.sh v0.1.0

# or omit to use the default in the script
./get_models.sh
```

You can also download individual models manually (see table below).  
By default, the CLIs look in `models/`. To change that, set `RTMO_MODELS_DIR=/path/to/models`.

---

## Models table (direct downloads)

Each file should be placed at `models/<name>/<name>.onnx`.  
Example: `models/rtmo_s_640x640_coco/rtmo_s_640x640_coco.onnx`.

> Replace `v0.1.0` with your chosen tag if needed.

| Size   | Dataset         | Input | Download |
|:------:|:----------------|:----:|:--|
| tiny   | body7           | 416  | https://github.com/namas191297/rtmo-ort/releases/download/v0.1.0/rtmo_t_416x416_body7.onnx |
| small  | coco            | 640  | https://github.com/namas191297/rtmo-ort/releases/download/v0.1.0/rtmo_s_640x640_coco.onnx |
| small  | crowdpose       | 640  | https://github.com/namas191297/rtmo-ort/releases/download/v0.1.0/rtmo_s_640x640_crowdpose.onnx |
| small  | body7           | 640  | https://github.com/namas191297/rtmo-ort/releases/download/v0.1.0/rtmo_s_640x640_body7.onnx |
| medium | coco            | 640  | https://github.com/namas191297/rtmo-ort/releases/download/v0.1.0/rtmo_m_640x640_coco.onnx |
| medium | body7           | 640  | https://github.com/namas191297/rtmo-ort/releases/download/v0.1.0/rtmo_m_640x640_body7.onnx |
| large  | coco            | 640  | https://github.com/namas191297/rtmo-ort/releases/download/v0.1.0/rtmo_l_640x640_coco.onnx |
| large  | crowdpose       | 640  | https://github.com/namas191297/rtmo-ort/releases/download/v0.1.0/rtmo_l_640x640_crowdpose.onnx |
| large  | body7           | 640  | https://github.com/namas191297/rtmo-ort/releases/download/v0.1.0/rtmo_l_640x640_body7.onnx |
| large  | body7_crowdpose | 640  | https://github.com/namas191297/rtmo-ort/releases/download/v0.1.0/rtmo_l_640x640_body7_crowdpose.onnx |

---

## Use the CLIs

All commands accept the same presets and thresholds:

- `--model-type {tiny,small,medium,large}` (default: `small`)
- `--dataset {coco,crowdpose,body7,body7_crowdpose}` (default: `coco`)
- `--no-letterbox` (disable square letterbox; default is letterbox on)
- `--score-thr`, `--kpt-thr`, `--max-det`
- `--device {cpu,cuda}`
- `--onnx /path/to/model.onnx` (overrides presets)
- `--models-dir /path/to/models` (default: `models`)

### Image
```bash
rtmo-image --model-type small --dataset coco \
  --input path/to/in.jpg --output out.jpg --device cpu
```

### Video
```bash
rtmo-video --model-type small --dataset coco \
  --input in.mp4 --output out.mp4 --device cuda
```

### Webcam
```bash
rtmo-webcam --model-type small --dataset coco --device cpu
# pick another camera:
# rtmo-webcam --cam 1
```

---

## Python API

```python
import cv2
from rtmo_ort import PoseEstimatorORT

onnx = "models/rtmo_s_640x640_coco/rtmo_s_640x640_coco.onnx"
pe = PoseEstimatorORT(onnx, device="cpu", letterbox=True)

img = cv2.imread("assets/demo.jpg")
boxes, kpts, scores = pe.infer(img)

vis = pe.annotate(img, boxes, kpts, scores)
cv2.imwrite("vis.jpg", vis)
```

**Outputs**
- `boxes`: `[N,4]` in xyxy
- `kpts`: `[N,K,3]` with `(x,y,score)` per keypoint
- `scores`: `[N]` person scores

---

## Notes and tips

- **NMS is fused** inside the ONNX models. Do not run NMS again.
- **Letterbox vs. stretch.** Letterbox (default) preserves aspect ratio and generally matches training; stretching (`--no-letterbox`) may reduce accuracy but can be fine for quick demos.
- **Keypoints count.** COCO = 17; CrowdPose = 14; Body7 is coarse. Some Body7 exports use 17-dim outputs for compatibility; semantics remain coarse.
- **GPU provider.** If you installed `onnxruntime-gpu`, use `--device cuda`. If CUDA isn’t found, ONNX Runtime falls back to CPU.
- **Codecs.** If `rtmo-video` fails to write a file, try `mp4v`, `XVID`, or install OS-level codecs.

---

## Project structure

```
rtmo_ort/
  ├─ estimator.py   # PoseEstimatorORT (ONNX Runtime + postprocess + drawing)
  ├─ cli.py         # rtmo-image / rtmo-video / rtmo-webcam
  └─ __init__.py
models/             # place ONNX files here (or use --onnx)
get_models.sh       # fetches model files for a given tag
```

---

If you want something specific, open an issue.

---

## Contributing & support

Issues and pull requests are welcome. If you found this useful, **star the repo** and consider sharing a short demo clip — it helps others discover it.

- Website: namasbhandari.in
- Repo: https://github.com/namas191297/rtmo-ort  
- Issues: https://github.com/namas191297/rtmo-ort/issues  
- License: Apache-2.0