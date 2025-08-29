# RTMO-ORT : RTMO via ONNXRuntime

Lightweight **RTMO** pose estimation on **pure ONNX Runtime**.  
Tiny Python class + simple CLIs: `rtmo-image`, `rtmo-video`, `rtmo-webcam`.

If this project helps you, please consider starring the repo — it improves discoverability.

---

## Install (pick ONE)

### 1) pip (quickest)
```bash
pip install -U pip
# CPU
pip install rtmo-ort[cpu]
# or GPU (CUDA) wheels if available for your platform
pip install rtmo-ort[gpu]
```
*If the pip package isn’t available yet for your platform, use option 2 (clone) or 3 (conda).*

### 2) Clone + editable install (dev mode)
```bash
git clone https://github.com/namas191297/rtmo-ort.git
cd rtmo-ort

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[cpu]"   # or ".[gpu]"
```

### 3) Conda environment
```bash
conda create -n rtmo-ort python=3.9 -y
conda activate rtmo-ort

# Inside this conda env, use pip to install the package
pip install -U pip
pip install rtmo-ort[cpu]   # or rtmo-ort[gpu]
```

> The package does **not** ship model files. See “Get the models” below.

---

## Get the models (required)

You need the RTMO **.onnx** files on disk. The **recommended** way is to run the helper script which downloads and places them under `models/`.

### Option A — Recommended script (from repo root)
```bash
./get_models.sh
```
This creates a layout like:
```
models/
  rtmo_s_640x640_coco/rtmo_s_640x640_coco.onnx
  rtmo_s_640x640_crowdpose/rtmo_s_640x640_crowdpose.onnx
  rtmo_s_640x640_body7/rtmo_s_640x640_body7.onnx
  rtmo_m_640x640_coco/rtmo_m_640x640_coco.onnx
  rtmo_m_640x640_body7/rtmo_m_640x640_body7.onnx
  rtmo_l_640x640_coco/rtmo_l_640x640_coco.onnx
  rtmo_l_640x640_crowdpose/rtmo_l_640x640_crowdpose.onnx
  rtmo_l_640x640_body7/rtmo_l_640x640_body7.onnx
  rtmo_t_416x416_body7/rtmo_t_416x416_body7.onnx
```

### Option B — Manual download
Download the `.onnx` files from **Releases** and place them under `models/` with the same folder structure as above.  
You can also point to a custom models folder using:
```bash
export RTMO_MODELS_DIR=/abs/path/to/my_models
```

---

## Model matrix & direct downloads

| Size   | Dataset    | Input | Keypoints | Download (.onnx) |
|:------:|:-----------|:-----:|:---------:|:-----------------|
| tiny   | body7      | 416   | 17        | [rtmo_t_416x416_body7.onnx](https://github.com/namas191297/rtmo-ort/releases/latest/download/rtmo_t_416x416_body7.onnx) |
| small  | coco       | 640   | 17        | [rtmo_s_640x640_coco.onnx](https://github.com/namas191297/rtmo-ort/releases/latest/download/rtmo_s_640x640_coco.onnx) |
| small  | crowdpose  | 640   | 14        | [rtmo_s_640x640_crowdpose.onnx](https://github.com/namas191297/rtmo-ort/releases/latest/download/rtmo_s_640x640_crowdpose.onnx) |
| small  | body7      | 640   | 17        | [rtmo_s_640x640_body7.onnx](https://github.com/namas191297/rtmo-ort/releases/latest/download/rtmo_s_640x640_body7.onnx) |
| medium | coco       | 640   | 17        | [rtmo_m_640x640_coco.onnx](https://github.com/namas191297/rtmo-ort/releases/latest/download/rtmo_m_640x640_coco.onnx) |
| medium | body7      | 640   | 17        | [rtmo_m_640x640_body7.onnx](https://github.com/namas191297/rtmo-ort/releases/latest/download/rtmo_m_640x640_body7.onnx) |
| large  | coco       | 640   | 17        | [rtmo_l_640x640_coco.onnx](https://github.com/namas191297/rtmo-ort/releases/latest/download/rtmo_l_640x640_coco.onnx) |
| large  | crowdpose  | 640   | 14        | [rtmo_l_640x640_crowdpose.onnx](https://github.com/namas191297/rtmo-ort/releases/latest/download/rtmo_l_640x640_crowdpose.onnx) |
| large  | body7      | 640   | 17        | [rtmo_l_640x640_body7.onnx](https://github.com/namas191297/rtmo-ort/releases/latest/download/rtmo_l_640x640_body7.onnx) |

> The table links always point to the latest Release. If a link 404s, refresh the Release or run `./get_models.sh`.

---

## Quickstart (CLI)

**Image → annotated image**
```bash
# Small COCO (17 kp), CPU
RTMO_MODELS_DIR="$(pwd)/models" \
rtmo-image --model-type small --dataset coco \
  --input assets/demo.jpg --output out.jpg --device cpu
```

**Video → annotated video**
```bash
# Medium COCO, GPU if available
RTMO_MODELS_DIR="$(pwd)/models" \
rtmo-video --model-type medium --dataset coco \
  --input input.mp4 --output out.mp4 --device cuda
```

**Webcam**
```bash
# Tiny Body7 (coarse skeleton), default camera 0
RTMO_MODELS_DIR="$(pwd)/models" \
rtmo-webcam --model-type tiny --dataset body7 --device cpu --cam 0
```

### Useful flags
- `--no-letterbox` : disable letterboxing (use simple resize)
- `--score-thr`, `--kpt-thr`, `--max-det` : tweak detection & keypoint thresholds
- `--size` : override input size (defaults: tiny=416, small/medium/large=640)
- `--onnx` : bypass presets and point directly to a specific `.onnx`

**Preset matrix** (what `--model-type` + `--dataset` map to):
| model-type | dataset      | default size | expected K |
|------------|--------------|--------------|------------|
| tiny       | body7        | 416          | 17         |
| small      | coco         | 640          | 17         |
| small      | crowdpose    | 640          | 14         |
| small      | body7        | 640          | 17         |
| medium     | coco/body7   | 640          | 17         |
| large      | coco/body7   | 640          | 17         |

> CrowdPose uses 14 keypoints (we connect `top_head` + neck + limbs).  
> COCO uses 17 keypoints.  
> Body7 exports in this repo contain 17 keypoints (coarser subset represented).

---

## Python API
```python
import cv2
from rtmo_ort import PoseEstimatorORT

onnx = "models/rtmo_s_640x640_coco/rtmo_s_640x640_coco.onnx"
pe = PoseEstimatorORT(onnx, device="cpu", letterbox=True, score_thr=0.25, kpt_thr=0.2)

img = cv2.imread("assets/demo.jpg")
boxes, kpts, scores = pe.infer(img)   # boxes[N,4], kpts[N,K,3], scores[N]

vis = pe.annotate(img, boxes, kpts, scores)
cv2.imwrite("vis.jpg", vis)
print(f"saved vis.jpg — persons: {len(boxes)}")
```

---

## Troubleshooting
- **No detections** on your image/video? Try lower thresholds:
  ```bash
  --score-thr 0.05 --kpt-thr 0.05
  ```
- **Webcam issues**: pass `--cam 1` (or another index), add `--window` to name the window.
- **OpenCV can’t open display** on some Linux servers: you may need `sudo apt-get install -y libgl1` or run headless and save to disk.
- **GPU not used**: check printed ONNX Runtime providers. If you only see `CPUExecutionProvider`, install the GPU wheel (`pip install onnxruntime-gpu`) and ensure CUDA/cuDNN are available.

---

## Notes
- ONNX Runtime version is not pinned; use the latest stable wheel.
- Letterboxing is optional; most RTMO exports assume external resize. `--no-letterbox` is available to experiment.
- You can store models anywhere and set `RTMO_MODELS_DIR` to point to that folder.

---

## Acknowledgments
- RTMO authors & MMPose contributors.
- ONNX Runtime team for the excellent inference engine.

---

## License
Apache-2.0. See [LICENSE](LICENSE).
