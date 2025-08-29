import argparse, cv2, time, os, re, glob
from .estimator import PoseEstimatorORT

# Map friendly types to RTMO letter + default input size
TYPE_MAP = {
    "tiny":   ("t", 416),
    "small":  ("s", 640),
    "medium": ("m", 640),
    "large":  ("l", 640),
}

DATASETS = {"coco", "body7", "crowdpose", "body7_crowdpose"}

def _scan_available(models_dir):
    """Return list of (model_type, dataset, size, folder) from 'models/rtmo_*'."""
    out = []
    # expected folder pattern: rtmo_<letter>_<WxH>_<dataset>
    pat = re.compile(r"^rtmo_([tsml])_(\d+)x\2_(.+)$")
    for d in glob.glob(os.path.join(models_dir, "rtmo_*")):
        if not os.path.isdir(d): continue
        name = os.path.basename(d)
        m = pat.match(name)
        if not m: continue
        letter, size, dataset = m.group(1), int(m.group(2)), m.group(3)
        t = {"t":"tiny","s":"small","m":"medium","l":"large"}.get(letter, None)
        if t is None: continue
        out.append((t, dataset, size, name))
    # stable ordering
    out.sort(key=lambda x: (["tiny","small","medium","large"].index(x[0]), x[1], x[2]))
    return out

def _resolve_onnx(args):
    """Return (onnx_path, inferred_size). If --onnx given, just use that."""
    if args.onnx:
        return args.onnx, args.size

    models_dir = args.models_dir or os.getenv("RTMO_MODELS_DIR", "models")
    if args.model_type not in TYPE_MAP:
        raise SystemExit(f"--model-type must be one of {list(TYPE_MAP)}")
    if args.dataset not in DATASETS:
        raise SystemExit(f"--dataset must be one of {sorted(DATASETS)}")

    letter, default_size = TYPE_MAP[args.model_type]
    size = args.size or default_size
    base = f"rtmo_{letter}_{size}x{size}_{args.dataset}"
    onnx = os.path.join(models_dir, base, f"{base}.onnx")

    if os.path.isfile(onnx):
        return onnx, size

    # Fallback: try to find a close match (scan by type+dataset, ignore size)
    candidates = [d for (t, ds, sz, name) in _scan_available(models_dir)
                  if t == args.model_type and ds == args.dataset]
    if candidates:
        # pick first candidate and build its onnx
        name = candidates[0]
        onnx_try = os.path.join(models_dir, name, f"{name}.onnx")
        if os.path.isfile(onnx_try):
            # parse size from folder name
            m = re.search(r"_(\d+)x\1_", name)
            inferred = int(m.group(1)) if m else size
            return onnx_try, inferred

    # Nothing found: print a friendly list
    have = _scan_available(models_dir)
    if not have:
        raise SystemExit(
            f"Could not find any RTMO models under '{models_dir}'. "
            f"Download zips from your GitHub Release and unzip into '{models_dir}/'."
        )
    msg = ["Requested preset not found.",
           f"Looked for: {base}/{base}.onnx under '{models_dir}'.",
           "Available models:"]
    for t, ds, sz, name in have:
        msg.append(f"  - type={t:6s} dataset={ds:14s} size={sz:4d}  ({name})")
    raise SystemExit("\n".join(msg))

def _build_estimator_from_args(args):
    onnx_path, inferred_size = _resolve_onnx(args)
    return PoseEstimatorORT(
        onnx_path,
        device=args.device,
        size=args.size or inferred_size,
        letterbox=not args.no_letterbox,
        score_thr=args.score_thr,
        kpt_thr=args.kpt_thr,
        max_det=args.max_det,
    )

def _add_common_preset_args(ap):
    ap.add_argument("--onnx", help="Path to ONNX (overrides presets).")
    ap.add_argument("--models-dir", default=os.getenv("RTMO_MODELS_DIR", "models"),
                    help="Folder containing rtmo_* model directories (default: %(default)s)")
    ap.add_argument("--model-type", choices=["tiny","small","medium","large"], default="small",
                    help="Model size preset (default: %(default)s)")
    ap.add_argument("--dataset", choices=sorted(list(DATASETS)), default="coco",
                    help="Training dataset preset (default: %(default)s)")
    ap.add_argument("--size", type=int, default=None,
                    help="Override input size (defaults: tiny=416, others=640)")
    ap.add_argument("--no-letterbox", action="store_true", help="Disable letterbox (stretch resize)")
    ap.add_argument("--score-thr", type=float, default=0.15)
    ap.add_argument("--kpt-thr", type=float, default=0.20)
    ap.add_argument("--max-det", type=int, default=5)
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")

def webcam_main():
    ap = argparse.ArgumentParser("RTMO webcam (ONNX Runtime) — presets or --onnx")
    _add_common_preset_args(ap)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--window", type=str, default="rtmo-ort")
    args = ap.parse_args()

    pe = _build_estimator_from_args(args)
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {args.cam}")
    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    t0, fps = time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok: break
        boxes, kpts, scores = pe.infer(frame)
        vis = pe.annotate(frame, boxes, kpts, scores)
        t1 = time.time(); fps = 0.9*fps + 0.1*(1.0/max(1e-3, t1-t0)); t0 = t1
        cv2.putText(vis, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imshow(args.window, vis)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()

def image_main():
    ap = argparse.ArgumentParser("RTMO image (ONNX Runtime) — presets or --onnx")
    _add_common_preset_args(ap)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="vis.jpg")
    args = ap.parse_args()

    pe = _build_estimator_from_args(args)
    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(args.input)
    boxes, kpts, scores = pe.infer(img)
    vis = pe.annotate(img, boxes, kpts, scores)
    cv2.imwrite(args.output, vis)
    print(f"Saved {args.output} — persons: {len(boxes)}")

def video_main():
    ap = argparse.ArgumentParser("RTMO video (ONNX Runtime) — presets or --onnx")
    _add_common_preset_args(ap)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="out.mp4")
    ap.add_argument("--show", action="store_true", help="also preview while writing")
    args = ap.parse_args()

    pe = _build_estimator_from_args(args)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(args.input)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.output, fourcc, fps, (W, H))
    if args.show:
        cv2.namedWindow("rtmo-video", cv2.WINDOW_NORMAL)

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        boxes, kpts, scores = pe.infer(frame)
        vis = pe.annotate(frame, boxes, kpts, scores)
        writer.write(vis); n += 1
        if args.show:
            cv2.imshow("rtmo-video", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    writer.release(); cap.release()
    if args.show: cv2.destroyAllWindows()
    print(f"Wrote {args.output} ({n} frames)")
