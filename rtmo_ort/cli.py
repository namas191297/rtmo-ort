import argparse, cv2, time, os
from .estimator import PoseEstimatorORT

def _build_estimator(args):
    return PoseEstimatorORT(
        args.onnx,
        device=args.device,
        size=args.size,
        letterbox=not args.no_letterbox,
        score_thr=args.score_thr,
        kpt_thr=args.kpt_thr,
        max_det=args.max_det,
    )

def webcam_main():
    ap = argparse.ArgumentParser("RTMO webcam (ONNX Runtime)")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    ap.add_argument("--size", type=int, default=None)  # auto if None
    ap.add_argument("--no-letterbox", action="store_true")
    ap.add_argument("--score-thr", type=float, default=0.15)
    ap.add_argument("--kpt-thr", type=float, default=0.20)
    ap.add_argument("--max-det", type=int, default=5)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--window", type=str, default="rtmo-ort")
    args = ap.parse_args()

    pe = _build_estimator(args)
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
    ap = argparse.ArgumentParser("RTMO image (ONNX Runtime)")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--no-letterbox", action="store_true")
    ap.add_argument("--score-thr", type=float, default=0.15)
    ap.add_argument("--kpt-thr", type=float, default=0.20)
    ap.add_argument("--max-det", type=int, default=5)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="vis.jpg")
    args = ap.parse_args()

    pe = _build_estimator(args)
    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(args.input)
    boxes, kpts, scores = pe.infer(img)
    vis = pe.annotate(img, boxes, kpts, scores)
    cv2.imwrite(args.output, vis)
    print(f"Saved {args.output} — persons: {len(boxes)}")

def video_main():
    ap = argparse.ArgumentParser("RTMO video (ONNX Runtime)")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--no-letterbox", action="store_true")
    ap.add_argument("--score-thr", type=float, default=0.15)
    ap.add_argument("--kpt-thr", type=float, default=0.20)
    ap.add_argument("--max-det", type=int, default=5)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="out.mp4")
    ap.add_argument("--show", action="store_true", help="also preview while writing")
    args = ap.parse_args()

    pe = _build_estimator(args)
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
