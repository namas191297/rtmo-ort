#!/usr/bin/env python3
import argparse, time, cv2, numpy as np, onnxruntime as ort

# minimal COCO-17 skeleton
# Full COCO-17 skeleton (adds face links)
SKELETON = [
    (0,1), (0,2), (1,3), (2,4),      # face: nose↔eyes↔ears
    (5,7), (7,9), (6,8), (8,10),     # arms
    (5,6),                           # shoulders
    (5,11), (6,12), (11,12),         # torso/hips
    (11,13), (13,15), (12,14), (14,16)  # legs
]


def letterbox(img, size=416, color=(114,114,114)):
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(round(w*r)), int(round(h*r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pw, ph = size - nw, size - nh
    pl, pr = pw//2, pw - pw//2
    pt, pb = ph//2, ph - ph//2
    out = cv2.copyMakeBorder(resized, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=color)
    return out, r, pl, pt  # pad-left, pad-top

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="path to end2end.onnx")
    ap.add_argument("--size", type=int, default=416, help="model input size (e.g., 416 or 640)")
    ap.add_argument("--score-thr", type=float, default=0.30, help="min box score to show")
    ap.add_argument("--kpt-thr",   type=float, default=0.30, help="min keypoint score to draw")
    ap.add_argument("--max-det",   type=int, default=5, help="show at most N detections")
    ap.add_argument("--no-letterbox", action="store_true",
                    help="do NOT letterbox; just resize to --size (may distort)")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--gpu", action="store_true", help="use CUDAExecutionProvider if available")
    args = ap.parse_args()

    providers = ["CUDAExecutionProvider","CPUExecutionProvider"] if args.gpu else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(args.onnx, providers=providers)
    in_name = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    if "dets" in outs and "keypoints" in outs:  # prefer explicit names
        outs = ["dets","keypoints"]

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open webcam {args.cam}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    cv2.namedWindow("rtmo-onnx", cv2.WINDOW_NORMAL)

    t0, fps = time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok: break

        # --- preprocess ---
        if args.no_letterbox:
            # straight resize (distorts aspect)
            resized = cv2.resize(frame, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
            blob = resized.astype(np.float32).transpose(2,0,1)[None, ...]  # 1x3xHxW
            # scales to map back
            sx = W / float(args.size)
            sy = H / float(args.size)
            pl = pt = None  # not used
        else:
            # letterbox to keep aspect (pads with 114)
            lb, r, pl, pt = letterbox(frame, args.size, (114,114,114))
            blob = lb.astype(np.float32).transpose(2,0,1)[None, ...]
            # single isotropic scale
            scale = r

        # --- inference ---
        dets, kpts = sess.run(outs, {in_name: blob})
        dets, kpts = dets[0], kpts[0]  # (N,D), (N,17,3)

        if dets.size:
            scores = dets[:,4] if dets.shape[1] > 4 else np.ones((dets.shape[0],), np.float32)
            keep = np.where(scores >= args.score_thr)[0]
            if keep.size:
                keep = keep[np.argsort(scores[keep])[::-1]][:args.max_det]
                boxes = dets[keep, :4].astype(np.float32)
                sc    = scores[keep].astype(np.float32)
                kp    = kpts[keep]

                # --- map back to original frame coords ---
                if args.no_letterbox:
                    # model coords -> original: multiply by per-axis scale
                    boxes[:,[0,2]] *= sx; boxes[:,[1,3]] *= sy
                    kp[...,0] *= sx;  kp[...,1] *= sy
                else:
                    # unletterbox: subtract pad then divide by scale
                    boxes[:,[0,2]] = (boxes[:,[0,2]] - pl) / max(1e-6, scale)
                    boxes[:,[1,3]] = (boxes[:,[1,3]] - pt) / max(1e-6, scale)
                    kp[...,0] = (kp[...,0] - pl) / max(1e-6, scale)
                    kp[...,1] = (kp[...,1] - pt) / max(1e-6, scale)

                boxes[:,[0,2]] = np.clip(boxes[:,[0,2]], 0, W-1)
                boxes[:,[1,3]] = np.clip(boxes[:,[1,3]], 0, H-1)
                kp[...,0] = np.clip(kp[...,0], 0, W-1)
                kp[...,1] = np.clip(kp[...,1], 0, H-1)

                # --- draw ---
                for b, ks, s in zip(boxes, kp, sc):
                    x1,y1,x2,y2 = map(int, b)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (60,220,60), 2)
                    cv2.putText(frame, f"{s:.2f}", (x1, max(0,y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60,220,60), 2, cv2.LINE_AA)
                    # skeleton lines
                    for (u,v) in SKELETON:
                        if ks[u,2] >= args.kpt_thr and ks[v,2] >= args.kpt_thr:
                            cv2.line(frame, (int(ks[u,0]),int(ks[u,1])),
                                     (int(ks[v,0]),int(ks[v,1])), (0,180,255), 2)
                    # keypoints
                    for x,y,ss in ks:
                        if ss >= args.kpt_thr:
                            cv2.circle(frame, (int(x),int(y)), 3, (255,255,255), -1)

        # FPS
        t1 = time.time(); fps = 0.9*fps + 0.1*(1.0/max(1e-3, t1-t0)); t0 = t1
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imshow("rtmo-onnx", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
