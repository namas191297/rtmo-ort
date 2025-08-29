from __future__ import annotations
import numpy as np, cv2
from typing import Tuple, Optional, List

try:
    import onnxruntime as ort
except Exception:
    ort = None

# COCO-17 skeleton (with simple face links)
SKELETON: List[Tuple[int, int]] = [
    (0,1),(0,2),(1,3),(2,4),
    (5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16)
]

class PoseEstimatorORT:
    """
    Minimal RTMO wrapper using ONNX Runtime.
    Assumes ONNX outputs: dets (N x D: [x1,y1,x2,y2,score,...]) and keypoints (N x 17 x 3).
    RTMO exports usually include NMS inside the ONNX.
    """
    def __init__(
        self,
        onnx_path: str,
        device: str = "cpu",       # "cpu" or "cuda"
        size: Optional[int] = None, # 416 or 640; auto if None
        letterbox: bool = True,
        score_thr: float = 0.15,
        kpt_thr: float = 0.20,
        max_det: int = 5,
    ):
        if ort is None:
            raise ImportError("onnxruntime not installed. Use extras: [cpu] or [gpu].")

        providers = ["CUDAExecutionProvider","CPUExecutionProvider"] if device.lower().startswith("cuda") \
                    else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        outs = [o.name for o in self.sess.get_outputs()]
        self.out_names = ["dets","keypoints"] if {"dets","keypoints"}.issubset(outs) else outs[:2]

        self.score_thr = float(score_thr)
        self.kpt_thr   = float(kpt_thr)
        self.max_det   = int(max_det)
        self.letterbox = bool(letterbox)
        self.size      = int(size) if size else self._infer_input_size()

    def _infer_input_size(self) -> int:
        shape = self.sess.get_inputs()[0].shape  # [1,3,H,W]
        try:
            return int(shape[-1])  # assume square
        except Exception:
            return 640

    @staticmethod
    def _letterbox(img: np.ndarray, size: int, color=(114,114,114)):
        h, w = img.shape[:2]
        r = min(size / w, size / h)
        nw, nh = int(round(w*r)), int(round(h*r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pw, ph = size - nw, size - nh
        pl, pr = pw//2, pw - pw//2
        pt, pb = ph//2, ph - ph//2
        out = cv2.copyMakeBorder(resized, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=color)
        return out, r, pl, pt

    def infer(self, frame_bgr: np.ndarray):
        """Returns (boxes Nx4, keypoints Nx17x3, scores Nx) in original image coords."""
        H, W = frame_bgr.shape[:2]

        if self.letterbox:
            lb, scale, pl, pt = self._letterbox(frame_bgr, self.size)
            blob = lb.astype(np.float32).transpose(2,0,1)[None, ...]
            post = ("lb", scale, pl, pt, W, H)
        else:
            resized = cv2.resize(frame_bgr, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            blob = resized.astype(np.float32).transpose(2,0,1)[None, ...]
            sx, sy = W/float(self.size), H/float(self.size)
            post = ("no_lb", sx, sy, W, H)

        dets, kpts = self.sess.run(self.out_names, {self.in_name: blob})
        dets, kpts = dets[0], kpts[0]  # (N,D), (N,17,3)

        if dets.size == 0:
            return (np.zeros((0,4), np.float32),
                    np.zeros((0,17,3), np.float32),
                    np.zeros((0,), np.float32))

        scores = dets[:,4] if dets.shape[1] > 4 else np.ones((dets.shape[0],), np.float32)
        keep = np.where(scores >= self.score_thr)[0]
        if keep.size:
            keep = keep[np.argsort(scores[keep])[::-1]][:self.max_det]
        else:
            return (np.zeros((0,4), np.float32),
                    np.zeros((0,17,3), np.float32),
                    np.zeros((0,), np.float32))

        boxes = dets[keep, :4].astype(np.float32)
        scores = scores[keep].astype(np.float32)
        kpts   = kpts[keep].astype(np.float32)

        if post[0] == "lb":
            _, scale, pl, pt, W, H = post
            boxes[:,[0,2]] = (boxes[:,[0,2]] - pl) / max(1e-6, scale)
            boxes[:,[1,3]] = (boxes[:,[1,3]] - pt) / max(1e-6, scale)
            kpts[...,0] = (kpts[...,0] - pl) / max(1e-6, scale)
            kpts[...,1] = (kpts[...,1] - pt) / max(1e-6, scale)
        else:
            _, sx, sy, W, H = post
            boxes[:,[0,2]] *= sx; boxes[:,[1,3]] *= sy
            kpts[...,0] *= sx;  kpts[...,1] *= sy

        boxes[:,[0,2]] = np.clip(boxes[:,[0,2]], 0, W-1)
        boxes[:,[1,3]] = np.clip(boxes[:,[1,3]], 0, H-1)
        kpts[...,0] = np.clip(kpts[...,0], 0, W-1)
        kpts[...,1] = np.clip(kpts[...,1], 0, H-1)
        return boxes, kpts, scores

    def annotate(self, frame_bgr, boxes, kpts, scores=None):
        for i, (b, ks) in enumerate(zip(boxes, kpts)):
            x1,y1,x2,y2 = map(int, b[:4])
            cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), (60,220,60), 2)
            if scores is not None:
                cv2.putText(frame_bgr, f"{float(scores[i]):.2f}", (x1, max(0,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60,220,60), 2, cv2.LINE_AA)
            for u,v in SKELETON:
                if ks[u,2] >= self.kpt_thr and ks[v,2] >= self.kpt_thr:
                    cv2.line(frame_bgr, (int(ks[u,0]),int(ks[u,1])),
                             (int(ks[v,0]),int(ks[v,1])), (0,180,255), 2, cv2.LINE_AA)
            for x,y,s in ks:
                if s >= self.kpt_thr:
                    cv2.circle(frame_bgr, (int(x),int(y)), 3, (255,255,255), -1)
        return frame_bgr
