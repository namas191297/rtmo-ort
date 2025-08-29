# rtmo_ort/estimator.py
import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, List

# ---------- Skeletons ----------
# COCO-17 indices (0..16): 5 face (0..4), shoulders 5,6; etc.
COCO17_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6), (0, 1), (0, 2), (1, 3), (2, 4),
]

# CrowdPose-14 (MMPose order):
# 0 LS,1 RS,2 LE,3 RE,4 LW,5 RW,6 LHip,7 RHip,8 LKnee,9 RKnee,
# 10 LAnkle,11 RAnkle,12 TopHead,13 Neck
CROWDPOSE14_EDGES = [
    (12, 13),
    (13, 0), (13, 1),
    (0, 2), (2, 4),
    (1, 3), (3, 5),
    (6, 8), (8,10),
    (7, 9), (9,11),
    (0, 1), (6, 7),
]

# Body7 (only used if K==7; many “body7” exports still output 17)
BODY7_EDGES = [
    (0, 1), (0, 2), (1, 2),
    (1, 3), (2, 4), (3, 4),
    (3, 5), (4, 6),
]

def _edges_for_k(K: int) -> List[tuple]:
    """Prefer choosing skeleton by actual number of keypoints."""
    if K >= 17: return COCO17_EDGES
    if K == 14: return CROWDPOSE14_EDGES
    if K == 7:  return BODY7_EDGES
    return []  # unknown -> draw points only

# ---------- Utilities ----------
def _letterbox(img: np.ndarray, new_size: int, color=(114,114,114)) -> Tuple[np.ndarray, float, Tuple[int,int]]:
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=img.dtype)
    px = (new_size - nw) // 2
    py = (new_size - nh) // 2
    canvas[py:py+nh, px:px+nw] = resized
    return canvas, scale, (px, py)

def _resize_stretch(img: np.ndarray, new_size: int) -> Tuple[np.ndarray, Tuple[float,float]]:
    h, w = img.shape[:2]
    resized = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
    sx = w / float(new_size)
    sy = h / float(new_size)
    return resized, (sx, sy)

def _to_chw(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.float32: img = img.astype(np.float32)
    return img.transpose(2, 0, 1)

def _providers(device: str):
    pref = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    avail = ort.get_available_providers()
    return [p for p in pref if p in avail] or ["CPUExecutionProvider"]

# ---------- Estimator ----------
class PoseEstimatorORT:
    def __init__(self, onnx_path: str, device="cpu", size=None, letterbox=True,
                 score_thr=0.15, kpt_thr=0.20, max_det=5, layout: str=None):
        self.onnx_path = onnx_path
        self.device = device
        self.letterbox = letterbox
        self.score_thr = float(score_thr)
        self.kpt_thr = float(kpt_thr)
        self.max_det = int(max_det)

        self.sess = ort.InferenceSession(onnx_path, providers=_providers(device))
        self.input_name = self.sess.get_inputs()[0].name
        ishape = self.sess.get_inputs()[0].shape  # [B,3,H,W] or dynamic

        self.size = size
        if self.size is None:
            try:
                H = int(ishape[2]) if isinstance(ishape[2], (int, np.integer)) else None
                W = int(ishape[3]) if isinstance(ishape[3], (int, np.integer)) else None
                if H and W and H == W: self.size = H
            except Exception: pass
        if self.size is None:
            base = os.path.basename(onnx_path).lower()
            self.size = 416 if "_416x416_" in base or "_t_" in base else 640

        outs = self.sess.get_outputs()
        self.name_dets = self.name_kpts = None
        for o in outs:
            n = o.name.lower()
            if "keypoint" in n or "kpt" in n: self.name_kpts = o.name
            if "det" in n or "bbox" in n:     self.name_dets = o.name
        if self.name_kpts is None:
            for o in outs:
                if len(o.shape) == 4 and o.shape[-1] == 3: self.name_kpts = o.name; break
        if self.name_dets is None:
            for o in outs:
                if len(o.shape) == 3 and (o.shape[-1] is None or o.shape[-1] >= 5): self.name_dets = o.name; break
        if self.name_dets is None or self.name_kpts is None:
            raise RuntimeError(f"Could not locate dets/keypoints outputs. Found {[o.name for o in outs]}")

    def _preprocess(self, img):
        H0, W0 = img.shape[:2]; S = self.size
        if self.letterbox:
            proc, scale, (px, py) = _letterbox(img, S)
            meta = dict(mode="letterbox", scale=scale, pad=(px, py), orig=(W0, H0))
        else:
            proc, (sx, sy) = _resize_stretch(img, S)
            meta = dict(mode="stretch", scale_xy=(sx, sy), orig=(W0, H0))
        blob = _to_chw(proc)[None]
        return blob, meta

    def _post_boxes(self, boxes, meta):
        if boxes.size == 0: return boxes
        boxes = boxes.copy()
        if meta["mode"] == "letterbox":
            px, py = meta["pad"]; s = max(meta["scale"], 1e-6)
            boxes[:, [0,2]] = (boxes[:, [0,2]] - px) / s
            boxes[:, [1,3]] = (boxes[:, [1,3]] - py) / s
        else:
            sx, sy = meta["scale_xy"]
            boxes[:, [0,2]] *= sx; boxes[:, [1,3]] *= sy
        W0, H0 = meta["orig"]
        boxes[:, [0,2]] = np.clip(boxes[:, [0,2]], 0, W0-1)
        boxes[:, [1,3]] = np.clip(boxes[:, [1,3]], 0, H0-1)
        return boxes

    def _post_kpts(self, kpts, meta):
        if kpts.size == 0: return kpts
        out = kpts.copy()
        if meta["mode"] == "letterbox":
            px, py = meta["pad"]; s = max(meta["scale"], 1e-6)
            out[..., 0] = (out[..., 0] - px) / s
            out[..., 1] = (out[..., 1] - py) / s
        else:
            sx, sy = meta["scale_xy"]
            out[..., 0] *= sx; out[..., 1] *= sy
        W0, H0 = meta["orig"]
        out[..., 0] = np.clip(out[..., 0], 0, W0-1)
        out[..., 1] = np.clip(out[..., 1], 0, H0-1)
        return out

    def infer(self, img):
        blob, meta = self._preprocess(img)
        dets, kpts = self.sess.run([self.name_dets, self.name_kpts], {self.input_name: blob})
        dets = np.asarray(dets); kpts = np.asarray(kpts)
        dets = dets[0] if dets.ndim == 3 else dets
        kpts = kpts[0] if kpts.ndim == 4 else kpts

        if dets.size == 0:
            return (np.zeros((0,4), np.float32),
                    np.zeros((0,0,3), np.float32),
                    np.zeros((0,), np.float32))

        boxes_xyxy = dets[:, :4].astype(np.float32)
        scores = dets[:, 4].astype(np.float32)

        keep = scores >= self.score_thr
        boxes_xyxy = boxes_xyxy[keep]
        if kpts.shape[0] == keep.shape[0]: kpts = kpts[keep]
        scores = scores[keep]

        if boxes_xyxy.shape[0] > self.max_det:
            idx = np.argsort(-scores)[: self.max_det]
            boxes_xyxy, kpts, scores = boxes_xyxy[idx], kpts[idx], scores[idx]

        boxes_xyxy = self._post_boxes(boxes_xyxy, meta)
        kpts = self._post_kpts(kpts, meta).astype(np.float32)
        return boxes_xyxy, kpts, scores

    def annotate(self, img, boxes, kpts, scores):
        vis = img.copy()
        for xyxy, sc in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.putText(vis, f"{sc:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,180,255), 1, cv2.LINE_AA)

        if kpts.size == 0: return vis
        N, K, _ = kpts.shape
        edges = _edges_for_k(K)  # choose by actual K

        for pid in range(N):
            ks = kpts[pid]
            for j in range(K):
                x, y, conf = ks[j]
                if conf < self.kpt_thr: continue
                cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
            for (u, v) in edges:
                if u >= K or v >= K: continue
                if ks[u, 2] >= self.kpt_thr and ks[v, 2] >= self.kpt_thr:
                    p1 = (int(ks[u, 0]), int(ks[u, 1])); p2 = (int(ks[v, 0]), int(ks[v, 1]))
                    cv2.line(vis, p1, p2, (0, 255, 255), 2)
        return vis
