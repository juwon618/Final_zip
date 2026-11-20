"""X-Guard AI AE (Inspection, origin CInspection).

Gets each image through the xrayImages subscription (sub-ai, step 5), runs
models/detect.pt on it, and writes the verdict to inspectionResult (step 6).
On FAIL it also adds a cin to defects.

MODEL_BACKEND in .env picks how the model is loaded:
  torchscript  default, torch.jit.load(detect.pt). See "detect.pt format" in the README.
  ultralytics  load the file as an ultralytics YOLO checkpoint
  dummy        no model; FAIL if the file name contains a class name

Run on the PC:  python ai/app.py
Run in Docker:  docker build -f ai/Dockerfile -t xguard-ai .  &&  docker run -p 9000:9000 --env-file .env xguard-ai
"""
import base64
import io
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import m2m  # noqa: E402

ORIG = "CInspection"
PORT = int(os.getenv("AI_PORT", "9000"))
MODEL_PATH = ROOT / os.getenv("MODEL_PATH", "models/detect.pt")
CLASSES_PATH = ROOT / os.getenv("CLASSES_PATH", "models/classes.txt")
BACKEND = os.getenv("MODEL_BACKEND", "torchscript").lower()
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
MODEL_INFO = MODEL_PATH.parent / "model_info.json"  # written by training (imgsz and so on)
if MODEL_INFO.exists():
    try:
        IMG_SIZE = int(json.loads(MODEL_INFO.read_text(encoding="utf-8")).get("imgsz", IMG_SIZE))
    except (ValueError, OSError):
        pass
CONF_TH = float(os.getenv("CONF_TH", "0.5"))
IOU_TH = float(os.getenv("IOU_TH", "0.45"))
MODEL_VER = os.getenv("MODEL_VER", "detect-xguard-1.0")
RESULT_MODE = os.getenv("RESULT_MODE", "flex")
# Only set these if the model was trained with mean/std normalization,
# e.g. NORM_MEAN=0.485,0.456,0.406 NORM_STD=0.229,0.224,0.225
NORM_MEAN = os.getenv("NORM_MEAN", "")
NORM_STD = os.getenv("NORM_STD", "")


def load_classes():
    if CLASSES_PATH.exists():
        names = [l.strip() for l in CLASSES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
        return dict(enumerate(names))
    return {0: "defect"}   # our dataset classes, same order as obj.names


# ====================================================================== pre/post-processing
def letterbox(img, size):
    """Resize to size x size keeping the aspect ratio and pad with gray (114).

    Returns (canvas, scale, pad_x, pad_y).
    """
    w, h = img.size
    s = min(size / w, size / h)
    nw, nh = round(w * s), round(h * s)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    return canvas, s, px, py


def to_tensor(img):
    import numpy as np
    import torch
    a = np.asarray(img, dtype=np.float32) / 255.0          # HWC, RGB, 0~1
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
    if NORM_MEAN and NORM_STD:
        mean = torch.tensor([float(v) for v in NORM_MEAN.split(",")]).view(1, 3, 1, 1)
        std = torch.tensor([float(v) for v in NORM_STD.split(",")]).view(1, 3, 1, 1)
        t = (t - mean) / std
    return t.contiguous()


def nms(boxes, scores, iou_th):
    import torch
    try:
        import torchvision
        return torchvision.ops.nms(boxes, scores, iou_th)
    except Exception:
        order = scores.argsort(descending=True)
        keep = []
        while order.numel():
            i = order[0]
            keep.append(int(i))
            if order.numel() == 1:
                break
            rest = order[1:]
            xx1 = torch.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = torch.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = torch.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = torch.minimum(boxes[i, 3], boxes[rest, 3])
            inter = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)
            area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
            iou = inter / (area(boxes[i]) + area(boxes[rest]) - inter + 1e-9)
            order = rest[iou <= iou_th]
        return torch.tensor(keep, dtype=torch.long)


def decode_output(out):
    """Turn raw model output into Tensor[N, 6] (x1, y1, x2, y2, conf, cls) in input (640) pixels.

    Accepted shapes:
      A) [N, 6] or [1, N, 6]        boxes with NMS already applied (preferred)
      B) [1, 4 + num_classes, 8400] raw YOLOv8 head (cx, cy, w, h + class scores); NMS is done here
    If the model returns a tuple or list, the first element is used.
    """
    import torch
    if isinstance(out, (list, tuple)):
        out = out[0]
    if isinstance(out, dict):
        out = out.get("boxes", next(iter(out.values())))
    out = out.detach().float().cpu()
    if out.dim() == 3 and out.shape[-1] == 6:
        out = out[0]
    if out.dim() == 2 and out.shape[-1] == 6:
        return out
    if out.dim() == 3:  # shape B
        p = out[0]
        if p.shape[0] > p.shape[1]:  # came out as [8400, 4+nc]
            p = p.T
        xywh, cls_scores = p[:4].T, p[4:].T
        conf, cls = cls_scores.max(1)
        m = conf >= min(CONF_TH, 0.25)
        xywh, conf, cls = xywh[m], conf[m], cls[m]
        if not len(conf):
            return torch.zeros((0, 6))
        xyxy = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
        keep = nms(xyxy + cls[:, None] * 4096, conf, IOU_TH)[:100]  # offset by class so NMS runs per class
        return torch.cat([xyxy[keep], conf[keep, None], cls[keep, None].float()], 1)
    raise ValueError(f"알 수 없는 모델 출력 모양: {tuple(out.shape)}")


# ====================================================================== model
def is_torchscript(path):
    """True if the file was saved with torch.jit.save (an ultralytics best.pt is not)."""
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            return any(n.endswith("constants.pkl") for n in z.namelist())
    except (zipfile.BadZipFile, OSError):
        return False


class Detector:
    def __init__(self):
        global MODEL_VER
        self.names = load_classes()
        self.model = None
        self.backend = BACKEND
        self.arch = ""
        if BACKEND == "dummy":
            print("MODEL_BACKEND=dummy (모델 없이 파일 이름으로 판정)")
            MODEL_VER = "dummy"
            return
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"모델 파일이 없어요: {MODEL_PATH}")
        # Trust the actual file type over MODEL_BACKEND. detect.pt is TorchScript and
        # best.pt is an ultralytics checkpoint; loading one as the other crashes on startup.
        ts = is_torchscript(MODEL_PATH)
        if ts and self.backend == "ultralytics":
            print(f"{MODEL_PATH.name} 은 TorchScript 라서 MODEL_BACKEND=ultralytics 대신 torchscript 로 읽어요", flush=True)
            self.backend = "torchscript"
        elif not ts and self.backend != "ultralytics":
            print(f"{MODEL_PATH.name} 은 ultralytics 체크포인트라 ultralytics 로 읽어요", flush=True)
            self.backend = "ultralytics"
        if self.backend != "ultralytics":
            import torch
            torch.set_grad_enabled(False)
            try:
                self.model = torch.jit.load(str(MODEL_PATH), map_location="cpu").eval()
            except Exception as e:  # not TorchScript after all (train.py left a best.pt), fall back
                print(f"TorchScript 로 못 읽어서 ultralytics 로 읽어요: {e!r}"[:200])
                self.backend = "ultralytics"
        if self.backend == "ultralytics":
            try:
                from ultralytics import YOLO
            except ImportError:
                raise SystemExit("!! 이 detect.pt 는 ultralytics 체크포인트라 ultralytics 가 필요해요: "
                                 "pip install ultralytics  (run_all.bat 은 자동 설치)")
            self.model = YOLO(str(MODEL_PATH))
            self.names = self.model.names
            try:
                base = Path(str((getattr(self.model, "ckpt", None) or {}).get("train_args", {}).get("model", ""))).stem
            except Exception:
                base = ""
            self.arch = base
        if MODEL_VER.strip().lower() in ("", "auto"):
            # MODEL_VER=auto: build the name from the loaded model. Use "name" from
            # model_info.json if there is one (e.g. yolov8n_400_single_b16), otherwise
            # something like yolov8m-detect-1116.
            info_name = ""
            if MODEL_INFO.exists():
                try:
                    info_name = str(json.loads(MODEL_INFO.read_text(encoding="utf-8")).get("name", ""))
                except (ValueError, OSError):
                    pass
            d = time.strftime("%m%d", time.localtime(MODEL_PATH.stat().st_mtime))
            MODEL_VER = info_name or "-".join(x for x in (self.arch, MODEL_PATH.stem, d) if x)
        print(f"모델 로드: {MODEL_PATH.name} ({self.backend}) 버전={MODEL_VER} 클래스={self.names} 입력={IMG_SIZE}",
              flush=True)
        # The first inference takes 1-2 s. Run one now so the first product doesn't time out.
        t0 = time.perf_counter()
        self(Image.new("RGB", (IMG_SIZE, IMG_SIZE * 5 // 8), (114, 114, 114)))
        print(f"예열 추론 {(time.perf_counter() - t0) * 1000:.0f}ms", flush=True)

    def __call__(self, img, filename=""):
        """img (PIL RGB) -> [[x1, y1, x2, y2, conf, "class"], ...] in original image pixels."""
        if BACKEND == "dummy":
            for i, n in self.names.items():
                if n.lower() in filename.lower():
                    w, h = img.size
                    return [[w * 0.3, h * 0.3, w * 0.6, h * 0.6, 0.91, n]]
            return []
        if self.backend == "ultralytics":
            r = self.model(img, imgsz=IMG_SIZE, conf=0.25, verbose=False)[0]
            return [[*map(float, b.xyxy[0]), float(b.conf), r.names[int(b.cls)]] for b in r.boxes]
        import torch
        lb, s, px, py = letterbox(img, IMG_SIZE)
        with torch.no_grad():
            det = decode_output(self.model(to_tensor(lb)))
        boxes = []
        w, h = img.size
        for x1, y1, x2, y2, conf, cls in det.tolist():
            # undo the letterbox: input (640) coords -> original coords
            x1, x2 = [min(max((v - px) / s, 0), w) for v in (x1, x2)]
            y1, y2 = [min(max((v - py) / s, 0), h) for v in (y1, y2)]
            boxes.append([x1, y1, x2, y2, conf, self.names.get(int(cls), str(int(cls)))])
        return boxes


# ====================================================================== verdict (rules from the spec)
def to_result(con, raw_boxes, ms):
    boxes = [[*[round(float(v), 1) for v in b[:4]], round(float(b[4]), 3), b[5]] for b in raw_boxes]
    top = max(boxes, key=lambda b: b[4], default=None)
    fail = top is not None and top[4] >= CONF_TH
    result = {
        "seq": int(con["seq"]), "productId": con.get("productId", ""),
        "verdict": "FAIL" if fail else "PASS",
        "defectClass": top[5] if fail else "",
        "confidence": top[4] if top else 0.0,
        "inferenceMs": int(ms), "modelVer": MODEL_VER,
    }
    return result, [b for b in boxes if b[4] >= CONF_TH]


def error_result(con):
    return {"seq": int(con.get("seq", 0)), "productId": con.get("productId", ""), "verdict": "ERROR",
            "defectClass": "", "confidence": 0.0, "inferenceMs": 0, "modelVer": MODEL_VER}


def publish(result, boxes):
    """Step 6: update inspectionResult, and add a defects cin on FAIL."""
    if RESULT_MODE == "flex":
        r = m2m.update("Inspection/inspectionResult", ORIG, {"xg:inRet": result})
    else:
        r = m2m.add_cin("Inspection/inspectionResult", ORIG, result)
    if m2m.rsc(r) not in (2001, 2004):
        print("!! inspectionResult 쓰기 실패", m2m.rsc(r), r.text[:200], flush=True)
    if result["verdict"] == "FAIL":
        m2m.add_cin("Inspection/defects", ORIG, {**result, "boxes": boxes})


# ====================================================================== server
app = Flask(__name__)
jobs = queue.Queue()
detector = None
stats = {"done": 0, "last": None}


def worker():
    while True:
        con = jobs.get()
        try:
            img = Image.open(io.BytesIO(base64.b64decode(con["img"]))).convert("RGB")
        except Exception as e:  # unreadable image -> ERROR (the rejector rejects ERROR too)
            print("이미지 읽기 실패:", e, flush=True)
            publish(error_result(con), [])
            continue
        try:
            t0 = time.perf_counter()
            raw = detector(img, con.get("file", ""))
            ms = (time.perf_counter() - t0) * 1000
            result, boxes = to_result(con, raw, ms)
        except Exception as e:
            print("추론 실패:", repr(e), flush=True)
            result, boxes = error_result(con), []
        publish(result, boxes)
        stats["done"] += 1
        stats["last"] = result
        print(f"⑤ #{result['seq']} {con.get('file', '')} -> ⑥ {result['verdict']} {result['defectClass']} "
              f"{result['confidence']} ({result['inferenceMs']}ms)", flush=True)


@app.post("/notify")
@app.post("/notify/<path:_rest>")  # ACME may append the nu path to the poa
def notify(_rest=None):
    rep, sur, vrq = m2m.parse_notification(request.get_json(silent=True))
    if not vrq:
        con = m2m.content_of(rep)
        if con and "img" in con:
            jobs.put(con)
    return m2m.notify_ok()


@app.get("/health")
def health():
    return jsonify({"backend": detector.backend, "model": MODEL_PATH.name, "modelVer": MODEL_VER,
                    "classes": detector.names, **stats})


if __name__ == "__main__":
    detector = Detector()
    threading.Thread(target=worker, daemon=True).start()
    print(f"AI AE: http://0.0.0.0:{PORT}/notify  CSE={m2m.CSE_URL}  RESULT_MODE={RESULT_MODE}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
