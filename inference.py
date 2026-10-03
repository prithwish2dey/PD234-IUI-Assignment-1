"""
Run the trained YOLOv8+DINOv2 hybrid detector on an image, a folder of
images, or a video, and save the annotated result.

Usage:
    python inference.py --source path/to/image.jpg  --weights models/best_yolov8_dinov2.pt
    python inference.py --source path/to/video.mp4   --weights models/best_yolov8_dinov2.pt
    python inference.py --source path/to/folder/     --weights models/best_yolov8_dinov2.pt

Output is written next to --out (default: test_outputs/).
"""
import argparse
import sys
from pathlib import Path

import cv2
import torch

# DINOv2 is loaded via torch.hub; its pickled classes need the cached repo
# on sys.path to be importable again when we unpickle a saved checkpoint.
_dino_hub_dir = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
if _dino_hub_dir.exists():
    sys.path.insert(0, str(_dino_hub_dir))

from models.dino_yolo import YOLOv8DINOv2
try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression
from ultralytics.data.augment import LetterBox

ROOT = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def load_model(weights, device):
    ckpt = torch.load(weights, map_location=device)
    model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = model.float().to(device).eval()
    names = model.names if hasattr(model, "names") else {i: str(i) for i in range(80)}
    return model, names


def preprocess(img, imgsz, device):
    lb = LetterBox(new_shape=(imgsz, imgsz), auto=False)
    im = lb(image=img)
    im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    im = torch.from_numpy(im.copy()).to(device).float() / 255.0
    return im.unsqueeze(0)


def draw(img, boxes, scores, classes, names, scale_x, scale_y):
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
        label = f"{names[int(c)]} {s:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 2, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


@torch.no_grad()
def predict_frame(model, frame, names, imgsz, device, conf, iou):
    h0, w0 = frame.shape[:2]
    im = preprocess(frame, imgsz, device)
    pred = model(im)
    pred = pred[0] if isinstance(pred, (list, tuple)) else pred
    det = non_max_suppression(pred, conf_thres=conf, iou_thres=iou)[0]
    out = frame.copy()
    if det is not None and len(det):
        scale_x, scale_y = w0 / imgsz, h0 / imgsz
        boxes = det[:, :4].cpu().numpy()
        scores = det[:, 4].cpu().numpy()
        classes = det[:, 5].cpu().numpy()
        out = draw(out, boxes, scores, classes, names, scale_x, scale_y)
    return out


def run_image(model, names, path, out_dir, imgsz, device, conf, iou):
    img = cv2.imread(str(path))
    out = predict_frame(model, img, names, imgsz, device, conf, iou)
    dst = out_dir / f"{path.stem}_pred.jpg"
    cv2.imwrite(str(dst), out)
    print(f"Saved: {dst}")


def run_video(model, names, path, out_dir, imgsz, device, conf, iou):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst = out_dir / f"{path.stem}_pred.mp4"
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = predict_frame(model, frame, names, imgsz, device, conf, iou)
        writer.write(out)
        n += 1
    cap.release()
    writer.release()
    print(f"Processed {n} frames. Saved: {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", default=str(ROOT / "models" / "best_yolov8_dinov2.pt"))
    ap.add_argument("--out", default=str(ROOT / "test_outputs"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model, names = load_model(args.weights, device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(args.source)

    if src.is_dir():
        for p in sorted(src.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                run_image(model, names, p, out_dir, args.imgsz, device, args.conf, args.iou)
    elif src.suffix.lower() in VID_EXTS:
        run_video(model, names, src, out_dir, args.imgsz, device, args.conf, args.iou)
    elif src.suffix.lower() in IMG_EXTS:
        run_image(model, names, src, out_dir, args.imgsz, device, args.conf, args.iou)
    else:
        raise ValueError(f"Unsupported source: {src}")


if __name__ == "__main__":
    main()
