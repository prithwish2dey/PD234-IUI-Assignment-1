"""
Picks a handful of COCO128 images, runs the trained hybrid model on them,
and saves the annotated predictions into test_outputs/images/ as the
"good output" sample set requested for grading/demo purposes.

Usage:
    python make_test_set.py --n 10
"""
import argparse
import random
from pathlib import Path

from inference import load_model, predict_frame
import cv2
import torch

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "datasets" / "coco128" / "images" / "train2017"))
    ap.add_argument("--weights", default=str(ROOT / "models" / "best_yolov8_dinov2.pt"))
    ap.add_argument("--out", default=str(ROOT / "test_outputs" / "images"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, names = load_model(args.weights, device)

    img_dir = Path(args.images)
    all_imgs = sorted(img_dir.glob("*.jpg"))
    random.seed(0)
    sample = random.sample(all_imgs, min(args.n, len(all_imgs)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in sample:
        img = cv2.imread(str(p))
        out = predict_frame(model, img, names, 640, device, args.conf, 0.45)
        dst = out_dir / f"{p.stem}_pred.jpg"
        cv2.imwrite(str(dst), out)
        print(f"Saved: {dst}")


if __name__ == "__main__":
    main()
