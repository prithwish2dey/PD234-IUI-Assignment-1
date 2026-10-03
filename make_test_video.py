"""
Builds a short demo video out of COCO128 images (since no video ships with
the dataset) so inference.py has something to run on for the video demo.

Usage:
    python make_test_video.py --out test_outputs/demo_input.mp4
"""
import argparse
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "datasets" / "coco128" / "images" / "train2017"))
    ap.add_argument("--out", default=str(ROOT / "test_outputs" / "demo_input.mp4"))
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    img_dir = Path(args.images)
    imgs = sorted(img_dir.glob("*.jpg"))[: args.n]
    if not imgs:
        raise RuntimeError(f"No images found in {img_dir}. Run train.py first to trigger the COCO128 download.")

    first = cv2.imread(str(imgs[0]))
    h, w = first.shape[:2]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    for p in imgs:
        img = cv2.imread(str(p))
        img = cv2.resize(img, (w, h))
        writer.write(img)
    writer.release()
    print(f"Saved demo video with {len(imgs)} frames to {out_path}")


if __name__ == "__main__":
    main()
