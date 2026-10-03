"""
Fine-tune the YOLOv8 + DINOv2 hybrid detector on COCO128, starting from
COCO-pretrained YOLOv8n weights (not from scratch).

Usage:
    source .venv/bin/activate
    python train.py --epochs 100 --imgsz 640 --batch 16

Results (weights, curves, confusion matrix, PR curves, metrics.csv) are
saved under results/<run-name>/ and copied into results/plots and
results/metrics for easy access.
"""
import argparse
import shutil
import sys
from pathlib import Path

# DINOv2 is loaded via torch.hub; its pickled classes need the cached repo
# on sys.path to be importable again when ultralytics unpickles a saved
# checkpoint (e.g. on --resume). Must happen before any ultralytics import
# that might load a checkpoint.
_dino_hub_dir = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
if _dino_hub_dir.exists():
    sys.path.insert(0, str(_dino_hub_dir))

import torch
from ultralytics.utils import RANK
try:
    from ultralytics.nn.tasks import attempt_load_one_weight
except ImportError:
    from ultralytics.nn.tasks import load_checkpoint as attempt_load_one_weight
from ultralytics.models.yolo.detect import DetectionTrainer

from models.dino_yolo import YOLOv8DINOv2

ROOT = Path(__file__).resolve().parent
PRETRAINED_YOLO = "yolov8n.pt"  # COCO-pretrained backbone/neck/head weights
INIT_FROM = None  # set from --continue_from; loads a prior hybrid-model checkpoint's weights


class DinoDetectionTrainer(DetectionTrainer):
    """DetectionTrainer that builds a YOLOv8DINOv2 hybrid model and warm
    starts its backbone/neck/head from COCO-pretrained YOLOv8n weights, so
    training is fine-tuning rather than learning from random init. Only the
    DINOv2 projection + fusion conv (not present in the pretrained
    checkpoint) start randomly initialized.

    If INIT_FROM is set (a previous run's last.pt/best.pt), its weights are
    loaded on top of the COCO-pretrained warm start, carrying over whatever
    that earlier run already learned (including the DINOv2 fusion layers).
    This is simpler and more robust than ultralytics' --resume (which
    requires the exact same run/epoch-budget/optimizer state); here we just
    start a fresh optimizer + LR schedule from better-than-random weights.
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = YOLOv8DINOv2(
            cfg=cfg or self.args.model,
            ch=3,
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )

        pretrained, _ = attempt_load_one_weight(PRETRAINED_YOLO)
        csd = pretrained.float().state_dict()
        model_sd = model.state_dict()
        matched = {k: v for k, v in csd.items() if k in model_sd and model_sd[k].shape == v.shape}
        model.load_state_dict(matched, strict=False)
        if verbose and RANK == -1:
            print(f"Loaded {len(matched)}/{len(model_sd)} tensors from {PRETRAINED_YOLO} "
                  f"(backbone/neck/head warm start; DINOv2 + fusion layers stay fresh).")

        if INIT_FROM:
            prev, _ = attempt_load_one_weight(INIT_FROM)
            prev_sd = prev.float().state_dict()
            model_sd = model.state_dict()
            matched2 = {k: v for k, v in prev_sd.items() if k in model_sd and model_sd[k].shape == v.shape}
            model.load_state_dict(matched2, strict=False)
            if verbose and RANK == -1:
                print(f"Loaded {len(matched2)}/{len(model_sd)} tensors from previous checkpoint {INIT_FROM} "
                      f"(continuing from where that run left off).")

        if weights:
            model.load(weights)
        return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "coco128.yaml"))
    ap.add_argument("--model", default="yolov8n.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="yolov8_dinov2_coco128")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--freeze", type=int, default=10,
                     help="Freeze the first N backbone layers (0-9 = full yolov8n backbone incl. SPPF) "
                          "so fine-tuning only updates the neck/head + new DINOv2 fusion layers. "
                          "Use --freeze 0 to fine-tune the whole network.")
    ap.add_argument("--lr0", type=float, default=0.001,
                     help="Lower initial LR than train-from-scratch default (0.01), appropriate for fine-tuning.")
    ap.add_argument("--continue_from", default=None,
                     help="Path to a previous run's checkpoint (e.g. results/<run>/weights/last.pt) whose "
                          "weights to carry over before starting this (fresh-schedule) training run.")
    args = ap.parse_args()

    global INIT_FROM
    INIT_FROM = args.continue_from

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {'cuda:0 (' + torch.cuda.get_device_name(0) + ')' if device == 0 else 'cpu'}")

    overrides = {
        "data": args.data,
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "project": str(ROOT / "results"),
        "name": args.name,
        "patience": args.patience,
        "plots": True,
        "val": True,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "auto",
        "lr0": args.lr0,
        "seed": 0,
    }
    if args.freeze > 0:
        overrides["freeze"] = args.freeze

    trainer = DinoDetectionTrainer(overrides=overrides)
    trainer.train()

    run_dir = Path(trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, ROOT / "models" / "best_yolov8_dinov2.pt")
        print(f"Best weights copied to models/best_yolov8_dinov2.pt")

    plots_dir = ROOT / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for png in run_dir.glob("*.png"):
        shutil.copy(png, plots_dir / png.name)
    for jpg in run_dir.glob("*.jpg"):
        shutil.copy(jpg, plots_dir / jpg.name)

    metrics_dir = ROOT / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv = run_dir / "results.csv"
    if csv.exists():
        shutil.copy(csv, metrics_dir / "results.csv")

    print(f"\nAll training artifacts saved under: {run_dir}")
    print(f"Plots copied to: {plots_dir}")
    print(f"Metrics copied to: {metrics_dir}")


if __name__ == "__main__":
    main()
