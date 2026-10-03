"""
Builds one clean, clearly-labeled results figure from results/metrics/results.csv
for use in the README / GitHub submission / assignment write-up.

Usage:
    python plot_results.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "results" / "metrics" / "results.csv"
OUT = ROOT / "results" / "plots" / "training_metrics.png"


def main():
    df = pd.read_csv(CSV)
    df.columns = [c.strip() for c in df.columns]
    epoch = df["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(epoch, df["train/box_loss"], label="Box loss (train)", color="#d62728")
    ax.plot(epoch, df["train/cls_loss"], label="Class loss (train)", color="#1f77b4")
    ax.plot(epoch, df["train/dfl_loss"], label="DFL loss (train)", color="#2ca02c")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("YOLOv8+DINOv2 Fine-Tuning Loss on COCO128")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epoch, df["metrics/precision(B)"], label="Precision", color="#9467bd")
    ax.plot(epoch, df["metrics/recall(B)"], label="Recall", color="#ff7f0e")
    ax.plot(epoch, df["metrics/mAP50(B)"], label="mAP@0.5", color="#2ca02c", linewidth=2)
    ax.plot(epoch, df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", color="#d62728", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("YOLOv8+DINOv2 Validation Metrics on COCO128")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle("YOLOv8 + DINOv2 Hybrid Detector — Fine-Tuning on COCO128", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
