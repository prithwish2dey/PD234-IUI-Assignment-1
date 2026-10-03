# YOLOv8 + DINOv2 Hybrid Object Detector (COCO128)

Implementation of the first work described at
https://cambum.net/I3DLab/AI4DM.htm — a YOLOv8 backbone fused with a
frozen DINOv2 foundation model's global embedding, injected at the
deepest backbone layer (SPPF output) to add semantic context on top of
YOLO's spatial features.

## Project layout

```
IUI-Assignment-1/
├── .venv/                      # virtual environment (create with setup below)
├── data/coco128.yaml           # dataset config (auto-downloads COCO128)
├── datasets/coco128/           # downloaded on first train.py run
├── models/
│   ├── dino_yolo.py            # YOLOv8DINOv2 hybrid model definition
│   └── best_yolov8_dinov2.pt   # best weights (created after training)
├── train.py                    # training entry point
├── inference.py                # run on an image / folder / video
├── make_test_video.py          # builds a demo video from COCO128 frames
├── make_test_set.py            # saves sample predicted images
├── results/
│   ├── <run-name>/              # full ultralytics run (weights, curves, logs)
│   ├── plots/                   # copied PNG/JPG plots (PR curve, confusion matrix, etc.)
│   └── metrics/results.csv      # per-epoch metrics
└── test_outputs/
    ├── images/                  # sample predicted images ("good output")
    ├── demo_input.mp4           # auto-built demo video
    └── demo_input_pred.mp4      # predicted/annotated demo video
```

## Setup

```bash
cd ~/Desktop/IUI-Assignment-1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# CUDA-enabled torch (adjust cu121 to your driver's CUDA version if needed):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Train

```bash
source .venv/bin/activate
python train.py --epochs 100 --imgsz 640 --batch 16
```

- First run auto-downloads COCO128 into `datasets/coco128/`.
- The DINOv2 backbone (`dinov2_vits14`) is downloaded from `facebookresearch/dinov2`
  via `torch.hub` and kept **frozen**; only the YOLOv8 backbone/neck/head,
  the DINOv2 projection conv and the fusion conv are trained.
- Training curves, PR/F1 curves, confusion matrix, and `results.csv` land in
  `results/yolov8_dinov2_coco128/` and are copied into `results/plots/` and
  `results/metrics/`.
- Best weights are copied to `models/best_yolov8_dinov2.pt`.

## Build a test set / demo video and run inference

```bash
# 10 sample predicted images into test_outputs/images/
python make_test_set.py --n 10

# build a short demo video out of COCO128 frames, then run detection on it
python make_test_video.py
python inference.py --source test_outputs/demo_input.mp4 --weights models/best_yolov8_dinov2.pt

# run on any single image
python inference.py --source path/to/image.jpg

# run on any video
python inference.py --source path/to/video.mp4

# run on a folder of images
python inference.py --source path/to/folder/
```

Predicted output is written to `test_outputs/` (`*_pred.jpg` / `*_pred.mp4`).

## Architecture notes

`models/dino_yolo.py` defines `YOLOv8DINOv2(DetectionModel)`:

1. Builds a standard YOLOv8n backbone/neck/head from `yolov8n.yaml`.
2. Loads a frozen `dinov2_vits14` backbone and a learnable 1x1 conv that
   projects its `[CLS]` embedding to the channel width of the backbone's
   deepest layer (SPPF, layer index 9 in yolov8n).
3. On the forward pass, after layer 9 produces its feature map, the DINOv2
   embedding is broadcast spatially and concatenated with that feature map,
   then reduced back to the original channel width with a learnable fusion
   conv — injecting global scene context before the neck/head continue as
   normal.

This mirrors the hybrid strategy in the reference work: keep YOLO's
speed (DINOv2 is frozen and only adds one extra forward pass, no extra
backward cost) while injecting the semantic understanding DINOv2 provides
to reduce false positives in cluttered scenes.
# PD234-IUI-Assignment-1
