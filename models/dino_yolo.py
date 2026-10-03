"""
YOLOv8 + DINOv2 hybrid object detector.

Implements the architecture described in the first work of
https://cambum.net/I3DLab/AI4DM.htm : a YOLOv8 backbone whose deepest
feature map is fused with a global semantic embedding produced by a
frozen DINOv2 foundation model. The DINOv2 embedding is projected to the
backbone's channel width, broadcast spatially, concatenated with the
backbone's deepest feature map (SPPF output) and fused back down with a
1x1 conv before continuing into the YOLOv8 neck/head.

This keeps YOLOv8's speed (DINOv2 only sees a small resized crop of the
same image and is frozen -> no extra backward cost) while injecting
global semantic context that helps reduce false positives in cluttered
scenes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel


class DINOv2Embedder(nn.Module):
    """Wraps a frozen DINOv2 backbone (via torch.hub) and projects its
    global [CLS] embedding to `c2` channels."""

    def __init__(self, c2, variant="dinov2_vits14", freeze=True):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", variant)
        self.hidden = self.backbone.embed_dim
        self.proj = nn.Conv2d(self.hidden, c2, kernel_size=1)
        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    @torch.no_grad()
    def _embed(self, img):
        # DINOv2 requires spatial dims divisible by patch size (14).
        h, w = img.shape[-2:]
        nh, nw = (h // 14) * 14 or 14, (w // 14) * 14 or 14
        img_r = F.interpolate(img, size=(max(nh, 224), max(nw, 224)), mode="bilinear", align_corners=False)
        feats = self.backbone.forward_features(img_r)
        cls_token = feats["x_norm_clstoken"]  # (B, hidden)
        return cls_token

    def forward(self, img, target_hw):
        if self.freeze:
            cls_token = self._embed(img)
        else:
            cls_token = self._embed.__wrapped__(self, img) if hasattr(self._embed, "__wrapped__") else self._embed(img)
        b, c = cls_token.shape
        x = cls_token.view(b, c, 1, 1)
        x = self.proj(x)
        x = x.expand(-1, -1, target_hw[0], target_hw[1])
        return x


class YOLOv8DINOv2(DetectionModel):
    """YOLOv8 detection model with a DINOv2 global-context injection at
    the deepest backbone layer (SPPF output)."""

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True,
                 dino_variant="dinov2_vits14", inject_layer=9, freeze_dino=True):
        self.inject_layer = inject_layer  # index of SPPF layer in self.model (yolov8n: layer 9)
        self._dino_ready = False  # super().__init__ probes forward() before dino/fuse exist
        super().__init__(cfg, ch, nc, verbose)
        deepest_c = self._infer_channels(inject_layer)
        self.dino = DINOv2Embedder(deepest_c, variant=dino_variant, freeze=freeze_dino)
        self.fuse_conv = nn.Conv2d(deepest_c * 2, deepest_c, kernel_size=1)
        nn.init.zeros_(self.fuse_conv.bias)
        self._dino_ready = True

    def _infer_channels(self, idx):
        layer = self.model[idx]
        for m in reversed(list(layer.modules())):
            if isinstance(m, nn.Conv2d):
                return m.out_channels
        raise RuntimeError(f"Could not infer channel width at layer {idx}")

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        raw_img = x
        y = []
        out = x
        for i, m in enumerate(self.model):
            if m.f != -1:
                out = y[m.f] if isinstance(m.f, int) else [out if j == -1 else y[j] for j in m.f]
            out = m(out)
            if i == self.inject_layer and self._dino_ready:
                dino_feat = self.dino(raw_img, out.shape[-2:])
                out = self.fuse_conv(torch.cat([out, dino_feat], dim=1))
            y.append(out if m.i in self.save else None)
        return out

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self._predict_once(x)
