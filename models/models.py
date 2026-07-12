"""
Model definitions for training and inference.

Models:
  - DINOv3Classifier: ConvNeXt-Base backbone for text tamper detection (88M params)
  - DINOClassifier:   DINOv2-small for unified face/photo detection (22M params)

Hardware used for training: Apple M3 Max (64GB RAM, MPS backend).
Inference benchmarked on NVIDIA A100-PCIE-40GB.

All models read HF_TOKEN from environment for gated model access.
At inference time with --network none, models load from cached HF + .pt checkpoints.
"""
from __future__ import annotations
import torch
import torch.nn as nn


# ── Text tamper: ConvNeXt-Base (dinov3-convnext-base-pretrain-lvd1689m) ──────

class DINOv3Classifier(nn.Module):
    """ConvNeXt-Base backbone + linear head. 88M params, hidden=1024.

    Freeze all but last `unfreeze_blocks` stages + layernorm + head.
    Uses pooler_output (global avg pool) for classification.
    """

    def __init__(self, unfreeze_blocks: int = 2, dropout: float = 0.15,
                 config_dir: str | None = None):
        super().__init__()
        import os
        from pathlib import Path
        from transformers import AutoModel, AutoConfig

        # Try local config first (no HF download needed), fallback to HF
        local_config = Path(config_dir) if config_dir else Path(__file__).parent / "configs" / "dinov3_convnext_base"
        if local_config.exists():
            config = AutoConfig.from_pretrained(str(local_config))
            self.backbone = AutoModel.from_config(config)
        else:
            token = os.environ.get("HF_TOKEN")
            kwargs = {"token": token} if token else {}
            self.backbone = AutoModel.from_pretrained(
                "facebook/dinov3-convnext-base-pretrain-lvd1689m", **kwargs)
        hidden = self.backbone.config.hidden_sizes[-1]  # 1024

        # Freeze everything
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Unfreeze last N stages + layernorm
        stages = self.backbone.model.stages
        for stage in stages[-unfreeze_blocks:]:
            for p in stage.parameters():
                p.requires_grad = True
        if hasattr(self.backbone, "layer_norm"):
            for p in self.backbone.layer_norm.parameters():
                p.requires_grad = True

        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"DINOv3Classifier: {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        feat = out.pooler_output  # (B, 1024)
        return self.head(feat).squeeze(-1)  # (B,)


# ── Face/photo: Unified DINOv2-small ─────────────────────────────────────────

class DINOClassifier(nn.Module):
    """DINOv2-small backbone + LayerNorm + Linear head for face/photo classification.

    All backbone parameters trainable (small model, 22M params).
    Differential LR: backbone_lr << head_lr.
    """

    def __init__(self, n_classes: int = 2,
                 model_name: str = "facebook/dinov2-small",
                 dropout: float = 0.1,
                 freeze_backbone: bool = False,
                 config_dir: str | None = None):
        super().__init__()
        import os
        from pathlib import Path
        from transformers import AutoModel, AutoConfig

        # Try local config first, fallback to HF
        local_config = Path(config_dir) if config_dir else Path(__file__).parent / "configs" / "dinov2_small"
        if local_config.exists():
            config = AutoConfig.from_pretrained(str(local_config))
            self.backbone = AutoModel.from_config(config)
        else:
            token = os.environ.get("HF_TOKEN")
            kwargs = {"token": token} if token else {}
            self.backbone = AutoModel.from_pretrained(model_name, **kwargs)
        hidden = self.backbone.config.hidden_size  # 384
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x, interpolate_pos_encoding=True)
        feat = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        return self.head(feat)

    def make_optimizer(self, backbone_lr: float = 1e-5, head_lr: float = 3e-4,
                       weight_decay: float = 1e-4) -> torch.optim.Optimizer:
        backbone_params = [p for n, p in self.named_parameters()
                           if p.requires_grad and n.startswith("backbone.")]
        head_params = [p for n, p in self.named_parameters()
                       if p.requires_grad and not n.startswith("backbone.")]
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})
        return torch.optim.AdamW(groups, weight_decay=weight_decay)
