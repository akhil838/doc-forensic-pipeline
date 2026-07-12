# FREUID Challenge 2026 — Reproducibility Package

## Solution overview

Binary fraud detection on identity documents. Pipeline:

1. **PaddleOCR TextDetection** — detect text field boxes per document
2. **Forensic panel generation** — each text crop → 224×1008 six-view composite (grayscale, ink mask, luminance residual, chroma residual, texture variance, edge magnitude)
3. **ConvNeXt-Base classifier** — fine-tuned on forensic panels (FREUID=0.0078 on public LB)
4. **Score aggregation** — top-3 mean of per-field tamper probabilities → document fraud score

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | GPU Docker image (CUDA 12.8, PyTorch, PaddlePaddle) |
| `prepare_submission.py` | Entrypoint: `/data/` images → `/submissions/submission.csv` |
| `model.py` | Model classes, forensic panel builder, text/face detection |
| `requirements.txt` | Python dependencies |
| `models/` | Trained weights (`.pt` files) |

## Model weights

Before building, place weights in `models/`:

```
models/
  convnextv2_base_tamper_clf.pt    # primary (335 MB, FREUID=0.0078)
  efficientnet_b0_tamper_clf.pt    # fallback (17 MB, FREUID=0.0141)
```

Copy from training artifacts:
```bash
cp ../artifacts/convnextv2_base_tamper_clf.pt models/
cp ../artifacts/efficientnet_b0_tamper_clf.pt models/
```

## Build

```bash
docker build -t freuid-repro:latest .
```

Build time: ~15 min (downloads PyTorch + PaddlePaddle).

## Run (local test)

```bash
# Prepare flat image directory (no CSV, no subfolders)
mkdir -p test_images
cp /path/to/some/images/*.jpeg test_images/

# Run with GPU, no network
docker run --rm \
  --gpus all \
  --network none \
  -v "$(pwd)/test_images:/data:ro" \
  -v "$(pwd)/output:/submissions" \
  freuid-repro:latest
```

Output: `output/submission.csv` with columns `id,label`.

## Run without GPU

```bash
docker run --rm \
  --network none \
  -v "$(pwd)/test_images:/data:ro" \
  -v "$(pwd)/output:/submissions" \
  freuid-repro:latest --device cpu
```

## Hardware requirements

- **GPU**: NVIDIA A100/T4/V100 with ≥8 GB VRAM (tested on A100-40GB)
- **RAM**: ≥16 GB
- **Time**: ~3–5h for 142k documents on A100

## External resources

| Resource | License | Usage |
|----------|---------|-------|
| `facebook/dinov3-convnext-base-pretrain-lvd1689m` | Apache 2.0 | Backbone (baked into checkpoint) |
| PaddleOCR PP-OCRv6 | Apache 2.0 | Text field detection |
| MediaPipe BlazeFace | Apache 2.0 | Face detection |
| ImageNet pretrained weights (via timm) | Various | EfficientNet-B0 init |

## Reproducibility

- **Seeds**: `SEED=42` for all random operations
- **Determinism**: inference is deterministic given same input order
- **Network**: `--network none` verified — no runtime downloads
- **Weights**: all baked into Docker image via `COPY models/ /models/`
