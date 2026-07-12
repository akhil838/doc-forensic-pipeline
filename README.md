# Document Forensic Pipeline

Two-branch fraud detection for identity documents: face/photo tampering + text field tampering.

**Public leaderboard: 0.294** (FREUID score, lower is better)

## Pipeline

1. **MediaPipe BlazeFace** — face detection (2ms/image, CPU)
2. **DINOv2-small** — portrait crop → photo fake probability (1.2ms/face, GPU)
3. **PaddleOCR PP-OCRv6** — text field detection (35ms/image, GPU)
4. **Forensic panels** — each field crop → 224×1008 six-view composite (grayscale, ink mask, L residual, chroma, texture, edge)
5. **ConvNeXt-Base** — panel → per-field tamper probability (1.5ms/field, GPU)
6. **Score** — `fraud_score = max(photo_prob, max(field_probs))`

Total inference: ~2.7h for 142k documents on A100.

## Repository Structure

```
├── prepare_submission.py           # Docker entrypoint
├── Dockerfile                      # GPU Docker image (CUDA 12.8)
├── requirements.txt                # Python dependencies
├── technical_report.pdf            # Compiled technical report
├── LICENSE                         # MIT
│
├── inference/
│   └── run_inference.py            # Full inference pipeline
│
├── models/
│   ├── models.py                   # DINOv3Classifier, DINOClassifier
│   └── weights/                    # Download from HuggingFace (see below)
│
├── train/
│   ├── train_text.py               # Train ConvNeXt-B text tamper detector
│   ├── train_face.py               # Train DINOv2-small face detector
│   ├── precache_boxes.py           # Pre-run PaddleOCR on training images
│   ├── train_labels.csv            # Competition training labels
│   ├── annotations/                # Subtask + field-level annotations
│   └── scripts/                    # Shared modules (panels, data, synth, metric)
│
├── technical_report/               # LaTeX source + figures
│   ├── technical_report.tex
│   ├── references.bib
│   └── figures/
│
└── how_to_train.md                 # Complete training guide
```

## Model Weights

Download from HuggingFace:

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('akhil838/doc-forensic-models-v1', local_dir='models/weights')
"
```

| Weight | Size | Description |
|--------|------|-------------|
| `dinov3_convnext_base_tamper_clf.pt` | 336 MB | Text tamper scorer (ConvNeXt-B, epoch 2) |
| `face/dinov2_small_unified_face.pt` | 85 MB | Face/photo classifier |
| `blaze_face.tflite` | 225 KB | MediaPipe face detection |
| `paddleocr_cache/` | 59 MB | PaddleOCR PP-OCRv6 detection model |

Weights: https://huggingface.co/akhil838/doc-forensic-models-v1

## Docker Build & Run

```bash
# Build (network available — downloads PyTorch, PaddlePaddle, caches HF models)
docker build --build-arg HF_TOKEN=hf_xxx -t freuid-repro .

# Run (no network)
docker run --rm --gpus all --network none \
  -v /path/to/test/images:/data:ro \
  -v $(pwd)/out:/submissions \
  freuid-repro:latest
```

Output: `/submissions/submission.csv` with columns `id,label`.

## Local Inference (without Docker)

```bash
export HF_TOKEN=hf_your_token
python inference/run_inference.py \
  --image-dir /path/to/images \
  --model-dir models/weights \
  --output submission.csv \
  --device cuda
```

## Training

See [how_to_train.md](how_to_train.md) for complete instructions.

## Hardware

- **Training**: Apple M3 Max (64GB RAM, MPS)
- **Inference**: NVIDIA A100-PCIE-40GB (~2.7h for 142k docs)
- **Also tested**: Kaggle T4 GPU

## External Resources

| Resource | License | Usage |
|----------|---------|-------|
| `facebook/dinov3-convnext-base-pretrain-lvd1689m` | Apache 2.0 | Text classifier backbone |
| `facebook/dinov2-small` | Apache 2.0 | Face classifier backbone |
| PaddleOCR PP-OCRv6 | Apache 2.0 | Text field detection |
| MediaPipe BlazeFace | Apache 2.0 | Face detection |

No external training data beyond the FREUID competition dataset.

## Contact

Akhil Kosuri — kosuriakhil19@gmail.com
