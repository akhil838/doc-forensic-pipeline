# How to Train & Run Inference

## Directory Structure

```
├── train/
│   ├── train_labels.csv                    # Competition training labels
│   ├── train_text.py                       # Train ConvNeXt-B text tamper detector
│   ├── train_face.py                       # Train unified DINOv2-small face detector
│   ├── precache_boxes.py                   # Pre-run PaddleOCR on training images
│   ├── annotations/
│   │   ├── subtask_annotations.csv         # Per-image photo/text labels (69k images)
│   │   ├── field_tamper_annotations.csv    # Per-field tamper labels (7k images)
│   │   ├── text_field_boxes.csv            # Known field box coordinates
│   │   └── paddle_cache/                   # PaddleOCR field boxes (.npy per image)
│   └── scripts/
│       ├── forensic_panels.py              # 6-view forensic panel builder
│       ├── data.py                         # Datasets, augmentations, field extraction
│       ├── synth_tamper.py                 # 9 synthetic tamper effects
│       ├── models.py → ../models/models.py # Shared model definitions
│       └── metric.py                       # FREUID metric computation
├── inference/
│   └── run_inference.py                    # Full inference pipeline
├── models/
│   ├── models.py                           # Model class definitions
│   └── weights/                            # Trained checkpoints (from HuggingFace)
├── prepare_submission.py                   # Docker entrypoint
└── Dockerfile
```

## Prerequisites

```bash
# GPU (A100/T4)
pip install torch==2.11.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/

# All other deps
pip install -r requirements.txt
```

## Step 1: Download Competition Data

Place the competition data so that `train_labels.csv` and `train/train/` are accessible:

```
/path/to/data/
├── train_labels.csv
├── train/train/*.jpeg          # 69,352 training images
└── public_test/public_test/*.jpeg  # 7,821 public test images (for inference only)
```

## Step 2: Pre-cache PaddleOCR Field Boxes

Generate PaddleOCR text field detections for all training images. Results are cached as `.npy` files so PaddleOCR only runs once per image.

```bash
# Text-fake images only (faster, minimum needed for training)
python train/precache_boxes.py \
  --data-root /path/to/data \
  --ann-dir train/annotations \
  --workers 4

# Include clean images too (recommended for more training negatives)
python train/precache_boxes.py \
  --data-root /path/to/data \
  --ann-dir train/annotations \
  --include-clean \
  --workers 4
```

On A100 GPU: ~15 min for all 69k images (35ms/image).
On CPU: ~8h with 4 workers (1.2s/image).

## Step 3: Train Text Tamper Detector (ConvNeXt-B)

Trains the ConvNeXt-Base model on forensic panels. The script:
1. Extracts real tampered fields using `field_tamper_annotations.csv`
2. Extracts genuine fields from clean documents
3. Generates synthetic tampered fields (9 effects)
4. Trains end-to-end with augmentations

```bash
python train/train_text.py \
  --data-root /path/to/data \
  --ann-dir train/annotations \
  --output-dir models/weights \
  --log train_text.log \
  --epochs 6 \
  --lr 2e-5 \
  --batch-size 12 \
  --neg-cap 20000 \
  --synth-on-real 10000 \
  --clean-neg-per-country 4000 \
  --clean-neg-cap 12000 \
  --clean-synth-pos 6000 \
  --synth-gen 8000 \
  --val-docs 3000 \
  --force-rebuild
```

Best checkpoint (by validation DOC AUC) saved to `models/weights/dinov3_convnext_base_tamper_clf.pt`.
Per-epoch checkpoints saved as `dinov3_convnext_base_tamper_clf_e{N}.pt`.

Training time: ~1.5h/epoch on MPS (M3 Max), ~20min/epoch on A100.

### Key hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--epochs` | 6 | Best model typically at epoch 2-3 |
| `--lr` | 2e-5 | AdamW learning rate |
| `--batch-size` | 12 | Per-GPU batch size |
| `--unfreeze-blocks` | 2 | Last 2 ConvNeXt stages unfrozen |
| `--neg-cap` | 20000 | Genuine field panels from tampered docs |
| `--synth-on-real` | 10000 | Synthetic effects applied to real fields |
| `--clean-neg-per-country` | 4000 | Clean docs sampled per country |
| `--synth-gen` | 8000 | Fully synthetic field panels |

### Field filtering (must match inference)

```
TOP_OCR_IGNORE_FRAC = 0.17    # Skip top 17% of image
MIN_W, MIN_H = 10, 8          # Skip tiny boxes
min_aspect = 0.5               # Skip vertical boxes (w/h < 0.5)
max_h_frac = 0.09              # Skip tall boxes (h > 9% of image)
max_w_frac = 0.85              # Skip wide boxes (w > 85% of image)
MAX_FIELDS = 40                # Cap per document
PAD_X, PAD_Y = 15, 5           # Context padding around crops
```

## Step 4: Train Face/Photo Detector (DINOv2-small)

```bash
python train/train_face.py \
  --data-root /path/to/data \
  --ann-csv train/annotations/subtask_annotations.csv \
  --output-dir models/weights/face \
  --epochs 8 \
  --batch-size 16
```

Saves `models/weights/face/unified_face.pt`.

## Step 5: Run Inference

### Full pipeline (from scratch)

```bash
python inference/run_inference.py \
  --image-dir /path/to/test/images \
  --model-dir models/weights \
  --output submission.csv \
  --box-cache train/annotations/paddle_cache \
  --log inference.log \
  --device cuda
```

### Pipeline steps

1. **MediaPipe face detection** → portrait crop (2ms/image, CPU)
2. **DINOv2-small face classifier** → `photo_prob` (1.2ms/face, GPU)
3. **PaddleOCR TextDetection** → field boxes (35ms/image on GPU, cached)
4. **Forensic panel generation** → 6-view 224×1008 composite per field
5. **ConvNeXt-B classifier** → per-field tamper probability (1.5ms/field, GPU)
6. **Aggregation** → `text_prob = max(field_probs)`, `fraud_score = max(photo_prob, text_prob)`

### Timing on A100 (142k images)

| Step | Time |
|------|------|
| Face detect + classify | 0.1h |
| PaddleOCR text detection | 1.4h |
| ConvNeXt-B text scoring | 1.2h |
| **Total** | **~2.7h** |

## Step 6: Docker Submission

```bash
# Build (network available)
docker build -t freuid-repro:latest .

# Run (no network)
docker run --rm --gpus all --network none \
  -v /path/to/test/images:/data:ro \
  -v $(pwd)/output:/submissions \
  freuid-repro:latest
```

Output: `/submissions/submission.csv` with columns `id,label`.

## Model Weights

Weights hosted on HuggingFace (private):

```bash
# Download weights
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('akhil838/doc-forensic-models-v1', local_dir='models/weights')
"
```

| Weight | Size | Description |
|--------|------|-------------|
| `dinov3_convnext_base_tamper_clf.pt` | 336 MB | Text tamper scorer (best epoch 2) |
| `timm_efficientnet_b0_tamper_clf.pt` | 17 MB | Fallback text scorer |
| `face/dinov2_small_unified_face.pt` | 85 MB | Unified face/photo classifier |
| `blaze_face.tflite` | 225 KB | MediaPipe face detection model |
| `paddleocr_cache/` | 59 MB | PaddleOCR PP-OCRv6 detection model |

## Reproducibility Notes

- **Seed**: 42 for all random operations
- **Determinism**: Inference is deterministic given same input order
- **Field filtering**: Training and inference use identical filtering parameters
- **Aggregation**: `max` (single highest-scoring field determines document score)
- **Score combination**: `fraud_score = max(photo_prob, text_prob)`
