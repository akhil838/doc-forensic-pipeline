# FREUID Challenge 2026 — reproducibility Dockerfile
#
# Build:  docker build -t freuid-repro:latest .
# Run:    docker run --rm --gpus all --network none \
#           -v /path/to/images:/data:ro -v $(pwd)/out:/submissions \
#           freuid-repro:latest
#
# Organizers run with: --network none -v DATA:/data:ro -v OUT:/submissions

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FREUID_DATA_DIR=/data \
    FREUID_OUTPUT_DIR=/submissions \
    FREUID_SUBMISSION_PATH=/submissions/submission.csv \
    FREUID_MODEL_DIR=/models \
    UV_PYTHON=3.12 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python via uv
RUN uv python install 3.12

# PyTorch (needs pytorch index)
RUN uv pip install --system --no-cache \
    torch==2.11.0+cu128 torchvision==0.22.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# PaddlePaddle GPU (needs paddle index)
RUN uv pip install --system --no-cache \
    paddlepaddle-gpu==3.3.0 \
    --index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/

# Fix NVIDIA libs that PaddlePaddle may overwrite
RUN uv pip install --system --no-cache \
    nvidia-nccl-cu12==2.28.9 nvidia-cudnn-cu12==9.19.0.56 2>/dev/null || true

# Remaining Python deps
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Offline mode: no downloads at runtime (configs are local, weights are baked in)
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# Copy code: inference + train/scripts + models/ (model classes)
COPY inference/ /app/inference/
COPY train/scripts/ /app/train/scripts/
COPY models/__init__.py models/models.py /app/models/
COPY models/configs/ /app/models/configs/
COPY prepare_submission.py /app/

# Model weights + PaddleOCR cache — baked into image, no runtime downloads
COPY models/weights/ /models/

RUN useradd --create-home --uid 1000 runner
USER runner

# Point PaddleOCR to pre-cached detection model
ENV PADDLEX_HOME=/models/paddleocr_cache

ENTRYPOINT ["python", "/app/prepare_submission.py"]
CMD []
