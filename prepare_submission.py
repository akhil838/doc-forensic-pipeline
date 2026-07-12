#!/usr/bin/env python3
"""
FREUID Challenge 2026 — Docker entrypoint.

Thin wrapper around inference/run_inference.py with Docker sandbox paths.
No internet access. All models loaded from /models.

Mounts:
  /data/           read-only   flat image files
  /submissions/    read-write  output submission.csv
  /models/         read-only   model weights (baked into image)
"""
import os
import sys

# Offline mode — no downloads at runtime
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLEX_HOME", "/models/paddleocr_cache")

# Add inference/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "inference"))

from run_inference import main as run_main

if __name__ == "__main__":
    # Inject Docker sandbox defaults into sys.argv if not already provided
    defaults = {
        "--image-dir": os.environ.get("FREUID_DATA_DIR", "/data"),
        "--model-dir": os.environ.get("FREUID_MODEL_DIR", "/models"),
        "--output": os.environ.get("FREUID_SUBMISSION_PATH", "/submissions/submission.csv"),
    }
    for flag, value in defaults.items():
        if flag not in sys.argv:
            sys.argv.extend([flag, value])

    run_main()
