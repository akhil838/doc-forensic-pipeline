# Docker reproducibility contract

This folder defines the **organizer-facing sandbox contract** for post-submission
verification. Teams may copy these files into their repository and adapt them.

Public copy: [freuid2026.microblink.com/reproducibility.html](https://freuid2026.microblink.com/reproducibility.html)

## Layout

| File | Purpose |
| ---- | ------- |
| `Dockerfile` | Minimal GPU-ready image; replace with your stack |
| `prepare_submission.py` | Reference entrypoint — writes `/submissions/submission.csv` |
| `requirements.txt` | Python deps for the reference script |
| `.dockerignore` | Keeps build context small |

## Data mounts (organizer side)

When organizers run your container:

```text
/host/private-test/images/   →  /data/          (read-only, flat image files only)
/host/output/                →  /submissions/   (read-write)
```

**There is no manifest or CSV inside `/data`.** Every test image is a file directly
under `/data/`. The row id is the filename **without extension**:

```text
/data/
  3f49e6921f3f4b10910738f7e9b476f3.jpeg
  c39d1c24d46a44a088991ddc7a072b7c.jpeg
  ...
```

Supported extensions: `.jpeg`, `.jpg`, `.png`, `.webp`, `.bmp`, `.tif`, `.tiff`.

## Output

The entrypoint **must** create:

```text
/submissions/submission.csv
```

Schema:

```csv
id,label
3f49e6921f3f4b10910738f7e9b476f3,0.99
```

- `id` — same as the image filename stem.
- `label` — finite float fraud score; **higher = more confident the document is fraudulent**.

## Build and local test

```bash
cd docker   # or copy these files into your repo

docker build -t freuid-repro:local .

# Flat directory of images only — no CSV
docker run --rm \
  --network none \
  -v /path/to/test_images:/data:ro \
  -v "$(pwd)/_local_out:/submissions" \
  freuid-repro:local
```

## Requirements for verification

1. Container starts with **no network** (`docker run --network none`).
2. No writes outside `/submissions/`.
3. Model weights bundled in the image or mounted read-only — **no runtime downloads**.
4. Document GPU vs CPU expectations in your technical report.
5. Pin base image digests when publishing pre-built images.
6. One output row per image file in `/data`; no missing or extra ids.

## Customization checklist

- [ ] Replace `requirements.txt` with your inference stack.
- [ ] Copy trained weights into the image (`COPY models/ /models/`) or document a read-only mount.
- [ ] Implement real inference in `predict_labels()` (keep CLI contract).
- [ ] Set `ENTRYPOINT` to your script; exit non-zero on failure.
- [ ] Test with `--network none` before replying on the Kaggle thread.
