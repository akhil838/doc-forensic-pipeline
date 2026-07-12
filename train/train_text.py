#!/usr/bin/env python
"""Train ConvNeXt-Base (DINOv3) text-field tamper detector on forensic panels.

Usage:
  python train_text.py --data-root /path/to/competition/data
  python train_text.py --data-root /path/to/competition/data --epochs 6 --lr 2e-5

Reads annotations from train/annotations/ and paddle boxes from train/annotations/paddle_cache/.
Saves best checkpoint to --output-dir (default: models/).
"""
from __future__ import annotations
import warnings, json, random, time, gc, os, sys
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path
import numpy as np, pandas as pd, cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Local modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.forensic_panels import forensic_panel_from_crop, real_ink_mask
from scripts.data import (
    load_field_annotations, load_paddle_boxes, extract_field_crops,
    split_long_crop, PanelDataset, SEED, TOP_OCR_IGNORE_FRAC,
)
from scripts.synth_tamper import (
    SYNTH_INK_TAMPER_REGISTRY, make_synth_dataset_sample,
    apply_synth_tamper_to_real,
)
from models.models import DINOv3Classifier
from scripts.metric import compute_freuid_score

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# PanelDataset, DINOv3Classifier, apply_synth_tamper_to_real, split_long_crop,
# forensic_panel_from_crop, real_ink_mask — all imported from scripts/ modules above.


# ---- Field extraction (uses paddle_cache .npy files) ----
def doc_field_crops(image_file, image_id, paddle_cache, top_ignore=TOP_OCR_IGNORE_FRAC,
                    min_w=10, min_h=8, max_fields=40,
                    min_aspect=0.5, max_h_frac=0.09, max_w_frac=0.85):
    """Extract text field crops from an image using cached paddle boxes.

    Filters (must match inference):
      - top header band (top_ignore=0.17)
      - tiny boxes (w<10 or h<8)
      - vertical/tall boxes (w/h < 0.5)
      - oversized boxes (h > 9% of H or w > 85% of W)
    """
    rgb = cv2.cvtColor(cv2.imread(str(image_file)), cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    boxes = load_paddle_boxes(paddle_cache, image_id)
    out = []
    for x0, y0, x1, y1 in boxes:
        bw, bh = x1 - x0, y1 - y0
        if y0 < top_ignore * H:
            continue
        if bw < min_w or bh < min_h:
            continue
        if bh > 0 and bw / bh < min_aspect:
            continue
        if bh > max_h_frac * H or bw > max_w_frac * W:
            continue
        cx0, cy0 = max(0, int(x0) - 15), max(0, int(y0) - 5)
        cx1, cy1 = min(W, int(x1) + 15), min(H, int(y1) + 5)
        crop = rgb[cy0:cy1, cx0:cx1]
        if crop.shape[0] >= 6 and crop.shape[1] >= 6:
            out.append(crop)
        if len(out) >= max_fields:
            break
    return out

# ---- Build training set ----
PANEL_CACHE_DIR = None  # set by --output-dir in main()

def _save_img(panel, folder, idx):
    """Save a forensic panel as JPEG."""
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{idx:06d}.jpg"
    cv2.imwrite(str(p), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return str(p)

def build_training_panels(args, ann, train_ids, rng, train_dir, paddle_cache):
    """Build training panels, save as images in a clean folder structure."""
    cache_dir = Path(args.output_dir) / "training_panels"
    manifest = cache_dir / "manifest.json"
    if manifest.exists() and not args.force_rebuild:
        mf = json.loads(manifest.read_text())
        panels, y = [], []
        for entry in mf["entries"]:
            img = cv2.imread(entry["path"])
            if img is not None:
                panels.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                y.append(entry["label"])
        if len(panels) == mf["total"]:
            y = np.array(y, dtype=np.float32)
            print(f"loaded cached panels: {len(panels)} from {cache_dir} (pos_frac={y.mean():.3f})")
            return panels, y
        print(f"cache incomplete ({len(panels)}/{mf['total']}), rebuilding...")

    import shutil
    if cache_dir.exists(): shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tmap = dict(zip(ann["id"], ann["tidx"]))
    path_of = {}
    for _, r in ann.iterrows():
        path_of[r["id"]] = str(train_dir / f"{r['id']}.jpeg")

    # --- Extract real fields ---
    neg_headroom = args.neg_cap + args.synth_on_real + args.long_split * 3
    real_fake, neg_crops = [], []
    real_genuine = []
    t0 = time.time()
    train_id_list = list(train_ids)
    for n, did in enumerate(train_id_list, 1):
        p = path_of.get(did)
        if not p or not Path(p).exists(): continue
        crops = doc_field_crops(p, did, paddle_cache)
        for idx, crop in enumerate(crops):
            if idx in tmap.get(did, set()):
                real_fake.append(forensic_panel_from_crop(crop))
            elif len(neg_crops) < neg_headroom:
                neg_crops.append(crop)
        if n % 500 == 0:
            print(f"  built fields {n}/{len(train_id_list)} ({time.time()-t0:.0f}s) fake={len(real_fake)}")

    random.Random(SEED).shuffle(neg_crops)

    # --- Genuine negatives (real + long-split) ---
    real_genuine = [forensic_panel_from_crop(c) for c in neg_crops[:args.neg_cap]]
    long_neg = []
    for crop in neg_crops[:args.long_split * 2]:
        for part in split_long_crop(crop, rng):
            long_neg.append(forensic_panel_from_crop(part))
            if len(long_neg) >= args.long_split: break
        if len(long_neg) >= args.long_split: break
    real_genuine.extend(long_neg)

    # --- Synth-on-real: apply tamper effects to clean crops ---
    synth_by_effect = {}
    for crop in neg_crops[:args.synth_on_real]:
        tampered, eff = apply_synth_tamper_to_real(crop, rng)
        if eff is not None:
            synth_by_effect.setdefault(eff, []).append(forensic_panel_from_crop(tampered))

    # --- Clean-country: genuine neg + synth-tampered pos ---
    data_root = Path(args.data_root)
    tl = pd.read_csv(data_root / "train_labels.csv", dtype={"id": str})
    cl = tl[tl["label"] == 0]
    cids = []
    for c in ["GUINEA/DL", "BENIN/DL", "MAURITIUS/ID", "EGYPT/DL", "MOZAMBIQUE/DL"]:
        cids += list(cl[cl["type"] == c].sort_values("id")["id"].head(args.clean_neg_per_country))
    random.Random(SEED).shuffle(cids)
    clean_crops, cap = [], args.clean_neg_cap + args.clean_synth_pos
    for cid in cids:
        p = train_dir / f"{cid}.jpeg"
        if not p.exists(): continue
        clean_crops.extend(doc_field_crops(str(p), cid, paddle_cache))
        if len(clean_crops) >= cap: break
    random.Random(SEED + 1).shuffle(clean_crops)
    for crop in clean_crops[:args.clean_synth_pos]:
        tampered, eff = apply_synth_tamper_to_real(crop, rng)
        if eff is not None:
            synth_by_effect.setdefault(eff, []).append(forensic_panel_from_crop(tampered))
    clean_neg = [forensic_panel_from_crop(c)
                 for c in clean_crops[args.clean_synth_pos:args.clean_synth_pos + args.clean_neg_cap]]
    real_genuine.extend(clean_neg)
    print(f"synth-on-real by effect: { {k: len(v) for k, v in synth_by_effect.items()} }")
    print(f"genuine negatives: {len(real_genuine)}")

    # --- Fully-synthetic: track effect per panel ---
    fully_synth_clean = []
    fully_synth_by_effect = {}
    TAMPER_TYPES = list(SYNTH_INK_TAMPER_REGISTRY.keys())
    for i in range(args.synth_gen):
        lbl = 1 if i < args.synth_gen // 2 else 0
        tt = rng.choice(TAMPER_TYPES) if lbl == 1 else None
        _, panel, meta = make_synth_dataset_sample(f"ft_{i:06d}", lbl, seed=SEED + i * 17, force_tamper_type=tt)
        if lbl == 0:
            fully_synth_clean.append(panel)
        else:
            eff = meta.get("tamper_type", "unknown")
            fully_synth_by_effect.setdefault(eff, []).append(panel)
    print(f"fully-synthetic clean: {len(fully_synth_clean)}, by effect: { {k: len(v) for k, v in fully_synth_by_effect.items()} }")

    # --- Hard negatives: genuine fields that the previous model scored high ---
    hard_neg = []
    hard_neg_path = Path("artifacts/hard_negatives_DISABLED.pkl")  # disabled: focal loss handles FPs instead
    if hard_neg_path.exists():
        import pickle
        hn = pickle.load(open(hard_neg_path, "rb"))
        hard_neg = hn["panels"]
        print(f"hard negatives loaded: {len(hard_neg)} panels")
    else:
        print("no hard negatives file found — skipping")

    # --- Save all images in clean folder structure ---
    entries = []
    def save_batch(panel_list, folder, label):
        for i, panel in enumerate(panel_list):
            p = _save_img(panel, folder, i)
            entries.append({"path": p, "label": label, "folder": str(folder.relative_to(cache_dir))})

    save_batch(real_fake, cache_dir / "train" / "fake", 1)
    save_batch(real_genuine, cache_dir / "train" / "real", 0)
    if hard_neg:
        save_batch(hard_neg, cache_dir / "train" / "hard_negatives", 0)
    for eff, panels_list in synth_by_effect.items():
        save_batch(panels_list, cache_dir / "train" / "synthetic" / eff, 1)
    save_batch(fully_synth_clean, cache_dir / "fully_synthetic" / "clean", 0)
    for eff, panels_list in fully_synth_by_effect.items():
        save_batch(panels_list, cache_dir / "fully_synthetic" / eff, 1)

    # --- Assemble flat arrays for training ---
    all_panels = real_fake + real_genuine + hard_neg
    all_labels = [1] * len(real_fake) + [0] * len(real_genuine) + [0] * len(hard_neg)
    for panels_list in synth_by_effect.values():
        all_panels.extend(panels_list); all_labels.extend([1] * len(panels_list))
    all_panels.extend(fully_synth_clean); all_labels.extend([0] * len(fully_synth_clean))
    for panels_list in fully_synth_by_effect.values():
        all_panels.extend(panels_list); all_labels.extend([1] * len(panels_list))

    y = np.array(all_labels, dtype=np.float32)
    manifest.write_text(json.dumps({"total": len(all_panels), "entries": entries}))
    saved_mb = sum(f.stat().st_size for f in cache_dir.rglob("*.jpg")) / 1e6
    print(f"TOTAL: {len(all_panels)} panels, pos_frac={y.mean():.3f}")
    print(f"saved to {cache_dir}/ ({saved_mb:.0f} MB)")
    return all_panels, y


def score_real_docs(model, doc_ids, train_dir, paddle_cache, batch_size=12):
    """Score real documents: extract fields → panels → model → max agg."""
    model.eval()
    scores, labels_out = [], []
    with torch.no_grad():
        for image_id, text_label in doc_ids:
            img_path = train_dir / f"{image_id}.jpeg"
            if not img_path.exists():
                continue
            crops = doc_field_crops(str(img_path), image_id, paddle_cache)
            if not crops:
                scores.append(0.0); labels_out.append(text_label); continue
            panels = [forensic_panel_from_crop(c) for c in crops]
            probs = []
            for i in range(0, len(panels), batch_size):
                batch = panels[i:i + batch_size]
                tensors = []
                for p in batch:
                    if p.shape[:2] != (224, 1008):
                        p = cv2.resize(p, (1008, 224), interpolation=cv2.INTER_AREA)
                    t = torch.from_numpy(p.astype(np.float32) / 255.0).permute(2, 0, 1)
                    t = (t - IMAGENET_MEAN.squeeze(0)) / IMAGENET_STD.squeeze(0)
                    tensors.append(t)
                x = torch.stack(tensors).to(DEVICE)
                logits = model(x)
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            scores.append(float(max(probs)))
            labels_out.append(text_label)
    return np.array(scores), np.array(labels_out)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train ConvNeXt-B text tamper detector.")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--ann-dir", type=str, default="annotations")
    ap.add_argument("--output-dir", type=str, default="models")
    ap.add_argument("--log", type=str, default=None)
    ap.add_argument("--neg-cap", type=int, default=14000)
    ap.add_argument("--synth-on-real", type=int, default=8000)
    ap.add_argument("--long-split", type=int, default=3000)
    ap.add_argument("--clean-neg-per-country", type=int, default=2500)
    ap.add_argument("--clean-neg-cap", type=int, default=8000)
    ap.add_argument("--clean-synth-pos", type=int, default=5000)
    ap.add_argument("--synth-gen", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--unfreeze-blocks", type=int, default=2)
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--val-docs", type=int, default=3000)
    args = ap.parse_args()

    # Logging
    import sys as _sys
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        _lf = open(args.log, 'w')
        class _Tee:
            def __init__(self, *s): self.s = s
            def write(self, m):
                for s in self.s: s.write(m); s.flush()
            def flush(self):
                for s in self.s: s.flush()
        _sys.stdout = _Tee(_sys.__stdout__, _lf)
        _sys.stderr = _Tee(_sys.__stderr__, _lf)
        print(f"Logging to {args.log}")

    data_root = Path(args.data_root)
    ann_dir = Path(args.ann_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = data_root / "train" / "train"
    if not train_dir.exists():
        train_dir = data_root / "train"
    paddle_cache = ann_dir / "paddle_cache"

    print(f"data_root:    {data_root}")
    print(f"train_dir:    {train_dir}")
    print(f"paddle_cache: {paddle_cache} ({len(list(paddle_cache.glob('*.npy')))} files)")
    print(f"output_dir:   {output_dir}")
    print(f"device:       {DEVICE}")

    # Load annotations (train-only)
    ann = load_field_annotations(ann_dir)
    ann = ann[ann["tidx"].map(len) > 0].reset_index(drop=True)
    ann_train = ann[ann["split"] == "train"].reset_index(drop=True)
    ids = list(ann_train["id"]); random.Random(SEED).shuffle(ids)
    n_val = int(len(ids) * 0.2)
    val_ids, train_ids = set(ids[:n_val]), set(ids[n_val:])
    print(f"annotated docs: train={len(train_ids)}, val={len(val_ids)}")

    # Build real-doc validation set: balanced from subtask_annotations
    sub = pd.read_csv(ann_dir / "subtask_annotations.csv", dtype={"id": str})
    n_half = args.val_docs // 2
    sub_fake = sub[sub["details_tamper"] == 1].sample(min(n_half, int((sub.details_tamper==1).sum())), random_state=SEED)
    sub_clean = sub[sub["details_tamper"] == 0].sample(min(n_half, int((sub.details_tamper==0).sum())), random_state=SEED)
    val_doc_ids = [(r["id"], int(r["details_tamper"])) for _, r in
                   pd.concat([sub_fake, sub_clean]).iterrows()
                   if (train_dir / f"{r['id']}.jpeg").exists() and
                      (paddle_cache / f"{r['id']}.npy").exists()]
    random.Random(SEED).shuffle(val_doc_ids)
    n_fake_val = sum(1 for _, l in val_doc_ids if l == 1)
    print(f"real-doc val: {len(val_doc_ids)} docs ({n_fake_val} text-fake, {len(val_doc_ids)-n_fake_val} clean)")

    rng = np.random.default_rng(SEED)
    panels, y = build_training_panels(args, ann_train, train_ids, rng, train_dir, paddle_cache)

    idx = np.arange(len(panels)); rng.shuffle(idx)
    n_val_panels = max(200, int(len(panels) * 0.1))
    val_idx, train_idx = idx[:n_val_panels], idx[n_val_panels:]
    train_ds = PanelDataset([panels[i] for i in train_idx], y[train_idx], augment=True)
    val_ds = PanelDataset([panels[i] for i in val_idx], y[val_idx], augment=False)
    print(f"train: {len(train_ds)} panels | val: {len(val_ds)} panels")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = DINOv3Classifier(unfreeze_blocks=args.unfreeze_blocks).to(DEVICE)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(DEVICE))
    save_path = output_dir / "dinov3_convnext_base_tamper_clf.pt"

    best_doc_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, n_batches = 0.0, 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            losses += loss.item(); n_batches += 1
            if n_batches % 50 == 0:
                print(f"  e{epoch} b{n_batches}/{len(train_loader)} loss={losses/n_batches:.4f}", flush=True)
        train_loss = losses / max(1, n_batches)
        dt = time.time() - t0

        # Panel-level val
        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                vp.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
                vl.extend(yb.numpy())
        vp, vl = np.array(vp), np.array(vl)
        panel_auc = roc_auc_score(vl, vp) if len(np.unique(vl)) > 1 else 0.0

        # Real-doc val (3k docs, max agg)
        t1 = time.time()
        ds, dl = score_real_docs(model, val_doc_ids, train_dir, paddle_cache, args.batch_size)
        dt_doc = time.time() - t1
        doc_auc = roc_auc_score(dl, ds) if len(np.unique(dl)) > 1 else 0.0
        if len(np.unique(dl)) > 1:
            fr = compute_freuid_score(dl, ds)
            tp = ((ds > 0.5) & (dl == 1)).sum()
            fp = ((ds > 0.5) & (dl == 0)).sum()
            fn = ((ds <= 0.5) & (dl == 1)).sum()
            print(f"\nepoch {epoch}/{args.epochs} ({dt:.0f}s train, {dt_doc:.0f}s val): loss={train_loss:.4f}")
            print(f"  PANEL: AUC={panel_auc:.4f}")
            print(f"  DOC:   AUC={doc_auc:.4f}  FREUID={fr['freuid_score']:.4f}  APCER@1%={fr['apcer_at_1_bpcer']:.4f}  TP={tp} FP={fp} FN={fn}")
        else:
            print(f"\nepoch {epoch}/{args.epochs} ({dt:.0f}s): loss={train_loss:.4f}  PANEL AUC={panel_auc:.4f}")

        # Save every epoch
        epoch_path = output_dir / f"dinov3_convnext_base_tamper_clf_e{epoch}.pt"
        torch.save(model.state_dict(), epoch_path)
        print(f"  saved -> {epoch_path}")

        metric = doc_auc if doc_auc > 0 else panel_auc
        if metric > best_doc_auc:
            best_doc_auc = metric
            torch.save(model.state_dict(), save_path)
            print(f"  ** BEST (DOC_AUC={doc_auc:.4f}) -> {save_path} **")

        gc.collect()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
        elif torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"\nbest doc AUC: {best_doc_auc:.4f}")
    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
