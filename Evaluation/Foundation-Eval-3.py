#!/usr/bin/env python3
"""
Evaluate timm models on a ShapeNet-render dataset described by metadata.csv.

Inputs:
- metadata.csv (semicolon-separated)
- ShapeNet-ImageNet1k-Mapping.with_index.json  (category -> {shapenet_synset, imagenet_class_indices, ...})
- ShapeNet_eval_masks_extended.json            (category -> {depth_1..depth_5: [0/1]*1000})

Outputs:
- predictions.csv        (per sample)
- metrics_per_class.csv  (ShapeNet synset one-vs-rest)
- metrics_per_group.csv  (all CSV columns aggregated)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

import timm
from timm.data import create_transform, resolve_data_config


# -----------------------------
# Utilities
# -----------------------------

def zfill_synset(x: str | int) -> str:
    """Convert '2958343' -> '02958343' (8 digits)."""
    s = str(x).strip()
    s = s.replace(".0", "")  # safety if read as float
    return s.zfill(8)

def safe_open_image(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        return None

def one_vs_rest_counts(y_true: np.ndarray, y_pred: np.ndarray, cls: str) -> Tuple[int, int, int, int]:
    """TP, TN, FP, FN for class cls in multi-class setting."""
    pos_true = (y_true == cls)
    pos_pred = (y_pred == cls)
    tp = int(np.logical_and(pos_true, pos_pred).sum())
    tn = int(np.logical_and(~pos_true, ~pos_pred).sum())
    fp = int(np.logical_and(~pos_true, pos_pred).sum())
    fn = int(np.logical_and(pos_true, ~pos_pred).sum())
    return tp, tn, fp, fn

def prf1(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    eps = 1e-12
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# -----------------------------
# Mapping & Masks
# -----------------------------

@dataclass(frozen=True)
class MappingIndex:
    # synset (8 digits) -> category key used in mapping/masks (e.g., "car")
    synset_to_category: Dict[str, str]
    # category -> list of ImageNet class indices associated to this category (from your mapping)
    category_to_imagenet_indices: Dict[str, List[int]]
    # category -> depth -> mask(1000,) as np.uint8
    category_depth_masks: Dict[str, Dict[str, np.ndarray]]

def load_mapping_and_masks(mapping_path: str, masks_path: str) -> MappingIndex:    
    with open(mapping_path, "r") as f:        
        mapping = json.load(f)    

    with open(masks_path, "r") as f:        
        masks = json.load(f)    

    synset_to_category = {}    
    category_to_imagenet_indices = {}    

    for category, payload in mapping.items():        
        syn = str(payload["shapenet_synset"])        
        syn = syn.zfill(8)        
        synset_to_category[syn] = category        
        category_to_imagenet_indices[category] = list(            
            payload.get("imagenet_class_indices", [])        
        )    

    category_depth_masks = {}    

    for category, payload in masks.items():        
        cat_masks = {}    

        # ---- exact / ancestor / sibling ----        
        for key in ["exact_mask", "ancestor_mask", "sibling_mask"]:            
            if key in payload:                
                arr = np.asarray(payload[key], dtype=np.uint8)                
                if arr.shape != (1000,):                    
                    raise ValueError(f"{category}:{key} has shape {arr.shape}")                
                cat_masks[key] = arr        
        
        # ---- depth masks ----        
        depth_masks = payload.get("depth_masks", {})        
        for depth_key, depth_mask in depth_masks.items():            
            arr = np.asarray(depth_mask, dtype=np.uint8)            
            if arr.shape != (1000,):                
                raise ValueError(f"{category}:{depth_key} has shape {arr.shape}")            
            cat_masks[depth_key] = arr        
        
        category_depth_masks[category] = cat_masks    
    
    return MappingIndex(        
        synset_to_category=synset_to_category,        
        category_to_imagenet_indices=category_to_imagenet_indices,        
        category_depth_masks=category_depth_masks,    
    )


# -----------------------------
# Model wrappers
# -----------------------------

@dataclass
class ModelBundle:
    model: torch.nn.Module
    transform: object
    device: torch.device
    mode: str  # "logits" or "knn"
    feature_dim: Optional[int] = None

def build_model(model_name: str, device: torch.device, mode: str) -> ModelBundle:
    """
    mode="logits": expect model(x) -> (B,1000)
    mode="knn":    use model as feature extractor (num_classes=0)
    """
    if mode == "logits":
        model = timm.create_model(model_name, pretrained=True)
    elif mode == "knn":
        # num_classes=0 makes timm remove classifier head when supported
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    model.eval().to(device)

    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg)

    # Infer feature dim if needed
    feature_dim = None
    if mode == "knn":
        with torch.no_grad():
            dummy = torch.zeros(1, 3, cfg["input_size"][1], cfg["input_size"][2], device=device)
            out = model(dummy)
            if out.ndim != 2:
                out = out.view(out.shape[0], -1)
            feature_dim = int(out.shape[-1])

    return ModelBundle(model=model, transform=transform, device=device, mode=mode, feature_dim=feature_dim)


# -----------------------------
# Evaluation
# -----------------------------

def predict_logits(
    bundle: ModelBundle,
    img_paths: List[str],
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      pred_idx: (N,) int
      pred_conf: (N,) float (softmax max prob)
    """
    model = bundle.model
    tfm = bundle.transform
    device = bundle.device

    pred_idx = np.full((len(img_paths),), -1, dtype=np.int32)
    pred_conf = np.full((len(img_paths),), np.nan, dtype=np.float32)

    batch_imgs = []
    batch_ids = []

    def flush():
        nonlocal batch_imgs, batch_ids
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs, dim=0).to(device)
        with torch.no_grad():
            logits = model(x)
            if logits.ndim != 2 or logits.shape[1] != 1000:
                raise RuntimeError(
                    f"logits mode requires output (B,1000). Got {tuple(logits.shape)}. "
                    f"Use --mode knn for non-ImageNet-head models."
                )
            probs = F.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        for i, sample_id in enumerate(batch_ids):
            pred_idx[sample_id] = int(idx[i].item())
            pred_conf[sample_id] = float(conf[i].item())

        batch_imgs, batch_ids = [], []

    for i, p in enumerate(tqdm(img_paths, desc="Inference (logits)")):
        img = safe_open_image(p)
        if img is None:
            continue
        x = tfm(img)
        batch_imgs.append(x)
        batch_ids.append(i)
        if len(batch_imgs) >= batch_size:
            flush()

    flush()
    return pred_idx, pred_conf


def extract_features(
    bundle: ModelBundle,
    img_paths: List[str],
    batch_size: int = 32,
) -> np.ndarray:
    """
    Returns features: (N, D) float32; rows for unreadable images are NaN.
    """
    assert bundle.mode == "knn"
    model = bundle.model
    tfm = bundle.transform
    device = bundle.device
    D = bundle.feature_dim or 0

    feats = np.full((len(img_paths), D), np.nan, dtype=np.float32)

    batch_imgs = []
    batch_ids = []

    def flush():
        nonlocal batch_imgs, batch_ids
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs, dim=0).to(device)
        with torch.no_grad():
            f = model(x)
            if f.ndim != 2:
                f = f.view(f.shape[0], -1)
            f = F.normalize(f, dim=1)
        f = f.detach().cpu().numpy().astype(np.float32)
        for j, sample_id in enumerate(batch_ids):
            feats[sample_id] = f[j]
        batch_imgs, batch_ids = [], []

    for i, p in enumerate(tqdm(img_paths, desc="Feature extraction (knn)")):
        img = safe_open_image(p)
        if img is None:
            continue
        x = tfm(img)
        batch_imgs.append(x)
        batch_ids.append(i)
        if len(batch_imgs) >= batch_size:
            flush()

    flush()
    return feats


def knn_predict(
    feats: np.ndarray,
    y_true: np.ndarray,
    train_frac: float = 0.7,
    k: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """
    Simple cosine kNN on your dataset labels (ShapeNet synsets).
    - Splits dataset into train/test by random permutation.
    - For train samples, stores features+labels.
    - For all samples, predicts via kNN among *train*.

    Returns:
      y_pred: (N,) predicted label (same space as y_true)
    """
    rng = np.random.default_rng(seed)
    N = feats.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)

    n_train = int(round(train_frac * N))
    train_idx = idx[:n_train]

    # Only keep rows with valid features
    train_idx = train_idx[~np.isnan(feats[train_idx]).any(axis=1)]
    if len(train_idx) == 0:
        raise RuntimeError("No valid training features found (all images failed to load?)")

    X_train = feats[train_idx]  # already normalized
    y_train = y_true[train_idx]

    # Precompute for efficient cosine similarity
    X_train_T = X_train.T  # (D, n_train)

    y_pred = np.array(["__invalid__"] * N, dtype=object)

    batch = 2048
    for start in tqdm(range(0, N, batch), desc=f"kNN predict (k={k})"):
        end = min(start + batch, N)
        X = feats[start:end]
        valid = ~np.isnan(X).any(axis=1)
        if not valid.any():
            continue

        Xv = X[valid]
        # cosine sim because both normalized
        sims = Xv @ X_train_T  # (b, n_train)

        # top-k indices
        topk = np.argpartition(-sims, kth=min(k, sims.shape[1]-1), axis=1)[:, :k]
        # majority vote
        for i_local, neigh in enumerate(topk):
            labels = y_train[neigh]
            # vote
            uniq, counts = np.unique(labels, return_counts=True)
            winner = uniq[np.argmax(counts)]
            y_pred[start + np.flatnonzero(valid)[i_local]] = winner

    return y_pred


# -----------------------------
# Metrics aggregation
# -----------------------------

def compute_metrics_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    classes = np.unique(y_true)
    rows = []
    for c in classes:
        tp, tn, fp, fn = one_vs_rest_counts(y_true, y_pred, c)
        m = prf1(tp, tn, fp, fn)
        rows.append({"class": c, "tp": tp, "tn": tn, "fp": fp, "fn": fn, **m})
    return pd.DataFrame(rows).sort_values(["f1", "precision", "recall"], ascending=False)

def compute_metrics_per_group(df: pd.DataFrame, correct_col: str = "is_correct") -> pd.DataFrame:
    """
    For every column in df (except Image), compute metrics for each value:
      n, correct, incorrect, accuracy
    """
    rows = []
    for col in df.columns:
        if col in ["Image"]:
            continue
        if col == correct_col:
            continue

        for val, g in df.groupby(col, dropna=False):
            n = int(len(g))
            corr = int(g[correct_col].sum())
            inc = n - corr
            acc = corr / max(n, 1)
            rows.append({
                "column": col,
                "value": str(val),
                "n": n,
                "correct": corr,
                "incorrect": inc,
                "accuracy": acc,
            })
    out = pd.DataFrame(rows)
    return out.sort_values(["column", "accuracy", "n"], ascending=[True, False, False])


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="Path to metadata.csv (semicolon-separated).")
    ap.add_argument("--image-root", default=".", help="Root directory prepended to metadata Image paths.")
    ap.add_argument("--model", required=True, help="timm model name, e.g. vit_small_patch16_224.dino")
    ap.add_argument("--mode", choices=["logits", "knn"], default="logits",
                    help="logits: model predicts ImageNet-1k indices; knn: feature extractor + kNN on dataset labels.")
    ap.add_argument("--mapping", required=True, help="ShapeNet-ImageNet mapping JSON (with indices).")
    ap.add_argument("--masks", required=True, help="ShapeNet eval masks JSON (extended).")
    ap.add_argument("--depth", default="depth_4", help="Which mask depth to use in logits mode: depth_1..depth_5")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--knn-train-frac", type=float, default=0.7)
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="eval_out", help="Output directory.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load metadata (your file is semicolon-separated)
    df = pd.read_csv(args.metadata, sep=";")
    if "Image" not in df.columns or "Class" not in df.columns:
        raise ValueError("metadata.csv must contain at least columns: Image, Class")

    # Resolve image paths
    img_paths = [os.path.join(args.image_root, p) for p in df["Image"].astype(str).tolist()]
    
    exists = np.array([os.path.exists(p) for p in img_paths])
    print(f"Image paths exist: {exists.sum()} / {len(img_paths)}")

    # Load mapping+mask index
    mi = load_mapping_and_masks(args.mapping, args.masks)

    # Map each row's ShapeNet synset -> category key (used by masks)
    synsets = df["Class"].apply(zfill_synset).astype(str).to_numpy()
    categories = np.array([mi.synset_to_category.get(s, "__unknown__") for s in synsets], dtype=object)
    df["shapenet_synset"] = synsets
    df["category_key"] = categories

    # Build model
    device = torch.device(args.device)
    bundle = build_model(args.model, device=device, mode=args.mode)

    if args.mode == "logits":
        pred_idx, pred_conf = predict_logits(bundle, img_paths, batch_size=args.batch_size)
        df["pred_imagenet_idx"] = pred_idx
        df["pred_conf"] = pred_conf

        # -------------------------------------------------
        # Correctness logic (auto-detect GT format)
        # ------------------------------------------------
        # Try to interpret GT as ImageNet index (0..999)
        cls_numeric = pd.to_numeric(df["Class"], errors="coerce")
        is_imagenet_index_gt = (    
            cls_numeric.notna().all()    
            and cls_numeric.between(0, 999).all()
        )
        
        if is_imagenet_index_gt:    
            # ---- Case A: GT = ImageNet class index ----    
            df["gt_imagenet_idx"] = cls_numeric.astype(int)    
            df["is_correct"] = (df["pred_imagenet_idx"] == df["gt_imagenet_idx"])
        else:    
            # ---- Case B: GT = ShapeNet synset ----    
            df["shapenet_synset"] = (        
                df["Class"]        
                .astype(str)        
                .str.replace(".0", "", regex=False)        
                .str.zfill(8)    
            )    
            df["category_key"] = (        
                df["shapenet_synset"]        
                .map(mi.synset_to_category)        
                .fillna("__unknown__")    
            )    
            depth = args.depth    
            is_correct = []    

            for cat, idx in zip(        
                df["category_key"].tolist(),        
                df["pred_imagenet_idx"].tolist()    
            ):        
                if cat == "__unknown__" or idx < 0:            
                    is_correct.append(False)            
                    continue        
                
                masks = mi.category_depth_masks.get(cat)        
                if masks is None or depth not in masks:            
                    is_correct.append(False)            
                    continue        
                
                is_correct.append(bool(masks[depth][idx] == 1))    
            df["is_correct"] = np.array(is_correct, dtype=bool)
        
        print("GT interpreted as ImageNet index:", is_imagenet_index_gt)
        print(df[["Class", "pred_imagenet_idx", "is_correct"]].head(5))

        # For per-class metrics in logits mode, we evaluate on ShapeNet synset space:
        #   y_pred_synset = argmax over allowed synsets? Not available from logits alone.
        # So instead we report one-vs-rest metrics over "correctness" per synset,
        # and also overall accuracy. If you want full multi-class confusion, use --mode knn.
        #
        # We'll still produce TP/TN/FP/FN per synset treating "correct for this synset" as positive.
        y_true = df["shapenet_synset"].to_numpy(dtype=object)
        y_pred = np.where(df["is_correct"].to_numpy(), y_true, "__wrong__").astype(object)

    else:  # knn mode
        feats = extract_features(bundle, img_paths, batch_size=args.batch_size)

        y_true = df["shapenet_synset"].to_numpy(dtype=object)
        y_pred = knn_predict(
            feats=feats,
            y_true=y_true,
            train_frac=args.knn_train_frac,
            k=args.knn_k,
            seed=args.seed,
        )

        df["pred_shapenet_synset"] = y_pred
        df["is_correct"] = (y_pred == y_true)

    # Overall metrics
    overall_acc = float(df["is_correct"].mean())
    print(f"\nOverall accuracy: {overall_acc:.4f}  (mode={args.mode}, model={args.model})\n")

    # Save per-sample predictions
    pred_path = os.path.join(args.outdir, "predictions.csv")
    df.to_csv(pred_path, index=False)

    # Per-class one-vs-rest (in ShapeNet synset space)
    per_class = compute_metrics_per_class(y_true=y_true, y_pred=y_pred)
    per_class_path = os.path.join(args.outdir, "metrics_per_class.csv")
    per_class.to_csv(per_class_path, index=False)

    # Per-group (every CSV column value)
    per_group = compute_metrics_per_group(df, correct_col="is_correct")
    per_group_path = os.path.join(args.outdir, "metrics_per_group.csv")
    per_group.to_csv(per_group_path, index=False)

    # Tiny summary json
    summary = {
        "model": args.model,
        "mode": args.mode,
        "overall_accuracy": overall_acc,
        "n": int(len(df)),
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Wrote:")
    print(f"  {pred_path}")
    print(f"  {per_class_path}")
    print(f"  {per_group_path}")
    print(f"  {os.path.join(args.outdir, 'summary.json')}")


if __name__ == "__main__":
    main()
