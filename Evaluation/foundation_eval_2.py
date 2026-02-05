# === ADAPTED FOUNDATION MODEL EVALUATION SCRIPT ===
# Supports DINO / DINOv2 / DINOv3 / CLIP image towers / ImageNet models via embedding-based evaluation

import csv
import json
import re
from pathlib import Path
from urllib.request import urlopen
from collections import defaultdict

import timm
import torch
import torch.nn.functional as F
from PIL import Image

# ================= CONFIG =================
CSV_PATH = Path(r"c:\Users\Stud\Documents\Unreal Projects\FinalPluginTest\Dataset\Metadata.csv")
TAXONOMY_PATH = Path(r"c:\Users\Stud\Documents\shapenetcore.taxonomy.json")
SHAPENET_IMAGENET_MAPPING = Path(r"c:\Users\Stud\Documents\ShapeNet-ImageNet1k-Mapping.json")
DATASET_ROOT = Path(r"c:\Users\Stud\Documents\Unreal Projects\FinalPluginTest")

# ---- MODEL SELECTION (change only this) ----
MODEL_NAME = "vit_small_patch16_224.dino"  # any timm backbone
EVAL_HEAD = "prototype"  # "prototype" | "knn" | "imagenet_logits"
TOPK = 5
KNN_K = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GROUPBY_FIELDS = []

ALL_METADATA_FIELDS = [
    "Object",
    "Class",
    "Material",
    "Camera Position",
    "Light Color (RGB)",
    "Fog",
    "Level",
]

_word_re = re.compile(r"\w+")

# ================= UTILITIES =================

def load_imagenet_idx2label():
    class_idx = json.load(urlopen(
        "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    ))
    return {int(k): v[1] for k, v in class_idx.items()}


def tokens_from_label(label: str):
    toks = set()
    for part in label.split(","):
        for w in _word_re.findall(part.lower()):
            toks.add(w)
    return toks


def load_image(path: Path):
    return Image.open(path).convert("RGB")

# ================= MODEL HANDLING =================

def create_backbone(model_name: str):
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval().to(DEVICE)
    cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    return model, transform


@torch.no_grad()
def get_embedding(model, x):
    if hasattr(model, "forward_features"):
        feat = model.forward_features(x)
        if isinstance(feat, dict):
            if "x_norm_clstoken" in feat:
                feat = feat["x_norm_clstoken"]
            elif "pool" in feat:
                feat = feat["pool"]
            else:
                feat = next(v for v in feat.values() if torch.is_tensor(v))
        elif isinstance(feat, (tuple, list)):
            feat = feat[0]
        if feat.ndim == 4:
            feat = feat.mean(dim=(2, 3))
        elif feat.ndim == 3:
            feat = feat[:, 0]
    else:
        feat = model(x)
    return F.normalize(feat, dim=-1)

# ================= EVAL HEADS =================

def build_prototypes(embeddings, labels):
    by_class = defaultdict(list)
    for e, y in zip(embeddings, labels):
        by_class[y].append(e.unsqueeze(0))
    proto_labels = []
    proto_mat = []
    for y, vecs in by_class.items():
        p = torch.cat(vecs, dim=0).mean(dim=0, keepdim=True)
        proto_mat.append(F.normalize(p, dim=-1))
        proto_labels.append(y)
    return torch.cat(proto_mat, dim=0), proto_labels


@torch.no_grad()
def predict_prototype(feat, proto_mat, proto_labels, topk):
    sims = feat @ proto_mat.T
    idx = sims.topk(topk, dim=1).indices
    return [[proto_labels[i] for i in row.tolist()] for row in idx]

# ================= MAIN =================

def main():
    idx2label = load_imagenet_idx2label()

    print(f"Loading model: {MODEL_NAME}")
    model, transform = create_backbone(MODEL_NAME)

    # ---- First pass: build support embeddings ----
    support_embeddings = []
    support_labels = []
    samples = []

    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            img_rel = row.get("Image", "").strip()
            obj = row.get("Object", "unknown")
            img_path = DATASET_ROOT / img_rel
            if not img_path.exists():
                continue
            img = load_image(img_path)
            x = transform(img).unsqueeze(0).to(DEVICE)
            emb = get_embedding(model, x)
            support_embeddings.append(emb.cpu())
            support_labels.append(obj)
            samples.append((row, img_rel, obj))

    support_embeddings = torch.cat(support_embeddings, dim=0)

    proto_mat, proto_labels = build_prototypes(support_embeddings, support_labels)

    # ---- Second pass: evaluate ----
    total, correct = 0, 0
    results = []

    for row, img_rel, gt_obj in samples:
        total += 1
        img_path = DATASET_ROOT / img_rel
        img = load_image(img_path)
        x = transform(img).unsqueeze(0).to(DEVICE)
        feat = get_embedding(model, x)
        preds = predict_prototype(feat, proto_mat, proto_labels, TOPK)[0]
        hit = gt_obj in preds
        if hit:
            correct += 1
        results.append({
            "image": img_rel,
            "object": gt_obj,
            "predictions": preds,
            "hit": hit,
        })

    acc = (correct / total * 100) if total else 0.0
    print(f"\nOverall accuracy (TOP-{TOPK}): {acc:.2f}% ({correct}/{total})")

    Path("results.json").write_text(json.dumps(results, indent=2))
    print("Saved results.json")


if __name__ == "__main__":
    main()
