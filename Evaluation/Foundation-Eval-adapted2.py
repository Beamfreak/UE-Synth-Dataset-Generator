
"""
Foundation model evaluation against ImageNet classes with ShapeNet->ImageNet mapping.

What this script can do out-of-the-box:
- For ImageNet-supervised timm models (logit head over 1000 classes): predict ImageNet top-k.
- For timm CLIP models that include text+image encoders: predict ImageNet top-k via zero-shot prompting.
- Score predictions against your ShapeNet classes using your ShapeNet->ImageNet mapping (wnids).

What it cannot do without extra data/training:
- Produce ImageNet class predictions from pure self-supervised DINO/DINOv2/DINOv3 *image-only* checkpoints
  (they output embeddings, not 1000-way logits). For those, you need either:
  (a) a linear probe trained on ImageNet, or
  (b) kNN retrieval against an ImageNet-labeled embedding database, or
  (c) a vision-language model (CLIP) for zero-shot labeling.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_model_data_config, create_transform
from PIL import Image

IMAGENET_CLASS_INDEX_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"

_word_re = re.compile(r"\w+")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_imagenet_class_index() -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Returns:
      idx2wnid: {0..999 -> 'n01440764'}
      idx2label: {0..999 -> 'tench'}
    """
    import urllib.request
    with urllib.request.urlopen(IMAGENET_CLASS_INDEX_URL) as resp:
        class_idx = json.load(resp)
    idx2wnid = {int(k): v[0] for k, v in class_idx.items()}
    idx2label = {int(k): v[1] for k, v in class_idx.items()}
    return idx2wnid, idx2label


def normalize_shapenet_synset(raw: str) -> str:
    # CSV sometimes stores without leading zero (e.g., 2958343 instead of 02958343)
    raw = str(raw).strip()
    raw = re.sub(r"\D", "", raw)
    return raw.zfill(8)


def extract_mapping(shapenet_to_imagenet: Any) -> Dict[str, List[str]]:
    """
    Coerce a ShapeNet->ImageNet mapping JSON into a usable dict.

    This script supports TWO mapping styles:

    (A) Synset->WNID mapping (recommended for rigorous scoring):
        { "02958343": ["n02691156", ...], ... }

    (B) ClassName->ImageNetLabelString mapping (your file format):
        {
          "airplane": ["airliner", "warplane, military plane", "aircraft"],
          "bag": ["bag", "backpack, ..."],
          ...
        }

    The returned dict will map either:
      - shapenet_synset8 -> [imagenet_wnid, ...]   (case A)
      - shapenet_class_name_lower -> [label_string, ...] (case B)
    """
    out: Dict[str, List[str]] = {}

    wnid_re = re.compile(r"^n\d{8}$")
    synset8_re = re.compile(r"^\d{8}$")

    def norm_synset(x: str) -> str:
        x = str(x).strip()
        x = re.sub(r"\D", "", x)
        return x.zfill(8)

    def norm_key_name(x: str) -> str:
        return str(x).strip().lower()

    def norm_label(x: str) -> str:
        # keep commas because ImageNet labels are comma-separated synonyms; we normalize whitespace
        return re.sub(r"\s+", " ", str(x).strip().lower())

    # ---- Dict input ----
    if isinstance(shapenet_to_imagenet, dict) and shapenet_to_imagenet:
        # Heuristic: decide if keys look like synsets (digits) or names (strings)
        sample_k = next(iter(shapenet_to_imagenet.keys()))
        sample_v = shapenet_to_imagenet[sample_k]

        # Case A: synset->wnids
        if (isinstance(sample_k, str) and re.sub(r"\D", "", sample_k) and
            (isinstance(sample_v, (list, tuple)) and sample_v and wnid_re.match(str(sample_v[0]).strip()) )):

            for k, v in shapenet_to_imagenet.items():
                sn = norm_synset(k)
                if isinstance(v, (list, tuple)):
                    wnids = [str(x).strip() for x in v if wnid_re.match(str(x).strip())]
                elif isinstance(v, dict):
                    cand = v.get("imagenet") or v.get("wnids") or v.get("imagenet_wnids") or v.get("imagenetWnids")
                    wnids = [str(x).strip() for x in (cand or []) if wnid_re.match(str(x).strip())]
                else:
                    wnids = []
                if wnids:
                    out[sn] = wnids
            return out

        # Case B: name->labelstrings (your mapping)
        if isinstance(sample_v, (list, tuple)) and sample_v and isinstance(sample_v[0], str) and not wnid_re.match(sample_v[0].strip()):
            for k, v in shapenet_to_imagenet.items():
                key = norm_key_name(k)
                labels = [norm_label(x) for x in v if isinstance(x, str) and str(x).strip()]
                if labels:
                    out[key] = labels
            return out

        # Fallback: try to interpret dict values as nested structures containing wnids
        for k, v in shapenet_to_imagenet.items():
            sn = norm_synset(k)
            wnids: List[str] = []
            if isinstance(v, dict):
                for kk in ("imagenet", "wnids", "imagenet_wnids", "imagenetWnids"):
                    if kk in v and isinstance(v[kk], (list, tuple)):
                        wnids = [str(x).strip() for x in v[kk] if wnid_re.match(str(x).strip())]
                        break
            if wnids:
                out[sn] = wnids
        return out

    # ---- List-of-records input (case A style) ----
    if isinstance(shapenet_to_imagenet, list):
        for rec in shapenet_to_imagenet:
            if not isinstance(rec, dict):
                continue
            sn_raw = rec.get("shapenetSynset") or rec.get("synset") or rec.get("shapenet_synset") or rec.get("id")
            if not sn_raw:
                continue
            sn = norm_synset(sn_raw)
            cand = rec.get("imagenet") or rec.get("imagenetWnids") or rec.get("imagenet_wnids") or rec.get("wnids") or rec.get("imagenetWnid")
            wnids: List[str] = []
            if isinstance(cand, str) and wnid_re.match(cand.strip()):
                wnids = [cand.strip()]
            elif isinstance(cand, (list, tuple)):
                wnids = [str(x).strip() for x in cand if wnid_re.match(str(x).strip())]
            if wnids:
                out[sn] = wnids
        return out

    return out

def load_taxonomy_name_map(taxonomy: Any) -> Dict[str, str]:
    """
    ShapeNet taxonomy is often a list of dicts containing synsetId + name.
    Returns { '02958343' : 'car, auto, automobile, machine, motorcar' } or similar.
    """
    out: Dict[str, str] = {}
    if isinstance(taxonomy, list):
        for rec in taxonomy:
            if not isinstance(rec, dict):
                continue
            sid = rec.get("synsetId") or rec.get("synset") or rec.get("id")
            name = rec.get("name") or rec.get("label") or rec.get("names")
            if sid and name:
                sn = normalize_shapenet_synset(sid)
                out[sn] = str(name)
    elif isinstance(taxonomy, dict):
        # sometimes nested; best-effort flatten
        for k, v in taxonomy.items():
            if isinstance(v, dict) and ("name" in v or "label" in v):
                out[normalize_shapenet_synset(k)] = str(v.get("name") or v.get("label"))
    return out


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def create_model_and_transform(model_name: str, device: torch.device):
    """
    Creates a timm model and its inference transform (size + normalization).
    For supervised ImageNet models we keep the default head.
    For CLIP models in timm, we keep as-is (they often have encode_image/encode_text).
    """
    model = timm.create_model(model_name, pretrained=True)
    model.eval().to(device)
    cfg = resolve_model_data_config(model)
    tfm = create_transform(**cfg, is_training=False)
    return model, tfm


@torch.no_grad()
def imagenet_logits_predict(model, x: torch.Tensor, topk: int) -> torch.Tensor:
    """
    Returns topk indices [B, topk] for ImageNet-1k logits models.
    """
    out = model(x)
    if out.ndim != 2 or out.shape[1] != 1000:
        raise ValueError(f"Model output is {tuple(out.shape)}; expected [B,1000] logits. "
                         f"This model likely does not have an ImageNet classifier head.")
    return out.topk(topk, dim=1).indices


@torch.no_grad()
def clip_zeroshot_predict(
    model,
    x: torch.Tensor,
    text_tokens: torch.Tensor,
    text_features: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """
    timm CLIP models typically expose encode_image / encode_text.
    We compare normalized image/text features.
    """
    if not (hasattr(model, "encode_image") and hasattr(model, "encode_text")):
        raise ValueError("Selected model does not look like a CLIP model in timm (missing encode_image/encode_text).")

    img_feat = model.encode_image(x)
    img_feat = F.normalize(img_feat, dim=-1)

    # text_features is precomputed normalized [1000, D]
    logits = img_feat @ text_features.T  # [B,1000]
    return logits.topk(topk, dim=1).indices


def build_imagenet_prompts(idx2label: Dict[int, str], template: str) -> List[str]:
    return [template.format(idx2label[i].replace("_", " ")) for i in range(1000)]


def tokenize_for_timm_clip(model, texts: List[str], device: torch.device) -> torch.Tensor:
    """
    timm CLIP models include a tokenizer util in some versions via model.tokenize.
    If not, we cannot do CLIP zero-shot within timm alone.
    """
    if hasattr(model, "tokenize"):
        return model.tokenize(texts).to(device)
    raise ValueError("This timm CLIP model does not expose a tokenizer via model.tokenize(). "
                     "Use a CLIP variant that does, or integrate open_clip/transformers.")


@torch.no_grad()
def compute_text_features(model, text_tokens: torch.Tensor) -> torch.Tensor:
    txt = model.encode_text(text_tokens)
    return F.normalize(txt, dim=-1)


def evaluate(
    csv_path: Path,
    dataset_root: Path,
    model_name: str,
    mode: str,
    mapping_path: Path,
    taxonomy_path: Optional[Path],
    topk: int,
    device: torch.device,
    prompt_template: str,
    limit: Optional[int],
) -> None:
    idx2wnid, idx2label = load_imagenet_class_index()

    mapping_raw = load_json(mapping_path)
    sn2im = extract_mapping(mapping_raw)
    if not sn2im:
        raise ValueError("Could not parse ShapeNet->ImageNet mapping JSON into a usable dict. "
                         "Please inspect the file format.")

    # Determine mapping style
    _wnid_re = re.compile(r'^n\d{8}$')
    mapping_is_wnid = any(isinstance(v, list) and v and _wnid_re.match(str(v[0])) for v in sn2im.values())


    sn2name = {}
    if taxonomy_path and taxonomy_path.exists():
        sn2name = load_taxonomy_name_map(load_json(taxonomy_path))

    model, transform = create_model_and_transform(model_name, device)

    # CLIP zero-shot precomputation
    text_tokens = None
    text_features = None
    if mode == "clip_zeroshot":
        prompts = build_imagenet_prompts(idx2label, prompt_template)
        text_tokens = tokenize_for_timm_clip(model, prompts, device)
        text_features = compute_text_features(model, text_tokens)  # [1000, D]

    total = 0
    top1 = 0
    topk_hits = 0

    per_image = []

    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            if limit is not None and total >= limit:
                break

            rel = (row.get("Image") or "").strip()
            if not rel:
                continue
            img_path = dataset_root / rel
            if not img_path.exists():
                continue

            sn = normalize_shapenet_synset(row.get("Class", ""))
            gt_obj = row.get("Object", "")
            gt_name = sn2name.get(sn, "")
            # Depending on mapping style, look up by synset or by class name
            if mapping_is_wnid:
                allowed_targets = sn2im.get(sn, [])
            else:
                key = (gt_name or row.get("ClassName") or row.get("Class") or "").strip().lower()
                allowed_targets = sn2im.get(key, [])
            if not allowed_targets:
                # No mapping for this sample; skip (or count as miss)
                continue

            img = load_image(img_path)
            x = transform(img).unsqueeze(0).to(device)

            if mode == "imagenet_logits":
                pred_idx = imagenet_logits_predict(model, x, topk)[0].tolist()
            elif mode == "clip_zeroshot":
                pred_idx = clip_zeroshot_predict(model, x, text_tokens, text_features, topk)[0].tolist()
            else:
                raise ValueError("mode must be 'imagenet_logits' or 'clip_zeroshot'")

            pred_wnids = [idx2wnid[i] for i in pred_idx]
            pred_labels = [idx2label[i] for i in pred_idx]

            if mapping_is_wnid:
                hit_k = any(w in allowed_targets for w in pred_wnids)
                hit_1 = pred_wnids[0] in allowed_targets
            else:
                hit_k = any(label_match(lbl, allowed_targets) for lbl in pred_labels)
                hit_1 = label_match(pred_labels[0], allowed_targets)

            total += 1
            topk_hits += int(hit_k)
            top1 += int(hit_1)

            per_image.append({
                "image": rel,
                "shapenet_synset": sn,
                "shapenet_object": gt_obj,
                "shapenet_name": gt_name,
                "mapped_imagenet_wnids": allowed_wnids,
                "pred_imagenet_wnids": pred_wnids,
                "pred_imagenet_labels": pred_labels,
                "top1_hit": hit_1,
                "topk_hit": hit_k,
            })

    if total == 0:
        raise RuntimeError("No evaluable samples found (missing images, mapping gaps, or CSV path issues).")

    top1_acc = 100.0 * top1 / total
    topk_acc = 100.0 * topk_hits / total

    print(f"\nModel: {model_name}")
    print(f"Mode:  {mode}")
    print(f"Samples evaluated: {total}")
    print(f"Top-1 accuracy (via mapping): {top1_acc:.2f}% ({top1}/{total})")
    print(f"Top-{topk} accuracy (via mapping): {topk_acc:.2f}% ({topk_hits}/{total})")

    out_path = Path("results_imagenet_mapping.json")
    out_path.write_text(json.dumps(per_image, indent=2), encoding="utf-8")
    print(f"Saved: {out_path.resolve()}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=r"c:\Users\Stud\Documents\Unreal Projects\FinalPluginTest\Dataset\Metadata.csv", help="Path to Metadata.csv")
    p.add_argument("--root", type=Path, default=r"c:\Users\Stud\Documents\Unreal Projects\FinalPluginTest", help="Dataset root to resolve image paths in CSV")
    p.add_argument("--model", type=str, required=True, help="timm model name")
    p.add_argument("--mode", type=str, default="imagenet_logits",
                   choices=["imagenet_logits", "clip_zeroshot"],
                   help="Prediction mode: ImageNet logits head or CLIP zero-shot.")
    p.add_argument("--mapping", type=Path, default=r"c:\Users\Stud\Documents\ShapeNet-ImageNet1k-Mapping.json", help="ShapeNet->ImageNet mapping JSON (wnids like n########).")
    p.add_argument("--taxonomy", type=Path, default=r"c:\Users\Stud\Documents\shapenetcore.taxonomy.json", help="ShapeNet taxonomy JSON (optional).")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--prompt_template", type=str, default="a photo of a {}",
                   help="Prompt template for CLIP zero-shot (uses ImageNet label).")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples for quick tests.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(
        csv_path=args.csv,
        dataset_root=args.root,
        model_name=args.model,
        mode=args.mode,
        mapping_path=args.mapping,
        taxonomy_path=args.taxonomy,
        topk=args.topk,
        device=device,
        prompt_template=args.prompt_template,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()