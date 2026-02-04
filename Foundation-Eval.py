import csv
import json
import re
from pathlib import Path
from urllib.request import urlopen

import timm
import torch
from PIL import Image

# CONFIG - adjust if needed
CSV_PATH = Path(r"c:\Users\Stud\Documents\Unreal Projects\FinalPluginTest\Dataset\Metadata.csv")
TAXONOMY_PATH = Path(r"c:\Users\Stud\Documents\shapenetcore.taxonomy.json")
DATASET_ROOT = Path(r"c:\Users\Stud\Documents\Unreal Projects\FinalPluginTest")
MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"  # swap for smaller model if needed
TOPK = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_word_re = re.compile(r"\w+")


def load_imagenet_idx2label():
    class_idx = json.load(urlopen("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"))
    return {int(k): v[1] for k, v in class_idx.items()}


def build_taxonomy_map(taxonomy_path: Path):
    tax = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    m = {}
    for entry in tax:
        meta = entry.get("metadata", {})
        name = meta.get("name")
        label = meta.get("label") or entry.get("text") or ""
        if name:
            try:
                m[int(name)] = label
            except Exception:
                # fallback if name not numeric
                m[name] = label
    return m


def tokens_from_label(label: str):
    # split on commas then extract word tokens
    toks = set()
    for part in label.split(","):
        for w in _word_re.findall(part.lower()):
            toks.add(w)
    return toks


def load_image(path: Path):
    return Image.open(path).convert("RGB")


def main():
    idx2label = load_imagenet_idx2label()
    tax_map = build_taxonomy_map(TAXONOMY_PATH)
    #print(tax_map)


    print(f"Loading model {MODEL_NAME} -> device {DEVICE}", flush=True)
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.eval().to(DEVICE)
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    #Test
    test_labels = model.pretrained_cfg['label_names']
    top_k = min(len(test_labels), 5)

    total = 0
    correct = 0
    results = []

    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            total += 1
            img_rel = row.get("Image", "").strip()
            class_val = row.get("Class", "").strip()
            obj_name = row.get("Object", "").strip()

            # resolve image path
            img_path = Path(img_rel)            
            img_path = DATASET_ROOT / img_rel
            if not img_path.exists():
                results.append({"image": img_rel, "error": "IMAGE_NOT_FOUND"})
                continue

            
            # find ground truth label by numeric metadata:name
            gt_label = None
            try:
                gt_key = int(class_val)
            except Exception:
                gt_key = class_val
            gt_label = tax_map.get(gt_key) or obj_name or str(class_val)

            gt_tokens = tokens_from_label(gt_label)
            

            # inference
            img = load_image(img_path)
            with torch.no_grad():
                inp = transform(img).unsqueeze(0).to(DEVICE)
                out = model(inp)
                
                #Test
                probabilites = torch.nn.functional.softmax(out[0], dim=0)
                values, indices = torch.topk(probabilites, top_k)
                predictions = [    
                    {"label": test_labels[i], "score": v.item()}    
                    for i, v in zip(indices, values)
                ]
                print(predictions)


                probs, inds = torch.topk(torch.softmax(out, dim=1), k=TOPK)
                probs = probs[0].cpu().tolist()
                inds = inds[0].cpu().tolist()

            pred_labels = [idx2label.get(int(i), "Unknown") for i in inds]
            pred_tokens = [tokens_from_label(p) for p in pred_labels]

            # match if any predicted token intersects gt tokens
            hit = any(bool(pt & gt_tokens) for pt in pred_tokens)

            if hit:
                correct += 1

            results.append({
                "image": img_rel,
                "class_value": class_val,
                "ground_truth_label": gt_label,
                "predictions": [{"label": lbl, "prob": float(p)} for lbl, p in zip(pred_labels, probs)],
                "hit": bool(hit)
            })

            if total % 50 == 0:
                print(f"Processed {total} images...", flush=True)

    acc = (correct / total * 100) if total else 0.0
    print(f"Total: {total}  Correct(top-{TOPK}): {correct}  Accuracy: {acc:.2f}%", flush=True)

    out_path = Path("classification_results.json")
    #out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote per-image results to {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()