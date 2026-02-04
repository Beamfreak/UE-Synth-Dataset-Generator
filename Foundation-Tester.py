import timm
import torch
from PIL import Image
from urllib.request import urlopen
import json

#test
print("Foundation-Tester.py loaded")
l = len(timm.list_models('*'))
print(f"Number of models available: {l}")

clip = timm.list_models('*clip*', pretrained=True)
#print(f"CLIP models available: {clip}")

dino = timm.list_models('*dino*', pretrained=True)
#print(f"DINO models available: {dino}")


print("\nSelecting model vit_base_patch16_dinov3.lvd1689m...")
try:
    model = timm.create_model('vit_base_patch16_dinov3.lvd1689m', pretrained=True)
    print("Model created successfully")
    model = model.eval()

    img = Image.open(urlopen('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png')).convert('RGB')
    
    print("Creating transform...")
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    print("Running inference...")
    output = model(transform(img).unsqueeze(0))
    top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
    print("Top-5 probabilities:", top5_probabilities)
    print("Top-5 labels:", top5_class_indices)

    # ImageNet class mapping
    class_idx = json.load(urlopen('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'))
    idx2label = {int(k): v[1] for k, v in class_idx.items()}

    labels = [idx2label.get(int(i), 'Unknown') for i in top5_class_indices[0].tolist()]
    probs = [float(p) for p in top5_probabilities[0].tolist()]

    print("Top-5 probabilities:", probs)
    print("Top-5 labels:", labels)
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()



'''
import csv
import json
from pathlib import Path
from urllib.request import urlopen

import timm
import torch
from PIL import Image

# CONFIG
CSV_PATH = Path(r"c:\Users\Stud\Documents\Unreal Projects\FinalPluginTest\Dataset\Metadata.csv")
TAXONOMY_PATH = Path(r"c:\Users\Stud\Documents\shapenetcore.taxonomy.json")
DATASET_ROOT = Path(r"c:\Users\Stud\Documents\Unreal Projects\FinalPluginTest\Dataset")  # images are relative to CSV Image column
MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"  # change to smaller model if needed
TOPK = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_imagenet_idx2label():
    class_idx = json.load(urlopen('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'))
    return {int(k): v[1] for k, v in class_idx.items()}

def build_taxonomy_map(taxonomy_path: Path):
    with taxonomy_path.open('r', encoding='utf-8') as f:
        tax = json.load(f)
    # map: metadata.name -> canonical label string; also collect searchable tokens
    meta_map = {}
    for entry in tax:
        meta = entry.get("metadata", {})
        name = meta.get("name")
        label = meta.get("label") or entry.get("text") or ""
        if name:
            meta_map[str(name)] = label
    return meta_map

def find_ground_truth_label(csv_class_value: str, object_name: str, meta_map: dict):
    # 1) direct match by key
    if str(csv_class_value) in meta_map:
        return meta_map[str(csv_class_value)]
    # 2) try match by substring tokens from object_name -> taxonomy labels
    obj_tokens = set(t.lower() for t in object_name.replace('_', ' ').split())
    for k, label in meta_map.items():
        lab_tokens = set(t.lower().strip() for t in label.split(','))
        if obj_tokens & lab_tokens:
            return label
    # 3) fallback: return object's raw name
    return object_name

def load_image(img_path: Path):
    return Image.open(img_path).convert("RGB")

def main():
    idx2label = load_imagenet_idx2label()
    meta_map = build_taxonomy_map(TAXONOMY_PATH)

    # create model
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.eval().to(DEVICE)
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    total = 0
    correct = 0
    per_row_results = []

    with CSV_PATH.open(encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)
        # find indices
        def idx_of(name):
            try:
                return header.index(name)
            except ValueError:
                return None
        img_idx = idx_of("Image") or 0
        obj_idx = idx_of("Object") or 1
        class_idx = idx_of("Class") or 3

        for row in reader:
            total += 1
            img_rel = row[img_idx].strip()
            obj = row[obj_idx].strip() if obj_idx is not None else ""
            cls = row[class_idx].strip() if class_idx is not None else ""

            img_path = (DATASET_ROOT / img_rel) if not Path(img_rel).is_absolute() else Path(img_rel)
            if not img_path.exists():
                per_row_results.append((img_rel, obj, cls, "IMAGE_NOT_FOUND"))
                continue

            gt_label = find_ground_truth_label(cls, obj, meta_map)

            img = load_image(img_path)
            with torch.no_grad():
                inp = transform(img).unsqueeze(0).to(DEVICE)
                out = model(inp)
                probs, inds = torch.topk(out.softmax(dim=1), k=TOPK)
                preds = [idx2label.get(int(i), "Unknown").lower() for i in inds[0].tolist()]

            # compare: consider correct if any predicted token matches any token in gt_label
            gt_tokens = set(t.strip().lower() for t in gt_label.split(','))
            hit = any(any(gt_tok in pred for gt_tok in gt_tokens) for pred in preds)

            if hit:
                correct += 1
            per_row_results.append((img_rel, obj, cls, gt_label, preds, hit))

    # summary
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"Processed: {total} images")
    print(f"Correct (top-{TOPK}): {correct}")
    print(f"Accuracy: {acc:.2f}%")

    # optional: write per-row results
    out_path = Path("classification_results.json")
    with out_path.open("w", encoding='utf-8') as outf:
        json.dump(per_row_results, outf, indent=2)
    print(f"Per-row results written to {out_path.resolve()}")

if __name__ == "__main__":
    main()
'''