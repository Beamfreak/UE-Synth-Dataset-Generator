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
SHAPENET_IMAGENET_MAPPING = Path(r"c:\Users\Stud\Documents\ShapeNet-ImageNet1k-Mapping.json")
DATASET_ROOT = Path(r"c:\Users\Stud\Documents\Unreal Projects\FinalPluginTest")
MODEL_NAME = "resnet50.a1_in1k"  # swap for smaller model if needed
TOPK = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Granularity config - which metadata fields to group evaluation by
# Options: "Material", "Level", "Camera Position", "Light Color (RGB)", "Fog", or combinations
# Empty list = overall accuracy, or specify fields like ["Material"], ["Level", "Material"], etc.
GROUPBY_FIELDS = []  # Change to ["Material"] for per-material accuracy, ["Level"] for per-level, etc.

# All metadata fields available for automatic per-field analysis
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


def build_shapenet_imagenet_map(mapping_path: Path):
    """Load ShapeNet to ImageNet mapping for better label matching."""
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    return mapping


def build_reverse_shapenet_map(shapenet_map: dict) -> dict:
    """Build reverse mapping: imagenet_label(lower) -> shapenet_key."""
    reverse = {}
    for skey, ilist in shapenet_map.items():
        if isinstance(ilist, list):
            for imlabel in ilist:
                reverse[imlabel.strip().lower()] = skey
        else:
            reverse[str(ilist).strip().lower()] = skey
    return reverse


def map_pred_label_to_shapenet(pred_label: str, reverse_map: dict, shapenet_map: dict):
    if not pred_label:
        return None
    key = pred_label.strip().lower()
    # direct lookup
    if key in reverse_map:
        return reverse_map[key]
    # fallback: token overlap with mapping values
    pred_toks = tokens_from_label(pred_label)
    for skey, ilist in shapenet_map.items():
        # join ilist into single string to token-compare
        texts = ", ".join(ilist) if isinstance(ilist, list) else str(ilist)
        if pred_toks & tokens_from_label(texts):
            return skey
    return None



def tokens_from_label(label: str):
    # split on commas then extract word tokens
    toks = set()
    for part in label.split(","):
        for w in _word_re.findall(part.lower()):
            toks.add(w)
    return toks


def load_image(path: Path):
    return Image.open(path).convert("RGB")


def get_group_key(row: dict, fields: list) -> str:
    """Generate a grouping key from specified metadata fields."""
    if not fields:
        return "overall"
    key_parts = []
    for field in fields:
        value = row.get(field, "unknown").strip()
        key_parts.append(f"{field}={value}")
    return " | ".join(key_parts)


def compute_group_stats(group_results: dict) -> dict:
    """Compute accuracy statistics for each group."""
    group_stats = {}
    for group_key, entries in group_results.items():
        total = len(entries)
        correct = sum(1 for e in entries if e["hit"])
        acc = (correct / total * 100) if total else 0.0
        group_stats[group_key] = {
            "total": total,
            "correct": correct,
            "accuracy": acc
        }
    return group_stats

def compute_group_stats(group_results: dict) -> dict:
    """Compute accuracy statistics for each group."""
    group_stats = {}
    for group_key, entries in group_results.items():
        total = len(entries)
        correct = sum(1 for e in entries if e["hit"])
        group_stats[group_key] = {
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": (correct / total * 100) if total else 0.0
        }
    return group_stats


def generate_analysis_table(results: list, groupby_fields: list = None) -> str:
    """Generate a detailed analysis table from results."""
    if groupby_fields is None:
        groupby_fields = []
    
    # Group results
    groups = {}
    for result in results:
        if "error" in result:
            continue
        
        # Build group key
        if groupby_fields:
            key_parts = [f"{field}={result.get(field, 'unknown')}" for field in groupby_fields]
            group_key = " | ".join(key_parts)
        else:
            group_key = "overall"
        
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(result)
    
    # Build table header
    lines = []
    lines.append("="*130)
    header_parts = ["Group"] if groupby_fields else ["Overall Results"]
    header_parts.extend(["Total", "Correct (TP)", "Incorrect (FP)", "Accuracy (%)", "Hit Rate"])
    lines.append(f"{'Group/Result':50} | {'Total':>8} | {'TP':>8} | {'FP':>8} | {'Accuracy':>10} | {'Hit Rate':>8}")
    lines.append("-"*130)
    
    # Add rows for each group
    for group_key in sorted(groups.keys()):
        entries = groups[group_key]
        total = len(entries)
        tp = sum(1 for e in entries if e["hit"])
        fp = total - tp
        accuracy = (tp / total * 100) if total else 0.0
        hit_rate = (tp / total) if total else 0.0
        
        lines.append(f"{group_key:50} | {total:>8d} | {tp:>8d} | {fp:>8d} | {accuracy:>9.2f}% | {hit_rate:>7.2%}")
    
    lines.append("="*130)
    return "\n".join(lines)


def generate_class_analysis(results: list, groupby_fields: list = None) -> str:
    """Generate per-class analysis for each group."""
    if groupby_fields is None:
        groupby_fields = []
    
    # Group results by (group_key, object_class)
    groups = {}
    for result in results:
        if "error" in result:
            continue
        
        obj_class = result.get("object", "unknown")
        
        if groupby_fields:
            key_parts = [f"{field}={result.get(field, 'unknown')}" for field in groupby_fields]
            group_key = " | ".join(key_parts)
        else:
            group_key = "overall"
        
        full_key = (group_key, obj_class)
        if full_key not in groups:
            groups[full_key] = []
        groups[full_key].append(result)
    
    # Build table
    lines = []
    lines.append("="*150)
    if groupby_fields:
        lines.append(f"Per-Class Breakdown by {groupby_fields}")
    else:
        lines.append("Per-Class Breakdown (Overall)")
    lines.append("="*150)
    lines.append(f"{'Group':40} | {'Class':20} | {'Total':>6} | {'TP':>6} | {'FP':>6} | {'Accuracy':>10} | {'Precision':>10}")
    lines.append("-"*150)
    
    for (group_key, obj_class) in sorted(groups.keys()):
        entries = groups[(group_key, obj_class)]
        total = len(entries)
        tp = sum(1 for e in entries if e["hit"])
        fp = total - tp
        accuracy = (tp / total * 100) if total else 0.0
        
        # Precision: TP / (TP + FP) - how many of our positive predictions were correct
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        
        group_display = group_key[:40] if group_key else "overall"
        lines.append(f"{group_display:40} | {obj_class:20} | {total:>6d} | {tp:>6d} | {fp:>6d} | {accuracy:>9.2f}% | {precision:>9.2f}%")
    
    lines.append("="*150)
    return "\n".join(lines)


def save_detailed_results(results: list, output_path: Path):
    """Save results to JSON with all details for later analysis."""
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved detailed results to {output_path.resolve()}")


def compute_confusion_from_results(results: list, groupby_fields: list, shapenet_keys: set):
    """Build per-group confusion matrices (true->pred counts)."""
    groups = {}
    for r in results:
        if "error" in r:
            continue
        if groupby_fields:
            key_parts = [f"{field}={r.get(field, 'unknown')}" for field in groupby_fields]
            group_key = " | ".join(key_parts)
        else:
            group_key = "overall"

        gt = r.get("gt_shapenet") or r.get("object", "unknown")
        pred = r.get("pred_shapenet") or "unknown"

        if group_key not in groups:
            groups[group_key] = {}
        mat = groups[group_key]
        if gt not in mat:
            mat[gt] = {}
        mat[gt][pred] = mat[gt].get(pred, 0) + 1
    return groups


def compute_metrics_from_confusion(confusion: dict) -> dict:
    """Given confusion matrix dict true->pred->count, compute TP/FP/FN/TN and derived metrics per class."""
    classes = set(confusion.keys())
    for t in confusion:
        classes.update(confusion[t].keys())
    total = sum(sum(preds.values()) for preds in confusion.values())
    metrics = {}
    for c in sorted(classes):
        tp = confusion.get(c, {}).get(c, 0)
        fp = sum(confusion.get(t, {}).get(c, 0) for t in confusion if t != c)
        fn = sum(v for p, v in confusion.get(c, {}).items() if p != c)
        tn = total - tp - fp - fn
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        npv = (tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        accuracy = ((tp + tn) / total) if total > 0 else 0.0
        metrics[c] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "npv": npv,
            "accuracy": accuracy,
            "support": sum(confusion.get(c, {}).values())
        }
    return metrics


def print_confusion_metrics_by_group(confusions: dict):
    for gk in sorted(confusions.keys()):
        print(f"\nGROUP: {gk}")
        conf = confusions[gk]
        metrics = compute_metrics_from_confusion(conf)
        print("Class".ljust(30) + " TP   FP   FN   TN   Prec    Rec    F1     Spec   NPV    Acc   Support")
        print("-"*110)
        for cls, m in sorted(metrics.items(), key=lambda x: x[0]):
            print(f"{cls:30} {m['TP']:4d} {m['FP']:4d} {m['FN']:4d} {m['TN']:4d} {m['precision']:6.2f} {m['recall']:6.2f} {m['f1']:6.2f} {m['specificity']:6.2f} {m['npv']:6.2f} {m['accuracy']:6.2f} {m['support']:8d}")

def main():
    idx2label = load_imagenet_idx2label()
    tax_map = build_taxonomy_map(TAXONOMY_PATH)
    shapenet_imagenet_map = build_shapenet_imagenet_map(SHAPENET_IMAGENET_MAPPING)
    reverse_shapenet_map = build_reverse_shapenet_map(shapenet_imagenet_map)

    print(f"Loading model {MODEL_NAME} -> device {DEVICE}", flush=True)
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.eval().to(DEVICE)
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    total = 0
    correct = 0
    results = []
    group_results = {}  # Track results by group for granular evaluation

    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            total += 1
            img_rel = row.get("Image", "").strip()
            class_val = row.get("Class", "").strip()
            obj_name = row.get("Object", "").strip()

            # resolve image path
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
            # keep original for shapenet matching
            orig_gt_label = gt_label
            gt_shapenet = None
            # Try to get ImageNet label mappings from ShapeNet mapping and record shapenet key
            for shapenet_key, imagenet_labels in shapenet_imagenet_map.items():
                if orig_gt_label and (orig_gt_label.lower() in shapenet_key.lower() or shapenet_key.lower() in orig_gt_label.lower()):
                    gt_shapenet = shapenet_key
                    gt_label = ", ".join(imagenet_labels) if isinstance(imagenet_labels, list) else imagenet_labels
                    break

            gt_tokens = tokens_from_label(gt_label)

            # inference
            img = load_image(img_path)
            with torch.no_grad():
                inp = transform(img).unsqueeze(0).to(DEVICE)
                out = model(inp)

                probs, inds = torch.topk(torch.softmax(out, dim=1), k=TOPK)
                probs = probs[0].cpu().tolist()
                inds = inds[0].cpu().tolist()

            pred_labels = [idx2label.get(int(i), "Unknown") for i in inds]
            pred_tokens = [tokens_from_label(p) for p in pred_labels]

            # match if any predicted token intersects gt tokens
            hit = any(bool(pt & gt_tokens) for pt in pred_tokens)

            if hit:
                correct += 1

            # Map predictions and ground-truth to ShapeNet keys where possible
            top_pred_label = pred_labels[0] if pred_labels else None
            pred_shapenet = map_pred_label_to_shapenet(top_pred_label, reverse_shapenet_map, shapenet_imagenet_map)

            # Get group key for granular evaluation
            group_key = get_group_key(row, GROUPBY_FIELDS)
            if group_key not in group_results:
                group_results[group_key] = []
            
            result_entry = {
                "image": img_rel,
                "class_value": class_val,
                "object": obj_name,
                "ground_truth_label": gt_label,
                "gt_shapenet": gt_shapenet,
                "pred_shapenet": pred_shapenet,
                "predictions": [{"label": lbl, "prob": float(p)} for lbl, p in zip(pred_labels, probs)],
                "hit": bool(hit)
            }
            
            # Include all metadata columns from the CSV for flexible grouping later
            for k, v in row.items():
                result_entry[k] = v.strip() if isinstance(v, str) else v
            
            results.append(result_entry)
            group_results[group_key].append(result_entry)

            if total % 50 == 0:
                print(f"Processed {total} images...", flush=True)

    # Compute and display overall accuracy
    acc = (correct / total * 100) if total else 0.0
    print(f"\n{'='*60}")
    print(f"Overall Results | Total: {total}  Correct(top-{TOPK}): {correct}  Accuracy: {acc:.2f}%")
    print(f"{'='*60}")
    
    # Compute and display grouped statistics if applicable
    if GROUPBY_FIELDS:
        print(f"\nAccuracy breakdown by {GROUPBY_FIELDS}:")
        group_stats = compute_group_stats(group_results)
        for group_key in sorted(group_stats.keys()):
            stats = group_stats[group_key]
            print(f"  {group_key:50s} | Total: {stats['total']:3d}  Correct: {stats['correct']:3d}  Accuracy: {stats['accuracy']:6.2f}%")
    
    # Save detailed results to JSON
    out_path = Path("classification_results.json")
    save_detailed_results(results, out_path)
    
    # Generate and display comprehensive analysis tables
    print("\n" + "="*130)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*130)
    
    # Overall summary
    acc = (correct / total * 100) if total else 0.0
    print(f"\nOVERALL RESULTS")
    print(f"Total samples: {total} | Correct (TP): {correct} | Incorrect (FP): {total - correct} | Accuracy: {acc:.2f}%")
    
    # Generate table for current grouping
    if GROUPBY_FIELDS:
        print(f"\n\nDETAILED BREAKDOWN BY {GROUPBY_FIELDS}:")
        print(generate_analysis_table(results, GROUPBY_FIELDS))
        
        print(f"\n\nPER-CLASS ANALYSIS BY {GROUPBY_FIELDS}:")
        print(generate_class_analysis(results, GROUPBY_FIELDS))
    else:
        print("\n\nDETAILED BREAKDOWN (Overall):")
        print(generate_analysis_table(results, []))
        
        print("\n\nPER-CLASS ANALYSIS (Overall):")
        print(generate_class_analysis(results, []))
    
    # Confusion-based metrics (TP/FP/TN/FN etc.)
    confusions = compute_confusion_from_results(results, GROUPBY_FIELDS, set(shapenet_imagenet_map.keys()))
    print("\n\nCONFUSION-BASED METRICS BY GROUP:")
    print_confusion_metrics_by_group(confusions)
    # save confusion summary
    conf_path = Path("confusion_matrices.json")
    conf_path.write_text(json.dumps(confusions, indent=2), encoding="utf-8")
    print(f"Saved confusion matrices to {conf_path.resolve()}")
    
    # Generate analysis for each individual metadata field
    for field in ALL_METADATA_FIELDS:
        san = field.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        print(f"\n\nGENERATING ANALYSIS FOR FIELD: {field}")
        # tables
        print("\n" + "="*100)
        print(f"ANALYSIS BY {field}")
        print("="*100)
        print(generate_analysis_table(results, [field]))
        print("\n")
        print(generate_class_analysis(results, [field]))

        # confusion
        conf = compute_confusion_from_results(results, [field], set(shapenet_imagenet_map.keys()))
        print_confusion_metrics_by_group(conf)

        # save artifacts
        out_conf = Path(f"confusion_matrices_{san}.json")
        out_conf.write_text(json.dumps(conf, indent=2), encoding="utf-8")
        out_txt = Path(f"analysis_{san}.txt")
        out_txt.write_text(generate_analysis_table(results, [field]) + "\n\n" + generate_class_analysis(results, [field]), encoding="utf-8")
        print(f"Saved per-field analysis to {out_txt.resolve()} and {out_conf.resolve()}")
    print("\n" + "="*130)
    print(f"Results saved to: {out_path.resolve()}")
    print("="*130)
    

if __name__ == "__main__":
    main()