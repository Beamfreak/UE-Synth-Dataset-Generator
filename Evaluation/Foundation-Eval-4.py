import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def load_masks(path: str) -> dict[str, any]:
	with open(path, "r") as f:
		return json.load(f)


def load_mapping(path: str) -> dict[str, any]:
	with open(path, "r") as f:
		return json.load(f)


def load_image(path: str, transform):
	img = Image.open(path).convert("RGB")
	return transform(img)


# --------------------------------------------------
# Logits-based evaluation (ResNet etc.)
# --------------------------------------------------
def eval_logits(model, df, transform, masks, depth, image_root):
	model.eval()
	preds = []
	for _, row in tqdm(df.iterrows(), total=len(df)):
		path = os.path.join(image_root, row["Image"])
		img = load_image(path, transform).unsqueeze(0).to(DEVICE)
		with torch.no_grad():
			logits = model(img)
			pred = torch.argmax(logits, dim=1).item()
		cat = row["Class"]
		mask = masks[cat][depth]
		correct = mask[pred] == 1
		preds.append(correct)
	acc = np.mean(preds)
	print(f"Accuracy ({depth}): {acc:.4f}")


# --------------------------------------------------
# DINO kNN evaluation
# --------------------------------------------------
def eval_knn(model, df, transform, image_root, k=5):
	model.eval()
	features = []
	labels = []
	for _, row in tqdm(df.iterrows(), total=len(df)):
		path = os.path.join(image_root, row["Image"])
		img = load_image(path, transform).unsqueeze(0).to(DEVICE)
		with torch.no_grad():
			feat = model.forward_features(img)
			feat = feat.mean(dim=1)
		features.append(feat.cpu().numpy().squeeze())
		labels.append(row["Class"])
	features = np.array(features)
	labels = np.array(labels)
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(features, labels)
	preds = knn.predict(features)
	acc = np.mean(preds == labels)
	print(f"kNN Accuracy: {acc:.4f}")


# --------------------------------------------------
# CLIP zero-shot evaluation
# --------------------------------------------------
def eval_clip(model, preprocess, df, masks, depth, image_root):
	model.eval()

	# Build text prompts from mapping keys
	classes = list(masks.keys())
	prompts = [f"a photo of a {c}" for c in classes]
	tokenizer = open_clip.get_tokenizer("ViT-B-32")
	text = tokenizer(prompts).to(DEVICE)
	with torch.no_grad():
		text_features = model.encode_text(text)
		text_features /= text_features.norm(dim=-1, keepdim=True)

	correct_list = []
	for _, row in tqdm(df.iterrows(), total=len(df)):
		path = os.path.join(image_root, row["Image"])
		img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
		with torch.no_grad():
			image_features = model.encode_image(img)
			image_features /= image_features.norm(dim=-1, keepdim=True)
			logits = 100.0 * image_features @ text_features.T
			pred_idx = logits.argmax(dim=1).item()
		pred_class = classes[pred_idx]
		mask = masks[row["Class"]][depth]
		# check if predicted class maps to valid ImageNet index
		correct_list.append(pred_class == row["Class"])
	acc = np.mean(correct_list)
	print(f"CLIP Zero-shot Accuracy: {acc:.4f}")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--model", required=True)
	ap.add_argument("--mode", choices=["logits", "knn", "clip"], required=True)
	ap.add_argument("--metadata", required=True)
	ap.add_argument("--image-root", required=True)
	ap.add_argument("--mapping", required=True)
	ap.add_argument("--masks", required=True)
	ap.add_argument("--depth", default="ancestor_mask")
	args = ap.parse_args()

	df = pd.read_csv(args.metadata)
	masks = load_masks(args.masks)

	if args.mode == "logits":
		model = timm.create_model(args.model, pretrained=True).to(DEVICE)
		transform = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))
		eval_logits(model, df, transform, masks, args.depth, args.image_root)
	elif args.mode == "knn":
		model = timm.create_model(args.model, pretrained=True, num_classes=0).to(DEVICE)
		transform = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))
		eval_knn(model, df, transform, args.image_root)
	elif args.mode == "clip":
		model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
		model = model.to(DEVICE)
		eval_clip(model, preprocess, df, masks, args.depth, args.image_root)


if __name__ == "__main__":
	main()