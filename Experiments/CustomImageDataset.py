import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, class_map=None):
        """
        Custom dataset for loading images with metadata from CSV.
        
        Args:
            csv_file: Path to CSV file with metadata
            root_dir: Root directory containing images
            transform: Optional transform to be applied on images
            class_map: Dictionary mapping custom classes to ImageNet indices
        """
        self.metadata = pd.read_csv(csv_file, delimiter=";")
        self.root_dir = root_dir
        self.transform = transform
        self.class_map = class_map or {}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.root_dir, row["Image"])
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # Get class and map to ImageNet if mapping provided
        raw_class = row["Class"]
        raw_class_str = str(raw_class).strip()
        raw_class_int = None
        if raw_class_str.isdigit():
            raw_class_int = int(raw_class_str)
        elif isinstance(raw_class, (int, float)):
            raw_class_int = int(raw_class)

        class_map_uses_strings = any(isinstance(k, str) for k in self.class_map.keys())
        if class_map_uses_strings:
            class_label = raw_class_str
        else:
            class_label = raw_class_int if raw_class_int is not None else raw_class_str

        imagenet_label = self.class_map.get(class_label, class_label)
        
        return {
            "image": image,
            "label": class_label,  # Original label
            "imagenet_label": imagenet_label,  # Mapped ImageNet label
            "idx": idx,
            # Store metadata as individual fields to avoid collation issues
            "image_path": row["Image"],
            "object": row.get("Object", ""),
            "level": row.get("World", row.get("Level", "")),
            "material": row.get("Material", ""),
            "camera_position": row.get("Camera Position", ""),
            "light_color": row.get("Light Color (RGB)", ""),
            "fog": row.get("Fog", ""),
        }
