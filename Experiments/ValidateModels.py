import torch
from ModelEvaluator import ModelEvaluator#
import json
import os

def main():

    # Configuration
    CSV_FILE = r"C:\Users\Stud\Documents\Unreal Projects\FinalPluginTest\Dataset\Metadata.csv"
    ROOT_DIR = r"C:\Users\Stud\Documents\Unreal Projects\FinalPluginTest" 
    BATCH_SIZE = 128

    cnn_models = [
        "resnet50",
        "resnet101",
        #"densenet201.tv_in1k",
        #"convnext_base.fb_in1k",
        #"convnext_large_mlp",
    ]
        
    vit_models = [
        #"vit_base_patch16_224",
        #"vit_large_patch16_224",
        #"swin_base_patch4_window7_224",
    ]

    # Gutbier Old custom : imageNet classes
    class_map = {
        2958343: 468,
        2691156: 895,
        404: 404,
        3948459: 763,
        3467517: 546
    }

    # custom : imageNet classes (auto-built from mapping file)
    #mapping_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Evaluation", "ShapeNet-ImageNet1k-Mapping-Indexed-subcategories4.json"))
    #with open(mapping_path, "r", encoding="utf-8") as f:
    #    mapping_data = json.load(f)

    #class_map = {}
    #for _, info in mapping_data.items():
    #    synset = info.get("shapenet_synset")
    #    indices = info.get("imagenet_class_indices", [])
    #    if synset and indices:
    #        for idx in indices:
    #            class_map[synset] = idx
    #print(f"Constructed class map from mapping file: {class_map}")


    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("Warning: CUDA not available, using CPU. This will be slow.")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(device=device, cnn_models=cnn_models, vit_models=vit_models, class_map=class_map)
    
    # Run evaluation
    evaluator.run_evaluation(CSV_FILE, ROOT_DIR, BATCH_SIZE)


if __name__ == "__main__":
    main()

        