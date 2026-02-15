activate env in Powershell: 
- Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
- . .\\.venv\Scripts\Activate.ps1

run Foundation-Eval-3.py
- Resnet:
    python .\Foundation-Eval-3.py --metadata 'C:\Users\Stud\Documents\Unreal Projects\FinalPluginTest\Dataset\Metadata.csv' --image-root 'C:\Users\Stud\Documents\Unreal Projects\FinalPluginTest\\' --model resnet50.a1_in1k --mode logits --mapping .\ShapeNet-ImageNet1k-Mapping-Indexed.json --masks .\ShapeNet_eval_masks_extended.json --depth ancestor_mask --outdir eval_resnet50
- CLIP:
    python .\Foundation-Eval-3.py --metadata 'C:\Users\Stud\Documents\Unreal Projects\FinalPluginTest\Dataset\Metadata.csv' --image-root 'C:\Users\Stud\Documents\Unreal Projects\FinalPluginTest\\' --model vit_base_patch16_clip_224.metaclip_2pt5b --mode knn --mapping .\ShapeNet-ImageNet1k-Mapping-Indexed.json --masks .\ShapeNet_eval_masks_extended.json --knn-k 5 --outdir eval_clip
- DINO:
    python .\Foundation-Eval-3.py --metadata 'C:\Users\Stud\Documents\Unreal Projects\FinalPluginTest\Dataset\Metadata.csv' --image-root 'C:\Users\Stud\Documents\Unreal Projects\FinalPluginTest\\' --model vit_base_patch16_dinov3.lvd1689m --mode knn --mapping .\ShapeNet-ImageNet1k-Mapping-Indexed.json --masks .\ShapeNet_eval_masks_extended.json --knn-k 5 --outdir eval_dino_knn