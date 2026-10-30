# Plant Identification Using Deep Learning and Image Processing

Species identification from a single leaf photograph on the [Leafsnap](https://www.kaggle.com/) dataset (185 tree species, 30,866 images: 23,147 lab + 7,719 field). Six deep-learning models are compared under one shared pipeline: same frozen train/validation/test split, same augmentation, same evaluation protocol.

> Major Project, B.E. CSE (AI & ML), Chandigarh University, October 2026.

## Models compared

| # | Notebook | Model | Framework |
|---|----------|-------|-----------|
| 1 | [`01_ResNet50_Leafsnap_Colab.ipynb`](notebooks/01_ResNet50_Leafsnap_Colab.ipynb) | ResNet-50 (classic baseline) | PyTorch / timm |
| 2 | [`02_InceptionResNetV2_Leafsnap_Colab.ipynb`](notebooks/02_InceptionResNetV2_Leafsnap_Colab.ipynb) | Inception-ResNet-v2 | PyTorch / timm |
| 3 | [`03_ViT_DINOv2_Leafsnap_Colab.ipynb`](notebooks/03_ViT_DINOv2_Leafsnap_Colab.ipynb) | ViT-B/14 + registers, DINOv2 pretrained (2-phase fine-tuning) | PyTorch / timm |
| 4 | [`04_TripletMetricLearning_Leafsnap_Colab.ipynb`](notebooks/04_TripletMetricLearning_Leafsnap_Colab.ipynb) | ResNet-50 embedding network, semi-hard triplet loss, species prototypes | PyTorch |
| 5 | [`05_ConvNeXtV2_Leafsnap.ipynb`](notebooks/05_ConvNeXtV2_Leafsnap.ipynb) | ConvNeXt V2 Tiny | PyTorch / timm |
| 6 | [`06_EfficientNetV2S_BotanicalFusion_Leafsnap.ipynb`](notebooks/06_EfficientNetV2S_BotanicalFusion_Leafsnap.ipynb) | EfficientNetV2-S with rembg background removal and 44 hand-crafted botanical features (shape, texture, colour, venation) fused with CNN features | TensorFlow |

## Image processing pipeline

1. Background removal with `rembg` (ONNX Runtime); the leaf is placed on a plain white canvas so models learn the leaf, not the surroundings.
2. Leakage-aware 70/15/15 split by specimen group (lab views of the same specimen stay in one split), saved to CSV and reused by every model.
3. Augmentation that preserves the whole leaf (flips, small affine, light colour jitter) and model-appropriate normalisation.
4. Evaluation: Top-1 / Top-5, macro & weighted F1, balanced accuracy, **lab vs. field** gap, per-class metrics, confusion pairs, Grad-CAM.

## Repository layout

```
notebooks/   one Colab notebook per model
docs/        final report (.docx), presentation (.pptx), project guide (.pdf)
requirements.txt
```

## Running

The notebooks are written for **Google Colab with a GPU** and save checkpoints/results to Google Drive.

- Notebooks 01-04 expect the dataset archive at `MyDrive/Major Project/archive.zip` and resume automatically after Colab disconnects.
- Notebooks 05-06 download Leafsnap through `kagglehub` (requires Kaggle access).
- Notebooks 01-04 share `splits/leafsnap_group_split_seed42.csv`; run any one first and the others will reuse the same split.

Results are not pre-filled in the notebooks 01-04; they come from the executed run. See the report in [`docs/`](docs) for the full methodology and comparison.

## Team

- Abhinab Jana (23BAI70305)
- Manjot Singh (23BAI70323)
- Paras Manak (23BAI70596)
- Ashmit Thakur (23BAI70503)
- Vashist Salar (23BAI70283)

## License

MIT, see [LICENSE](LICENSE). Dataset and pretrained weights are subject to their own licenses.
