# Project Report — Brain Tumor MRI Classification (summary)

This short report summarizes the current pipeline (from the notebook), dataset, models, and recommended next steps for improving performance and reproducibility.

## Dataset & statistics (as discovered in notebook)
- Dataset: `masoudnickparvar/brain-tumor-mri-dataset` (Kaggle)
- Classes: `glioma`, `meningioma`, `notumor`, `pituitary`
- Example counts reported in the notebook run (training/testing split present):
  - Training: 1400 images per class (example)
  - Testing: 400 images per class (example)
- Default train/val/test split in notebook: 80% / 10% / 10% (stratified)

## Models evaluated (notebook)
- CNNBaseline: 4 conv-block network (from-scratch) — establishes baseline lower bound
- ResNet50 (transfer learning) — final fc replaced to match class count
- EfficientNet‑B0 (transfer learning)
- ViT (experimental) via torchvision's `vit_b_16`

## Training setup (defaults)
- Image size: 224×224
- Batch size: 32
- Epochs: default 5 (for quick tests); recommended 20–50 for real runs
- Optimizer: Adam (default learning rate 3e-4)
- Weight decay: 1e-4
- Label smoothing: 0.05
- Early stopping patience: configurable (default 6)
- Mixed precision: enabled only on CUDA

## Metrics & evaluation
- Evaluation uses accuracy, precision, recall, F1, confusion matrix (notebook calculates these)
- For clinical tasks, per-class precision/recall and confusion matrix inspection are crucial to detect systematic class confusions (e.g., tumor vs no-tumor)

## Explainability
- Grad‑CAM and Captum saliency maps implemented in notebook, with model.target_layer property where relevant.
- Visualization helps inspect whether model attends to tumor regions vs spurious cues (image borders, artifacts).

## Observations & recommended next steps
1. Dataset hygiene
   - Verify image resolution and remove near-duplicate or corrupted images.
   - Consider additional preprocessing: CLAHE / contrast normalization for MRI scans.
2. Data augmentation & balancing
   - Add more robust augmentations (random crops, elastic transforms, intensity augmentations).
   - If class imbalance appears, consider oversampling, weighted sampling, or focal loss.
3. Model improvements
   - Fine-tune EfficientNet/ResNet with a lower LR and longer schedules.
   - Experiment with larger backbones or ensembles.
4. Evaluation improvements
   - Add cross‑validation (k-fold) to better estimate generalization.
   - Save per-epoch metrics and confusion matrices for later analysis.
5. Reproducibility & MLOps
   - Extract notebook into a package (train.py / eval.py) with YAML/argparse configs.
   - Add experiment tracking (W&B / TensorBoard) and save checkpoints with metadata.
   - Provide a small minimal test subset and unit tests for the dataset loader.

## Future Work Scope

- **Multimodal fusion** — combine imaging features with structured clinical metadata (age, tumor location, prior history) where available, moving toward the multimodal learning direction relevant to broader medical AI research.
- **Segmentation extension** — extend from classification to full BraTS-style tumor segmentation (necrotic core / edema / enhancing tumor) using a U-Net or nnU-Net backbone, with Grad-CAM-informed region proposals as a bridge.
- **Uncertainty quantification** — add Monte Carlo Dropout or deep ensembles to produce calibrated confidence estimates, critical for any clinical decision-support framing.
- **Self-supervised pretraining** — pretrain on unlabeled MRI slices (SimCLR / MAE) before fine-tuning, to test whether domain-specific pretraining beats ImageNet transfer on this task.
- **External validation** — evaluate the best checkpoint on a fully held-out dataset (e.g. a different institution's scans) to test generalization beyond this dataset's specific acquisition protocol.
- **Clinical-grade explainability evaluation** — quantitatively score Grad-CAM region overlap against radiologist-annotated tumor masks (where available), rather than relying on qualitative visual inspection alone.
