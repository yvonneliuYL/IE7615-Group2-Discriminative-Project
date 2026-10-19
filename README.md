# IE 7615 Discriminative Deep Learning Project — Group 2

**Members:** Yi-Fan Liu, Pooja Yogesh Patil, Hiral Hitesh Rana, Wen-Hsin Su

Each student in the class photographed one unique object under different lighting conditions, angles and backgrounds, which gives 73 Object IDs (OBJ001 to OBJ073). We used the final merged dataset and the final Object IDs published by the instructors. Each object has its own folder, and the folder name is the class label.

The project has three milestones, each building on the one before:

| Milestone | Task | Final model | Main test result |
|---|---|---|---|
| [1](#milestone-1-single-object-image-id) | **Identify** the object in a single-object photo (73-class classification) | EfficientNetB0 (transfer learning) | 97.4% accuracy, macro F1 0.974 |
| [2](#milestone-2-multi-object-detection-and-location) | **Identify and locate** every object in a multi-object image | YOLOv8s (transfer learning) | see Milestone 3 |
| [3](#milestone-3-final-model) | **Final model**: evaluation, error analysis and live demo | **YOLOv8s** | **mAP@0.5 0.912, mAP@0.5:0.95 0.856** |

**Our final model is YOLOv8s.** EfficientNetB0 is very accurate, but a classifier gives only one label per image and cannot say where an object is. YOLOv8s handles both demo tasks with one model: it identifies a single-object photo, and it identifies and locates every object in a multi-object image.

---

## Milestone 1: Single-object image ID

73-class image classification (OBJ001–OBJ073). Background-only shots (files marked `_bg_`) contain no object, so they were excluded. This leaves 6,964 object images, with 90 to 106 images per class (mean 95.4). The classes are close to balanced, so no class weighting was needed. All images were resized to 224 × 224 pixels.

The data was split 70 / 15 / 15 into training (4,874 images), validation (1,045) and test (1,045) sets. The split is stratified by class with a fixed seed (42), so every object appears in all three sets in the same proportions, and every model was trained and evaluated on the same images. The test set was used only for final reporting, never for training or model selection.

Training images were augmented on the fly with random horizontal flips, rotation (±8%), zoom (±15%), translation (±10%), brightness (±20%) and contrast (±20%). Augmentation is active only during training.

All five models are built, trained and evaluated in one notebook, [`milestone1/Week1_CNN_Comparison.ipynb`](milestone1/Week1_CNN_Comparison.ipynb). The saved outputs show every training run.

| Model | Type | Val acc | Test acc | Top-5 | Macro F1 | Params (M) |
|---|---|---|---|---|---|---|
| **EfficientNetB0** (selected) | Transfer learning, fine-tuned from `block6a` | 98.2% | **97.4%** | 99.7% | 0.974 | 4.14 |
| ResNet50 | Transfer learning, fine-tuned `conv5` | 97.5% | 97.0% | 99.6% | 0.970 | 23.74 |
| MobileNetV2 | Transfer learning, fine-tuned from `block_13` | 96.2% | 96.2% | 99.3% | 0.962 | 2.35 |
| VGG16 | Transfer learning, fine-tuned `block5` | 95.5% | 95.0% | 98.7% | 0.950 | 14.75 |
| Custom CNN | From scratch (5 × Conv-BN-ReLU-MaxPool) | 77.0% | 78.5% | 92.6% | 0.785 | 1.61 |

The final model was chosen by validation accuracy; the test set was used only for reporting.

**What we learned:** with only ~67 training photos per class, transfer learning from ImageNet clearly beats training from scratch. This idea carries over to the detector in Milestone 2.

---

## Milestone 2: Multi-object detection and location

Given one image that contains many objects, the model must report **every object's ID and its bounding box**. The dataset only has class labels, so the pipeline has four steps.

**1. Clean the single photos.** Every photo is converted to 224 × 224 RGB with its EXIF rotation applied. OBJ054_017 is excluded (instructor note), and exact duplicate photos are removed **before** the split (17 photos removed, 6,947 left). The split is 70 / 15 / 15, stratified, seed 42, and a check confirms that no photo (or a copy of it) is in more than one split.

**2. Create a bounding box for every photo.** Two open-vocabulary detectors, **YOLO-World v2** and **Grounding DINO (tiny)**, are prompted with each object's name from the class sheet, and they check each other:

| Status | Rule | Photos |
|---|---|---|
| Models agree | top boxes overlap with IoU ≥ 0.5 (or a 2nd/3rd candidate matches) | 5,280 (76%) |
| One model only | no agreement, but one model is confident | 1,188 |
| Rescued | kept so that no Object ID loses more than 20% of its photos | 130 |
| Left out | neither model found the object reliably | 349 |

Where the models agree, their boxes overlap with a mean IoU of 0.944. In total, 6,598 photos have a box. Label-check figures (random sample, disagreements, and our group's objects) are saved for manual review.

**3. Build multi-object grid images.** Photos are placed side by side in grids, following the instructors' sizes:

| Grid | Objects | Image size | Train / val / test images |
|---|---|---|---|
| 2×2 | 4 | 448 × 448 | 240 / 30 / 30 |
| 3×3 | 9 | 672 × 672 | 240 / 30 / 30 |
| 4×4 | 16 | 896 × 896 | 240 / 30 / 30 |
| 5×4 | 20 | 1120 × 896 | 240 / 30 / 30 |
| 5×5 | 25 | 1120 × 1120 | 240 / 30 / 30 |

That gives 1,200 train / 150 val / 150 test grids (17,760 / 2,220 / 2,220 objects). Train grids use only train photos, val only val photos, and test only test photos. Train tiles are randomly flipped and get random brightness, contrast and colour changes (±25%).

**4. Fine-tune YOLOv8s.** YOLOv8s pretrained on COCO (11.2 M parameters), with a new head for 73 Object IDs: `imgsz=1120`, `batch=8`, up to 60 epochs with early stopping (patience 15). The best epoch is chosen on the validation grids (val mAP@0.5 = 0.921). Training took 56.6 min on a T4.

---

## Milestone 3: Final model

The YOLOv8s model is evaluated once on the 150 held-out test grids (2,220 objects).

| Test set | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| **All test grids** | **0.919** | **0.883** | **0.912** | **0.856** |
| 2×2 | 0.885 | 0.861 | 0.954 | 0.902 |
| 3×3 | 0.895 | 0.872 | 0.927 | 0.881 |
| 4×4 | 0.919 | 0.882 | 0.923 | 0.875 |
| 5×4 | 0.919 | 0.862 | 0.908 | 0.855 |
| 5×5 | 0.912 | 0.890 | 0.911 | 0.855 |

**At confidence 0.5**, an object counts as correct when its ID is right and its box overlaps the true box with IoU ≥ 0.5. On that basis, 1,958 of 2,220 objects are correct (precision 0.926, recall 0.882), and 25 of the 150 grids are completely correct.

**Error analysis.** The weakest Object IDs (AP@0.5) are OBJ040, OBJ010, OBJ065, OBJ068 and OBJ051 (0.63–0.70). Most errors are **missed objects** (262) rather than wrong IDs. Confusions between IDs are rare, and no pair occurs more than twice (e.g. OBJ003 → OBJ012, OBJ060 → OBJ014). Performance stays stable from 2×2 to 5×5 grids.

**Live demo.** The last cell of the final notebook loads the trained weights and runs on uploaded images:
- **(a)** a single-object photo (≤ 336 px) is run at `imgsz=448` and returns its Object ID;
- **(b)** a multi-object image is run at `imgsz=1120` and returns every object's ID, confidence, box (x1, y1, x2, y2) and grid row / column.

The results are also saved as a CSV.

---

## Repository contents

```
milestone1/
├── Week1_CNN_Comparison.ipynb      # data pipeline, all 5 models, training, evaluation, error analysis
└── results/
    ├── model_comparison.csv            # metrics for all 5 models
    ├── model_comparison_cpu_latency.csv
    ├── per_class_report_best.csv       # precision / recall / F1 per class (EfficientNetB0)
    ├── class_names.json, split.csv     # class order and the exact train / val / test split
    ├── figures/                        # accuracy comparison, learning curves, confusion matrix, errors
    └── <model>/                        # metrics.json, history.json (per-epoch curves), summary.txt (architecture)
milestone2/
└── Week2_MultiObject_v2.ipynb      # Milestone 2 detection notebook
milestone3/
└── Week3_finalmodel.ipynb          # final YOLOv8s: labelling, grids, training, test results, error analysis, live demo
```

## How to run

### Milestone 1
1. Put the class dataset as `database.zip` in your Google Drive (one folder per object: `database/images_OBJxxx/`). The dataset is not included here because it is the class's shared photo collection.
2. Open the notebook in Google Colab and select **Runtime → Change runtime type → T4 GPU**.
3. In the first code cell, set `DRIVE_DIR` to the Drive folder that contains `database.zip`.
4. **Runtime → Run all.** Training all five models takes about 75 minutes on a T4. Models and results are written to `DRIVE_DIR/week1_results/`.

### Milestones 2 and 3 (final model)
1. Put `database.zip` (or the unzipped `images_OBJxxx` folders) anywhere in My Drive, up to 3 folders deep. The notebook finds it automatically, or you can set `DATASET_SOURCE` in the settings cell.
2. Open `milestone3/Week3_finalmodel.ipynb` in Colab, select **T4 GPU**, then **Runtime → Run all**.
3. Results go to `My Drive/milestone2_final/`. If Colab disconnects, run all again: box labels are reloaded from Drive and training resumes from the last epoch.
4. A full run takes about 2 hours (labelling ~42 min, training ~57 min). With saved labels and a finished training run, it takes about 10–15 min.
5. **Demo only:** run the last cell. It loads `milestone2_final/runs/final_yolov8s/weights/best.pt` and asks for images to upload.

Built on Google Colab with TensorFlow 2.20 / Keras 3 (Milestone 1) and PyTorch with Ultralytics 8.4 and Hugging Face Transformers (Milestones 2–3).

**Notes**
- Trained model files are not in the repository because of GitHub's 100 MB file limit. Running the notebooks recreates them.
- Nothing is written to the shared class dataset folder. All outputs go to each user's own Drive.
- The latency column in `model_comparison.csv` was measured with single-image calls on the GPU, which mostly reflects layer count rather than compute. `model_comparison_cpu_latency.csv` adds a single-image CPU latency measured on a laptop (Intel i5), which is a fairer speed comparison.