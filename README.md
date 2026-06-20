# Green AI — EuroSAT Land Use Classification with Lightweight CNNs

A comparative study of **energy-efficient deep learning models** for satellite image classification using the **EuroSAT RGB dataset**, with a focus on minimizing CO₂ emissions while maintaining high accuracy.

---

##  Project Overview

This project benchmarks three lightweight CNN architectures on the EuroSAT dataset to evaluate the trade-off between **accuracy, computational cost, and carbon footprint**:

| Model | FLOPs | Parameters | Accuracy (Mean ± SD) | CO₂ Emission |
|---|---|---|---|---|
| **EfficientNet-B0** | 0.211 GFLOPs | 4.02 M | 0.9531 ± 0.0011 | 0.038109 kg CO₂eq |
| **MobileNetV3-Large** | 0.121 GFLOPs | 4.21 M | 0.9630 ± 0.0001 | 0.077924 kg CO₂eq |
| **ShufflNetV2-x1.0** | 0.077 GFLOPs | 1.264 M | 0.8825 ± 0.0019 | 0.032030 kg CO₂eq |

---

##  Repository Structure

```
Green_AI/
│
├── EfficientNET-B0.ipynb       # EfficientNet-B0 training & evaluation
├── MobileNetV3 (1).ipynb       # MobileNetV3-Large training & evaluation
├── ShufflNetV2.ipynb           # ShuffleNetV2 training & evaluation
└── README.md
```

---

## Dataset — EuroSAT RGB

- **Source:** EuroSAT RGB (Sentinel-2 satellite imagery)
- **Total Images:** 27,000
- **Classes (10):**
  - AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial
  - Pasture, PermanentCrop, Residential, River, SeaLake
- **Image Resolution:** 3 × 160 × 160 (resized)
- **Normalization:** Mean `[0.485, 0.456, 0.406]` | Std `[0.229, 0.224, 0.225]` (ImageNet standard)

### Data Splits

| Notebook | Train | Validation | Test |
|---|---|---|---|
| EfficientNet-B0 | 70% (18,900) | 15% (4,050) | 15% (4,050) |
| MobileNetV3 | 70% (18,900) | 15% (4,050) | 15% (4,050) |
| ShuffleNetV2 | 80% (21,600) | 10% (2,700) | 10% (2,700) |

---

## Model Architectures

### 1. EfficientNet-B0
- **Scaling:** Compound scaling (Depth × Width × Resolution)
- **Compound Coefficients:** α=1.2, β=1.1, γ=1.15
- **Transfer Learning:** ImageNet-1K pretrained weights
- **Fine-tuning Strategy:** Partial — last block (`features.8`) unfrozen + new classifier head
- **Frozen Parameters:** 3,595,388 | **Trainable:** 424,970

### 2. MobileNetV3-Large
- **Architecture:** Inverted residuals with hard swish activation
- **Transfer Learning:** ImageNet-1K pretrained weights
- **Fine-tuning Strategy:** Partial — last two blocks (`features.15` & `features.16`) unfrozen + new classifier head

### 3. ShuffleNetV2-x1.0
- **Architecture:** Channel shuffle operations for efficient feature mixing
- **Transfer Learning:** ImageNet pretrained weights (DEFAULT)
- **Fine-tuning Strategy:** Feature extractor frozen; only classifier (`fc` layer) trainable

---

##  Training Configuration

| Hyperparameter | EfficientNet-B0 | MobileNetV3 | ShuffleNetV2 |
|---|---|---|---|
| **Epochs** | 10 | 10 | 10 |
| **Batch Size** | 64 | 32 | 64 |
| **Learning Rate** | 3e-4 | 3e-4 | 1e-3 |
| **Optimizer** | Adam | Adam | Adam |
| **Loss Function** | CrossEntropyLoss | CrossEntropyLoss | CrossEntropyLoss |
| **Precision** | FP32 | FP32 | FP32 |
| **Seeds** | [42, 123, 777] | [42, 123, 777] | [42, 123, 777] |
| **Hardware** | Tesla T4 GPU | Tesla T4 GPU | Tesla T4 GPU |

---

##  Results

### EfficientNet-B0 (Best Run — Seed 777)

```
Accuracy  : 0.9531 ± 0.0011
Precision : 0.9517 ± 0.0014
Recall    : 0.9517 ± 0.0013
F1 Score  : 0.9516 ± 0.0013
CO₂       : 0.038109 kg CO₂eq
```

### MobileNetV3-Large (Best Run — Seed 777)

```
Accuracy  : 0.9630 ± 0.0001
Precision : 0.9629 ± 0.0004
Recall    : 0.9617 ± 0.0005
F1 Score  : 0.9622 ± 0.0003
CO₂       : 0.077924 kg CO₂eq
```

### ShuffleNetV2-x1.0 (Best Run — Seed 777)

```
Accuracy  : 0.8825 ± 0.0019
Precision : 0.8823 ± 0.0016
Recall    : 0.8774 ± 0.0027
F1 Score  : 0.8785 ± 0.0021
CO₂       : 0.032030 kg CO₂eq
```

> All metrics are reported as **Mean ± Standard Deviation** across 3 independent seeds for reproducibility.

---

##  Green AI Philosophy

This project embraces the **Green AI** paradigm — evaluating not just model accuracy but also its **environmental cost**:

-  **FLOPs** — Floating point operations per forward pass
- **Parameters** — Total model size
- **CO₂ Emissions** — Measured using [CodeCarbon](https://github.com/mlco2/codecarbon)

**Key Finding:** ShuffleNetV2 offers the **lowest carbon footprint** (0.032 kg CO₂eq) with only ~1.26M parameters, making it the most eco-friendly option — though at a cost to accuracy. MobileNetV3 achieves the **highest accuracy** (96.3%) while maintaining a compact architecture.

---

##  Dependencies

```bash
pip install torch torchvision
pip install codecarbon
pip install thop
pip install scikit-learn matplotlib seaborn
```

---

##  How to Run

1. Mount Google Drive and set `DATA_DIR` to the path of your EuroSAT dataset
2. Open any notebook in Google Colab with a T4 GPU runtime
3. Run all cells sequentially

---

##  Dataset Download

EuroSAT RGB dataset can be downloaded from:
- [EuroSAT GitHub](https://github.com/phelber/EuroSAT)
- [Kaggle EuroSAT](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)

---

## 👤 Author

**SaKeT039**

---

## 📜 License

This project is for academic and research purposes.
