# 🦋 Assignment Project: Butterfly Image Classification with MobileNetV2

This repository documents the implementation of a **deep learning pipeline for butterfly species classification**, completed as part of an AI course assignment. The assignment required the following workflow:

1. **Choose a Topic** → I selected *Butterfly Image Classification* within computer vision.
2. **Review Literature** → Read and summarized **10 research papers** related to image classification, transfer learning, and accuracy improvement techniques.
3. **Dataset Selection** → Identified a butterfly species dataset from Kaggle.
4. **Baseline Code** → Chose an existing Kaggle notebook implementation.
5. **Reimplementation & Improvement** → Rewrote the code into a clean, modular pipeline with enhancements in data quality, augmentation, evaluation, and accuracy.

---

## 📌 Project Overview

The goal of this project is to classify butterfly images into multiple species using a **transfer learning** approach with **MobileNetV2**. The workflow follows two-stage training:

* **Stage 1:** Train a new classification head while keeping the MobileNetV2 backbone frozen.
* **Stage 2:** Unfreeze the top layers of the backbone and fine-tune with a reduced learning rate.

This staged approach allows the model to preserve pretrained ImageNet features while adapting to butterfly-specific patterns.

---

## ✨ Key Features

* **Dataset Handling**: Supports both folder-structured and CSV-structured datasets. Includes checks for corruption and duplicates (SHA-1 hashes). Class imbalance addressed with computed class weights.
* **Preprocessing & Augmentation**: Images resized to 224×224, normalized to \[0,1], cached for faster training. Augmentations include random flip, rotation, zoom, contrast, and translation.
* **Transfer Learning**: MobileNetV2 backbone pretrained on ImageNet with custom classification head: `GlobalAveragePooling2D → Dropout → Dense(softmax)`.
* **Training Strategy**: Two-stage pipeline with OOM-safe batch fallback, AdamW optimizer, label smoothing, and learning rate schedules.
* **Regularization**: Dropout, L2 weight decay, and early stopping ensure generalization.
* **Evaluation Tools**: Classification report, confusion matrix, hardest example mining, and automated metrics logging.
* **Performance Optimizations**: Mixed precision training, Colab GPU-ready, and reproducible seeds.

---

## 📊 Alignment with 8 Key Factors for ML/DL Accuracy

The improvements were guided by the **8 Key Factors for ML/DL Accuracy** provided in the assignment:

| Factor                                 | Implementation in Project                                                                                                  | Notes                                                       |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **1. Data Quality & Quantity**         | ✔ Removed duplicates and corrupt files<br>✔ Class weights for imbalance<br>✔ Consistent preprocessing (resize + normalize) | ✘ Limited to Kaggle dataset; no extra data collected        |
| **2. Feature Engineering**             | ✔ Automated normalization and scaling<br>✔ Dimensionality reduction via GAP                                                | ✘ No manual feature creation (CNN learns features directly) |
| **3. Model Selection & Architecture**  | ✔ MobileNetV2 backbone<br>✔ Shallow, regularized head design                                                               | ✘ Did not compare with other architectures due to time      |
| **4. Hyperparameter Tuning**           | ✔ AdamW optimizer<br>✔ Label smoothing<br>✔ Batch size fallback<br>✔ ReduceLROnPlateau                                     | ✘ No full hyperparameter sweep (compute-limited)            |
| **5. Data Augmentation**               | ✔ Applied augmentations (flip, rotation, zoom, etc.)<br>✔ Implemented MixUp (optional)                                     | ✘ Text/audio augmentation not applicable                    |
| **6. Regularization & Generalization** | ✔ Dropout (0.25)<br>✔ Weight Decay<br>✔ Early Stopping                                                                     | ✘ Ensemble methods not implemented                          |
| **7. Evaluation & Feedback Loop**      | ✔ Stratified validation split<br>✔ Confusion matrix & hardest examples<br>✔ JSON/CSV/PNG artifacts saved                   | ✘ No automated retraining loop                              |
| **8. Computational Resources**         | ✔ Mixed precision for GPU efficiency<br>✔ Early stopping for time savings                                                  | ✘ Did not use TPUs or multi-GPU training                    |


---

## 🚀 Getting Started

### Run in Google Colab

1. Upload your `kaggle.json` API key.
2. Run notebook cells sequentially.
3. Dataset auto-downloads and extracts.

### Local Setup

```bash
# Clone repository
git clone https://github.com/your-username/butterfly-image-classification-mobilenetv2.git
cd butterfly-image-classification-mobilenetv2

# Create environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧠 Model Architecture

* **Backbone:** MobileNetV2 (ImageNet pretrained)
* **Head:** GlobalAveragePooling2D → Dropout(0.25) → Dense(NUM\_CLASSES, softmax with L2 regularization)
* **Optimizer:** AdamW (lr=3e-4, wd=1e-4)
* **Loss Function:** Categorical Crossentropy (label smoothing=0.1)
* **Training:** Two stages (frozen backbone → fine-tuning top \~40%)

---

## 📈 Training Process

1. **Stage 1:** Train only the classification head with backbone frozen.
2. **Stage 2:** Unfreeze last 40% of MobileNetV2 layers, reduce learning rate, fine-tune.
3. **Reason for Two Stages:** Directly training the whole model can destroy pretrained features; staged training preserves general patterns while adapting to butterfly images.

---

## 📊 Evaluation Artifacts

All evaluation results are stored under `results/`:

* `metrics.json` — class-level + overall accuracy
* `confusion_matrix.png` — visual class confusions
* `hardest_examples_topN.json` — misclassified high-loss examples
* `test_predictions.csv` — predictions for unlabeled test set

---

## 🛠 Requirements

* Python 3.9+
* TensorFlow 2.15+
* scikit-learn
* pandas, numpy, matplotlib
* TensorFlow Probability (optional, for MixUp)

---

## 🗺 Future Improvements

* Add support for **EfficientNet** and other backbones
* Perform **hyperparameter sweeps** for LR, dropout, smoothing
* Enable **test-time augmentation (TTA)**
* Explore **ensembles** for improved robustness

---

## 📝 License

MIT License
