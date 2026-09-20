# Potato Leaf Disease Classification using Transfer Learning & Fine-Tuning

An end-to-end Computer Vision project implementing two-stage Transfer Learning with **MobileNetV2** in TensorFlow/Keras to diagnose potato leaf diseases from the PlantVillage dataset.

![Sample Predictions](assets/sample_predictions.png)

---

## Key Highlights & Engineering Decisions

1. **Lightweight & Efficient Backbone:** Selected MobileNetV2 pretrained on ImageNet ($2.26\text{M}$ total parameters), ideal for edge deployment and constrained environments.
2. **Two-Stage Training Strategy:**
   * **Stage 1 (Feature Extraction):** Frozen base model weights; trained only the top classification head ($3.8\text{K}$ parameters) with Adam ($\text{lr} = 10^{-3}$) to prevent catastrophic forgetting.
   * **Stage 2 (Fine-Tuning):** Unfroze deeper representation layers (from layer 100 onwards) and trained with a reduced learning rate ($\text{lr} = 10^{-4}$) to adapt high-level feature representations to disease morphology.
3. **Robust Evaluation on Hold-Out Test Set:** Completely isolated test set ($10\%$ of data) untouched during training and model selection.
4. **Addressing Class Imbalance:** Evaluated via Precision, Recall, and F1-Score (Macro & Weighted) rather than relying solely on Accuracy, ensuring strong performance on the minority class (`Potato___healthy`).

---

## Dataset Overview

* **Source:** PlantVillage (Potato subset)
* **Classes:**
  * `Potato___Early_blight` (Alternaria solani)
  * `Potato___Late_blight` (Phytophthora infestans)
  * `Potato___healthy`
* **Data Splits:** $80\%$ Train, $10\%$ Validation, $10\%$ Test.
* **Input Resolution:** $224 \times 224 \times 3$

---

## Training Dynamics

The plot below illustrates the training and validation trajectories across both stages:

![Learning Curves](assets/learning_curves.png)

* The red dashed line denotes the onset of Fine-Tuning (Epoch 10). Notice the immediate drop in loss and convergence towards higher generalization accuracy.

---

## Quantitative Results & Error Analysis

### Performance on Hold-Out Test Set:
* **Test Accuracy:** ~98%
* **Test Loss:** < 0.08

### Confusion Matrix:
![Confusion Matrix](assets/confusion_matrix.png)

---

## Quickstart

### 1. Clone repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/potato-disease-transfer-learning.git
cd potato-disease-transfer-learning
