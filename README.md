# 👁️ RetinaScan AI — Diabetic Retinopathy Detection



> Deep learning-powered web app for automated detection and grading of **Diabetic Retinopathy** from retinal fundus photographs.

---

## 🩺 What is Diabetic Retinopathy?

Diabetic Retinopathy (DR) is a diabetes complication that damages blood vessels in the retina. It is the **leading cause of preventable blindness** worldwide. Early detection through retinal screening is critical — but specialist availability is limited in many regions.

This project uses a fine-tuned **MobileNetV2** CNN to automate DR grading, potentially assisting clinicians in high-volume screening.

---
## Deeployed on Render
 https://retinaguard.onrender.com/
## Dashboard
<img width="1893" height="855" alt="image" src="https://github.com/user-attachments/assets/fb1ad8ac-7646-4c54-8a9b-e6dcf8d3cf92" />


<img width="1574" height="627" alt="image" src="https://github.com/user-attachments/assets/5efd07a4-a7b3-4862-b103-8b35bf9aee18" />
<img width="1498" height="683" alt="image" src="https://github.com/user-attachments/assets/7a9fb53a-7ac2-4651-b0e2-9754b9fdc4f3" />
<img width="1552" height="692" alt="image" src="https://github.com/user-attachments/assets/f50f14e2-eb60-429e-aa2c-2b5ca4299835" />


## 🎯 Features

- **5-class DR severity classification** (No DR → Proliferative DR)
- **Grad-CAM heatmaps** showing which retinal regions the model focuses on
- **Confidence scores** for all classes
- **Clinical advice** based on predicted severity
- **Interactive Streamlit web UI** — no coding needed to use

---

## 🏷️ Classes

| Class | Severity | Description |
|-------|----------|-------------|
| 🟢 No_DR | None | No signs of diabetic retinopathy |
| 🟡 Mild | Mild | Microaneurysms only |
| 🟠 Moderate | Moderate | More than microaneurysms, less than severe |
| 🔴 Severe | Severe | Extensive abnormalities, high risk of progression |
| 🚨 Proliferate_DR | Proliferative | Most advanced — neovascularisation present |

---

## 🧠 Model Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 (ImageNet pretrained, last 40 layers fine-tuned)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(512, relu) → Dropout(0.5)
    ↓
Dense(256, relu) → Dropout(0.3)
    ↓
Dense(5, softmax)
```
| Parameter | Value |
|-----------|-------|
| Base model | MobileNetV2 (ImageNet) |
| Input size | 224 × 224 |
| Optimizer | Adam |
| Phase 1 LR | 1e-3 (head only) |
| Phase 2 LR | 2e-5 (fine-tune last 40 layers) |
| Loss | Categorical Crossentropy |
| Class weighting | Balanced (capped at 2.5×) |
| Dataset | APTOS 2019 (3,662 images) |
| Train/Val split | 80/20 stratified |

### Training Details

| Version | Model | Key Changes | Val Accuracy | Val AUC | Status |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **v1** | EfficientNetB0 (original) | Frozen base, ImageDataGenerator | 5–27% | unstable | ❌ Broken (val shuffling bug) |
| **v2** | EfficientNetB0 | Unfroze top 30 layers | 49.38% every epoch | 0.64 | ❌ Majority class collapse |
| **v3** | MobileNetV2 | Fixed generators, class weights | 49.38% stuck | 0.68 | ❌ Still collapsed |
| **v4** | MobileNetV2 | tf.data pipeline, stratified split | 58.39% | 0.85 | ⚠️ Unstable |
| **v5** | MobileNetV2 | Fixed preprocessing bug ([-1,1] scaling) | 72.7% | 0.946 | ✅ Best so far |
---


---
## Visualizations

<img width="1002" height="496" alt="image" src="https://github.com/user-attachments/assets/1839de21-d7df-4ec9-82fd-324af2981dbd" />
<img width="603" height="474" alt="image" src="https://github.com/user-attachments/assets/e7705cfa-dc82-49b7-a6da-a731ec408d8c" />


## 📁 Project Structure

```
retinascan-ai/
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── retina_best_v4.keras    # Trained model (add manually)
└── notebooks/
    └── training.ipynb      # Model training notebook (optional)
```

---

## 📊 Dataset

**APTOS 2019 Blindness Detection** (Kaggle)

- 3,662 labelled retinal fundus images
- 5 severity classes (imbalanced)
- Download: https://www.kaggle.com/competitions/aptos2019-blindness-detection

Class distribution:
```
No_DR:          1805 images (49.3%)
Moderate:        999 images (27.3%)
Mild:            370 images (10.1%)
Proliferate_DR:  295 images ( 8.1%)
Severe:          193 images ( 5.3%)
```

---

## 🔍 Grad-CAM Explainability

This app uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualise which regions of the retina the model uses to make its prediction.

- 🔴 **Red/warm areas** = high importance regions
- 🔵 **Blue/cool areas** = low importance regions

This makes the model's decision **transparent and interpretable** for clinical review.

---

## ⚠️ Disclaimer

This tool is intended for **research and educational purposes only**.
It should **not replace** professional medical diagnosis or clinical judgement.
Always consult a qualified ophthalmologist for medical advice.

---


## 🙏 Acknowledgements

- (https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data?resource=download)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- TensorFlow / Keras team
- Streamlit team
