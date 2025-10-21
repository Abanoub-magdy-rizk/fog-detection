 🧠 FoG Detection & Prediction

Freezing of Gait (**FoG**) is one of the most disabling motor symptoms in patients with **Parkinson’s disease (PD)**. It causes a sudden and temporary inability to move forward while walking, which increases the risk of **falls, injuries**, and reduced mobility.

This project aims to **detect** and **predict** FoG episodes using wearable sensor signals (e.g., accelerometers). By applying signal processing, feature engineering, and machine learning / deep learning techniques, we classify gait into:

- ✅ **Normal Gait**
- ✅ **Pre-FoG** 
- ✅ **FoG Event**

The long-term goal is to support real-time systems for **early intervention** and fall prevention.

---

## 📌 Objectives

| Goal | Description |
|---------|------------|
| Preprocessing | Filter and normalize raw accelerometer signals |
| Windowing | Segment data into sliding windows with labels |
| Feature Extraction | Time, frequency, and time-frequency domain |
| Modeling | ML/DL for detection and prediction |
| Evaluation | F1-Score, Sensitivity, Specificity, PR-Curve |
| Deployment-Ready | Support for real-time inference |

---

## 📡 Dataset
This project uses the **Daphnet Freezing of Gait Dataset**, collected from Parkinson’s disease subjects using wearable sensors during walking tasks.

> Note: Dataset is not included in the repository due to size — instructions for download will be provided in `data/raw/`.

---

## 📂 Project Structure (Planned)

FoG_Prediction_Project/
├─ data/                      # Raw and processed dataset files
│  ├─ raw/                    # Original Daphnet dataset (not included)
│  └─ processed/              # Cleaned and preprocessed data
│
├─ scripts/                   # Main project pipeline
│  ├─ 01_preprocess.py        # Filtering, normalization, and signal cleaning
│  ├─ 02_windowing.py         # Sliding-window segmentation and labeling
│  ├─ 03_features.py          # Feature extraction (time/freq/time-frequency)
│  ├─ 04_images_generator.py  # Generate RP/STFT/PWVD images
│  ├─ 05_train_models.py      # ML/DL model training
│  └─ 06_ensemble_eval.py     # Final evaluation and ensemble learning
│
├─ models/                    # Saved machine/deep learning trained models
│
├─ results/                   # Plots, metrics, confusion matrix, PR-curves
│
├─ requirements.txt           # Python dependencies
├─ README.md                  # Project documentation
└─ .gitignore                 # Files/folders excluded from git tracking
