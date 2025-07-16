# Skin Cancer Detection via Ensemble Deep Learning

This project introduces a **research-oriented deep learning pipeline** designed for the automated classification of skin lesions.  
Rather than relying on conventional single-stage classifiers, it explores a **multi-stage ensemble approach** that mirrors a structured, diagnostic reasoning process similar to how a clinician might evaluate a lesion.  
The system integrates pretrained convolutional neural networks (CNNs) and weighted ensemble strategies to differentiate between melanoma, nevus, benign vs malignant lesions, and their respective subtypes.

This work is part of an ongoing research effort in medical image analysis and aims to support the early and reliable detection of skin cancer through AI-assisted diagnostic systems.

---

## Project Goals

This project is designed as a **research-driven initiative** to explore novel strategies in medical image classification — particularly for skin cancer detection. While many existing solutions focus on isolated models or single-stage pipelines, this project takes a **multi-stage ensemble-based approach** with the goal of mimicking a more human-like, hierarchical diagnostic process.

We aim to test whether breaking the classification into meaningful stages (melanoma → nevus → benign/malignant → subtype) can lead to better generalization, explainability, and robustness, especially when dealing with real-world noisy data and class imbalance.

Key objectives:

- Build a reliable classification system for common skin cancer types using a structured, step-wise decision pipeline.
- Improve classification accuracy by combining multiple deep learning models via weighted ensemble strategies.
- Offer a modular, extensible framework for medical imaging researchers who want to experiment with various model combinations and decision flows.
- Provide a reproducible platform for evaluating model performance across ISIC datasets (2018–2020).
- Encourage community collaboration for model benchmarking, validation on new datasets, and shared improvements.

> This is not just a software tool — it is a research playground designed to test **alternative ensemble and decision-routing strategies** in medical AI.

---

## Multi-Stage Classification Pipeline

The ensemble pipeline performs **five sequential classification tasks**:

1. **Melanoma Classification**  
   → Binary classification to detect melanoma.

2. **Nevus Classification**  
   → Binary classification to detect nevus when melanoma is not detected.

3. **Benign vs Malignant Classification**  
   → Binary classification for remaining undecided cases.

4. **Malignant Subtype Classification**  
   → Multi-class classification between `akiec` and `bcc`.

5. **Benign Subtype Classification**  
   → Multi-class classification between `bkl`, `df`, and `vasc`.

Each stage uses an ensemble of three deep learning models (Xception, DenseNet121, CNN) with weighted soft voting.

> The entire pipeline has been developed and tested on **Google Colab**, utilizing Google Drive for model and data storage.

---

## Project Structure

```bash
SKIN-CANCER-DETECTION/
│
├── 0_melanoma_classification/         # Melanoma training & ensemble notebooks
│   └── melanoma_ensemble.ipynb
│
├── 1_nevus_classification/            # Nevus classification notebooks
│   └── nevus_ensemble.ipynb
│
├── 2_binary_classification/           # Benign vs Malignant classification
│   └── binary_ensemble.ipynb
│
├── 3_benign_classification/           # Benign subtype classification
│   └── benign_ensemble.ipynb
│
├── 4_malignant_classification/        # Malignant subtype classification
│   └── malignant_ensemble.ipynb
│
├── data/                              # Datasets and processed classification folders
│   ├── ISIC_2018/
│   ├── ISIC_2019/
│   ├── ISIC_2020/
│   ├── all_classification/
│   ├── benign_classification/
│   ├── binary_classification/
│   ├── malignant_classification/
│   ├── mel_classification/
│   ├── nev_classification/
│   └── README.md                      # Dataset instructions
│
├── models/                            # Saved Keras model files
│   ├── benign_models/
│   ├── binary_models/
│   ├── malignant_models/
│   ├── melanoma_models/
│   ├── nevus_models/
│   └── README.md                      # Model structure explanation
│
├── scripts/                           # Helper scripts and main pipeline
│   ├── copy_files.py                  # File mover by class or split
│   ├── resize_images.py               # Resize utility for datasets
│
├── data_exploration.ipynb             # Dataset stats & cleaning
├── ensemble_pipeline.py               # Final ensemble classification pipeline
├── ensemble_validation.ipynb          # Evaluation & reports on test data
│
├── .gitignore
└── README.md                          # This file
```

## Technologies Used

- Python 3.10+
- TensorFlow / Keras
- NumPy, Pandas
- Scikit-learn
- Google Colab & Google Drive
- ISIC Dataset (2018, 2019, 2020)

---

## Model Ensemble Strategy

Each classification stage uses a soft-voting ensemble with fixed or grid-optimized weights:

**Model Backbones:**

- Xception
- DenseNet121
- Custom CNN

**Activation Functions:**

- Sigmoid for binary outputs
- Softmax for multi-class subtypes

---

## Evaluation Metrics

- Accuracy
- Precision & Recall
- F1-Score
- AUC (for binary tasks)
- Confusion Matrix

All evaluations are handled in `ensemble_validation.ipynb`.

---

## License

This project is released under the MIT License.

---

## Citation

If you use this project in a scientific publication, please consider citing it.  
BibTeX entry will be added upon publication.
