# Data Directory

This directory contains all the image data and associated classification metadata used in the project.

Please organize the datasets into subfolders according to their classification purposes:

- `all_classification/`: Contains all images and their combined metadata.
- `mel_classification/`, `nev_classification/`: Used for binary classification tasks of melanoma and nevus detection.
- `binary_classification/`: For benign vs malignant classification.
- `malignant_classification/`: Subtype classification for malignant samples (e.g., AKIEC vs BCC).
- `benign_classification/`: Subtype classification for benign samples (e.g., BKL, DF, VASC).
- `ISIC_2018/`, `ISIC_2019/`, `ISIC_2020/`: Raw ISIC dataset folders.

> Make sure that each folder contains a `metadata.csv` file with at least the following columns: `path`, `diagnosis`, and any relevant label (e.g., `benign_malignant`).
