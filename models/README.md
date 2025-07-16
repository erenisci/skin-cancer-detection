# Models Directory

This directory holds the trained Keras models used for ensemble predictions in the classification pipeline.

Organize your models into the following subfolders:

- `melanoma_models/`: Models trained to detect melanoma (binary).
- `nevus_models/`: Models trained to detect nevus (binary).
- `binary_models/`: Models trained for general benign vs malignant classification.
- `malignant_models/`: Multi-class models for malignant subtypes (AKIEC, BCC).
- `benign_models/`: Multi-class models for benign subtypes (BKL, DF, VASC).

> Each folder should include `.keras` model files that are ready to be loaded using `tf.keras.models.load_model()`.

Ensure model file names follow consistent naming, e.g., `xception_finetuned.keras`, `densenet_finetuned.keras`, etc.
