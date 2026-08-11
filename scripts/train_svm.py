from pathlib import Path

import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


PROJECT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_DIR / "processed"
MODELS_DIR = PROJECT_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)

CLASS_NAMES = [
    "empty",
    "standing",
    "walking",
    "sitting"
]


# ==========================================
# LOAD DATA
# ==========================================

X_train = np.load(PROCESSED_DIR / "X_train.npy")
y_train = np.load(PROCESSED_DIR / "y_train.npy")

X_val = np.load(PROCESSED_DIR / "X_val.npy")
y_val = np.load(PROCESSED_DIR / "y_val.npy")

X_test = np.load(PROCESSED_DIR / "X_test.npy")
y_test = np.load(PROCESSED_DIR / "y_test.npy")


print("\nTRAIN:", X_train.shape)
print("VAL  :", X_val.shape)
print("TEST :", X_test.shape)


# ==========================================
# MODEL
# ==========================================

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "svm",
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42
        )
    )
])


# ==========================================
# TRAIN
# ==========================================

print("\nTraining SVM...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ==========================================
# VALIDATION
# ==========================================

val_pred = model.predict(X_val)

val_accuracy = accuracy_score(
    y_val,
    val_pred
)

print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

print(
    f"Validation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    classification_report(
        y_val,
        val_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# ==========================================
# TEST
# ==========================================

test_pred = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_pred
)

print("\n" + "=" * 60)
print("TEST")
print("=" * 60)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    classification_report(
        y_test,
        test_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    test_pred,
    labels=[0, 1, 2, 3]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(figsize=(8, 7))

display.plot(
    ax=ax,
    values_format="d"
)

plt.title(
    "Wi-Fi CSI Activity Recognition - SVM"
)

plt.tight_layout()

output = (
    MODELS_DIR
    / "svm_confusion_matrix.png"
)

plt.savefig(
    output,
    dpi=200
)

plt.close()


# ==========================================
# SAVE MODEL
# ==========================================

model_path = (
    MODELS_DIR
    / "svm_csi.joblib"
)

joblib.dump(
    model,
    model_path
)

print("\nModel saved:")
print(model_path)

print("\nConfusion matrix:")
print(output)