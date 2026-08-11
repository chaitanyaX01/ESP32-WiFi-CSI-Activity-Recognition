from pathlib import Path

import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =====================================
# PATHS
# =====================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_DIR / "processed"
MODELS_DIR = PROJECT_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)


# =====================================
# CLASS NAMES
# =====================================

CLASS_NAMES = [
    "empty",
    "standing",
    "walking",
    "sitting"
]


# =====================================
# LOAD DATA
# =====================================

print("\nLoading dataset...")

X_train = np.load(PROCESSED_DIR / "X_train.npy")
y_train = np.load(PROCESSED_DIR / "y_train.npy")

X_val = np.load(PROCESSED_DIR / "X_val.npy")
y_val = np.load(PROCESSED_DIR / "y_val.npy")

X_test = np.load(PROCESSED_DIR / "X_test.npy")
y_test = np.load(PROCESSED_DIR / "y_test.npy")


print("\nDataset shapes")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# =====================================
# RANDOM FOREST
# =====================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# =====================================
# VALIDATION
# =====================================

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

val_pred = model.predict(X_val)

val_accuracy = accuracy_score(
    y_val,
    val_pred
)

print(
    f"\nValidation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        val_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# =====================================
# FINAL TEST
# =====================================

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

test_pred = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_pred
)

print(
    f"\nTest Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(
    y_test,
    test_pred,
    labels=[0, 1, 2, 3]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(
    figsize=(8, 7)
)

display.plot(
    ax=ax,
    values_format="d"
)

plt.title(
    "Wi-Fi CSI Activity Recognition\nRandom Forest"
)

plt.tight_layout()

confusion_path = (
    MODELS_DIR
    / "random_forest_confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=200
)

plt.close()


# =====================================
# SAVE MODEL
# =====================================

model_path = (
    MODELS_DIR
    / "random_forest_csi.joblib"
)

joblib.dump(
    model,
    model_path
)


# =====================================
# FEATURE IMPORTANCE
# =====================================

importance = model.feature_importances_

indices = np.argsort(
    importance
)[::-1][:20]

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    range(len(indices)),
    importance[indices]
)

plt.xlabel("Top Feature")
plt.ylabel("Importance")

plt.title(
    "Top 20 CSI Feature Importances"
)

plt.tight_layout()

importance_path = (
    MODELS_DIR
    / "feature_importance.png"
)

plt.savefig(
    importance_path,
    dpi=200
)

plt.close()


# =====================================
# FINISH
# =====================================

print("\n" + "=" * 60)

print("MODEL SAVED")

print("\nModel:")
print(model_path)

print("\nConfusion Matrix:")
print(confusion_path)

print("\nFeature Importance:")
print(importance_path)

print("=" * 60)