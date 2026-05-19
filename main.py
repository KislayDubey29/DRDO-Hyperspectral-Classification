from src.visualize import save_metrics_json
from src.visualize import save_accuracy_txt

from src.cnn_3d import run_cnn_3d
from src.data_loader import load_indian_pines
from src.preprocess import apply_pca, create_patches

from src.svm_classifier import run_svm
from src.cnn_2d import run_cnn_2d


# ==============================
# LOAD DATASET
# ==============================

X, y = load_indian_pines()


# ==============================
# APPLY PCA
# ==============================

X_pca = apply_pca(
    X,
    num_components=30
)


# ==============================
# CREATE PATCHES
# ==============================

X_patches, y_patches = create_patches(
    X_pca,
    y,
    window_size=5
)

print("Patch Shape:", X_patches.shape)


# ==============================
# RUN SVM
# ==============================

print("\n==============================")
print("RUNNING SVM")
print("==============================")

svm_accuracy = run_svm(
    X_patches,
    y_patches
)


# ==============================
# RUN CNN-2D
# ==============================

print("\n==============================")
print("RUNNING CNN-2D")
print("==============================")

cnn2d_accuracy = run_cnn_2d(
    X_patches,
    y_patches
)


# ==============================
# RUN CNN-3D
# ==============================

print("\n==============================")
print("RUNNING CNN-3D")
print("==============================")

cnn3d_accuracy = run_cnn_3d(
    X_patches,
    y_patches
)


# ==============================
# FINAL RESULTS
# ==============================

print("\n==============================")
print("FINAL RESULTS")
print("==============================")

print("SVM Accuracy:", svm_accuracy)
print("CNN-2D Accuracy:", cnn2d_accuracy)
print("CNN-3D Accuracy:", cnn3d_accuracy)


# ==============================
# SAVE METRICS
# ==============================

metrics = {
    "SVM Accuracy": float(svm_accuracy),
    "CNN2D Accuracy": float(cnn2d_accuracy),
    "CNN3D Accuracy": float(cnn3d_accuracy)
}

save_metrics_json(metrics)

save_accuracy_txt(metrics)

print("\nMetrics files saved!")