import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# -----------------------------
# Create folders automatically
# -----------------------------
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)


# -----------------------------
# Accuracy & Loss Graphs
# -----------------------------
def plot_training_history(history, model_name):

    # Accuracy Plot
    plt.figure(figsize=(8, 5))

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.legend()

    plt.savefig(
        f"results/figures/{model_name.lower()}_accuracy.png"
    )

    plt.close()

    # Loss Plot
    plt.figure(figsize=(8, 5))

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()

    plt.savefig(
        f"results/figures/{model_name.lower()}_loss.png"
    )

    plt.close()


# -----------------------------
# Confusion Matrix
# -----------------------------
def save_confusion_matrix(y_true, y_pred, model_name):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=False,
        cmap='Blues'
    )

    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.savefig(
        f"results/figures/{model_name.lower()}_confusion_matrix.png"
    )

    plt.close()


# -----------------------------
# Classification Report
# -----------------------------
def save_classification_report(y_true, y_pred, model_name):

    report = classification_report(
        y_true,
        y_pred
    )

    with open(
        f"results/metrics/{model_name.lower()}_classification_report.txt",
        "w"
    ) as f:

        f.write(report)


# -----------------------------
# Save Accuracy JSON
# -----------------------------
def save_metrics_json(metrics_dict):

    with open(
        "results/metrics/metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics_dict,
            f,
            indent=4
        )


# -----------------------------
# Save Accuracy TXT
# -----------------------------
def save_accuracy_txt(metrics_dict):

    with open(
        "results/metrics/accuracies.txt",
        "w"
    ) as f:

        for key, value in metrics_dict.items():

            f.write(
                f"{key}: {value}\n"
            )