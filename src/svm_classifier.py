import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


import numpy as np


def run_svm(X, y):

    # Flatten patches
    X = X.reshape(X.shape[0], -1)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training Samples:", X_train.shape)
    print("Testing Samples:", X_test.shape)

    # SVM Model
    svm = SVC(kernel='rbf')

    print("\nTraining SVM...")

    svm.fit(X_train, y_train)
    joblib.dump(svm, "results/models/svm_model.pkl")
    print("SVM model saved!")

    print("SVM Training Complete!")

    # Prediction
    y_pred = svm.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\nSVM Accuracy:", accuracy)

    # Classification Report
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    return accuracy
