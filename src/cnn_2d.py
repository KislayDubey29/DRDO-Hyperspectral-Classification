import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from src.visualize import *


def run_cnn_2d(X, y):

    # One-hot encoding
    y_cat = to_categorical(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_cat,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("CNN Training Shape:", X_train.shape)

    # CNN Model
    model = Sequential()

    model.add(
        Conv2D(
            32,
            (3, 3),
            activation='relu',
            input_shape=X_train.shape[1:]
        )
    )

    model.add(
        MaxPooling2D((2, 2))
    )

    model.add(Flatten())

    model.add(
        Dense(128, activation='relu')
    )

    model.add(
        Dense(y_cat.shape[1], activation='softmax')
    )

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    print("\nTraining CNN-2D...\n")

    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1
    )

    # Save trained model
    model.save("results/models/cnn2d_model.h5")

    print("CNN-2D model saved!")

    # Evaluate
    loss, accuracy = model.evaluate(
        X_test,
        y_test
    )

    print("\nCNN-2D Accuracy:", accuracy)

    # Predictions
    y_pred = model.predict(X_test)

    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Save plots
    plot_training_history(
        history,
        "CNN2D"
    )

    # Save confusion matrix
    save_confusion_matrix(
        y_true,
        y_pred,
        "CNN2D"
    )

    # Save classification report
    save_classification_report(
        y_true,
        y_pred,
        "CNN2D"
    )

    return accuracy