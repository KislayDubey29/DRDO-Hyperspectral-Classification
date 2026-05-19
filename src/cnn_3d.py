import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from src.visualize import *


def run_cnn_3d(X, y):

    # Add channel dimension
    X = X.reshape(
        X.shape[0],
        X.shape[1],
        X.shape[2],
        X.shape[3],
        1
    )

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

    print("3D CNN Train Shape:", X_train.shape)
    print("3D CNN Test Shape:", X_test.shape)

    # CNN-3D Model
    model = Sequential()

    model.add(
        Conv3D(
            filters=8,
            kernel_size=(3, 3, 3),
            activation='relu',
            input_shape=X_train.shape[1:]
        )
    )

    model.add(
        MaxPooling3D(
            pool_size=(2, 2, 2)
        )
    )

    model.add(Flatten())

    model.add(
        Dense(128, activation='relu')
    )

    model.add(
        Dropout(0.4)
    )

    model.add(
        Dense(
            y_cat.shape[1],
            activation='softmax'
        )
    )

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    print("\nTraining CNN-3D...\n")

    # Train
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )

    # Save trained model
    model.save(
        "results/models/cnn3d_model.h5"
    )

    print("CNN-3D model saved!")

    # Evaluate
    loss, accuracy = model.evaluate(
        X_test,
        y_test
    )

    print("\nCNN-3D Accuracy:", accuracy)

    # Predictions
    y_pred = model.predict(X_test)

    y_pred = np.argmax(
        y_pred,
        axis=1
    )

    y_true = np.argmax(
        y_test,
        axis=1
    )

    # Save plots
    plot_training_history(
        history,
        "CNN3D"
    )

    # Save confusion matrix
    save_confusion_matrix(
        y_true,
        y_pred,
        "CNN3D"
    )

    # Save classification report
    save_classification_report(
        y_true,
        y_pred,
        "CNN3D"
    )

    return accuracy