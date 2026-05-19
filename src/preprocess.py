import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def apply_pca(X, num_components=30):

    h, w, bands = X.shape

    X_reshaped = X.reshape(-1, bands)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    pca = PCA(n_components=num_components)

    X_pca = pca.fit_transform(X_scaled)

    X_pca = X_pca.reshape(h, w, num_components)

    return X_pca


def create_patches(X, y, window_size=5):

    margin = window_size // 2

    padded_X = np.pad(
        X,
        ((margin, margin), (margin, margin), (0, 0)),
        mode='constant'
    )

    patches = []
    labels = []

    for r in range(margin, padded_X.shape[0] - margin):
        for c in range(margin, padded_X.shape[1] - margin):

            label = y[r - margin, c - margin]

            if label == 0:
                continue

            patch = padded_X[
                r - margin:r + margin + 1,
                c - margin:c + margin + 1
            ]

            patches.append(patch)
            labels.append(label - 1)

    return np.array(patches), np.array(labels)