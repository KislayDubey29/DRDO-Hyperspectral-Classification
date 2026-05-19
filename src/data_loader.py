import scipy.io as sio
import numpy as np


def load_indian_pines():
    data = sio.loadmat("datasets/IndianPines/Indian_pines_corrected.mat")
    gt = sio.loadmat("datasets/IndianPines/Indian_pines_gt.mat")

    X = data['indian_pines_corrected']
    y = gt['indian_pines_gt']

    return X, y


def load_paviaU():
    data = sio.loadmat("datasets/PaviaU/PaviaU.mat")
    gt = sio.loadmat("datasets/PaviaU/PaviaU_gt.mat")

    X = data['paviaU']
    y = gt['paviaU_gt']

    return X, y


def load_salinas():
    data = sio.loadmat("datasets/Salinas/Salinas_corrected.mat")
    gt = sio.loadmat("datasets/Salinas/Salinas_gt.mat")

    X = data['salinas_corrected']
    y = gt['salinas_gt']

    return X, y