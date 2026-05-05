import os
import numpy as np
import json

def load_ucihar(base_dir="data/UCI_HAR_dataset"):
    signals = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z"
    ]
    X_list, y_list = [], []
    for split in ["train", "test"]:
        sigs = []
        for sig in signals:
            path = os.path.join(base_dir, split, "Inertial Signals", f"{sig}_{split}.txt")
            sigs.append(np.loadtxt(path))
        X = np.stack(sigs, axis=-1)  # (N, 128, 9)
        y = np.loadtxt(os.path.join(base_dir, split, f"y_{split}.txt"))
        X_list.append(X)
        y_list.append(y)
    
    X_train, X_test = X_list[0], X_list[1]
    y_train, y_test = y_list[0], y_list[1]

    # Normalisation (moyenne/écart-type par canal)
    mu, sigma = X_train.mean(axis=(0,1)), X_train.std(axis=(0,1)) + 1e-8
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    # Labels 1-based → 0-based + one-hot
    y_train, y_test = y_train - 1, y_test - 1
    from tensorflow.keras.utils import to_categorical
    y_train = to_categorical(y_train, num_classes=6)
    y_test = to_categorical(y_test, num_classes=6)

    return X_train, y_train, X_test, y_test, mu.tolist(), sigma.tolist()