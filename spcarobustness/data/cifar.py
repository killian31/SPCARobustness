from typing import Optional, Tuple

import numpy as np


def load_cifar10_binary_airplane_frog(n_samples: Optional[int] = None, seed: int = 29):
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor

    root = "./cifar_data"
    train_ds = CIFAR10(root=root, train=True, download=True, transform=ToTensor())
    test_ds = CIFAR10(root=root, train=False, download=True, transform=ToTensor())

    keep = {0: 0, 6: 1}

    def ds_to_numpy_binary(ds):
        xs, ys = [], []
        for img, lbl in ds:
            if lbl in keep:
                xs.append(img.view(-1).numpy())
                ys.append(keep[lbl])
        X = np.stack(xs, axis=0).astype(np.float32)
        y = np.array(ys, dtype=np.int32)
        return X, y

    X_train_full, y_train_full = ds_to_numpy_binary(train_ds)
    X_test_full, y_test_full = ds_to_numpy_binary(test_ds)

    if n_samples is not None:
        n_train = int(n_samples * 0.8)
        n_test = int(n_samples * 0.2)
        rng = np.random.RandomState(seed)
        idx_train = rng.choice(
            len(X_train_full), size=min(n_train, len(X_train_full)), replace=False
        )
        idx_test = rng.choice(
            len(X_test_full), size=min(n_test, len(X_test_full)), replace=False
        )
        X_train, y_train = X_train_full[idx_train], y_train_full[idx_train]
        X_test, y_test = X_test_full[idx_test], y_test_full[idx_test]
        return X_train, X_test, y_train, y_test, (len(X_train) + len(X_test))
    else:
        return (
            X_train_full,
            X_test_full,
            y_train_full,
            y_test_full,
            (len(X_train_full) + len(X_test_full)),
        )
