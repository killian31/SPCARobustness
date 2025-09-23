from typing import Tuple, Optional
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_mnist(
    n_samples: Optional[int] = None, seed: int = 29
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    if n_samples is None:
        n_samples = len(X)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), n_samples, replace=False)
    X = X[idx].astype(np.float32) / 255.0
    y = y[idx].astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    return X_train, X_test, y_train, y_test, n_samples
