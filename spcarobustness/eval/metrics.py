from typing import Dict, List, Tuple

import numpy as np


def evaluate_robustness(X_clean, X_adv, y, classifier) -> Tuple[float, float]:
    clean_predictions = classifier.predict(X_clean)
    adv_predictions = classifier.predict(X_adv)
    clean_predictions = np.argmax(clean_predictions, axis=1)
    adv_predictions = np.argmax(adv_predictions, axis=1)
    y = y.astype(int)
    clean_accuracy = np.mean(clean_predictions == y) * 100.0
    adv_accuracy = np.mean(adv_predictions == y) * 100.0
    return clean_accuracy, adv_accuracy
