from __future__ import annotations
import os
import json
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from spcarobustness.models.components import (
    FixedLinear,
    FixedScaler,
    ClassifierNN,
    PipelineModel,
    ImageFlattenWrapper,
)


def model_path(
    models_dir: str,
    dataset: str,
    algo: str,
    n_components: int,
    num_classes: int,
    input_dim: int,
    spca_mode: Optional[str] = None,
) -> str:
    os.makedirs(models_dir, exist_ok=True)
    algo_label = (
        algo if algo != "spca" else (f"spca-{spca_mode}" if spca_mode else "spca")
    )
    fname = f"{dataset}_{algo_label}_ncomp-{n_components}_classes-{num_classes}_input-{input_dim}.pt"
    return os.path.join(models_dir, fname)


def save_classifier(
    path: str,
    *,
    algo: str,
    n_components: int,
    input_dim: int,
    num_classes: int,
    components: np.ndarray,
    mean_vec: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    classifier_state: Dict[str, Any],
    spca_mode: Optional[str] = None,
):
    payload = {
        "algo": algo,
        "spca_mode": spca_mode,
        "n_components": int(n_components),
        "input_dim": int(input_dim),
        "num_classes": int(num_classes),
        "components": components.astype(np.float32),
        "mean_vec": mean_vec.astype(np.float32),
        "scaler_mean": scaler_mean.astype(np.float32),
        "scaler_scale": scaler_scale.astype(np.float32),
        "classifier_state": classifier_state,
    }
    torch.save(payload, path)


def load_classifier(
    path: str,
    device: torch.device,
):
    if not os.path.exists(path):
        return None
    payload = torch.load(path, map_location=device)

    algo = payload["algo"]
    n_components = int(payload["n_components"])
    input_dim = int(payload["input_dim"])
    num_classes = int(payload["num_classes"])
    components = torch.tensor(payload["components"], dtype=torch.float32)
    mean_vec = torch.tensor(payload["mean_vec"], dtype=torch.float32)
    scaler_mean = payload["scaler_mean"]
    scaler_scale = payload["scaler_scale"]
    classifier_state = payload["classifier_state"]

    weight = components
    bias = -mean_vec @ components.T
    fixed_transform = FixedLinear(weight, bias)
    fixed_scaler = FixedScaler(scaler_mean, scaler_scale)
    classifier_nn = ClassifierNN(input_dim=n_components, num_classes=num_classes)
    model = PipelineModel(fixed_transform, fixed_scaler, classifier_nn).to(device)
    model.classifier.load_state_dict(classifier_state)

    # Build fresh ART classifier for attacks
    import torch.optim as optim
    import torch.nn as nn
    from art.estimators.classification import PyTorchClassifier

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Determine image shape if known (so attacks like SquareAttack accept input)
    dataset_guess = os.path.basename(path).split("_")[0]
    image_shape = None
    if dataset_guess == "mnist" or input_dim == 784:
        image_shape = (1, 28, 28)
    elif dataset_guess.startswith("cifar") or input_dim == 3072:
        image_shape = (3, 32, 32)

    wrapped_model = model if image_shape is None else ImageFlattenWrapper(model)
    input_shape = (input_dim,) if image_shape is None else image_shape

    art_classifier = PyTorchClassifier(
        model=wrapped_model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=num_classes,
        clip_values=(0, 1),
    )
    meta = {
        "algo": algo,
        "n_components": n_components,
        "input_dim": input_dim,
        "num_classes": num_classes,
    }
    return art_classifier, meta
