"""
Benchmark PCA vs SparsePCA pipelines on MNIST under FGSM, PGD, and MIM attacks.

- FGSM  : Goodfellow et al., 2015
- PGD   : Madry et al., 2018
- MIM   : Dong et al., 2018 (momentum iterative method)

Usage:
  Set `attack_name` in __main__ to one of {"FGSM","PGD","MIM"}.
  Adjust eps_list, n_components_list, and optional attack_params as desired.
"""

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from art.attacks.evasion import (
    FastGradientMethod,
    MomentumIterativeMethod,
    ProjectedGradientDescent,
)
from art.estimators.classification import PyTorchClassifier
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, SparsePCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# -----------------------------
# Data
# -----------------------------
def load_mnist(n_samples=None, seed: int = 29):
    print("Fetching MNIST dataset...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    print("Scaling and splitting data...")
    if n_samples is None:
        n_samples = len(X)
    rng = np.random.RandomState(seed)
    random_indices = rng.choice(len(X), n_samples, replace=False)
    X = X[random_indices]
    y = y[random_indices].astype(np.int32)

    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    return X_train, X_test, y_train, y_test, n_samples


# -----------------------------
# Fixed preprocessing layers
# -----------------------------
class FixedLinear(nn.Module):
    """
    Fixed linear transform using precomputed weights and bias.
    Used to apply PCA or SparsePCA as the first layer.
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        self.linear = nn.Linear(in_features, out_features, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.linear(x)


class FixedScaler(nn.Module):
    """
    Fixed StandardScaler: (x - mean) / scale.
    """

    def __init__(self, mean, scale, eps: float = 1e-12):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        return (x - self.mean) / (self.scale + self.eps)


# -----------------------------
# Classifier head
# -----------------------------
class ClassifierNN(nn.Module):
    """
    A simple two-layer MLP classifier.
    """

    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class PipelineModel(nn.Module):
    """
    Raw (flattened) image -> fixed PCA/SPCA -> fixed scaler -> trainable NN.
    """

    def __init__(self, fixed_transform, fixed_scaler, classifier):
        super().__init__()
        self.fixed_transform = fixed_transform
        self.fixed_scaler = fixed_scaler
        self.classifier = classifier

    def forward(self, x):
        x = self.fixed_transform(x)
        x = self.fixed_scaler(x)
        x = self.classifier(x)
        return x


# -----------------------------
# Training
# -----------------------------
def train_pipeline_model(
    model,
    X_train,
    y_train,
    device,
    num_epochs=10,
    batch_size=128,
    lr=1e-3,
    verbose=False,
):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        if verbose:
            avg_loss = epoch_loss / len(loader.dataset)
            print(f"Epoch {epoch}: Loss {avg_loss:.4f}")
    return model, optimizer, criterion


def setup_pipeline_classifier(
    transformer, X_train, y_train, device=torch.device("cpu"), num_epochs=20
):
    """
    Fit transformer (PCA/SparsePCA) and scaler, build composite PyTorch model,
    train it, and wrap it with ART's PyTorchClassifier so attacks can backprop
    through the fixed preprocessing (end-to-end differentiable).
    """
    transformer.fit(X_train)
    X_train_trans = transformer.transform(X_train)

    scaler = StandardScaler()
    scaler.fit(X_train_trans)
    n_comp = X_train_trans.shape[1]

    # mean for PCA exists; SparsePCA doesn't expose mean_ (assume zero-centering there).
    mean_val = (
        transformer.mean_
        if hasattr(transformer, "mean_")
        else np.zeros(X_train.shape[1])
    )

    # Transform: (x - mean) @ components_.T  -> weight=components_, bias=-(mean @ components_.T)
    weight = torch.tensor(transformer.components_, dtype=torch.float32)
    bias = -torch.matmul(
        torch.tensor(mean_val, dtype=torch.float32),
        torch.tensor(transformer.components_.T, dtype=torch.float32),
    )

    fixed_transform = FixedLinear(weight, bias)
    fixed_scaler = FixedScaler(scaler.mean_, scaler.scale_)

    classifier_nn = ClassifierNN(input_dim=n_comp, num_classes=10)
    model = PipelineModel(fixed_transform, fixed_scaler, classifier_nn).to(device)

    model, optimizer, criterion = train_pipeline_model(
        model, X_train, y_train, device, num_epochs=num_epochs, verbose=True
    )

    art_classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(784,),
        nb_classes=10,
        clip_values=(0, 1),
    )
    return art_classifier


# -----------------------------
# Attacks
# -----------------------------
def make_attack(
    classifier,
    attack_name: str,
    eps: float,
    attack_params: Dict = None,
):
    """
    Build an ART attack for a given attack name and epsilon.
    `attack_params` can include shared or attack-specific knobs, e.g.:
      - 'batch_size' (shared)
      - 'norm' (shared; default np.inf)
      - For PGD/MIM: 'eps_step' (or 'eps_step_ratio'), 'max_iter'
      - For MIM: 'decay' (momentum)
    """
    if attack_params is None:
        attack_params = {}
    batch_size = int(attack_params.get("batch_size", 128))
    norm = attack_params.get("norm", np.inf)

    if attack_name.upper() == "FGSM":
        # Single-step sign gradient
        return FastGradientMethod(classifier, eps=eps, batch_size=batch_size)

    # For iterative attacks, choose eps_step either from explicit value or ratio of eps.
    eps_step = attack_params.get("eps_step", None)
    if eps_step is None:
        eps_step_ratio = attack_params.get(
            "eps_step_ratio", 0.25 if attack_name.upper() == "PGD" else 0.1
        )
        eps_step = float(max(eps * eps_step_ratio, 1e-6))
    max_iter = int(
        attack_params.get("max_iter", 40 if attack_name.upper() == "PGD" else 10)
    )

    if attack_name.upper() == "PGD":
        return ProjectedGradientDescent(
            classifier,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=False,
            batch_size=batch_size,
        )
    elif attack_name.upper() == "MIM":
        decay = float(attack_params.get("decay", 1.0))  # commonly 1.0
        return MomentumIterativeMethod(
            classifier,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            decay=decay,
            max_iter=max_iter,
            targeted=False,
            batch_size=batch_size,
        )
    else:
        raise ValueError(
            f"Unsupported attack_name '{attack_name}'. Choose from FGSM, PGD, MIM."
        )


def generate_adversarial_samples(
    classifier, X: np.ndarray, attack_name: str, eps: float, attack_params: Dict = None
):
    attack = make_attack(classifier, attack_name, eps, attack_params)
    X_adv = attack.generate(x=X)
    X_adv = np.clip(X_adv, 0, 1)
    return X_adv


# -----------------------------
# Evaluation + Benchmark
# -----------------------------
def evaluate_robustness(X_clean, X_adv, y, classifier):
    clean_predictions = classifier.predict(X_clean)
    adv_predictions = classifier.predict(X_adv)

    clean_predictions = np.argmax(clean_predictions, axis=1)
    adv_predictions = np.argmax(adv_predictions, axis=1)

    y = y.astype(int)
    clean_accuracy = np.mean(clean_predictions == y) * 100.0
    adv_accuracy = np.mean(adv_predictions == y) * 100.0
    return clean_accuracy, adv_accuracy


def benchmark_robustness(
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier_pca_dict: Dict[int, PyTorchClassifier],
    classifier_spca_dict: Dict[int, PyTorchClassifier],
    eps_list: List[float],
    n_samples: int,
    attack_name: str,
    attack_params: Dict,
    save_samples: bool = True,
):
    """
    For each n_components and each epsilon, generate adversarial samples using attack_name
    and compute accuracy for PCA and SPCA versions.
    """
    # Baseline clean accuracies (epsilon=0)
    pca_clean_acc_dict = {}
    spca_clean_acc_dict = {}
    for n_comp in classifier_pca_dict.keys():
        pca_clean_acc, _ = evaluate_robustness(
            X_test, X_test, y_test, classifier_pca_dict[n_comp]
        )
        spca_clean_acc, _ = evaluate_robustness(
            X_test, X_test, y_test, classifier_spca_dict[n_comp]
        )
        pca_clean_acc_dict[n_comp] = pca_clean_acc
        spca_clean_acc_dict[n_comp] = spca_clean_acc

    pca_accuracies_dict = {n_comp: [acc] for n_comp, acc in pca_clean_acc_dict.items()}
    spca_accuracies_dict = {
        n_comp: [acc] for n_comp, acc in spca_clean_acc_dict.items()
    }

    pbar = tqdm(
        total=len(eps_list) * len(classifier_pca_dict.keys()),
        desc=f"[{attack_name}] Testing epsilon values",
    )

    # Optional saving of adversarial panels (first test image only)
    if save_samples:
        n_components_list = list(classifier_pca_dict.keys())
        X_adv_pcas = {ncp: [] for ncp in n_components_list}
        X_adv_spcas = {ncp: [] for ncp in n_components_list}

    for n_comp in classifier_pca_dict.keys():
        for eps in eps_list:
            pbar.set_description(f"[{attack_name}] eps={eps:.3f}, n_comp={n_comp}")
            X_adv_pca = generate_adversarial_samples(
                classifier_pca_dict[n_comp], X_test, attack_name, eps, attack_params
            )
            X_adv_spca = generate_adversarial_samples(
                classifier_spca_dict[n_comp], X_test, attack_name, eps, attack_params
            )

            _, pca_adv_acc = evaluate_robustness(
                X_test, X_adv_pca, y_test, classifier_pca_dict[n_comp]
            )
            _, spca_adv_acc = evaluate_robustness(
                X_test, X_adv_spca, y_test, classifier_spca_dict[n_comp]
            )

            pca_accuracies_dict[n_comp].append(pca_adv_acc)
            spca_accuracies_dict[n_comp].append(spca_adv_acc)

            if save_samples:
                X_adv_pcas[n_comp].append(X_adv_pca[0])
                X_adv_spcas[n_comp].append(X_adv_spca[0])
            pbar.update(1)
    pbar.close()

    # Save side-by-side adversarial panels for first image across epsilons
    if save_samples:
        n_components_list = list(classifier_pca_dict.keys())
        directory = (
            f"adv_samples_{attack_name.lower()}_eps_{eps_list[0]}_to_{eps_list[-1]}_"
            f"ncomp_{min(n_components_list)}_to_{max(n_components_list)}_nsamples_{n_samples}"
        )
        os.makedirs(directory, exist_ok=True)
        X_image = X_test[0]
        for ncp in n_components_list:
            show_adversarial_samples(
                X_image,
                X_adv_pcas[ncp],
                X_adv_spcas[ncp],
                eps_list,
                ncp,
                directory,
                attack_name,
            )

    epsilons = np.concatenate(([0], eps_list))
    return epsilons, pca_accuracies_dict, spca_accuracies_dict


# -----------------------------
# Plotting
# -----------------------------
def plot_benchmark(
    epsilons: np.ndarray,
    pca_accuracies_dict: Dict[int, List[float]],
    spca_accuracies_dict: Dict[int, List[float]],
    n_components_list: List[int],
    n_samples: int,
    attack_name: str,
):
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(n_components_list), max(n_components_list))

    # To avoid colorbar mismatches, ensure n_components_list is sorted
    n_components_list = sorted(n_components_list)

    for idx, n_comp in enumerate(n_components_list):
        color = cmap(norm(n_comp))
        ax.plot(
            epsilons,
            pca_accuracies_dict[n_comp],
            "--o",
            color=color,
            label="PCA" if idx == 0 else "_nolegend_",
        )
        ax.plot(
            epsilons,
            spca_accuracies_dict[n_comp],
            "-o",
            color=color,
            label="SPCA" if idx == 0 else "_nolegend_",
        )

    ax.set_xlabel("Epsilon (ε)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Accuracy vs. Attack Strength ({attack_name})")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Number of Components")
    ax.legend()
    ax.grid(True)
    out_name = (
        f"{attack_name.lower()}_eps_{epsilons[1]}_to_{epsilons[-1]}_"
        f"ncomp_{min(n_components_list)}_to_{max(n_components_list)}_nsamples_{n_samples}.png"
    )
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def show_adversarial_samples(
    X: np.ndarray,
    X_adv_pcas: List[np.ndarray],
    X_adv_spcas: List[np.ndarray],
    eps_list: List[float],
    ncp: int,
    directory: str,
    attack_name: str,
):
    """
    Save a panel: original image + adversarials for PCA (top) and SPCA (bottom)
    across eps_list, for a given n_components.
    """
    n_eps = len(X_adv_pcas)
    fig, axes = plt.subplots(2, n_eps + 1, figsize=(1.8 * (n_eps + 1), 6))

    # Originals
    axes[0, 0].imshow(X.reshape(28, 28), cmap="gray")
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Original")
    axes[1, 0].imshow(X.reshape(28, 28), cmap="gray")
    axes[1, 0].axis("off")
    axes[1, 0].set_title("Original")

    # Adversarials
    for i, (X_adv_pca, X_adv_spca) in enumerate(zip(X_adv_pcas, X_adv_spcas)):
        axes[0, i + 1].imshow(X_adv_pca.reshape(28, 28), cmap="gray")
        axes[0, i + 1].axis("off")
        axes[0, i + 1].set_title(f"ε={eps_list[i]:.2f}")
        axes[1, i + 1].imshow(X_adv_spca.reshape(28, 28), cmap="gray")
        axes[1, i + 1].axis("off")
        axes[1, i + 1].set_title(f"ε={eps_list[i]:.2f}")

    axes[0, 0].set_ylabel("PCA")
    axes[1, 0].set_ylabel("SPCA")
    plt.suptitle(f"{attack_name}: Adversarial Samples (n_components={ncp})", y=0.98)
    plt.tight_layout()
    out_png = (
        f"{directory}/adversarial_samples_{attack_name.lower()}_"
        f"{eps_list[0]:.2f}_to_{eps_list[-1]:.2f}_ncomp_{ncp}.png"
    )
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Orchestration
# -----------------------------
def main(
    eps_list: np.ndarray,
    n_components_list: List[int],
    n_samples,
    attack_name: str,
    attack_params: Dict = None,
    save_samples: bool = True,
):
    X_train, X_test, y_train, y_test, n_samples = load_mnist(n_samples=n_samples)

    # Normalize each split to [0,1] (kept from original script)
    X_train = ((X_train - X_train.min()) / (X_train.max() - X_train.min())).astype(
        np.float32
    )
    X_test = ((X_test - X_test.min()) / (X_test.max() - X_test.min())).astype(
        np.float32
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Initializing and training PCA and SPCA pipeline models...")

    classifier_pca_dict: Dict[int, PyTorchClassifier] = {}
    classifier_spca_dict: Dict[int, PyTorchClassifier] = {}

    for n_comp in tqdm(n_components_list, desc="Training per n_components"):
        pca_transformer = PCA(n_components=n_comp)
        spca_transformer = SparsePCA(
            n_components=n_comp, random_state=29, max_iter=100, alpha=1
        )

        classifier_pca_dict[n_comp] = setup_pipeline_classifier(
            pca_transformer, X_train, y_train, device=device, num_epochs=20
        )
        classifier_spca_dict[n_comp] = setup_pipeline_classifier(
            spca_transformer, X_train, y_train, device=device, num_epochs=20
        )

    print(f"Running benchmark tests for {attack_name}...")
    epsilons, pca_accuracies_dict, spca_accuracies_dict = benchmark_robustness(
        X_test,
        y_test,
        classifier_pca_dict,
        classifier_spca_dict,
        eps_list,
        n_samples,
        attack_name=attack_name,
        attack_params=attack_params or {},
        save_samples=save_samples,
    )

    print("\nBenchmark Results:")
    print(f"Attack: {attack_name}")
    print("Epsilon values:", [round(i, 3) for i in epsilons])
    for n_comp in n_components_list:
        print(f"\nResults for {n_comp} components:")
        print(f"PCA accuracies:  {[round(i, 2) for i in pca_accuracies_dict[n_comp]]}")
        print(f"SPCA accuracies: {[round(i, 2) for i in spca_accuracies_dict[n_comp]]}")

    plot_benchmark(
        epsilons,
        pca_accuracies_dict,
        spca_accuracies_dict,
        n_components_list,
        n_samples,
        attack_name=attack_name,
    )


def unit_test_benchmark():
    # Quick smoke test on small config
    eps_list = np.arange(0.1, 1.0, 0.1)
    n_components_list = [150, 191]
    attack_name = "FGSM"
    main(eps_list, n_components_list, 1000, attack_name=attack_name, save_samples=True)


if __name__ == "__main__":
    test_mode = False
    if test_mode:
        unit_test_benchmark()
        raise SystemExit

    # -----------------------------
    # EXPERIMENT CONFIG
    # -----------------------------
    # Choose one: "FGSM", "PGD", or "MIM"
    attack_name = "FGSM"

    # PCA dimensionalities to compare
    n_components_list = [100, 150, 200]  # 191 ~ 95% explained variance for PCA on MNIST

    # Attack strength sweep
    eps_list = np.arange(0.01, 0.21, 0.01)

    # Optional attack-specific knobs (sensible defaults given).
    # For PGD: default max_iter=40, eps_step=eps*0.25
    # For MIM: default max_iter=10,  eps_step=eps*0.10, decay=1.0
    attack_params = {
        # "batch_size": 128,
        # "norm": np.inf,
        # "eps_step": 0.02,          # OR use ratio below
        # "eps_step_ratio": 0.25,    # if 'eps_step' not set, uses ratio*eps
        # "max_iter": 40,
        # "decay": 1.0,              # (MIM only)
    }

    # Dataset size (None => full MNIST from OpenML)
    n_samples = None

    main(
        eps_list=eps_list,
        n_components_list=n_components_list,
        n_samples=n_samples,
        attack_name=attack_name,
        attack_params=attack_params,
        save_samples=True,
    )
