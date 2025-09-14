"""
CIFAR-10 (Binary: Airplane vs Frog) — PCA vs SparsePCA under FGSM, PGD, MIM.

- FGSM  : Goodfellow et al., 2015
- PGD   : Madry et al., 2018
- MIM   : Dong et al., 2018

Pipeline:
  raw 32x32x3 (flattened, [0,1]) ->
  Fixed (PCA/SPCA) linear projection ->
  Fixed StandardScaler ->
  small MLP (2 classes)

Switch attack via `attack_name` in __main__.
Epsilon is capped at 0.1 for natural images.
"""

import os
from typing import Dict, List, Optional

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
from sklearn.decomposition import PCA, MiniBatchSparsePCA, SparsePCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# -----------------------------
# Utils & Reproducibility
# -----------------------------
def set_seed(seed: int = 29):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -----------------------------
# Data (CIFAR-10 via torchvision) — Binary Airplane vs Frog
# -----------------------------
def load_cifar10_binary_airplane_frog(n_samples: Optional[int] = None, seed: int = 29):
    """
    Returns airplane (label 0) vs frog (label 1) only.

    Output:
        X_train, X_test: float32 in [0,1], shape (N, 3072)
        y_train, y_test: int labels in {0,1}
        n_total: number of samples actually used (train+test)
    """
    set_seed(seed)
    try:
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import ToTensor
    except Exception as e:
        raise RuntimeError(
            "torchvision is required. Please `pip install torchvision`."
        ) from e

    root = "./cifar_data"
    train_ds = CIFAR10(root=root, train=True, download=True, transform=ToTensor())
    test_ds = CIFAR10(root=root, train=False, download=True, transform=ToTensor())

    # CIFAR-10 class indices: airplane=0, frog=6
    keep = {0: 0, 6: 1}  # map airplane->0, frog->1

    def ds_to_numpy_binary(ds):
        xs, ys = [], []
        for img, lbl in ds:
            if lbl in keep:
                xs.append(img.view(-1).numpy())  # 3072, in [0,1]
                ys.append(keep[lbl])
        X = np.stack(xs, axis=0).astype(np.float32)
        y = np.array(ys, dtype=np.int32)
        return X, y

    X_train_full, y_train_full = ds_to_numpy_binary(train_ds)  # ~10k (5k/5k)
    X_test_full, y_test_full = ds_to_numpy_binary(test_ds)  # ~2k  (1k/1k)

    if n_samples is not None:
        # Split n_samples into 80% train / 20% test
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


# -----------------------------
# Fixed preprocessing layers
# -----------------------------
class FixedLinear(nn.Module):
    """Fixed linear transform using precomputed weights and bias (PCA/SPCA)."""

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
    """Fixed StandardScaler: (x - mean) / scale."""

    def __init__(self, mean, scale, eps: float = 1e-12):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        return (x - self.mean) / (self.scale + self.eps)


# -----------------------------
# Classifier head (2 classes)
# -----------------------------
class ClassifierNN(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        hidden = int(min(1024, max(256, input_dim // 2)))
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class PipelineModel(nn.Module):
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
    num_epochs=15,
    batch_size=256,
    lr=2e-3,
    verbose=False,
):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

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


def _linear_projection_features(
    X: np.ndarray, components: np.ndarray, mean_vec: np.ndarray
):
    """Compute (X - mean) @ components.T in numpy."""
    return (X - mean_vec) @ components.T


def setup_pipeline_classifier(
    transformer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    device=torch.device("cpu"),
    num_epochs=15,
    input_dim: int = 3072,
    num_classes: int = 2,
    verbose: bool = False,
):
    """
    Fit transformer (PCA/SparsePCA/MiniBatchSparsePCA), compute linear projection
    (X - mean) @ components_.T, fit scaler on those features, build model, train,
    and wrap in ART's PyTorchClassifier (nb_classes=2).
    """
    transformer.fit(X_train)

    # print cumulative explained variance
    if hasattr(transformer, "explained_variance_ratio_"):
        cum_var = np.cumsum(transformer.explained_variance_ratio_)
        print(
            f"Cumulative explained variance by {transformer.n_components_} components: "
            f"{cum_var[-1]*100:.2f}%"
        )

    components = transformer.components_
    if hasattr(transformer, "mean_"):
        mean_val = transformer.mean_
    else:
        mean_val = np.zeros(input_dim, dtype=np.float32)

    X_train_proj = _linear_projection_features(X_train, components, mean_val)
    scaler = StandardScaler()
    scaler.fit(X_train_proj)

    n_comp = components.shape[0]
    weight = torch.tensor(components, dtype=torch.float32)
    bias = -torch.tensor(mean_val, dtype=torch.float32) @ torch.tensor(
        components.T, dtype=torch.float32
    )

    fixed_transform = FixedLinear(weight, bias)
    fixed_scaler = FixedScaler(scaler.mean_, scaler.scale_)
    classifier_nn = ClassifierNN(input_dim=n_comp, num_classes=num_classes)
    model = PipelineModel(fixed_transform, fixed_scaler, classifier_nn).to(device)

    model, optimizer, criterion = train_pipeline_model(
        model, X_train, y_train, device, num_epochs=num_epochs, verbose=verbose
    )

    art_classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(input_dim,),
        nb_classes=num_classes,
        clip_values=(0, 1),
    )
    return art_classifier


# -----------------------------
# Attacks
# -----------------------------
def make_attack(classifier, attack_name: str, eps: float, attack_params: Dict = None):
    if attack_params is None:
        attack_params = {}

    batch_size = int(attack_params.get("batch_size", 129))
    norm = attack_params.get("norm", np.inf)

    # Cast to plain Python float (avoid np.float32/64 mismatches)
    eps = float(eps)

    if attack_name.upper() == "FGSM":
        # Make sure eps_step matches type -> also a plain float
        eps_step = float(attack_params.get("eps_step", eps))
        return FastGradientMethod(
            classifier, eps=eps, eps_step=eps_step, batch_size=batch_size
        )

    # Iterative attacks: compute eps_step and cast to float too
    eps_step = attack_params.get("eps_step", None)
    if eps_step is None:
        eps_step_ratio = attack_params.get(
            "eps_step_ratio", 0.25 if attack_name.upper() == "PGD" else 0.10
        )
        eps_step = eps * float(eps_step_ratio)
    eps_step = float(max(eps_step, 1e-6))
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
        decay = float(attack_params.get("decay", 1.0))
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
    eps_list: np.ndarray,
    n_samples: int,
    attack_name: str,
    attack_params: Dict,
    save_samples: bool = True,
    verbose: bool = False,
):
    # Clean accuracies (epsilon=0)
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

    if verbose:
        print("Clean accuracies (no attack):")
        for n_comp in classifier_pca_dict.keys():
            print(
                f"  n_components={n_comp}: PCA acc={pca_clean_acc_dict[n_comp]:.2f}%, "
                f"SPCA acc={spca_clean_acc_dict[n_comp]:.2f}%"
            )

    pca_accuracies_dict = {n_comp: [acc] for n_comp, acc in pca_clean_acc_dict.items()}
    spca_accuracies_dict = {
        n_comp: [acc] for n_comp, acc in spca_clean_acc_dict.items()
    }

    pbar = tqdm(
        total=len(eps_list) * len(classifier_pca_dict.keys()),
        desc=f"[{attack_name}] Testing epsilon values",
    )

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

    if save_samples:
        n_components_list = list(classifier_pca_dict.keys())
        directory = (
            f"cifar10_binary_airplane_frog_adv_{attack_name.lower()}_"
            f"eps_{eps_list[0]:.3f}_to_{eps_list[-1]:.3f}_"
            f"ncomp_{min(n_components_list)}_to_{max(n_components_list)}_nsamples_{n_samples}"
        )
        os.makedirs(directory, exist_ok=True)
        X_image = X_test[0]
        for ncp in n_components_list:
            show_adversarial_samples_cifar_binary(
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
    ax.set_title(f"CIFAR-10 Binary (Airplane vs Frog): Accuracy vs ε — {attack_name}")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="# PCA Components")
    ax.legend()
    ax.grid(True)
    out_name = (
        f"cifar10_binary_airplane_frog_{attack_name.lower()}_"
        f"eps_{epsilons[1]:.3f}_to_{epsilons[-1]:.3f}_"
        f"ncomp_{min(n_components_list)}_to_{max(n_components_list)}_nsamples_{n_samples}.png"
    )
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def show_adversarial_samples_cifar_binary(
    X: np.ndarray,
    X_adv_pcas: List[np.ndarray],
    X_adv_spcas: List[np.ndarray],
    eps_list: List[float],
    ncp: int,
    directory: str,
    attack_name: str,
):
    """Panels for color images (32x32x3)."""
    n_eps = len(X_adv_pcas)
    fig, axes = plt.subplots(2, n_eps + 1, figsize=(2.0 * (n_eps + 1), 6))

    def to_img(arr):
        return arr.reshape(3, 32, 32).transpose(1, 2, 0)  # HWC

    axes[0, 0].imshow(to_img(X))
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Original")
    axes[1, 0].imshow(to_img(X))
    axes[1, 0].axis("off")
    axes[1, 0].set_title("Original")

    for i, (X_adv_pca, X_adv_spca) in enumerate(zip(X_adv_pcas, X_adv_spcas)):
        axes[0, i + 1].imshow(to_img(X_adv_pca))
        axes[0, i + 1].axis("off")
        axes[0, i + 1].set_title(f"ε={eps_list[i]:.2f}")
        axes[1, i + 1].imshow(to_img(X_adv_spca))
        axes[1, i + 1].axis("off")
        axes[1, i + 1].set_title(f"ε={eps_list[i]:.2f}")

    axes[0, 0].set_ylabel("PCA")
    axes[1, 0].set_ylabel("SPCA")
    plt.suptitle(
        f"{attack_name}: Adv Samples (Airplane vs Frog, n_components={ncp})", y=0.98
    )
    plt.tight_layout()
    out_png = (
        f"{directory}/cifar10_binary_airplane_frog_adversarial_samples_"
        f"{attack_name.lower()}_{eps_list[0]:.3f}_to_{eps_list[-1]:.3f}_ncomp_{ncp}.png"
    )
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Orchestration
# -----------------------------
def main(
    eps_list: np.ndarray,
    n_components_list: List[int],
    attack_name: str,
    attack_params: Dict = None,
    n_samples: Optional[int] = None,
    spca_mode: str = "minibatch",  # {"minibatch","sparse"}
    spca_fit_samples: Optional[
        int
    ] = 10000,  # fit SPCA on subset for speed (None => full)
    pca_fit_samples: Optional[int] = None,  # usually None (full)
    num_epochs: int = 15,
    save_samples: bool = True,
    verbose: bool = True,
):
    set_seed(29)
    if verbose:
        print(f"Attack: {attack_name}, Params: {attack_params}")
        print(f"Epsilon values: {[round(i,3) for i in eps_list]}")
        print(f"n_components_list: {n_components_list}")
        print(f"n_samples: {n_samples} (None => full binary set)")
        print(f"SPCA mode: {spca_mode}, SPCA fit samples: {spca_fit_samples}")
        print(f"PCA fit samples: {pca_fit_samples}")
        print(f"Num epochs (PCA): {num_epochs}, (SPCA): {num_epochs*2}")
        print(f"Save adversarial samples: {save_samples}")
    eps_list = np.array(eps_list, dtype=np.float32)

    X_train_full, X_test, y_train_full, y_test, n_total = (
        load_cifar10_binary_airplane_frog(n_samples=n_samples, seed=29)
    )

    # Optionally restrict fitting sets for PCA/SPCA (reduce compute)
    if pca_fit_samples is not None and pca_fit_samples < len(X_train_full):
        rng = np.random.RandomState(29)
        idx = rng.choice(len(X_train_full), size=pca_fit_samples, replace=False)
        X_train_pca = X_train_full[idx]
        y_train_pca = y_train_full[idx]
    else:
        X_train_pca = X_train_full
        y_train_pca = y_train_full

    if spca_fit_samples is not None and spca_fit_samples < len(X_train_full):
        rng = np.random.RandomState(29)
        idx = rng.choice(len(X_train_full), size=spca_fit_samples, replace=False)
        X_train_spca = X_train_full[idx]
        y_train_spca = y_train_full[idx]
    else:
        X_train_spca = X_train_full
        y_train_spca = y_train_full

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        "Initializing and training PCA and SPCA pipeline models (binary: airplane vs frog)..."
    )

    classifier_pca_dict: Dict[int, PyTorchClassifier] = {}
    classifier_spca_dict: Dict[int, PyTorchClassifier] = {}
    input_dim = X_train_full.shape[1]  # 3072

    for n_comp in tqdm(n_components_list, desc="Training per n_components"):
        assert 1 <= n_comp <= input_dim, f"n_components must be in [1,{input_dim}]"

        # PCA
        pca_transformer = PCA(
            n_components=n_comp, svd_solver="randomized", random_state=29
        )
        classifier_pca_dict[n_comp] = setup_pipeline_classifier(
            pca_transformer,
            X_train_pca,
            y_train_pca,
            device=device,
            num_epochs=num_epochs,
            input_dim=input_dim,
            num_classes=2,
            verbose=verbose,
        )

        # SPCA (MiniBatchSparsePCA default; version-agnostic: max_iter vs n_iter)
        if spca_mode.lower() == "minibatch":
            try:
                spca_transformer = MiniBatchSparsePCA(
                    n_components=n_comp,
                    alpha=1,
                    max_iter=200,
                    batch_size=256,
                    random_state=29,
                    verbose=verbose,
                )
            except TypeError:
                spca_transformer = MiniBatchSparsePCA(
                    n_components=n_comp,
                    alpha=1.0,
                    n_iter=200,
                    batch_size=256,
                    random_state=29,
                    verbose=verbose,
                )
        else:
            spca_transformer = SparsePCA(
                n_components=n_comp,
                random_state=29,
                max_iter=200,
                alpha=1,
                verbose=verbose,
            )

        classifier_spca_dict[n_comp] = setup_pipeline_classifier(
            spca_transformer,
            X_train_spca,
            y_train_spca,
            device=device,
            num_epochs=num_epochs * 2,
            input_dim=input_dim,
            num_classes=2,
            verbose=verbose,
        )

    print(f"Running benchmark tests for {attack_name} (ε ≤ 0.1)...")
    epsilons, pca_accuracies_dict, spca_accuracies_dict = benchmark_robustness(
        X_test,
        y_test,
        classifier_pca_dict,
        classifier_spca_dict,
        eps_list,
        n_total,
        attack_name=attack_name,
        attack_params=attack_params or {},
        save_samples=save_samples,
        verbose=verbose,
    )

    print("\nBenchmark Results (CIFAR-10 binary: airplane vs frog):")
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
        n_total,
        attack_name=attack_name,
    )


def unit_test_benchmark():
    # Quick smoke test
    eps_list = np.linspace(0.01, 0.1, 5)
    n_components_list = [128, 256]
    main(
        eps_list=eps_list,
        n_components_list=n_components_list,
        attack_name="FGSM",
        attack_params={},
        n_samples=6000,  # ~4800 train / 1200 test
        spca_mode="minibatch",
        spca_fit_samples=4000,
        pca_fit_samples=4000,
        num_epochs=8,
        save_samples=True,
    )


if __name__ == "__main__":
    test_mode = False
    if test_mode:
        unit_test_benchmark()
        raise SystemExit

    # -----------------------------
    # EXPERIMENT CONFIG (Binary Airplane vs Frog)
    # -----------------------------
    attack_names = ["FGSM", "PGD", "MIM"]

    n_components_list = [100, 150, 200]

    eps_list = np.arange(0.01, 0.101, 0.005)

    for attack_name in attack_names:
        print(f"\n\n=== Running benchmark for {attack_name} ===")
        print(f"Epsilon values: {[round(i,3) for i in eps_list]}")
        print(f"n_components_list: {n_components_list}")
        print("n_samples: None (full binary set)")
        print("SPCA mode: minibatch, SPCA fit samples: 10000")
        print("PCA fit samples: None (full)")
        print("Num epochs (PCA): 20, (SPCA): 40")
        print("Save adversarial samples: True")
        print("=============================================\n")
        # Optional per-attack knobs:
        #   PGD:  max_iter=40, eps_step=eps*0.25
        #   MIM:  max_iter=10, eps_step=eps*0.10, decay=1.0
        attack_params = {
            # "batch_size": 128,
            # "norm": np.inf,
            # "eps_step": 0.01,
            # "eps_step_ratio": 0.25,
            # "max_iter": 40,
            # "decay": 1.0,  # (MIM only)
        }

        # Dataset size (None => full binary set: ~12k total (10k train / 2k test))
        n_samples = None

        # SPCA fitting shortcuts
        spca_mode = "minibatch"  # "minibatch" or "sparse"
        spca_fit_samples = 10000
        pca_fit_samples = None  # fit PCA on full train

        # Epochs
        num_epochs = 20

        main(
            eps_list=eps_list,
            n_components_list=n_components_list,
            attack_name=attack_name,
            attack_params=attack_params,
            n_samples=n_samples,
            spca_mode=spca_mode,
            spca_fit_samples=spca_fit_samples,
            pca_fit_samples=pca_fit_samples,
            num_epochs=num_epochs,
            save_samples=True,
            verbose=False,
        )
