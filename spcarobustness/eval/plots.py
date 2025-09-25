import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def plot_benchmark(
    epsilons: np.ndarray,
    pca_accuracies_dict: Dict[int, List[float]],
    spca_accuracies_dict: Dict[int, List[float]],
    n_components_list: List[int],
    n_samples: int,
    attack_name: str,
    norm_label: str,
    prefix: str = "",
):
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.get_cmap("viridis")
    norm = Normalize(min(n_components_list), max(n_components_list))

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
    ax.set_title(f"Accuracy vs. Attack Strength ({attack_name}, norm={norm_label})")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Number of Components")
    ax.legend()
    ax.grid(True)
    out_name = (
        f"{prefix}{attack_name.lower()}_norm_{norm_label}_"
        f"eps_{epsilons[1]}_to_{epsilons[-1]}_"
        f"ncomp_{min(n_components_list)}_to_{max(n_components_list)}_nsamples_{n_samples}.png"
    )
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def show_adversarial_samples_mnist(
    X: np.ndarray,
    X_adv_pcas: List[np.ndarray],
    X_adv_spcas: List[np.ndarray],
    eps_list: List[float],
    ncp: int,
    directory: str,
    attack_name: str,
    norm_label: str,
):
    n_eps = len(X_adv_pcas)
    fig, axes = plt.subplots(2, n_eps + 1, figsize=(1.8 * (n_eps + 1), 6))

    axes[0, 0].imshow(X.reshape(28, 28), cmap="gray")
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Original")
    axes[1, 0].imshow(X.reshape(28, 28), cmap="gray")
    axes[1, 0].axis("off")
    axes[1, 0].set_title("Original")

    for i, (xa, xb) in enumerate(zip(X_adv_pcas, X_adv_spcas)):
        axes[0, i + 1].imshow(xa.reshape(28, 28), cmap="gray")
        axes[0, i + 1].axis("off")
        axes[0, i + 1].set_title(f"ε={eps_list[i]:.2f}")
        axes[1, i + 1].imshow(xb.reshape(28, 28), cmap="gray")
        axes[1, i + 1].axis("off")
        axes[1, i + 1].set_title(f"ε={eps_list[i]:.2f}")

    axes[0, 0].set_ylabel("PCA")
    axes[1, 0].set_ylabel("SPCA")
    plt.suptitle(
        f"{attack_name} (norm={norm_label}): Adv Samples (n_components={ncp})", y=0.98
    )
    plt.tight_layout()
    os.makedirs(directory, exist_ok=True)
    out_png = (
        f"{directory}/mnist_adversarial_samples_{attack_name.lower()}_norm_{norm_label}_"
        f"{eps_list[0]:.2f}_to_{eps_list[-1]:.2f}_ncomp_{ncp}.png"
    )
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def show_adversarial_samples_cifar_binary(
    X: np.ndarray,
    X_adv_pcas: List[np.ndarray],
    X_adv_spcas: List[np.ndarray],
    eps_list: List[float],
    ncp: int,
    directory: str,
    attack_name: str,
    norm_label: str,
):
    n_eps = len(X_adv_pcas)
    fig, axes = plt.subplots(2, n_eps + 1, figsize=(2.0 * (n_eps + 1), 6))

    def to_img(arr):
        return arr.reshape(3, 32, 32).transpose(1, 2, 0)

    axes[0, 0].imshow(to_img(X))
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Original")
    axes[1, 0].imshow(to_img(X))
    axes[1, 0].axis("off")
    axes[1, 0].set_title("Original")

    for i, (xa, xb) in enumerate(zip(X_adv_pcas, X_adv_spcas)):
        axes[0, i + 1].imshow(to_img(xa))
        axes[0, i + 1].axis("off")
        axes[0, i + 1].set_title(f"ε={eps_list[i]:.2f}")
        axes[1, i + 1].imshow(to_img(xb))
        axes[1, i + 1].axis("off")
        axes[1, i + 1].set_title(f"ε={eps_list[i]:.2f}")

    axes[0, 0].set_ylabel("PCA")
    axes[1, 0].set_ylabel("SPCA")
    plt.suptitle(
        f"{attack_name} (norm={norm_label}): Adv Samples (Airplane vs Frog, n_components={ncp})",
        y=0.98,
    )
    plt.tight_layout()
    os.makedirs(directory, exist_ok=True)
    out_png = (
        f"{directory}/cifar10_binary_airplane_frog_adversarial_samples_"
        f"{attack_name.lower()}_norm_{norm_label}_{eps_list[0]:.3f}_to_{eps_list[-1]:.3f}_ncomp_{ncp}.png"
    )
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
