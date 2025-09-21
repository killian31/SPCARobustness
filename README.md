# SPCARobustness

Benchmark PCA vs. SparsePCA pipelines for image classification under common adversarial attacks. This repo provides two runnable scripts:

- `all_attacks.py`: MNIST (10-class) with FGSM, PGD, and MIM attacks.
- `all_attacks_cifar.py`: CIFAR-10 (binary: airplane vs frog) with FGSM, PGD, and MIM attacks.

Both scripts build an end-to-end differentiable pipeline:

1) Fixed linear projection (PCA or SparsePCA) as the first layer
2) Fixed StandardScaler
3) Small trainable MLP classifier

Attacks are implemented with IBM ART (Adversarial Robustness Toolbox), enabling gradients to flow through the fixed preprocessing.


## Requirements

- Python 3.10+ recommended
- pip 22+ recommended
- Optional: CUDA-capable GPU and matching PyTorch build (CPU works too)

Python dependencies are listed in `requirements.txt` (PyTorch, torchvision, scikit-learn, matplotlib, tqdm, ART, etc.). Datasets are downloaded automatically:

- MNIST is fetched via `sklearn.fetch_openml`.
- CIFAR-10 is downloaded via `torchvision.datasets.CIFAR10` into `./cifar_data`.


## Installation

Create and activate a virtual environment, then install requirements.

### Windows PowerShell

```powershell
# From the repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Optional (GPU): install a CUDA-specific PyTorch if you have a supported NVIDIA GPU and drivers. Replace the torch/torchvision install with the matching wheel (example for CUDA 12.1):

```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### WSL / Ubuntu (bash)

```bash
# From the repo root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional (GPU with CUDA 12.1):

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```


## How to run

You can run either script directly. Both will train PCA and SPCA pipelines and then evaluate robustness across a sweep of epsilon values, saving plots and sample adversarial images.

### MNIST (10-class)

```powershell
# In the activated environment
python all_attacks.py
```

- Default attack is FGSM. To switch to PGD or MIM, open `all_attacks.py` and change the `attack_name` in the `__main__` section:
  - `attack_name = "FGSM"` | `"PGD"` | `"MIM"`
- You can also adjust:
  - `n_components_list` (number of PCA/SPCA components)
  - `eps_list` (attack strength sweep)
  - `attack_params` (e.g., `max_iter`, `eps_step` or `eps_step_ratio`, `decay`)
  - `n_samples` (set to a smaller integer for a quick run; `None` uses full MNIST)

Outputs:
- A plot like `fgsm_eps_0.01_to_0.2_ncomp_100_to_200_nsamples_<N>.png`
- If enabled (`save_samples=True`), per-setting adversarial panels saved under a folder like `adv_samples_fgsm_eps_.../`

Quick smoke test (optional): set `test_mode = True` at the top of `__main__` to run a short unit test.

### CIFAR-10 (Binary airplane vs frog)

```powershell
# In the activated environment
python all_attacks_cifar.py
```

- By default, this script loops over `FGSM`, `PGD`, and `MIM`, using a reasonable epsilon range (≤ 0.1 for natural images) and `n_components_list = [100, 150, 200]`.
- You can tune runtime vs. quality using parameters in the `__main__` block or `main()`:
  - `n_samples`: limit dataset size (e.g., `6000`) for faster iterations
  - `spca_mode`: `"minibatch"` (default, faster) or `"sparse"`
  - `spca_fit_samples` / `pca_fit_samples`: optionally fit PCA/SPCA on a subset for speed
  - `num_epochs`: training epochs for the MLP
  - `attack_params`: knobs like `max_iter`, `eps_step`/`eps_step_ratio`, `decay`

Outputs:
- A plot like `cifar10_binary_airplane_frog_fgsm_eps_0.01_to_0.1_ncomp_100_to_200_nsamples_<N>.png`
- If enabled (`save_samples=True`), per-setting adversarial panels saved under a folder like `cifar10_binary_airplane_frog_adv_fgsm_eps_.../`

Quick smoke test (optional): set `test_mode = True` near the bottom to run a short benchmark.


## Runtime tips

- GPU will significantly speed up training; both scripts automatically use CUDA if `torch.cuda.is_available()`.
- For quick iteration:
  - Reduce `n_components_list` (e.g., `[64]` or `[64, 128]`).
  - Set smaller `n_samples` (e.g., `2000` for MNIST; `6000` for CIFAR binary).
  - Decrease `num_epochs` in CIFAR or in `setup_pipeline_classifier()` for MNIST.
- ART attacks are vectorized; use `attack_params["batch_size"]` to tune memory vs. speed.


## Troubleshooting

- PyTorch install issues: ensure you install a build matching your platform and CUDA version. See https://pytorch.org/get-started/locally/ for exact commands.
- `torchvision`/`Pillow` import errors: upgrade `pip` and reinstall `torchvision` (it will bring a compatible `Pillow`).
- OpenML download hiccups for MNIST: transient network issues can occur; retry later or behind a stable connection.
- Long runtimes: start with fewer components, fewer epochs, or a smaller dataset; enable `test_mode` where available.


## Project structure

- `all_attacks.py` — MNIST PCA/SPCA vs FGSM/PGD/MIM
- `all_attacks_cifar.py` — CIFAR-10 binary (airplane vs frog) PCA/SPCA vs FGSM/PGD/MIM
- Output figures and adversarial panels are written to the repo root under descriptive file and folder names.
