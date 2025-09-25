# SPCARobustness

[![python](https://img.shields.io/badge/python-3.10%2B-green)]()
[![pytorch](https://img.shields.io/badge/pytorch-2.2%2B-orange)]()

A unified framework for benchmarking PCA vs SparsePCA pipelines under adversarial attacks. This repository evaluates the robustness of dimensionality reduction techniques by building end-to-end differentiable pipelines and testing them against various adversarial attacks.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Windows PowerShell](#windows-powershell)
  - [WSL/Ubuntu](#wsluubuntu)
- [Quick Start](#quick-start)
  - [MNIST Example](#mnist-example)
  - [CIFAR-10 Binary Example](#cifar-10-binary-example)
- [Command Line Interface](#command-line-interface)
  - [Available Datasets](#available-datasets)
  - [Available Attacks](#available-attacks)
  - [Common Parameters](#common-parameters)
- [Advanced Usage](#advanced-usage)
  - [Model Caching](#model-caching)
  - [Performance Tuning](#performance-tuning)
  - [Attack Configuration](#attack-configuration)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)

## Overview

The framework builds end-to-end differentiable pipelines consisting of:

1. Fixed linear projection (PCA or SparsePCA) as the first layer
2. Fixed StandardScaler for normalization
3. Small trainable MLP classifier

Adversarial attacks are implemented using IBM's Adversarial Robustness Toolbox (ART), enabling gradients to flow through the fixed preprocessing layers. The system supports multiple datasets and attack types with comprehensive evaluation metrics.

## Requirements

- Python 3.10+ recommended
- pip 22+ recommended  
- Optional: CUDA-capable GPU with matching PyTorch build

All Python dependencies are listed in `requirements.txt`. Datasets are downloaded automatically:

- MNIST via `sklearn.fetch_openml`
- CIFAR-10 via `torchvision.datasets.CIFAR10` into `./cifar_data`

## Installation

### Windows PowerShell

```powershell
# Clone and navigate to the repository
git clone <repository-url>
cd SPCARobustness

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU acceleration with CUDA 12.1:

```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### WSL/Ubuntu

```bash
# Clone and navigate to the repository
git clone <repository-url>
cd SPCARobustness

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU acceleration with CUDA 12.1:

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### MNIST Example

Run MNIST experiments with all available attacks:

```powershell
python main.py --dataset mnist --attacks ALL
```

Run with specific attacks and custom parameters:

```powershell
python main.py --dataset mnist --attacks FGSM PGD MIM --n-components 100 150 200 --eps-start 0.01 --eps-end 0.2 --save-samples
```

### CIFAR-10 Binary Example

Run CIFAR-10 binary classification (airplane vs frog) with all attacks:

```powershell
python main.py --dataset cifar-binary --attacks ALL --eps-end 0.1
```

Run with specific configuration for faster execution:

```powershell
python main.py --dataset cifar-binary --attacks FGSM PGD MIM SQUARE --n-components 100 150 --eps-start 0.01 --eps-end 0.1 --n-samples 5000 --save-models
```

## Command Line Interface

### Available Datasets

- `mnist`: MNIST 10-class digit classification (28x28 grayscale)
- `cifar-binary`: CIFAR-10 binary classification (airplane vs frog, 32x32 RGB)

### Available Attacks

- `FGSM`: Fast Gradient Sign Method
- `PGD`: Projected Gradient Descent  
- `MIM`: Momentum Iterative Method
- `SQUARE`: Square Attack (black-box)

Use `--attacks ALL` to run all attacks, or specify individual attacks: `--attacks FGSM PGD MIM`

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset choice (`mnist`, `cifar-binary`) | `mnist` |
| `--attacks` | List of attacks or `ALL` | Uses `--attack` value |
| `--attack` | Single attack if `--attacks` not specified | `FGSM` |
| `--norm` | Attack norm (2 for L2, inf for L∞) | `2` |
| `--n-components` | PCA/SPCA component counts | `[100, 150, 200]` |
| `--eps-start` | Starting epsilon value | `0.01` |
| `--eps-end` | Ending epsilon value | `0.2` |
| `--eps-step` | Epsilon step size | `0.01` |
| `--n-samples` | Limit dataset size for speed | `None` (full dataset) |
| `--epochs` | Training epochs for MLP | `20` |
| `--save-samples` | Save adversarial sample visualizations | `False` |
| `--save-models` | Cache trained models to disk | `False` |

## Advanced Usage

### Model Caching

Enable model caching for faster repeated experiments:

```powershell
python main.py --dataset mnist --attacks ALL --save-models --models-dir cached_models
```

### Performance Tuning

For faster execution during development:

```powershell
python main.py --dataset mnist --attacks FGSM PGD --n-components 64 128 --n-samples 2000 --epochs 10 --attack-n-test 1000
```

### Attack Configuration

Fine-tune attack parameters:

```powershell
python main.py --dataset cifar-binary --attacks SQUARE --square-max-iter 500 --square-restarts 3 --attack-batch-size 128
```

## Output Files

The system generates several types of output files:

**Robustness Plots**:

- Format: `{dataset}_{attack}_norm_{norm}_eps_{start}_to_{end}_ncomp_{components}_nsamples_{n}.png`
- Example: `mnist_fgsm_norm_l2_eps_0.01_to_0.2_ncomp_100_to_200_nsamples_60000.png`

**Adversarial Sample Visualizations** (when `--save-samples` is enabled):

- Directory format: `adv_samples_{attack}_norm_{norm}_eps_{start}_to_{end}_ncomp_{components}_nsamples_{n}/`
- Contains side-by-side comparisons of clean vs adversarial examples

**Cached Models** (when `--save-models` is enabled):

- Directory: `models/` (or custom via `--models-dir`)
- Contains serialized PCA/SPCA transformations and trained classifiers

## Troubleshooting

**PyTorch Installation Issues**: Ensure you install a build matching your platform and CUDA version. See [PyTorch installation guide](https://pytorch.org/get-started/locally/).

**Memory Issues**: Reduce batch sizes or limit test samples:

```powershell
python main.py --attack-batch-size 64 --attack-n-test 2000
```

**Long Runtimes**: Use fewer components and smaller datasets for initial testing:

```powershell
python main.py --n-components 64 --n-samples 5000 --epochs 5
```

**CUDA Out of Memory**: Disable GPU or reduce batch sizes:

```powershell
# Force CPU usage
CUDA_VISIBLE_DEVICES="" python main.py --dataset mnist
```
