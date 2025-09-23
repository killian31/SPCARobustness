from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Fixed StandardScaler: (x - mean) / (scale + eps)."""

    def __init__(self, mean, scale, eps: float = 1e-12):
        super().__init__()
        self.mean: torch.Tensor
        self.scale: torch.Tensor
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, x):
        return (x - self.mean) / (self.scale + self.eps)


class ClassifierNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 10, hidden: Optional[int] = 128):
        super().__init__()
        hidden = hidden or 128
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class PipelineModel(nn.Module):
    def __init__(self, fixed_transform: nn.Module, fixed_scaler: nn.Module, classifier: nn.Module):
        super().__init__()
        self.fixed_transform = fixed_transform
        self.fixed_scaler = fixed_scaler
        self.classifier = classifier

    def forward(self, x):
        x = self.fixed_transform(x)
        x = self.fixed_scaler(x)
        x = self.classifier(x)
        return x


class ImageFlattenWrapper(nn.Module):
    """
    Wrap a base model that expects flattened vectors so it can accept images (NCHW or NHWC),
    by flattening the input in forward(). Exposes `classifier` attribute if present on base.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        # expose inner classifier if available (helps with saving/loading utilities)
        if hasattr(base, "classifier"):
            self.classifier = base.classifier

    def forward(self, x):
        # Flatten all but batch dim
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.base(x)
