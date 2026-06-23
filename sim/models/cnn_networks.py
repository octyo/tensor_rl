import math
import torch
import torch.nn as nn
from .tensor_layers import CPLinear, TuckerLinear, TTLinear, MPSLinear, CPConv2d, TuckerConv2d


class CNNBackbone(nn.Module):
    """Standard CNN feature extractor: Conv2d → ReLU blocks + Flatten.

    Expects input shape (batch, C, H, W).
    """

    def __init__(self, in_channels: int, hidden_channels=(32, 64),
                 kernel_sizes=(3, 3), strides=(1, 1)):
        super().__init__()
        layers = []
        ch_in = in_channels
        for ch_out, k, s in zip(hidden_channels, kernel_sizes, strides):
            layers.append(nn.Conv2d(ch_in, ch_out, kernel_size=k, stride=s, padding=k // 2))
            layers.append(nn.ReLU())
            ch_in = ch_out
        layers.append(nn.Flatten())
        self.net = nn.Sequential(*layers)
        self._out_ch = ch_in
        self._kernel_sizes = kernel_sizes
        self._strides = strides

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def output_dim(self, obs_shape: tuple) -> int:
        """Compute flattened output size for a given (C, H, W) obs_shape."""
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            return self.forward(dummy).shape[1]


class TensorizedCNNBackbone(nn.Module):
    """CNN backbone with CP- or Tucker-decomposed conv layers.

    Expects input shape (batch, C, H, W).
    """

    def __init__(self, in_channels: int, hidden_channels=(32, 64),
                 kernel_sizes=(3, 3), strides=(1, 1),
                 rank: int = 4, cnn_decomp: str = "cp"):
        super().__init__()
        layers = []
        ch_in = in_channels
        for ch_out, k, s in zip(hidden_channels, kernel_sizes, strides):
            if cnn_decomp == "cp":
                layers.append(CPConv2d(ch_in, ch_out, kernel_size=k, rank=rank,
                                       stride=s, padding=k // 2))
            elif cnn_decomp == "tucker":
                layers.append(TuckerConv2d(ch_in, ch_out, kernel_size=k,
                                           ranks=min(rank, min(ch_in, ch_out)),
                                           stride=s, padding=k // 2))
            else:
                raise ValueError(f"cnn_decomp must be 'cp' or 'tucker', got '{cnn_decomp}'")
            layers.append(nn.ReLU())
            ch_in = ch_out
        layers.append(nn.Flatten())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def output_dim(self, obs_shape: tuple) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            return self.forward(dummy).shape[1]


class CNNQNetwork(nn.Module):
    """Q-network for spatial (image-like) observations.

    obs_shape: (C, H, W) — channels-first.

    cnn_mode:
        "backbone"   — standard CNN extracts features, then a factorized linear
                        (network_type in cp/tucker/tt/mps/standard) maps to Q-values.
        "tensorized" — CP/Tucker decomposed conv layers extract features,
                        then a standard linear maps to Q-values.

    Architecture:
        backbone:    CNNBackbone → [CPLinear/TuckerLinear/...] → action_dim
        tensorized:  TensorizedCNNBackbone → nn.Linear → action_dim
    """

    def __init__(self, obs_shape: tuple, action_dim: int, rank: int = 4,
                 network_type: str = "cp", cnn_mode: str = "backbone",
                 hidden_sizes=(128,), tt_dims: tuple = None):
        super().__init__()
        self.obs_shape = tuple(obs_shape)   # (C, H, W)
        self.action_dim = action_dim
        self.cnn_mode = cnn_mode

        # --- Build backbone ---
        if cnn_mode == "backbone":
            backbone = CNNBackbone(in_channels=obs_shape[0])
            feat_dim = backbone.output_dim(obs_shape)
            self.backbone = backbone
        elif cnn_mode == "tensorized":
            backbone = TensorizedCNNBackbone(in_channels=obs_shape[0],
                                             rank=rank, cnn_decomp=network_type
                                             if network_type in ("cp", "tucker") else "cp")
            feat_dim = backbone.output_dim(obs_shape)
            self.backbone = backbone
        else:
            raise ValueError(f"cnn_mode must be 'backbone' or 'tensorized', got '{cnn_mode}'")

        # --- Build head ---
        head_layers = []
        prev = feat_dim
        for h in hidden_sizes:
            head_layers.append(self._make_linear(prev, h, network_type, rank, tt_dims,
                                                 tensorize=(cnn_mode == "backbone")))
            head_layers.append(nn.ReLU())
            prev = h
        # Output layer is always standard
        head_layers.append(nn.Linear(prev, action_dim))
        self.head = nn.Sequential(*head_layers)

    @staticmethod
    def _make_linear(in_dim, out_dim, network_type, rank, tt_dims, tensorize=True):
        if not tensorize or network_type == "standard":
            return nn.Linear(in_dim, out_dim)
        elif network_type == "cp":
            return CPLinear(in_dim, out_dim, rank=rank)
        elif network_type == "tucker":
            return TuckerLinear(in_dim, out_dim,
                                ranks=(min(rank, out_dim), min(rank, in_dim)))
        elif network_type == "tt":
            return TTLinear(in_dim, out_dim, rank=rank)
        elif network_type == "mps":
            return MPSLinear(in_dim, out_dim, rank=rank, tt_dims=tt_dims)
        else:
            raise ValueError(f"Unknown network_type: {network_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (C, H, W) single obs — add batch dim
        if x.dim() == len(self.obs_shape):
            x = x.unsqueeze(0)
        features = self.backbone(x)
        return self.head(features)
