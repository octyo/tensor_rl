import torch
import torch.nn as nn
from .tensor_layers import (CPLinear, TuckerLinear, TTLinear, MPSLinear,
                            TTEmbedding, CPEmbedding, TuckerEmbedding)


def parse_tensorize_layers(spec: str, n_layers: int) -> set:
    """Convert a --tensorize_layers string to a set of layer indices to tensorize.

    Indexing: 0 = first hidden layer, 1 = second hidden, ..., n_layers-1 = output layer.
    'all'  -> every layer
    'none' -> no layers (equivalent to --network standard)
    '0,2'  -> only layers 0 and 2
    """
    if spec == "all":
        return set(range(n_layers))
    if spec == "none":
        return set()
    indices = set()
    for part in spec.split(","):
        idx = int(part.strip())
        if not (0 <= idx < n_layers):
            raise ValueError(
                f"Layer index {idx} out of range. "
                f"Valid indices: 0..{n_layers - 1} "
                f"(0=first hidden, {n_layers - 1}=output)"
            )
        indices.add(idx)
    return indices


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(128, 128),
                 network_type="standard", rank=4, tensorize_layers="all",
                 tt_dims=None):
        super().__init__()
        self.network_type = network_type
        self.action_dim = action_dim
        self._tt_dims = tt_dims

        n_layers = len(hidden_sizes) + 1
        self._tz = parse_tensorize_layers(str(tensorize_layers), n_layers)

        layers = []
        prev_dim = state_dim
        for layer_idx, h in enumerate(hidden_sizes):
            layers.append(self._build_layer(prev_dim, h, network_type, rank, layer_idx))
            layers.append(nn.ReLU())
            prev_dim = h

        output_idx = len(hidden_sizes)
        layers.append(self._build_layer(prev_dim, action_dim, network_type, rank, output_idx))

        self.model = nn.Sequential(*layers)

    def _build_layer(self, in_dim, out_dim, net_type, rank, layer_idx):
        if layer_idx not in self._tz or net_type == "standard":
            return nn.Linear(in_dim, out_dim)
        elif net_type == "cp":
            return CPLinear(in_dim, out_dim, rank=rank)
        elif net_type == "tucker":
            return TuckerLinear(in_dim, out_dim, ranks=(min(rank, out_dim), min(rank, in_dim)))
        elif net_type == "tt":
            return TTLinear(in_dim, out_dim, rank=rank)
        elif net_type == "mps":
            return MPSLinear(in_dim, out_dim, rank=rank, tt_dims=self._tt_dims)
        else:
            raise ValueError(f"Unknown network type: {net_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DuelingQNetwork(nn.Module):
    """Dueling architecture: shared trunk → separate value V(s) and advantage A(s,a) heads.
    Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))

    tensorize_layers indices refer to the trunk hidden layers only (0..len(hidden_sizes)-1).
    The value/advantage heads are always nn.Linear.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(128, 128),
                 network_type="standard", rank=4, tensorize_layers="all",
                 tt_dims=None):
        super().__init__()
        self.network_type = network_type
        self.action_dim = action_dim
        self._tt_dims = tt_dims

        n_trunk_layers = len(hidden_sizes)
        self._tz = parse_tensorize_layers(str(tensorize_layers), n_trunk_layers)

        trunk_layers = []
        prev_dim = state_dim
        for layer_idx, h in enumerate(hidden_sizes):
            trunk_layers.append(self._build_layer(prev_dim, h, network_type, rank, layer_idx))
            trunk_layers.append(nn.ReLU())
            prev_dim = h
        self.trunk = nn.Sequential(*trunk_layers)

        self.value_head = nn.Linear(prev_dim, 1)
        self.advantage_head = nn.Linear(prev_dim, action_dim)

    def _build_layer(self, in_dim, out_dim, net_type, rank, layer_idx):
        if layer_idx not in self._tz or net_type == "standard":
            return nn.Linear(in_dim, out_dim)
        elif net_type == "cp":
            return CPLinear(in_dim, out_dim, rank=rank)
        elif net_type == "tucker":
            return TuckerLinear(in_dim, out_dim, ranks=(min(rank, out_dim), min(rank, in_dim)))
        elif net_type == "tt":
            return TTLinear(in_dim, out_dim, rank=rank)
        elif net_type == "mps":
            return MPSLinear(in_dim, out_dim, rank=rank, tt_dims=self._tt_dims)
        else:
            raise ValueError(f"Unknown network type: {net_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class StructuredQNetwork(nn.Module):
    """Q-network with a genuine tensor-network first layer that preserves spatial structure.

    The input is kept as (batch, d1, d2, ..., dN) — e.g. (batch, 7, 7, 3) for MiniGrid.
    A TTEmbedding or CPEmbedding contracts over each mode independently, producing
    (batch, hidden_size). This is the genuine TN claim: spatial axes are treated as
    separate tensor modes, not flattened into an undifferentiated vector.

    Architecture:
        (batch, d1, d2, d3)
          → TTEmbedding / CPEmbedding  → (batch, hidden_size)   [structured TN layer]
          → ReLU
          → Linear(hidden_size, hidden_size)                     [standard hidden]
          → ReLU
          → Linear(hidden_size, action_dim)                      [output head]

    Only the first layer uses tensor structure. The rest is standard — we are testing
    whether structured input encoding alone changes sample/parameter efficiency.
    """

    def __init__(self, mode_dims: tuple, action_dim: int, hidden_size: int = 128,
                 embedding_type: str = "tt", rank: int = 4):
        super().__init__()
        self.mode_dims = tuple(mode_dims)
        self.action_dim = action_dim
        self.embedding_type = embedding_type

        if embedding_type == "tt":
            self.embedding = TTEmbedding(mode_dims, out_features=hidden_size, rank=rank)
        elif embedding_type == "cp":
            self.embedding = CPEmbedding(mode_dims, out_features=hidden_size, rank=rank)
        elif embedding_type == "tucker":
            self.embedding = TuckerEmbedding(mode_dims, out_features=hidden_size, ranks=rank)
        else:
            raise ValueError(f"embedding_type must be 'tt', 'cp', or 'tucker', got '{embedding_type}'")

        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output_head = nn.Linear(hidden_size, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handles both (d1,d2,d3) single-step and (batch,d1,d2,d3) batch inputs.
        # TTEmbedding / CPEmbedding internally unsqueeze when no batch dim is present.
        h = torch.relu(self.embedding(x))
        h = torch.relu(self.hidden(h))
        return self.output_head(h)
