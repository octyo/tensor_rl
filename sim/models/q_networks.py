import torch
import torch.nn as nn
from .tensor_layers import CPLinear, TuckerLinear, TTLinear


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
                 network_type="standard", rank=4, tensorize_layers="all"):
        super().__init__()
        self.network_type = network_type

        # Total layers = one per hidden size + one output
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
                 network_type="standard", rank=4, tensorize_layers="all"):
        super().__init__()
        self.network_type = network_type
        self.action_dim = action_dim

        # Dueling: only hidden trunk layers are tensorizable (heads are always Linear)
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
        else:
            raise ValueError(f"Unknown network type: {net_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
