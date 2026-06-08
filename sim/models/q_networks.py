import torch
import torch.nn as nn
from .tensor_layers import CPLinear, TuckerLinear, TTLinear

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(128, 128), network_type="standard", rank=4):
        super().__init__()
        self.network_type = network_type
        
        layers = []
        prev_dim = state_dim
        
        for h in hidden_sizes:
            layers.append(self._build_layer(prev_dim, h, network_type, rank))
            layers.append(nn.ReLU())
            prev_dim = h
            
        # Final output layer
        layers.append(self._build_layer(prev_dim, action_dim, network_type, rank))
        
        self.model = nn.Sequential(*layers)

    def _build_layer(self, in_dim, out_dim, net_type, rank):
        if net_type == "standard":
            return nn.Linear(in_dim, out_dim)
        elif net_type == "cp":
            return CPLinear(in_dim, out_dim, rank=rank)
        elif net_type == "tucker":
            # For simplicity, using same rank for in and out (rank, rank)
            # In practice, usually out_dim and in_dim determine max rank, so we clamp if necessary
            r1 = min(rank, out_dim)
            r2 = min(rank, in_dim)
            return TuckerLinear(in_dim, out_dim, ranks=(r1, r2))
        elif net_type == "tt":
            return TTLinear(in_dim, out_dim, rank=rank)
        else:
            raise ValueError(f"Unknown network type: {net_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DuelingQNetwork(nn.Module):
    """Dueling architecture: shared trunk → separate value V(s) and advantage A(s,a) heads.
    Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(128, 128),
                 network_type="standard", rank=4):
        super().__init__()
        self.network_type = network_type
        self.action_dim = action_dim

        # Shared feature trunk (all hidden layers except last)
        trunk_layers = []
        prev_dim = state_dim
        for h in hidden_sizes:
            trunk_layers.append(self._build_layer(prev_dim, h, network_type, rank))
            trunk_layers.append(nn.ReLU())
            prev_dim = h
        self.trunk = nn.Sequential(*trunk_layers)

        # Value and advantage heads use standard linear for simplicity
        self.value_head = nn.Linear(prev_dim, 1)
        self.advantage_head = nn.Linear(prev_dim, action_dim)

    def _build_layer(self, in_dim, out_dim, net_type, rank):
        if net_type == "standard":
            return nn.Linear(in_dim, out_dim)
        elif net_type == "cp":
            from .tensor_layers import CPLinear
            return CPLinear(in_dim, out_dim, rank=rank)
        elif net_type == "tucker":
            from .tensor_layers import TuckerLinear
            r1 = min(rank, out_dim)
            r2 = min(rank, in_dim)
            return TuckerLinear(in_dim, out_dim, ranks=(r1, r2))
        elif net_type == "tt":
            from .tensor_layers import TTLinear
            return TTLinear(in_dim, out_dim, rank=rank)
        else:
            raise ValueError(f"Unknown network type: {net_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
