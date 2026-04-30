import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import parafac, tucker, tensor_train

# Set tensorly backend to PyTorch
tl.set_backend('pytorch')

class CPLinear(nn.Module):
    """
    A linear layer approximated using CP decomposition.
    Weight tensor W (out_features, in_features) is decomposed as sum of rank rank-1 tensors.
    """
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Decomposed factors of the weight matrix W
        # CP decomposition models an order-N tensor, here N=2 (matrix).
        # W ~ U @ V.T, where U parameterizes (out_features, rank) and V (in_features, rank)
        self.factor_out = nn.Parameter(torch.Tensor(out_features, rank))
        self.factor_in = nn.Parameter(torch.Tensor(in_features, rank))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Glorot initialization equivalent
        stdv = 1. / (self.in_features ** 0.5)
        self.factor_in.data.uniform_(-stdv, stdv)
        self.factor_out.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch_size, in_features)
        # Using equivalent to x @ W^T = x @ (factor_out @ factor_in^T)^T = x @ factor_in @ factor_out^T
        out = x.matmul(self.factor_in).matmul(self.factor_out.t())
        out += self.bias
        return out


class TuckerLinear(nn.Module):
    """
    A linear layer approximated using Tucker decomposition.
    """
    def __init__(self, in_features: int, out_features: int, ranks: tuple):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # ranks should be a tuple of (rank_out, rank_in)
        self.rank_out, self.rank_in = ranks
        
        self.core = nn.Parameter(torch.Tensor(self.rank_out, self.rank_in))
        self.factor_out = nn.Parameter(torch.Tensor(out_features, self.rank_out))
        self.factor_in = nn.Parameter(torch.Tensor(in_features, self.rank_in))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.in_features ** 0.5)
        self.core.data.uniform_(-stdv, stdv)
        self.factor_in.data.uniform_(-stdv, stdv)
        self.factor_out.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_features)
        # W = factor_out @ core @ factor_in^T
        # y = x @ W^T = x @ factor_in @ core^T @ factor_out^T
        out = x.matmul(self.factor_in)
        out = out.matmul(self.core.t())
        out = out.matmul(self.factor_out.t())
        out += self.bias
        return out


class TTLinear(nn.Module):
    """
    A linear layer approximated using Tensor Train decomposition for 2D weight matrix.
    Note: For a pure matrix, TT reduces into two cores (similar to SVD or truncated linear).
    """
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        self.core1 = nn.Parameter(torch.Tensor(1, out_features, rank))
        self.core2 = nn.Parameter(torch.Tensor(rank, in_features, 1))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.in_features ** 0.5)
        self.core1.data.uniform_(-stdv, stdv)
        self.core2.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_features)
        # Contract cores: core1 is (1, out_features, rank) and core2 is (rank, in_features, 1)
        # W: (out_features, in_features) = (core1.squeeze(0)) @ (core2.squeeze(2))
        W = self.core1.squeeze(0).matmul(self.core2.squeeze(2))
        out = x.matmul(W.t()) + self.bias
        return out
