import math
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

        self.factor_out = nn.Parameter(torch.Tensor(out_features, rank))
        self.factor_in = nn.Parameter(torch.Tensor(in_features, rank))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.in_features ** 0.5)
        self.factor_in.data.uniform_(-stdv, stdv)
        self.factor_out.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        # W ~ factor_out @ factor_in^T  →  y = x @ factor_in @ factor_out^T
        return x.matmul(self.factor_in).matmul(self.factor_out.t()) + self.bias


class TuckerLinear(nn.Module):
    """A linear layer approximated using Tucker decomposition."""
    def __init__(self, in_features: int, out_features: int, ranks: tuple):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
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
        # x: (batch, in_features)
        # W = factor_out @ core @ factor_in^T
        # y = x @ factor_in @ core^T @ factor_out^T
        out = x.matmul(self.factor_in)
        out = out.matmul(self.core.t())
        out = out.matmul(self.factor_out.t())
        return out + self.bias


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
        # W: (out_features, in_features) = core1.squeeze(0) @ core2.squeeze(2)
        W = self.core1.squeeze(0).matmul(self.core2.squeeze(2))
        return x.matmul(W.t()) + self.bias


# ---------------------------------------------------------------------------
# Structured multi-mode embedding layers
# ---------------------------------------------------------------------------

class TTEmbedding(nn.Module):
    """Tensor-Train (MPS) embedding for multi-dimensional structured input.

    Takes x of shape (batch, d1, d2, ..., dN) and produces (batch, out_features)
    by contracting a separate TT core over each mode axis.

    Core shapes:
        cores[0]:   (1,    d0, rank)           — left-boundary core
        cores[n]:   (rank, dn, rank)            — middle cores  (n = 1 .. N-2)
        cores[N-1]: (rank, d_{N-1}, out_features) — right-boundary core

    The bond dimensions between all adjacent cores are set uniformly to `rank`.
    The final bond dimension equals `out_features`, so no extra projection is needed.

    Contraction is performed left-to-right, progressively reducing each spatial
    mode while passing the bond dimension forward.

    Parameter count (for (7,7,3), out_features=128, rank=4):
        7*4 + 4*7*4 + 4*3*128 + 128 = 28 + 112 + 1536 + 128 = 1804
        vs nn.Linear(147, 128): 18,944  →  ~10.5x compression
    """

    def __init__(self, mode_dims: tuple, out_features: int, rank: int):
        super().__init__()
        self.mode_dims = tuple(mode_dims)
        self.out_features = out_features
        self.rank = rank
        N = len(mode_dims)
        assert N >= 2, "TTEmbedding requires at least 2 modes."

        cores = []
        for n, d in enumerate(mode_dims):
            if n == 0:
                cores.append(nn.Parameter(torch.empty(1, d, rank)))
            elif n == N - 1:
                cores.append(nn.Parameter(torch.empty(rank, d, out_features)))
            else:
                cores.append(nn.Parameter(torch.empty(rank, d, rank)))
        self.cores = nn.ParameterList(cores)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        scale = 1.0 / math.sqrt(math.prod(self.mode_dims))
        for core in self.cores:
            nn.init.uniform_(core, -scale, scale)

    @property
    def parameter_count(self) -> int:
        return sum(c.numel() for c in self.cores) + self.bias.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept both (d1,...,dN) and (batch, d1,...,dN)
        if x.dim() == len(self.mode_dims):
            x = x.unsqueeze(0)
        batch = x.shape[0]

        # Contract mode 0 with cores[0] of shape (1, d0, rank).
        # h[b, r, rest] = sum_{i} x[b, i, ...].flat * cores[0][0, i, r]
        d0 = self.mode_dims[0]
        h = x.reshape(batch, d0, -1)                          # (batch, d0, rest)
        h = torch.einsum('bip, ir -> brp', h, self.cores[0][0])  # (batch, rank, rest)

        # Contract each middle mode n with cores[n] of shape (rank, dn, rank).
        # h[b, r_new, rest] = sum_{r_old, j} h[b, r_old, j, rest] * core[r_old, j, r_new]
        for n in range(1, len(self.mode_dims) - 1):
            dn = self.mode_dims[n]
            remaining = h.shape[2] // dn
            h = h.reshape(batch, self.rank, dn, remaining)    # (batch, r, dn, rest)
            h = torch.einsum('bsip, sir -> brp', h, self.cores[n])  # (batch, rank, rest)

        # Contract last mode with cores[-1] of shape (rank, d_{N-1}, out_features).
        # h[b, o] = sum_{r, k} h[b, r, k] * cores[-1][r, k, o]
        d_last = self.mode_dims[-1]
        h = h.reshape(batch, self.rank, d_last)               # (batch, rank, d_last)
        h = torch.einsum('bsk, sko -> bo', h, self.cores[-1]) # (batch, out_features)

        return h + self.bias


class CPEmbedding(nn.Module):
    """CP (CANDECOMP/PARAFAC) embedding for multi-dimensional structured input.

    Takes x of shape (batch, d1, d2, ..., dN) and produces (batch, out_features).

    For each CP rank component r, the response to input x is the inner product of x
    with the rank-1 tensor factor_1[:,r] ⊗ factor_2[:,r] ⊗ ... ⊗ factor_N[:,r].
    This contracts each spatial mode against its factor matrix and takes the elementwise
    product across modes (Hadamard), giving a (batch, rank) feature vector.
    A final linear projection maps to out_features.

    Contraction for 3 modes:
        h[b, r] = sum_{i,j,k} x[b,i,j,k] * f1[i,r] * f2[j,r] * f3[k,r]
    Implemented progressively:
        h = einsum over mode 0  →  (batch, rank, d1, d2)
        h = einsum over mode 1  →  (batch, rank, d2)
        h = einsum over mode 2  →  (batch, rank)
        out = h @ W_out         →  (batch, out_features)

    Parameter count (for (7,7,3), out_features=128, rank=4):
        7*4 + 7*4 + 3*4 + 4*128 + 128 = 28+28+12+512+128 = 708
        vs nn.Linear(147, 128): 18,944  →  ~26.8x compression
    """

    def __init__(self, mode_dims: tuple, out_features: int, rank: int):
        super().__init__()
        self.mode_dims = tuple(mode_dims)
        self.out_features = out_features
        self.rank = rank

        self.factors = nn.ParameterList([
            nn.Parameter(torch.empty(d, rank)) for d in mode_dims
        ])
        self.W_out = nn.Parameter(torch.empty(rank, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        scale = 1.0 / math.sqrt(math.prod(self.mode_dims))
        for f in self.factors:
            nn.init.uniform_(f, -scale, scale)
        nn.init.uniform_(self.W_out, -scale, scale)

    @property
    def parameter_count(self) -> int:
        return (sum(f.numel() for f in self.factors)
                + self.W_out.numel() + self.bias.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept both (d1,...,dN) and (batch, d1,...,dN)
        if x.dim() == len(self.mode_dims):
            x = x.unsqueeze(0)
        batch = x.shape[0]

        # Contract mode 0: h[b, r, rest] = sum_i x[b, i, ...] * factors[0][i, r]
        d0 = self.mode_dims[0]
        h = x.reshape(batch, d0, -1)                               # (batch, d0, rest)
        h = torch.einsum('bip, ir -> brp', h, self.factors[0])     # (batch, rank, rest)

        # Contract each subsequent mode n: multiply-and-sum over mode dimension
        for n in range(1, len(self.mode_dims)):
            dn = self.mode_dims[n]
            remaining = h.shape[2] // dn if n < len(self.mode_dims) - 1 else 1
            if n < len(self.mode_dims) - 1:
                h = h.reshape(batch, self.rank, dn, remaining)     # (batch, rank, dn, rest)
                h = torch.einsum('brip, ir -> brp', h, self.factors[n])  # (batch, rank, rest)
            else:
                # Last mode: shape is (batch, rank, dn)
                h = h.reshape(batch, self.rank, dn)
                h = torch.einsum('brk, kr -> br', h, self.factors[n])   # (batch, rank)

        # Project to out_features
        return h.matmul(self.W_out) + self.bias
