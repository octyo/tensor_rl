"""True MPS (Matrix Product State) Q-network.

Unlike MPSLinear which parameterizes a weight matrix W as a TT-core chain
and then computes y = Wx + b, this module IS the tensor network: the input
tensor flows directly through connected MPS cores via contraction. No weight
matrix is materialized and there are no activation functions between cores.

Two variants are provided:

1. StructuredMPSQNetwork: operates on the natural (d1, d2, ..., dN) tensor
   structure of the observation (e.g. (7,7,3) for MiniGrid). Each spatial
   mode is one MPS site with physical dimension = mode size. The last core
   outputs Q-values. This is a multilinear function of the input modes.

2. FeatureMapMPSQNetwork: groups the flat input into chunks, applies a local
   feature map to each chunk, then contracts through MPS cores. Reduces the
   number of sites to avoid the vanishing gradient problem of per-scalar MPS.
"""

import math
import torch
import torch.nn as nn


class StructuredMPSQNetwork(nn.Module):
    """True MPS Q-network operating on structured observations.

    Takes input of shape (batch, d1, d2, ..., dN) and contracts through
    N MPS cores, one per spatial mode. The last core outputs action_dim
    Q-values directly.

    Core shapes:
        core[0]:   (1,    d0, rank)
        core[n]:   (rank, dn, rank)        for n = 1..N-2
        core[N-1]: (rank, d_{N-1}, action_dim)

    The contraction is:
        h = x[:, :, j2, ..., jN] @ core_0[:, :, :]   (contract mode 0)
        h = h contracted with core_1 over mode 1, etc.
        Q = final contraction gives (batch, action_dim)

    This is a multilinear function of the input — nonlinearity comes from
    the multi-mode multiplicative interactions, not from activation functions.

    Args:
        mode_dims:  Tuple of spatial dimensions, e.g. (7, 7, 3).
        action_dim: Number of actions.
        rank:       Bond dimension between adjacent cores.
    """

    def __init__(self, mode_dims: tuple, action_dim: int, rank: int = 4):
        super().__init__()
        self.mode_dims = tuple(mode_dims)
        self.action_dim = action_dim
        self.rank = rank
        N = len(mode_dims)

        cores = []
        for n, d in enumerate(mode_dims):
            r_left = 1 if n == 0 else rank
            r_right = action_dim if n == N - 1 else rank
            cores.append(nn.Parameter(torch.empty(r_left, d, r_right)))
        self.cores = nn.ParameterList(cores)
        self.bias = nn.Parameter(torch.zeros(action_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for core in self.cores:
            scale = 1.0 / math.sqrt(core.shape[1])
            nn.init.uniform_(core, -scale, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == len(self.mode_dims):
            x = x.unsqueeze(0)
        batch = x.shape[0]
        N = len(self.mode_dims)

        # x: (batch, d0, d1, ..., d_{N-1})
        # Contract mode 0: h[b, r] = sum_i x[b, i, ...] * core_0[0, i, r]
        d0 = self.mode_dims[0]
        h = x.reshape(batch, d0, -1)  # (batch, d0, rest)
        h = torch.einsum('bip, ir -> brp', h, self.cores[0].squeeze(0))

        for n in range(1, N - 1):
            dn = self.mode_dims[n]
            remaining = h.shape[2] // dn
            h = h.reshape(batch, self.rank, dn, remaining)
            h = torch.einsum('bsip, sir -> brp', h, self.cores[n])

        # Last core: (rank, d_{N-1}, action_dim)
        d_last = self.mode_dims[-1]
        h = h.reshape(batch, self.rank, d_last)
        h = torch.einsum('bsk, sko -> bo', h, self.cores[-1])

        return h + self.bias

    @property
    def parameter_count(self) -> int:
        return sum(c.numel() for c in self.cores) + self.bias.numel()


class FeatureMapMPSQNetwork(nn.Module):
    """True MPS Q-network with local feature map on grouped inputs.

    Groups the flat input vector into n_sites chunks, applies a learned
    linear embedding to each chunk (the "feature map"), then contracts
    through MPS cores. This keeps the number of sites manageable (avoiding
    the vanishing gradient of 147-site per-scalar MPS) while maintaining
    the tensor network structure.

    Architecture:
        1. Partition input (147,) into n_sites groups of size ~group_size
        2. Embed each group: phi_n = W_n @ x_group_n + b_n  -> (phys_dim,)
        3. MPS contraction over n_sites cores using phi as physical indices
        4. Output: Q-values of shape (batch, action_dim)

    The learned embedding replaces the fixed feature map and provides the
    nonlinearity (the embedding is linear per-group, but the MPS contraction
    makes the overall function multilinear in the embeddings).

    Args:
        input_dim:  Total input dimension (e.g. 147).
        action_dim: Number of actions.
        rank:       Bond dimension.
        n_sites:    Number of MPS sites (input groups). Default: auto.
        phys_dim:   Embedding dimension per site. Default: 8.
    """

    def __init__(self, input_dim: int, action_dim: int, rank: int = 4,
                 n_sites: int = None, phys_dim: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.rank = rank
        self.phys_dim = phys_dim

        if n_sites is None:
            n_sites = max(3, round(math.sqrt(input_dim)))
        self.n_sites = n_sites

        # Compute group sizes (distribute input_dim across n_sites)
        base = input_dim // n_sites
        remainder = input_dim % n_sites
        self.group_sizes = []
        for i in range(n_sites):
            self.group_sizes.append(base + (1 if i < remainder else 0))

        # Learned per-site embedding: each maps (group_size,) -> (phys_dim,)
        self.embeddings = nn.ModuleList()
        for gs in self.group_sizes:
            self.embeddings.append(nn.Linear(gs, phys_dim))

        # MPS cores
        N = n_sites
        cores = []
        for n in range(N):
            r_left = 1 if n == 0 else rank
            r_right = action_dim if n == N - 1 else rank
            cores.append(nn.Parameter(torch.empty(r_left, phys_dim, r_right)))
        self.cores = nn.ParameterList(cores)
        self.bias = nn.Parameter(torch.zeros(action_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # Cores: init so contraction preserves magnitude.
        # After embedding, phi has std ~1/sqrt(phys_dim) (from Linear init).
        # sum_d phi_d * core[:, d, :] should have spectral radius ~1.
        for core in self.cores:
            r_left, d, r_right = core.shape
            # Orthogonal-like: for each physical index, init as scaled identity
            with torch.no_grad():
                nn.init.normal_(core, 0, 0.01)
                r_min = min(r_left, r_right)
                for i in range(r_min):
                    for k in range(d):
                        core[i, k, i % r_right] += 1.0 / d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch = x.shape[0]

        # Split input into groups and embed
        chunks = torch.split(x, self.group_sizes, dim=1)
        phis = [emb(chunk) for emb, chunk in zip(self.embeddings, chunks)]
        # phis[n]: (batch, phys_dim)

        # Left-to-right MPS contraction
        # core[0]: (1, d, r) -> squeeze -> (d, r)
        h = torch.einsum('bd, dr -> br', phis[0], self.cores[0].squeeze(0))

        for n in range(1, self.n_sites - 1):
            h = torch.einsum('br, rds, bd -> bs', h, self.cores[n], phis[n])

        # Last core: (rank, d, action_dim)
        h = torch.einsum('br, rda, bd -> ba',
                         h, self.cores[-1], phis[-1])

        return h + self.bias

    @property
    def parameter_count(self) -> int:
        p = sum(c.numel() for c in self.cores) + self.bias.numel()
        p += sum(e.weight.numel() + e.bias.numel() for e in self.embeddings)
        return p
