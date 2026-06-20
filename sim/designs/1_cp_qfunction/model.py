"""Design 1: CP Q-Function — the Q-function IS a CP tensor, not a neural net.

Q(s,a) = Σᵣ uᵣ(s_row) · vᵣ(s_col) · wᵣ(s_type) · zᵣ(a)

Four CP factors: 3 spatial modes + 1 action mode.
No projection layer, no activations, no neural net.
"""

import math
import torch
import torch.nn as nn


class CPQFunction(nn.Module):
    def __init__(self, mode_dims, action_dim, rank):
        super().__init__()
        self.action_dim = action_dim
        self.mode_dims = tuple(mode_dims)
        self.rank = rank

        self.spatial_factors = nn.ParameterList([
            nn.Parameter(torch.empty(d, rank)) for d in mode_dims
        ])
        self.action_factor = nn.Parameter(torch.empty(action_dim, rank))
        self.bias = nn.Parameter(torch.zeros(action_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        scale = 1.0 / math.sqrt(math.prod(self.mode_dims))
        for f in self.spatial_factors:
            nn.init.uniform_(f, -scale, scale)
        nn.init.uniform_(self.action_factor, -scale, scale)

    def forward(self, x):
        if x.dim() == len(self.mode_dims):
            x = x.unsqueeze(0)
        batch = x.shape[0]

        # Progressive contraction over spatial modes -> (batch, rank)
        d0 = self.mode_dims[0]
        h = x.reshape(batch, d0, -1)                            # (batch, d0, rest)
        h = torch.einsum('bip, ir -> brp', h, self.spatial_factors[0])  # (batch, rank, rest)

        for n in range(1, len(self.mode_dims)):
            dn = self.mode_dims[n]
            if n < len(self.mode_dims) - 1:
                remaining = h.shape[2] // dn
                h = h.reshape(batch, self.rank, dn, remaining)
                h = torch.einsum('brip, ir -> brp', h, self.spatial_factors[n])
            else:
                h = h.reshape(batch, self.rank, dn)
                h = torch.einsum('brk, kr -> br', h, self.spatial_factors[n])  # (batch, rank)

        # Action factor: Q[b, a] = Σᵣ h[b, r] · z[a, r]
        q = torch.einsum('br, ar -> ba', h, self.action_factor)  # (batch, action_dim)
        return q + self.bias
