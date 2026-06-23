"""Design 2: Neural Network Weight Decomposition.

Sub-case A: Standard MLP with CPLinear/TuckerLinear weight factorization (flat input).
Sub-case B: TensorizedLinear operating on unflattened (7,7,3) tensor input via mode products.
"""

import math
import torch
import torch.nn as nn


class TensorizedLinear(nn.Module):
    """Y = X ×₁ A ×₂ B ×₃ C — independent linear map per mode.

    Input: (batch, d1, d2, ..., dN)
    Output: (batch, r1 * r2 * ... * rN)
    """
    def __init__(self, mode_dims, ranks):
        super().__init__()
        if isinstance(ranks, int):
            ranks = (ranks,) * len(mode_dims)
        self.mode_dims = tuple(mode_dims)
        self.ranks = tuple(ranks)
        self.factors = nn.ParameterList([
            nn.Parameter(torch.empty(d, r)) for d, r in zip(mode_dims, ranks)
        ])
        self.bias = nn.Parameter(torch.zeros(math.prod(ranks)))
        self._reset_parameters()

    def _reset_parameters(self):
        for f in self.factors:
            nn.init.kaiming_uniform_(f, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if x.dim() == len(self.mode_dims):
            x = x.unsqueeze(0)
        batch = x.shape[0]
        h = x.reshape(batch, *self.mode_dims)
        for n in range(len(self.mode_dims)):
            if n > 0:
                h = h.movedim(n + 1, 1)
            h = torch.tensordot(h, self.factors[n], dims=([1], [0]))
            h = h.movedim(-1, n + 1)
        return h.reshape(batch, -1) + self.bias


class TensorizedQNetwork(nn.Module):
    """Sub-case B: TensorizedLinear → ReLU → Linear → ReLU → Linear(action_dim).
    Input is NEVER flattened — operates on structured (7,7,3) tensor.
    """
    def __init__(self, mode_dims, action_dim, rank, hidden_size=128):
        super().__init__()
        self.action_dim = action_dim
        out_dim = rank ** len(mode_dims)
        self.tensor_layer = TensorizedLinear(mode_dims, rank)
        self.hidden = nn.Linear(out_dim, hidden_size)
        self.output = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        h = torch.relu(self.tensor_layer(x))
        h = torch.relu(self.hidden(h))
        return self.output(h)
