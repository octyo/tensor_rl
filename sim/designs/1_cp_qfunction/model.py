"""Design 1: CP Q-Function — the Q-function IS a CP tensor, not a neural net.

Q(s,a) = sum_r U_r(s_row) * V_r(s_col) * W_r(s_type) * Z_r(a)

Uses CPEmbedding with out_features=action_dim as the entire model.
No activations, no hidden layers. Pure multilinear function.
"""

import torch.nn as nn
from models.tensor_layers import CPEmbedding


class CPQFunction(nn.Module):
    def __init__(self, mode_dims, action_dim, rank):
        super().__init__()
        self.action_dim = action_dim
        self.cp = CPEmbedding(mode_dims, out_features=action_dim, rank=rank)

    def forward(self, x):
        return self.cp(x)
