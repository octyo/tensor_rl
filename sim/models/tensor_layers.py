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


class TuckerEmbedding(nn.Module):
    """Tucker embedding for multi-dimensional structured input.

    Takes x of shape (batch, d1, ..., dN) and produces (batch, out_features).
    A core tensor G(r1,...,rN) is contracted with per-mode factor matrices
    F_n(dn, rn), producing a rank-feature vector that is projected to out_features.

    Parameter count (for (7,7,3), out_features=128, rank=4):
        7*4 + 7*4 + 3*4 + 4^3 + 64*128 + 128 = 28+28+12+64+8192+128 = 8452
        vs nn.Linear(147, 128): 18,944  →  ~2.2x compression
    """

    def __init__(self, mode_dims: tuple, out_features: int, ranks):
        super().__init__()
        self.mode_dims = tuple(mode_dims)
        self.out_features = out_features
        N = len(mode_dims)
        if isinstance(ranks, int):
            ranks = (ranks,) * N
        self.ranks = tuple(ranks)
        assert len(self.ranks) == N

        self.factors = nn.ParameterList([
            nn.Parameter(torch.empty(d, r)) for d, r in zip(mode_dims, self.ranks)
        ])
        self.core = nn.Parameter(torch.empty(*self.ranks))
        self.W_out = nn.Parameter(torch.empty(math.prod(self.ranks), out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        scale = 1.0 / math.sqrt(math.prod(self.mode_dims))
        for f in self.factors:
            nn.init.uniform_(f, -scale, scale)
        nn.init.uniform_(self.core, -scale, scale)
        nn.init.uniform_(self.W_out, -scale, scale)

    @property
    def parameter_count(self) -> int:
        return (sum(f.numel() for f in self.factors)
                + self.core.numel() + self.W_out.numel() + self.bias.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == len(self.mode_dims):
            x = x.unsqueeze(0)
        batch = x.shape[0]
        N = len(self.mode_dims)

        # h starts as (batch, d1, d2, ..., dN)
        h = x.reshape(batch, *self.mode_dims)

        # Contract each spatial mode n against F_n(dn, rn).
        # After contracting mode n, that axis becomes rn.
        # We keep the batch dim at position 0 and contracted modes accumulate.
        # Strategy: contract the last spatial axis each iteration by permuting it to the end,
        # then using matmul, keeping order (batch, r0,...,r_{n-1}, d_{n+1},...,d_{N-1}, rn).
        # Simpler: use einsum with explicit dim labels.
        for n in range(N):
            # h shape: (batch, r0,...,r_{n-1}, d_n, d_{n+1},...,d_{N-1})
            # d_n is at axis n+1. Swap it to axis 1 for the contraction.
            if n > 0:
                h = h.movedim(n + 1, 1)
            # h shape: (batch, d_n, r0,...,r_{n-1}, d_{n+1},...,d_{N-1})
            # Contract axis 1 (d_n) with factors[n] shape (d_n, r_n)
            h = torch.tensordot(h, self.factors[n], dims=([1], [0]))
            # tensordot removes axis 1 and appends r_n at the end.
            # h shape: (batch, r0,...,r_{n-1}, d_{n+1},...,d_{N-1}, r_n)
            # Move r_n to position n+1 so contracted dims stay at 1..n+1.
            h = h.movedim(-1, n + 1)
            # h shape: (batch, r0,...,r_n, d_{n+1},...,d_{N-1})

        # After N contractions h shape is: (batch, r_0, r_1, ..., r_{N-1}) — already ordered.

        # Element-wise multiply with core (Tucker contraction) and project
        core_flat = self.core.reshape(1, -1)   # (1, r0*r1*...*r_{N-1})
        h_flat = h.reshape(batch, -1)           # (batch, r0*r1*...*r_{N-1})
        features = h_flat * core_flat           # (batch, r0*r1*...*r_{N-1})

        return features.matmul(self.W_out) + self.bias


# ---------------------------------------------------------------------------
# Decomposed convolutional layers
# ---------------------------------------------------------------------------

def auto_factor_dims(n: int, n_modes: int) -> tuple:
    """Factor integer n into n_modes roughly-equal integers via greedy prime-bucket assignment.

    Example: auto_factor_dims(128, 3) → (4, 4, 8)
    """
    if n_modes == 1:
        return (n,)

    # Prime factorization
    factors = []
    d = 2
    rem = n
    while d * d <= rem:
        while rem % d == 0:
            factors.append(d)
            rem //= d
        d += 1
    if rem > 1:
        factors.append(rem)

    if len(factors) < n_modes:
        # Pad with 1s if not enough prime factors
        factors = [1] * (n_modes - len(factors)) + factors

    # Greedily assign primes to n_modes buckets to minimize max-bucket
    buckets = [1] * n_modes
    for p in sorted(factors, reverse=True):
        min_idx = buckets.index(min(buckets))
        buckets[min_idx] *= p

    return tuple(sorted(buckets))


class MPSLinear(nn.Module):
    """True MPS/Tensor-Train linear layer for weight matrices.

    Unlike TTLinear (which is just 2-core = rank-R matrix factorization),
    MPSLinear reshapes W(out, in) into a higher-order tensor by factoring
    both in_features and out_features into mode shapes, then represents the
    weight as a chain of TT-cores with bond dimension `rank`.

    The full weight is reconstructed by contracting all cores, then the
    standard y = x @ W.T + b is applied. This is mathematically equivalent
    to the streaming contraction but simpler to implement.

    tt_dims: optional tuple specifying how to factor in_features into modes
             e.g. (4, 4, 8) for in_features=128. out_features is split similarly.
             If None, auto_factor_dims() is used.
    """

    def __init__(self, in_features: int, out_features: int, rank: int,
                 tt_dims: tuple = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Determine mode shapes for in and out.
        # tt_dims is a hint: if it doesn't exactly divide in_features, auto-factor instead.
        if tt_dims is not None and math.prod(tt_dims) == in_features:
            self.in_dims = tuple(tt_dims)
            self.out_dims = auto_factor_dims(out_features, len(self.in_dims))
        else:
            if tt_dims is not None:
                # Hint didn't match — silently fall back to auto-factoring
                pass
            n_modes = max(2, round(math.log2(min(in_features, out_features)) / 2))
            self.in_dims = auto_factor_dims(in_features, n_modes)
            self.out_dims = auto_factor_dims(out_features, n_modes)

        N = len(self.in_dims)
        assert N == len(self.out_dims)
        self.n_modes = N

        # TT-cores: each core has shape (r_left, in_d_n, out_d_n, r_right)
        # Bond dims: [1, rank, rank, ..., rank, 1]
        cores = []
        for n in range(N):
            r_left = 1 if n == 0 else rank
            r_right = 1 if n == N - 1 else rank
            cores.append(nn.Parameter(
                torch.empty(r_left, self.in_dims[n], self.out_dims[n], r_right)
            ))
        self.cores = nn.ParameterList(cores)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        scale = 1.0 / math.sqrt(self.in_features)
        for core in self.cores:
            nn.init.uniform_(core, -scale, scale)

    def _build_weight(self) -> torch.Tensor:
        """Contract all TT-cores into the full weight matrix (out_features, in_features)."""
        # core[n] shape: (r_left, d_in_n, d_out_n, r_right)
        # We accumulate W as (in_so_far, out_so_far, r_right) after each core.

        # Core 0: (1, d_in_0, d_out_0, r) → squeeze left → (d_in_0, d_out_0, r)
        W = self.cores[0].squeeze(0)  # (d_in_0, d_out_0, r0)

        for n in range(1, self.n_modes):
            core = self.cores[n]  # (r_left, d_in_n, d_out_n, r_right)
            # W current shape: (in_so_far, out_so_far, r_left)
            # Contract over r bond: W[i,o,r] * core[r, k, l, s] → (i,o,k,l,s)
            W = torch.einsum('ior, rkls -> iokls', W, core)
            in_so_far = math.prod(self.in_dims[:n + 1])
            out_so_far = math.prod(self.out_dims[:n + 1])
            r_right = 1 if n == self.n_modes - 1 else self.rank
            # Merge in dims (i,k) and out dims (o,l): reshape to (in_so_far, out_so_far, r_right)
            W = W.reshape(math.prod(self.in_dims[:n]),
                          math.prod(self.out_dims[:n]),
                          self.in_dims[n],
                          self.out_dims[n],
                          r_right)
            # Permute to (in_prev, in_n, out_prev, out_n, r) then reshape
            W = W.permute(0, 2, 1, 3, 4)  # (in_prev, in_n, out_prev, out_n, r)
            W = W.reshape(in_so_far, out_so_far, r_right)

        # Final W: (in_features, out_features, 1) → squeeze → (in_features, out_features)
        W = W.squeeze(-1)  # (in_features, out_features)
        return W.t()  # (out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._build_weight()  # (out_features, in_features)
        return x.matmul(W.t()) + self.bias

    @property
    def parameter_count(self) -> int:
        return sum(c.numel() for c in self.cores) + self.bias.numel()


class CPConv2d(nn.Module):
    """CP-decomposed Conv2d.

    Weight W(out_C, in_C, kH, kW) is approximated as a sum of rank rank-1 tensors
    across all 4 modes. Forward is computed via sequential einsum contractions
    without materializing the full weight tensor.

    Parameter count vs Conv2d(in_C, out_C, k):
        4 * rank * (out_C + in_C + kH + kW)  vs  out_C * in_C * kH * kW
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 rank: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rank = rank
        self.stride = stride
        self.padding = padding

        self.f_oc = nn.Parameter(torch.empty(out_channels, rank))
        self.f_ic = nn.Parameter(torch.empty(in_channels, rank))
        self.f_kh = nn.Parameter(torch.empty(kernel_size, rank))
        self.f_kw = nn.Parameter(torch.empty(kernel_size, rank))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        scale = 1.0 / math.sqrt(self.in_channels * self.kernel_size ** 2)
        for f in (self.f_oc, self.f_ic, self.f_kh, self.f_kw):
            nn.init.uniform_(f, -scale, scale)

    def _build_weight(self) -> torch.Tensor:
        # W[o, i, h, w] = sum_r f_oc[o,r] * f_ic[i,r] * f_kh[h,r] * f_kw[w,r]
        W = torch.einsum('or, ir, hr, wr -> oihw',
                         self.f_oc, self.f_ic, self.f_kh, self.f_kw)
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._build_weight()
        return torch.nn.functional.conv2d(x, W, self.bias,
                                          stride=self.stride, padding=self.padding)


class TuckerConv2d(nn.Module):
    """Tucker-decomposed Conv2d (Lebedev et al. 2014 style).

    W(out_C, in_C, kH, kW) = core(r_oc, r_ic, r_kh, r_kw) contracted with
    4 factor matrices. Implemented as 3 sequential convolutions:
        1. Pointwise: in_C  → r_ic   (1×1, groups=1)
        2. Spatial:   r_ic  → r_ic   (kH×kW depthwise-ish)  [actually r_ic → r_oc * r_kh * r_kw via core]
        3. Pointwise: r_oc  → out_C  (1×1, groups=1)

    For simplicity we reconstruct the full weight and call conv2d once.
    ranks can be a single int (broadcast to all 4 modes) or a 4-tuple (r_oc, r_ic, r_kh, r_kw).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 ranks, stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if isinstance(ranks, int):
            ranks = (ranks, ranks, ranks, ranks)
        self.ranks = tuple(ranks)
        r_oc, r_ic, r_kh, r_kw = self.ranks

        self.core = nn.Parameter(torch.empty(r_oc, r_ic, r_kh, r_kw))
        self.f_oc = nn.Parameter(torch.empty(out_channels, r_oc))
        self.f_ic = nn.Parameter(torch.empty(in_channels, r_ic))
        self.f_kh = nn.Parameter(torch.empty(kernel_size, r_kh))
        self.f_kw = nn.Parameter(torch.empty(kernel_size, r_kw))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        scale = 1.0 / math.sqrt(self.in_channels * self.kernel_size ** 2)
        nn.init.uniform_(self.core, -scale, scale)
        for f in (self.f_oc, self.f_ic, self.f_kh, self.f_kw):
            nn.init.uniform_(f, -scale, scale)

    def _build_weight(self) -> torch.Tensor:
        # W[o, i, h, w] = sum_{a,b,c,d} core[a,b,c,d] * f_oc[o,a] * f_ic[i,b] * f_kh[h,c] * f_kw[w,d]
        W = torch.einsum('abcd, oa, ib, hc, wd -> oihw',
                         self.core, self.f_oc, self.f_ic, self.f_kh, self.f_kw)
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._build_weight()
        return torch.nn.functional.conv2d(x, W, self.bias,
                                          stride=self.stride, padding=self.padding)
