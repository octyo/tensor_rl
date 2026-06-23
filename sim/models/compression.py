"""Post-hoc weight compression for trained Q-networks.

compress_network() replaces each nn.Linear in a model with a factorized equivalent
(CPLinear, TuckerLinear, or MPSLinear) by decomposing the trained weight matrix.

Usage:
    compressed = compress_network(trained_model, method="cp", rank=4)
    compressed = compress_network(trained_model, method="mps", rank=4, tt_dims=(4, 4, 8))
    fine_tuned = finetune(compressed, env, args, n_episodes=50)
"""

import copy
import math
import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import parafac, tucker, tensor_train

from .tensor_layers import CPLinear, TuckerLinear, MPSLinear, auto_factor_dims

tl.set_backend('pytorch')


def _decompose_linear(weight: torch.Tensor, bias: torch.Tensor,
                      method: str, rank: int,
                      tt_dims: tuple = None) -> nn.Module:
    """Decompose one linear layer's weight into a factorized layer."""
    out_features, in_features = weight.shape
    W = weight.detach().float()

    if method == "cp":
        # parafac returns CPTensor with .weights (rank,) and .factors (unit-norm columns).
        # Reconstruction: W ≈ (factors[0] * weights) @ factors[1].T
        # Absorb weights into factor_out so CPLinear.forward (x @ f_in @ f_out.T) is exact.
        cp_tensor = parafac(W, rank=rank, init='svd', n_iter_max=100, tol=1e-6)
        factors  = cp_tensor.factors   # [out×R, in×R], unit-norm columns
        weights  = cp_tensor.weights   # (R,) scale factors
        layer = CPLinear(in_features, out_features, rank=rank)
        with torch.no_grad():
            layer.factor_out.copy_((factors[0] * weights.unsqueeze(0)).float())
            layer.factor_in.copy_(factors[1].float())
            if bias is not None:
                layer.bias.copy_(bias.detach().float())
            else:
                layer.bias.zero_()
        # Sanity: reconstruction error should be small
        W_hat = (layer.factor_out @ layer.factor_in.t())
        err = (W_hat - W).abs().max().item()
        if err > 0.5:
            print(f"  [compress] CP recon error={err:.4f} (rank={rank} may be too low)")
        return layer

    elif method == "tucker":
        core_t, factors = tucker(W, rank=[min(rank, out_features), min(rank, in_features)],
                                 init='svd', n_iter_max=100, tol=1e-6)
        rank_out, rank_in = core_t.shape
        layer = TuckerLinear(in_features, out_features, ranks=(rank_out, rank_in))
        with torch.no_grad():
            layer.core.copy_(core_t.float())
            layer.factor_out.copy_(factors[0].float())
            layer.factor_in.copy_(factors[1].float())
            if bias is not None:
                layer.bias.copy_(bias.detach().float())
            else:
                layer.bias.zero_()
        # Sanity: W ≈ factor_out @ core @ factor_in.T
        W_hat = layer.factor_out @ layer.core @ layer.factor_in.t()
        err = (W_hat - W).abs().max().item()
        if err > 0.5:
            print(f"  [compress] Tucker recon error={err:.4f} (rank={rank} may be too low)")
        return layer

    elif method in ("tt", "mps"):
        # Match MPSLinear's core format: N cores each (r_left, d_in_n, d_out_n, r_right).
        # Strategy: reshape W.T → (*in_dims, *out_dims), permute to interleaved
        # (d_in_0, d_out_0, ...), merge each (d_in_n, d_out_n) into one mode, then
        # apply tensor_train to get N cores of shape (r_left, d_in_n*d_out_n, r_right),
        # finally reshape each core to (r_left, d_in_n, d_out_n, r_right).
        if tt_dims is not None and math.prod(tt_dims) == in_features:
            in_dims = tuple(tt_dims)
            out_dims = auto_factor_dims(out_features, len(in_dims))
        else:
            # Try decreasing n_modes until the effective (achievable) rank >= requested.
            # auto_factor_dims can pad with 1s, which creates tiny merged modes that cap bonds.
            n_modes_base = max(2, round(math.log2(min(in_features, out_features)) / 2))
            in_dims, out_dims = None, None
            eff_rank = 1
            for nm in range(n_modes_base, 1, -1):
                _in  = auto_factor_dims(in_features, nm)
                _out = auto_factor_dims(out_features, nm)
                merged = [di * do for di, do in zip(_in, _out)]
                # max achievable bond at position i: min(prod_left, prod_right)
                er = rank
                for i in range(nm - 1):
                    left  = math.prod(merged[:i + 1])
                    right = math.prod(merged[i + 1:])
                    er = min(er, left, right)
                in_dims, out_dims, eff_rank = _in, _out, er
                if er >= rank:
                    break

        N = len(in_dims)
        merged_shape = [in_dims[n] * out_dims[n] for n in range(N)]
        # Clamp rank to what's actually achievable for this tensor shape
        eff_rank = rank
        for i in range(N - 1):
            eff_rank = min(eff_rank,
                           math.prod(merged_shape[:i + 1]),
                           math.prod(merged_shape[i + 1:]))

        # W.T: (in_features, out_features) → (*in_dims, *out_dims)
        W_nd = W.t().reshape(*in_dims, *out_dims)
        # Permute to interleaved: (d_in_0, d_out_0, d_in_1, d_out_1, ...)
        perm = []
        for i in range(N):
            perm.extend([i, N + i])
        W_interleaved = W_nd.permute(*perm).contiguous()
        # Merge each (d_in_n, d_out_n) pair into one mode
        W_merged = W_interleaved.reshape(*merged_shape)

        # TT decomposition: uniform bond = eff_rank
        tt_rank = [1] + [eff_rank] * (N - 1) + [1]
        tt_tensor = tensor_train(W_merged, rank=tt_rank)

        # Pass the in_dims we actually used for decomposition so MPSLinear uses the same modes.
        layer = MPSLinear(in_features, out_features, rank=eff_rank, tt_dims=in_dims)
        with torch.no_grad():
            for n, core in enumerate(tt_tensor.factors):
                # core: (r_left, d_in_n * d_out_n, r_right)
                r_l, _, r_r = core.shape
                reshaped = core.reshape(r_l, in_dims[n], out_dims[n], r_r).float()
                layer.cores[n].data.copy_(reshaped)
            if bias is not None:
                layer.bias.copy_(bias.detach().float())
            else:
                layer.bias.zero_()
        # Sanity: build weight from cores and compare
        W_hat = layer._build_weight()
        err = (W_hat - W).abs().max().item()
        if err > 0.5:
            print(f"  [compress] MPS recon error={err:.4f} (eff_rank={eff_rank})")
        return layer

    else:
        raise ValueError(f"Unknown compression method: {method!r}. "
                         f"Choose from 'cp', 'tucker', 'tt', 'mps'.")


def compress_network(model: nn.Module, method: str, rank: int,
                     tt_dims: tuple = None,
                     layer_filter=None) -> nn.Module:
    """Return a new model with all nn.Linear layers replaced by factorized equivalents.

    Args:
        model:        Trained model. Not modified in-place.
        method:       "cp" | "tucker" | "tt" | "mps"
        rank:         CP-rank / Tucker rank / TT bond dimension.
        tt_dims:      For method="mps"/"tt": optional mode shape tuple for in_features.
                      If None, auto_factor_dims() is used.
        layer_filter: Optional callable(name, layer) -> bool.
                      Return False to skip that layer. Default: compress all nn.Linear.

    Returns:
        Compressed deep copy of the model.
    """
    compressed = copy.deepcopy(model)

    # Collect (parent_module, attr_name, layer) for all nn.Linear targets
    replacements = []
    for name, module in compressed.named_modules():
        for attr_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                full_name = f"{name}.{attr_name}" if name else attr_name
                if layer_filter is None or layer_filter(full_name, child):
                    replacements.append((module, attr_name, child, full_name))

    for parent, attr_name, layer, full_name in replacements:
        try:
            new_layer = _decompose_linear(
                layer.weight, layer.bias, method, rank, tt_dims
            )
            setattr(parent, attr_name, new_layer)
        except Exception as e:
            # Skip layers too small to decompose at the requested rank
            print(f"  [compress_network] Skipping {full_name} ({layer.in_features}→"
                  f"{layer.out_features}): {e}")

    return compressed


def finetune(model: nn.Module, env, args, n_episodes: int = 50) -> nn.Module:
    """Fine-tune a compressed model for n_episodes using a fresh DQN agent.

    Args:
        model:      Compressed Q-network (nn.Module).
        env:        Gymnasium environment (already created, will be reset each episode).
        args:       Namespace with gamma, lr (or defaults from config).
        n_episodes: Number of fine-tuning episodes.

    Returns:
        The fine-tuned model (same object, modified in-place by the agent's optimizer).
    """
    import random
    import numpy as np
    from agents.dqn_agent import DQNAgent
    from config import LR, GAMMA, BUFFER_SIZE, BATCH_SIZE, REPLAY_MIN_SIZE

    lr = getattr(args, 'lr', LR)
    gamma = getattr(args, 'gamma', GAMMA)
    epsilon_decay = max(n_episodes * 5, 1000)

    agent = DQNAgent(
        model,
        lr=lr,
        gamma=gamma,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        replay_min_size=min(REPLAY_MIN_SIZE, 500),
        epsilon_start=0.3,
        epsilon_end=0.05,
        epsilon_decay_steps=epsilon_decay,
        device="cpu",
    )

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        state = obs.flatten().astype(np.float32) if hasattr(obs, 'flatten') else obs
        done = False
        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = (next_obs.flatten().astype(np.float32)
                          if hasattr(next_obs, 'flatten') else next_obs)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state

    return model
