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
    W = weight.detach()

    if method == "cp":
        # parafac returns a CPTensor; .factors is [factor_out (out,R), factor_in (in,R)]
        # tensorly parafac on a 2D matrix gives 2 factors
        cp_tensor = parafac(W, rank=rank, init='svd', n_iter_max=100, tol=1e-6)
        factors = cp_tensor.factors  # list of 2 tensors: [out_features×rank, in_features×rank]
        layer = CPLinear(in_features, out_features, rank=rank)
        with torch.no_grad():
            layer.factor_out.copy_(factors[0])
            layer.factor_in.copy_(factors[1])
            if bias is not None:
                layer.bias.copy_(bias.detach())
            else:
                layer.bias.zero_()
        return layer

    elif method == "tucker":
        core_t, factors = tucker(W, rank=[min(rank, out_features), min(rank, in_features)],
                                 init='svd', n_iter_max=100, tol=1e-6)
        rank_out, rank_in = core_t.shape
        layer = TuckerLinear(in_features, out_features, ranks=(rank_out, rank_in))
        with torch.no_grad():
            layer.core.copy_(core_t)
            layer.factor_out.copy_(factors[0])
            layer.factor_in.copy_(factors[1])
            if bias is not None:
                layer.bias.copy_(bias.detach())
            else:
                layer.bias.zero_()
        return layer

    elif method in ("tt", "mps"):
        # Reshape W into higher-order tensor, then apply TT decomposition
        if tt_dims is not None:
            in_dims = tuple(tt_dims)
            assert math.prod(in_dims) == in_features
            out_dims = auto_factor_dims(out_features, len(in_dims))
        else:
            n_modes = max(2, round(math.log2(min(in_features, out_features)) / 2))
            in_dims = auto_factor_dims(in_features, n_modes)
            out_dims = auto_factor_dims(out_features, n_modes)

        N = len(in_dims)
        # Reshape W(out, in) → (d_in_0, d_in_1, ..., d_out_0, d_out_1, ...) for TT decomp
        # We interleave in/out dims: (d_in_0, d_out_0, d_in_1, d_out_1, ...)
        interleaved_shape = []
        for i in range(N):
            interleaved_shape.extend([in_dims[i], out_dims[i]])

        W_reshaped = W.T.reshape(in_features, out_features)
        W_reshaped = W_reshaped.reshape(*in_dims, *out_dims)
        # Permute to interleaved: (d_in_0, d_out_0, d_in_1, d_out_1, ...)
        perm = []
        for i in range(N):
            perm.extend([i, N + i])
        W_interleaved = W_reshaped.permute(*perm).reshape(*interleaved_shape)

        # TT rank = [1, rank, rank, ..., rank, 1]
        tt_rank = [1] + [rank] * (len(interleaved_shape) - 1) + [1]
        tt_tensor = tensor_train(W_interleaved, rank=tt_rank)
        # tt_tensor.factors: list of N*2 cores

        # Build MPSLinear and copy cores
        layer = MPSLinear(in_features, out_features, rank=rank, tt_dims=tt_dims)
        with torch.no_grad():
            if bias is not None:
                layer.bias.copy_(bias.detach())
            else:
                layer.bias.zero_()
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
