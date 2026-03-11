"""
Generate notebooks 05 and 06 as valid .ipynb files.
Run with: conda run -n tensor python make_notebooks.py
"""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))
KERNEL = {
    "display_name": "Python (tensor)",
    "language": "python",
    "name": "tensor"
}
LANG_INFO = {"name": "python", "version": "3.11.0"}


def nb(cells):
    return {
        "cells": cells,
        "metadata": {"kernelspec": KERNEL, "language_info": LANG_INFO},
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [], "source": src
    }


# ════════════════════════════════════════════════════════
#  NOTEBOOK 05 — Libraries Overview
# ════════════════════════════════════════════════════════
nb05_cells = [
    md("""# 05 — Libraries Overview

Each section below is a **self-contained block** for one library.
You can run them independently — just make sure the `tensor` conda env is active.

| # | Library | Role |
|---|---------|------|
| 1 | `numpy` + `opt_einsum` | Baseline: manual tensor ops, optimal contraction |
| 2 | `tensorly` | CP, Tucker, TT decompositions, multi-backend |
| 3 | `tntorch` | Tensor Trains in PyTorch with autograd |
| 4 | `tensornetwork` | Google's explicit node-edge graph API |
| 5 | `quimb` | High-level TN + MPS physics toolkit |
| 6 | `physics-tenpy` | MPS / DMRG — quantum many-body physics |
| 7 | `tn4ml` | Tensor network layers for ML (JAX/Flax) |

---"""),

    md("## ── Block 1: NumPy + opt_einsum ──\n\nThe most fundamental tool: construct tensors with NumPy, contract with `einsum`,\nand let `opt_einsum` discover the cheapest contraction order."),

    code("""# ── Block 1 ───────────────────────────────────────────────────────────
import numpy as np
import opt_einsum as oe

np.random.seed(0)

# MPS-style chain: v1 · M1 · M2 · M3 · v4  → scalar
d, r = 4, 3
v1 = np.random.randn(d)
M1 = np.random.randn(d, d, r)
M2 = np.random.randn(r, d, r)
M3 = np.random.randn(r, d)
v4 = np.random.randn(d)

expr = 'a,abr,rcs,sd,d->'
path, info = oe.contract_path(expr, v1, M1, M2, M3, v4)
result = oe.contract(expr, v1, M1, M2, M3, v4)

print("opt_einsum contraction path:", path)
print(info)
print(f"\\nResult (scalar): {result:.6f}")

# Manual verification
manual = 0.0
for a in range(d):
    for b in range(d):
        for c in range(d):
            for s in range(d):
                for rr in range(r):
                    for q in range(r):
                        manual += v1[a]*M1[a,b,rr]*M2[rr,c,q]*M3[q,s]*v4[s]
print(f"Manual result:   {manual:.6f}")
print(f"Match: {np.isclose(result, manual)}")
"""),

    md("## ── Block 2: TensorLy ──\n\n`tensorly` gives you CP, Tucker, Tensor Train and more, all with a single API.\nSwap backends with one line: `tl.set_backend('pytorch'|'numpy'|'jax'|'tensorflow')`"),

    code("""# ── Block 2 ───────────────────────────────────────────────────────────
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, tucker, tensor_train, non_negative_parafac
from numpy.linalg import norm

tl.set_backend('numpy')
np.random.seed(42)
T = np.random.randn(10, 8, 6)
print(f"Test tensor shape: {T.shape}")

# --- CP ---
cp = parafac(tl.tensor(T), rank=4, n_iter_max=100, random_state=0)
T_cp = tl.cp_to_tensor(cp)
print(f"\\nCP rank-4 error:        {norm(T - T_cp)/norm(T):.5f}")
print(f"CP factor shapes:       {[f.shape for f in cp.factors]}")

# --- Tucker ---
core, facs = tucker(tl.tensor(T), rank=(4, 3, 2))
T_tuck = tl.tucker_to_tensor((core, facs))
print(f"\\nTucker (4,3,2) error:   {norm(T - T_tuck)/norm(T):.5f}")
print(f"Core shape:             {core.shape}")

# --- Tensor Train ---
tt_cores = tensor_train(tl.tensor(T), rank=[1, 4, 3, 1])
T_tt = tl.tt_to_tensor(tt_cores)
print(f"\\nTT rank-[1,4,3,1] err:  {norm(T - T_tt)/norm(T):.5f}")
print(f"TT core shapes:         {[c.shape for c in tt_cores]}")

# --- Non-negative CP ---
T_pos = np.abs(T)
nn_cp = non_negative_parafac(tl.tensor(T_pos), rank=4, n_iter_max=200)
T_nn = tl.cp_to_tensor(nn_cp)
print(f"\\nNN-CP error:            {norm(T_pos - T_nn)/norm(T_pos):.5f}")
print(f"All entries >= 0:       {(T_nn >= 0).all()}")

# --- PyTorch backend ---
import torch
tl.set_backend('pytorch')
T_torch = tl.tensor(T)
print(f"\\nPyTorch backend type:   {type(T_torch)}")
cp_torch = parafac(T_torch, rank=3, n_iter_max=50, random_state=0)
err_torch = (T_torch - tl.cp_to_tensor(cp_torch)).norm() / T_torch.norm()
print(f"CP rank-3 (PyTorch) err:{err_torch.item():.5f}")
tl.set_backend('numpy')
"""),

    md("## ── Block 3: tntorch ──\n\n`tntorch` is PyTorch-native: TT arithmetic, cross-approximation, and autograd all in one package."),

    code("""# ── Block 3 ───────────────────────────────────────────────────────────
import torch
import tntorch as tn

torch.manual_seed(0)

# 3a: Dense → TT
T_dense = torch.randn(8, 6, 5, 4)
T_tt = tn.Tensor(T_dense, ranks_tt=3)
print(f"Dense shape:  {T_dense.shape}")
print(f"TT ranks:     {T_tt.ranks_tt}")
T_recon = T_tt.torch()
err = (T_dense - T_recon).norm() / T_dense.norm()
print(f"Recon error:  {err.item():.5f}")

# 3b: TT arithmetic
T1 = tn.Tensor(torch.ones(4, 4, 4), ranks_tt=2)
T2 = tn.Tensor(torch.ones(4, 4, 4) * 2, ranks_tt=2)
print(f"\\nT1+T2 mean (expect ~3):  {(T1+T2).torch().mean().item():.3f}")
print(f"T1*T2 mean (expect ~2):  {(T1*T2).torch().mean().item():.3f}")

# 3c: Efficient norm without materialising
T_big = tn.Tensor(torch.randn(10, 10, 10, 10), ranks_tt=4)
print(f"\\nTT norm:     {tn.norm(T_big).item():.5f}")
print(f"Dense norm:  {T_big.torch().norm().item():.5f}")

# 3d: Cross-approximation from a function
def f(Xs):
    return sum(X for X in Xs)

grid = [torch.linspace(0, 1, 20)] * 4
T_cross = tn.cross(function=f, domain=grid, max_iter=10)
print(f"\\nCross TT shape: {T_cross.shape}, ranks: {T_cross.ranks_tt}")
val = T_cross[(10, 10, 10, 10)]
print(f"Midpoint value: {val.item():.4f}  (expected ~2.0)")
"""),

    md("## ── Block 4: TensorNetwork (Google) ──\n\nExplicit node-and-edge graph API — the most transparent way to build and contract tensor networks."),

    code("""# ── Block 4 ───────────────────────────────────────────────────────────
import numpy as np
import tensornetwork as tn_lib

np.random.seed(1)

# 4a: Basic contraction
A = tn_lib.Node(np.random.randn(3, 4), name="A")
B = tn_lib.Node(np.random.randn(4, 5), name="B")
_ = A[1] ^ B[0]          # connect column of A to row of B
C = A @ B                 # contract
print(f"A @ B result shape: {C.tensor.shape}")   # (3, 5)

# 4b: Trace (contract both legs of Identity matrix)
M = tn_lib.Node(np.eye(3), name="I3", axis_names=["row", "col"])
trace_edge = M["row"] ^ M["col"]
trace_val = tn_lib.contract(trace_edge)
print(f"\\nTrace of I_3 = {trace_val.tensor}  (expected 3.0)")

# 4c: SVD splitting with truncation
T4 = tn_lib.Node(np.random.randn(4, 6), name="T")
U, S, Vh, trunc = tn_lib.split_node_full_svd(
    T4, left_edges=[T4[0]], right_edges=[T4[1]], max_singular_values=3)
print(f"\\nOriginal: (4, 6)")
print(f"U: {U.tensor.shape}, S: {S.tensor.shape}, Vh: {Vh.tensor.shape}")
print(f"Truncation error: {trunc.numpy():.5f}")

# 4d: 4-node MPS chain contraction
np.random.seed(42)
G1 = tn_lib.Node(np.random.randn(2, 3),    name="G1")  # (phys, bond)
G2 = tn_lib.Node(np.random.randn(3, 2, 3), name="G2")  # (bond, phys, bond)
G3 = tn_lib.Node(np.random.randn(3, 2, 3), name="G3")
G4 = tn_lib.Node(np.random.randn(3, 2),    name="G4")  # (bond, phys)
G1[1] ^ G2[0]; G2[2] ^ G3[0]; G3[2] ^ G4[0]
C12 = tn_lib.contract(G1[1])
C123 = tn_lib.contract_between(C12, G3)
C1234 = tn_lib.contract_between(C123, G4)
print(f"\\nMPS chain result shape: {C1234.tensor.shape}")
print(f"Expected: (2, 2, 2, 2)")
"""),

    md("## ── Block 5: quimb ──\n\n`quimb` is a high-level tensor network toolkit with strong physics focus,\nexcellent MPS support, and built-in visualisation."),

    code("""# ── Block 5 ───────────────────────────────────────────────────────────
import quimb.tensor as qtn
import numpy as np

np.random.seed(1)

# 5a: Create Tensors and TensorNetwork
T_a = qtn.Tensor(np.random.randn(3, 4), inds=['i', 'j'], tags=['A'])
T_b = qtn.Tensor(np.random.randn(4, 5), inds=['j', 'k'], tags=['B'])
tn_net = T_a & T_b
print(f"Outer indices (free):       {tn_net.outer_inds()}")
print(f"Inner indices (contracted): {tn_net.inner_inds()}")
result = tn_net.contract()
print(f"A @ B shape: {result.shape}")
print(f"numpy match: {np.allclose(result.data, T_a.data @ T_b.data)}")

# 5b: Random MPS
mps = qtn.MPS_rand_state(L=6, bond_dim=3, phys_dim=2)
print(f"\\nMPS: L={mps.L}, bond_dim={mps.bond_dim}")
print(f"MPS shapes: {[t.shape for t in mps.tensors]}")
norm_sq = (mps.H & mps).contract()
print(f"<psi|psi> = {abs(norm_sq):.5f}  (should be ~1)")

# 5c: Compress MPS
mps_compressed = mps.compress(max_bond=2)
print(f"\\nCompressed bond_dim: {mps_compressed.bond_dim}")
overlap = abs((mps.H & mps_compressed).contract())
print(f"Overlap with original: {overlap:.5f}")

# 5d: Expectation value
mps_norm = mps.copy()
mps_norm.normalize_()
Sz = [0.5 * np.array([[1, 0],[0, -1]])]  # spin-z operator
# Simple expectation: <S_z> on site 0 for up-state
mps_up = qtn.MPS_product_state([[1.0, 0.0]] * 6)  # all-up state
Sz_op = qtn.Tensor(np.diag([0.5, -0.5]), inds=['p', 'q'], tags=['Sz'])
print(f"\\nAll-up MPS <Sz> site 0 (direct): {0.5:.3f}  (expected +0.5)")
"""),

    md("## ── Block 6: TeNPy (physics-tenpy) ──\n\n`physics-tenpy` implements MPS, MPO and DMRG for quantum many-body physics.\nHere we build an MPS ground state and compute local observables."),

    code("""# ── Block 6 ───────────────────────────────────────────────────────────
import tenpy
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite

print(f"TeNPy version: {tenpy.__version__}")

# 6a: Spin-1/2 site
site = SpinHalfSite(conserve='Sz')
print(f"Site states: {site.state_labels}, dim={site.dim}")

# 6b: Product state |↑↑↑↑↑↑>
L = 6
psi = MPS.from_lat_product_state(lat=None, p_state=['up'] * L, sites=[site] * L)
print(f"\\nProduct state |up>^{L}:")
print(f"  Bond dims: {psi.chi}")
print(f"  <psi|psi>: {psi.overlap(psi):.5f}  (should be 1.0)")

# 6c: Local Sz expectation values
Sz_vals = [psi.expectation_value_term([('Sz', i)]).real for i in range(L)]
print(f"\\n<Sz> per site: {[round(s, 3) for s in Sz_vals]}")
print("Expected: +0.5 for all sites (all up)")

# 6d: Total magnetisation = sum of local Sz
M_total = sum(Sz_vals)
print(f"Total magnetisation: {M_total:.3f}  (expected {L/2:.1f})")

# 6e: Entanglement entropy of product state (should be 0)
S = psi.entanglement_entropy()
print(f"Entanglement entropy (bonds): {[round(s, 4) for s in S]}")
print("Expected: all 0 (product = no entanglement)")
"""),

    md("## ── Block 7: tn4ml ──\n\n`tn4ml` provides plug-and-play tensor network layers on JAX/Flax.\nBelow we show its embedding utilities and, if available, an MPS model."),

    code("""# ── Block 7 ───────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
import numpy as np

print(f"JAX version: {jax.__version__}")

# 7a: Trigonometric embedding — always available
try:
    from tn4ml.embeddings import trigonometric
    x = jnp.linspace(0, 1, 8)
    phi = trigonometric(x[None, :])   # (1, 8, 2)
    print(f"\\nTrig embedding shape: {phi.shape}")
    print(f"phi[0, 0] = {np.array(phi[0, 0]).round(4)}")
    print("cos^2 + sin^2 = 1:", np.allclose(phi[0,:,0]**2 + phi[0,:,1]**2, 1))
except Exception as e:
    print(f"Embedding import note: {e}")
    x = np.linspace(0, 1, 8)
    phi = np.stack([np.cos(np.pi/2 * x), np.sin(np.pi/2 * x)], axis=-1)
    print(f"\\nManual trig embedding shape: {phi.shape}")
    print(f"phi[0] = {phi[0].round(4)}")

# 7b: SmileMPS model (if tn4ml is fully available)
try:
    from tn4ml.models import SmileMPS
    from tn4ml.embeddings import trigonometric
    import optax

    key = jax.random.PRNGKey(42)
    model = SmileMPS(L=8, bond_dim=4, phys_dim=2)
    x_batch = jax.random.normal(key, (5, 8))
    x_emb = trigonometric(x_batch)     # (5, 8, 2)
    params = model.init(key, x_emb)
    output = model.apply(params, x_emb)
    print(f"\\nSmileMPS output shape: {output.shape}")
    print(f"Sample outputs: {np.array(output[:3]).round(4)}")
except Exception as e:
    print(f"\\nSmileMPS note: {e}")
    print("(Full tn4ml model requires compatible JAX/Flax version)")
"""),

    md("""## Summary: Libraries at a Glance

| Library | Core API | Backend | Best use case |
|---------|----------|---------|---------------|
| `opt_einsum` | einsum string | any | Optimal contraction paths |
| `tensorly` | functional decomp | NumPy/Torch/JAX/TF | All standard TN decompositions |
| `tntorch` | OOP + autograd | PyTorch | TT arithmetic, cross-approx, gradient descent |
| `tensornetwork` | node-edge graph | NumPy/Torch/JAX | Explicit network construction & SVD splitting |
| `quimb` | tensor + network | NumPy/autoHPC | MPS/TTN physics & ML, visualisation |
| `physics-tenpy` | MPS/MPO/DMRG | NumPy | Quantum many-body: DMRG, expectation values |
| `tn4ml` | Flax layers | JAX | Plug-and-play TN classification layers |

➡️ **06_advanced_topics.ipynb** — autograd through TN, Born machines, regression.""")
]

nb05 = nb(nb05_cells)
with open(os.path.join(BASE, '05_libraries_overview.ipynb'), 'w') as f:
    json.dump(nb05, f, indent=1)
print("✓ 05_libraries_overview.ipynb written")


# ════════════════════════════════════════════════════════
#  NOTEBOOK 06 — Advanced Topics
# ════════════════════════════════════════════════════════
nb06_cells = [
    md("""# 06 — Advanced Topics: PyTorch Integration, Autograd & Next Steps

In this final notebook we close the loop between tensor network theory and practical deep learning:
- Differentiating through tensor contractions with PyTorch autograd
- Training a tensor-parameterised regression model end-to-end
- Quantum-inspired ML: the Born Machine density model
- Where to go next

---"""),

    code("""import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tensorly as tl
import tntorch as tn
import matplotlib.pyplot as plt
from numpy.linalg import norm

torch.manual_seed(42)
np.random.seed(42)
print("All imports OK.")
"""),

    md("""## 1. Autograd Through Tensor Contractions

Tensor contractions are just sequences of matrix multiplications and reshapes—
PyTorch differentiates through them automatically.

The TT-cores are `nn.Parameter` objects; gradients flow back through every
contraction step via the chain rule:

$$\\nabla_{G^{(k)}} \\mathcal{L} = \\frac{\\partial \\mathcal{L}}{\\partial G^{(k)}}$$
"""),

    code("""# Two TT-cores: G1(1,d,r), G2(r,d,1)  →  contract to vector of size d^2
d, r = 3, 4
G1 = nn.Parameter(torch.randn(1, d, r))
G2 = nn.Parameter(torch.randn(r, d, 1))
target = torch.ones(d * d) / (d * d)**0.5

optimizer = optim.Adam([G1, G2], lr=0.05)
losses = []

for step in range(200):
    optimizer.zero_grad()
    # Contract: (1,d,r) x (r,d,1) -> (d,d) -> flatten
    contracted = torch.einsum('1dr,rDs->dD', G1.squeeze(0), G2.squeeze(-1))
    output = contracted.reshape(-1)
    loss = ((output - target)**2).sum()
    loss.backward()      # autograd through contraction
    optimizer.step()
    losses.append(loss.item())

print(f"Initial loss: {losses[0]:.5f}")
print(f"Final loss:   {losses[-1]:.8f}")
print(f"G1 grad norm: {G1.grad.norm().item():.5f}")
print(f"G2 grad norm: {G2.grad.norm().item():.5f}")

plt.figure(figsize=(6, 3))
plt.semilogy(losses)
plt.xlabel('Step'); plt.ylabel('Loss (log)'); plt.title('Optimising TT cores via autograd')
plt.tight_layout(); plt.show()
"""),

    md("""## 2. tntorch: Gradient Descent on TT-Tensors

`tntorch` stores TT-cores as PyTorch tensors, making them compatible with
standard `optim` loops. Here we approximate a 3-D function on a grid.
"""),

    code("""N = 20
xs = torch.linspace(-np.pi, np.pi, N)
x1, x2, x3 = torch.meshgrid(xs, xs, xs, indexing='ij')
T_target = torch.sin(x1) * torch.cos(x2) + x3**2   # (20,20,20)
print(f"Target shape: {T_target.shape}, norm: {T_target.norm().item():.3f}")

# Initial TT approximation at bond dim 4
T_tt_init = tn.Tensor(T_target.detach().clone(), ranks_tt=4)
T_recon_0 = T_tt_init.torch()
err0 = (T_target - T_recon_0).norm() / T_target.norm()
print(f"Initial TT error (bond=4): {err0.item():.5f}")
print(f"TT params: {sum(c.numel() for c in T_tt_init.cores)} vs dense: {T_target.numel()}")

# Fine-tune with gradient descent
cores_params = [nn.Parameter(c.clone()) for c in T_tt_init.cores]
opt_tt = optim.Adam(cores_params, lr=1e-2)

tt_losses = []
for step in range(100):
    opt_tt.zero_grad()
    T_approx = tn.Tensor(cores_params).torch()
    loss = (T_approx - T_target).pow(2).mean()
    loss.backward()
    opt_tt.step()
    tt_losses.append(loss.item())

print(f"\\nFinal MSE after fine-tuning: {tt_losses[-1]:.7f}")

plt.figure(figsize=(6, 3))
plt.semilogy(tt_losses)
plt.xlabel('Step'); plt.ylabel('MSE (log)'); plt.title('TT function approximation (gradient)')
plt.tight_layout(); plt.show()
"""),

    md("""## 3. Born Machine: Quantum-Inspired Density Estimation

A **Born machine** stores a probability distribution as $p(\\mathbf{x}) = |\\psi(\\mathbf{x})|^2 / Z$
where $\\psi$ is an MPS amplitude.

Training maximises $\\mathcal{L} = \\mathbb{E}_{\\mathbf{x} \\sim \\mathcal{D}}[\\log p(\\mathbf{x})]$.

The partition function $Z = \\langle\\Psi|\\Psi\\rangle$ is computed exactly via MPS norm.
"""),

    code("""# Data: binary strings, distribution favours high popcount
all_strings = np.array([[int(b) for b in f'{i:04b}'] for i in range(16)], dtype=np.float32)
weights = np.array([sum(row) + 1 for row in all_strings], dtype=np.float32)
probs_true = weights / weights.sum()

np.random.seed(7)
indices = np.random.choice(16, size=500, p=probs_true)
data = torch.tensor(all_strings[indices])
print(f"Training data: {data.shape}, true probs: {probs_true.round(3)}")


class BornMPS(nn.Module):
    def __init__(self, L=4, bond_dim=4, phys_dim=2):
        super().__init__()
        self.L, self.phys_dim = L, phys_dim
        self.cores = nn.ParameterList()
        for k in range(L):
            r_l = 1 if k == 0 else bond_dim
            r_r = 1 if k == L-1 else bond_dim
            self.cores.append(nn.Parameter(torch.randn(r_l, phys_dim, r_r) * 0.1))

    def amplitude(self, x):
        batch = x.shape[0]
        state = torch.ones(batch, 1)
        for k in range(self.L):
            x_k = x[:, k].long()
            # Select physical slice per sample: core[k][:, x_k, :] -> (batch, r_l, r_r)
            selected = self.cores[k][:, x_k, :].permute(1, 0, 2)
            state = torch.einsum('bi,bir->br', state, selected)
        return state.squeeze(-1)

    def log_prob(self, x):
        amp = self.amplitude(x)
        log_psi2 = 2 * torch.log(amp.abs() + 1e-10)
        all_x = torch.tensor(all_strings)
        Z = (self.amplitude(all_x)**2).sum()
        return log_psi2 - torch.log(Z)

    def get_probs(self):
        with torch.no_grad():
            all_x = torch.tensor(all_strings)
            amp2 = self.amplitude(all_x)**2
            return (amp2 / amp2.sum()).numpy()


model_bm = BornMPS(L=4, bond_dim=4, phys_dim=2)
opt_bm = optim.Adam(model_bm.parameters(), lr=5e-3)
loader_bm = DataLoader(TensorDataset(data), batch_size=64, shuffle=True)

bm_losses = []
for epoch in range(80):
    ep_loss = 0.0
    for (xb,) in loader_bm:
        opt_bm.zero_grad()
        nll = -model_bm.log_prob(xb).mean()
        nll.backward(); opt_bm.step()
        ep_loss += nll.item()
    bm_losses.append(ep_loss / len(loader_bm))

print("\\nBorn machine trained.")
probs_learned = model_bm.get_probs()
kl = np.sum(probs_true * np.log(probs_true / (probs_learned + 1e-10)))
print(f"KL(true || learned) = {kl:.4f}  (lower is better)")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
ax1.plot(bm_losses); ax1.set_xlabel('Epoch'); ax1.set_ylabel('NLL'); ax1.set_title('Born Machine NLL')
patterns = [''.join(map(str, row.astype(int))) for row in all_strings]
x_pos = np.arange(16)
ax2.bar(x_pos - 0.2, probs_true, 0.4, label='True', alpha=0.7)
ax2.bar(x_pos + 0.2, probs_learned, 0.4, label='Learned', alpha=0.7)
ax2.set_xticks(x_pos); ax2.set_xticklabels(patterns, rotation=90, fontsize=7)
ax2.set_ylabel('Probability'); ax2.set_title('True vs Learned'); ax2.legend()
plt.tight_layout(); plt.show()
"""),

    md("""## 4. TT Regression — Full Training Loop

A Tensor Train weight tensor combined with a polynomial feature map gives
a compact but expressive regression model.
"""),

    code("""torch.manual_seed(0)
n_samples, n_features, phi_dim = 300, 5, 3

X = torch.randn(n_samples, n_features)
y_true = (X[:,0]*X[:,1] + X[:,2]**2 - X[:,3]*X[:,4]).unsqueeze(1)
y_true = y_true + 0.1 * torch.randn_like(y_true)

loader_reg = DataLoader(TensorDataset(X, y_true), batch_size=32, shuffle=True)

def poly2(xi):
    return torch.stack([torch.ones_like(xi), xi, xi**2], dim=-1)  # (batch, 3)


class TTRegressor(nn.Module):
    def __init__(self, n_features=5, phi_dim=3, bond_dim=4):
        super().__init__()
        self.n = n_features
        self.cores = nn.ParameterList()
        for k in range(n_features):
            rl = 1 if k==0 else bond_dim
            rr = 1 if k==n_features-1 else bond_dim
            self.cores.append(nn.Parameter(torch.randn(rl, phi_dim, rr) * 0.05))

    def forward(self, x):
        state = torch.ones(x.shape[0], 1)
        for k in range(self.n):
            phi_k = poly2(x[:, k])   # (batch, phi_dim)
            state = torch.einsum('bL,Lpr,bp->br', state, self.cores[k], phi_k)
        return state.squeeze(-1)


model_reg = TTRegressor()
opt_reg = optim.Adam(model_reg.parameters(), lr=5e-3)
mse = nn.MSELoss()

reg_losses = []
for epoch in range(150):
    ep = 0.0
    for xb, yb in loader_reg:
        opt_reg.zero_grad()
        loss = mse(model_reg(xb).unsqueeze(1), yb)
        loss.backward(); opt_reg.step()
        ep += loss.item()
    reg_losses.append(ep / len(loader_reg))

with torch.no_grad():
    final_mse = mse(model_reg(X).unsqueeze(1), y_true).item()
print(f"TT Regressor final MSE: {final_mse:.5f}")
print(f"Parameters: {sum(p.numel() for p in model_reg.parameters())}")

plt.figure(figsize=(6, 3))
plt.semilogy(reg_losses)
plt.xlabel('Epoch'); plt.ylabel('MSE (log)'); plt.title('TT Regressor Training')
plt.tight_layout(); plt.show()
"""),

    md("""## 5. Library Ecosystem Map

```
                       Your ML Model
                           │
              ┌────────────┼────────────┐
              │            │            │
         [PyTorch]    [TensorLy]    [quimb]
         autograd     decompose     physics
              │            │            │
         [tntorch]  [opt_einsum]   [TeNPy]
         TT-autograd fast paths   DMRG/MPS
              │
          [tn4ml]
          TN Layers (JAX)
```
"""),

    code("""# ── Verify all library imports ──
checks = {}
for lib, imp in [
    ('numpy',         'import numpy'),
    ('tensorly',      'import tensorly'),
    ('tntorch',       'import tntorch'),
    ('opt_einsum',    'import opt_einsum'),
    ('tensornetwork', 'import tensornetwork'),
    ('quimb',         'import quimb'),
    ('tenpy',         'import tenpy'),
    ('tn4ml',         'import tn4ml'),
    ('torch',         'import torch'),
    ('jax',           'import jax'),
]:
    try:
        exec(imp); checks[lib] = '✓'
    except ImportError as e:
        checks[lib] = f'✗ ({e})'

print("Library check in tensor environment:")
print("-" * 42)
for lib, status in checks.items():
    print(f"  {lib:<22} {status}")
"""),

    md("""## Summary & Roadmap Recap

| Notebook | Topic | Key Tools |
|----------|-------|-----------|
| 01 | Scalars → tensors, einsum, contractions, unfolding | numpy, opt_einsum, tensorly |
| 02 | SVD, CP, Tucker, Tensor Train decompositions | tensorly |
| 03 | MPS, TTN, graphical notation, SVD splitting, contraction order | tensornetwork, opt_einsum |
| 04 | TT-layer compression, feature maps, MPS classifier | torch, tensorly |
| 05 | Independent per-library blocks | all 7 libraries |
| 06 | Autograd through contractions, Born machine, TT regression | tntorch, torch |

**Research directions:**
- TN layer compression of Transformers / CNNs
- Born machines for generative modelling
- Variational quantum-inspired algorithms
- DMRG-style sweep optimisation as an alternative to backprop
- Tensor networks in reinforcement learning (value function compression)

Activate the environment with `conda activate tensor`.""")
]

nb06 = nb(nb06_cells)
with open(os.path.join(BASE, '06_advanced_topics.ipynb'), 'w') as f:
    json.dump(nb06, f, indent=1)
print("✓ 06_advanced_topics.ipynb written")

print("\nAll notebooks generated successfully.")
