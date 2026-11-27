# Transition charges and transition dipoles (`tensornet_transition`)

Model family for excited-state 0 → n transition properties, built on the QM-only
O(3) TensorNet representation (`mlpui/models/transition/model.py`,
backend `mlpui/backends/transition.py`, alias `transition`).

## Model

For `num_states` transitions the per-atom head outputs raw scores `[A, S]`, which
are projected per structure and state to

    q_i^(n) = raw_i^(n) - mean_j raw_j^(n),      Σ_i q_i^(n) = 0 exactly

(the transition density integrates to ⟨Ψ0|Ψn⟩ = 0). The dipole is derived, not a
second head:

    μ^(n) = Σ_i q_i^(n) r_i  [+ Σ_i p_i^(n)]

so q and μ share one phase and μ is origin independent. With
`atomic_dipoles: true`, `p_i = Σ_j w_ij (r_j − r_i)` over the representation
neighbour graph (scalar edge weights from TensorNet features ⊙ radial basis,
cosine cutoff, last layer zero-initialised, i.e. training starts from the
pure point-charge dipole). Use it when the charge labels cannot reproduce the
QM transition dipole (e.g. Mulliken/Löwdin transition charges).

### Symmetric molecules: `atom_index_embedding`

The representation and heads are permutation equivariant and O(3) invariant, so
two atoms related by a symmetry operation of the molecule (`P(i)`) always get the
same predicted scalar: `q̂_P(i) = q̂_i`. Transition charges of a state whose
symmetry differs from the ground state are instead antisymmetric,
`q_P(i) = −q_i` (e.g. the short-axis-polarised transition of anthracene). The
best an invariant model can do is then `q̂ = 0`, and the loss stalls at the label
variance (training loss can still drop by memorising thermal distortions, while
validation gets worse). Making the network equivariant internally or allowing
pseudoscalar outputs does not help: the molecular C2 rotation is a proper rotation, and
a planar molecule's own mirror plane maps every atom onto itself. The per-frame phase-less
sign cannot fix it either, because it flips all atoms together. (With the phase-less
loss and large thermal distortions an invariant model can partly fake the antisymmetry by
choosing "which side is positive" from the distortion; that function is discontinuous near
the symmetric geometry and generalises poorly. In a test on distorted ethylene the plain
model's validation RMSE was 0.38 / 0.78 / 0.99 of the label RMS for 0.05 / 0.02 / 0.005 Å
distortions, vs 0.07 / 0.03 / 0.006 with the index embedding.)

`atom_index_embedding: N` adds a learned vector per within-structure atom index
(0..N−1, N ≥ atoms per structure) to the TensorNet features before the heads
(zero-initialised). This breaks permutation symmetry on purpose, so it is valid
only when every frame uses the same atom ordering (single-molecule datasets); the
phase convention ("which side is positive") is then the one carried by the labels.
Rotation/translation invariance of q and equivariance of μ are unchanged.
`0` (default) disables it and keeps old configs and checkpoints valid.

Outputs: `transition_charges [A, S]`, `transition_dipole [B, S, 3]`
(`[S, 3]` for one structure), and `transition_atomic_dipoles [A, S, 3]` when enabled.
The model is gas phase only (MM fields are rejected) and nonperiodic.

## Phase-less loss

Every eigenvector is defined up to a sign, so each (frame, state) label carries
an arbitrary s_n = ±1, shared by q and μ. `Trainer` calls the backend hook
`align_targets(model, predictions, targets, loss_weights)` per structure before
the MSE (both the batched and the per-structure path). For the Trainer loss

    L = w_q mean_{i,n}(q̂ − s q)² + w_μ mean_{n,a}(μ̂ − s μ)²

the exact minimiser is `s_n = sign(w_q/N · q̂_n·q_n + w_μ/3 · μ̂_n·μ_n)` (ties → +1,
sign detached). Evaluation metrics therefore are RMSEs after optimal sign.
Backends without `align_targets` are unaffected.

## Dataset

| file | shape | unit |
|---|---|---|
| `transition_charges.npy` | dense `[frames, N, S]` or ragged `[A, S]`; `[frames, N]`/`[A]` if S = 1 | e |
| `transition_dipole.npy` | `[frames, S, 3]`; `[frames, 3]` if S = 1 | e·Å (scaled with `length_scale`) |

Both must agree on S, and `num_states` in the model config must equal it.
Convert atomic-unit dipoles with 1 e·bohr = 0.529177 e·Å.

Not handled here (do upstream): state tracking across energy-ordered state swaps,
near-degenerate / conical-intersection frames (no globally smooth sign exists),
state–state (m, n ≥ 1) properties whose sign is s_m·s_n.

Example config: `configs/train_tensornet_transition.yaml`. Tests: `tests/test_transition.py`.

## Global heads for chromophores

A purely local per-atom head cannot make all atoms agree on molecule-level state mixing
(which configurations contribute to state n). Two optional heads add it; both require
`atom_index_embedding` (fixed atom ordering) and are mutually exclusive.

### `global_basis: M`

q_i^(n) += Σ_a C_na(R) b_a(i): M learned per-atom-index basis vectors b_a (+ a zero-initialised
local correction) mixed by coefficients C(R) [S, M] read out from attention-pooled atom
features. `model.init_global_basis(b)` sets b [N, M], e.g. from a sign-invariant PCA of the
training labels.

### `diabatic_states: M` (M ≥ num_states) and target `excitation_energies`

A symmetric M × M diabatic Hamiltonian is built from atomic contributions,
H(R) = diag(ε⁰) + Σ_i h_i(x_i), and diagonalised, H = U diag(E) Uᵀ. The lowest S eigenvalues
are predicted as `excitation_energies` [S] (eV after dataset conversion), and
q_i^(n) = Σ_a U_an b_a(i). Mixing near avoided crossings then follows the energies; eigenvector
signs are arbitrary and absorbed by the phase-less loss; M > S leaves room for higher states.
The eigen-decomposition backward replaces 1/(E_j − E_i) by a Lorentzian-broadened
d/(d² + ε²) (`TransitionTensorNet.EIG_BROADENING`, default 0.01) so near-degenerate spectra
keep finite gradients. `model.init_diabatic(b, eps0)` sets the basis [N, M] and energies [M]
(e.g. per-state sign-aligned label templates and mean excitation energies).

Dataset field: `excitation_energies.npy` `[frames, S]` (or `[frames]` if S = 1), energy unit,
scaled with `energy_scale`; it must agree with the other transition fields on S. Training it
requires `diabatic_states > 0`.

Observed on TDA/PBE0 labels of three dyes (BODIPY, PHBDI, rhodamine; 2400 training frames,
cutoff 8 Å, 3 layers, 3-seed ensembles): the diabatic head (M = 6–8) lowered transition-charge
test RMSE by 20–41 % relative to the local head, and supervising excitation energies lowered it
further for every molecule and both M tested. Tests: `tests/test_transition_global.py`,
`tests/test_transition_diabatic.py`.
