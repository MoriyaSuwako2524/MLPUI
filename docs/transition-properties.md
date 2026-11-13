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
