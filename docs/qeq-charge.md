# QEq charge electrostatics (`qeq_charge`)

Electrostatic component of the decomposition

    E_QMMM = E_gas (UMA or reference) + E_charge (this model) + E_residual (TensorNet-QMMM, residual_only)

## Model

For each structure the QM charges minimise

    E_QEq(q; phi) = sum_i chi_i q_i + 1/2 q^T H q + sum_i q_i phi_i,   sum_i q_i = Q

* chi_i(R) (electronegativity, eV/e) and J_i(R) (hardness, eV/e^2, bounded by
  `hardness_bounds`) are a per-element value plus a QM-only TensorNet correction
  (the last layer starts at zero, i.e. training starts from the element model).
* Atoms carry Gaussian charges with per-element width sigma (bounded by `sigma_bounds`).
  H_ii = J_i + k/(sigma_i sqrt(pi)); H_ij = k erf(r_ij/sqrt(2(sigma_i^2+sigma_j^2)))/r_ij.
  H is positive definite, so the constrained linear system always has a unique solution.
* phi_i = k sum_M Q_M erf(r_iM/(sqrt(2) sigma_i))/r_iM is the potential of the fixed MM
  point charges at the Gaussian of atom i. MM charges/coordinates are inputs only.

The returned energy is the environment part `E_charge = E_QEq(q*; phi) - E_QEq(q0; 0)`,
which is exactly zero without MM charges; it contains the gas-charge Coulomb term and
the polarization energy `-1/2 phi^T Pi phi` (the 1/2 factor is automatic).

## Derivatives (why no dq/dR)

q* and q0 are stationary points of their functionals, so the first derivatives of
E_charge with respect to QM coordinates, MM coordinates and MM charges contain no
dq/dR, dq/dr_MM or dq/dQ_MM terms. Autograd through `torch.linalg.solve` gives exactly
these Hellmann-Feynman derivatives (tested). Outputs:

| output | definition |
|---|---|
| `energy` | E_charge per structure (eV) |
| `forces` | -dE_charge/dR (eV/angstrom) |
| `mm_forces` | -dE_charge/dr_MM = -Q_MM * mm_potential_gradient |
| `mm_potential` | dE/dQ_MM = ESP of q* at MM sites (eV/e) |
| `mm_potential_gradient` | spatial gradient of that ESP with q* fixed (= -field), the same fixed-density definition as Q-Chem ESP_EFIELD labels |
| `gas_mm_potential`, `gas_mm_potential_gradient` | the same for the gas charges q0 |
| `charges`, `gas_charges` | q* (polarized) and q0 (gas) |

MM forces need no separate label: they follow from the potential gradient.

## Data and training

Dual-region NpyDataset (`mm_pos`, `mm_charges`, `mm_offsets`), labels in eV/angstrom/e:
`energy = E_QMMM - E_gas`, `forces = F_QMMM - F_gas`, QM/MM `mm_potential` and
`mm_potential_gradient`, and the new optional per-MM-site targets `gas_mm_potential` /
`gas_mm_potential_gradient` (gas-density ESP at the same sites). See
`configs/train_qeq_charge.yaml`. MK/ESP-fitted atomic charges are not required; they are
ill-conditioned for buried atoms, and fitting the MM-site ESP directly is preferred.

## Residual stage

`python -m scripts.qeq_residual CHECKPOINT SOURCE OUTPUT` writes a copy of a dataset
with energy, forces, mm_forces, mm_potential and mm_potential_gradient replaced by
label - QEq prediction (plus `qeq_*` prediction arrays and `residual_manifest.json` with
the checkpoint SHA256). Train TensorNet-QMMM with `baseline: {mode: residual_only}` on it.
