"""Output heads for UMA/eSCNMD models.

The backbone (eSCNMDBackbone) returns raw node embeddings. This module
provides the energy (and gradient-based force) head that converts those
embeddings into physical observables.

Architecture mirrors fairchem MLP_EFS_Head / DatasetSpecificSingleHeadWrapper:
    energy_block: Linear(C→C) → SiLU → Linear(C→C) → SiLU → Linear(C→1)

Forces are obtained as -∂E/∂pos (conservative, no separate force head).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EnergyAndForceHead(nn.Module):
    """Energy output head for UMA eSCNMD backbone.

    Takes l=0 scalar node embeddings, predicts per-atom energies, aggregates
    to system energy, then optionally computes forces via autograd.

    Normalisation applied during forward:
        E_physical = E_raw * rmsd + mean + sum(element_refs[atomic_numbers])

    Parameters
    ----------
    sphere_channels:
        Width of the backbone's node embedding (matches backbone config).
    """

    def __init__(self, sphere_channels: int = 128) -> None:
        super().__init__()
        # Indices 0, 2, 4 are Linear; 1, 3 are SiLU — matches checkpoint keys.
        self.energy_block = nn.Sequential(
            nn.Linear(sphere_channels, sphere_channels),
            nn.SiLU(),
            nn.Linear(sphere_channels, sphere_channels),
            nn.SiLU(),
            nn.Linear(sphere_channels, 1),
        )
        # Normaliser scalars (identical across all UMA tasks in this checkpoint)
        self.register_buffer("normalizer_rmsd", torch.tensor(1.0))
        self.register_buffer("normalizer_mean", torch.tensor(0.0))
        # Per-dataset element references registered as named buffers at load time
        self._dataset_names: list[str] = []

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def load_normalizer(
        self,
        rmsd: float,
        mean: float,
        dataset_element_refs: dict[str, list[float]],
    ) -> None:
        """Populate normaliser and per-dataset element references.

        Parameters
        ----------
        rmsd, mean:
            Normaliser parameters from the checkpoint tasks_config.
        dataset_element_refs:
            Mapping of dataset name → list of per-element reference energies
            (100 values, indexed by atomic number).
        """
        self.normalizer_rmsd.fill_(rmsd)
        self.normalizer_mean.fill_(mean)

        for dataset, refs in dataset_element_refs.items():
            t = torch.tensor(refs, dtype=torch.float32)
            # Ensure we have at least 120 slots to avoid index-out-of-range
            if len(t) < 120:
                pad = torch.zeros(120 - len(t), dtype=torch.float32)
                t = torch.cat([t, pad])
            buf_name = f"elemrefs_{dataset}"
            self.register_buffer(buf_name, t[:120])
            if dataset not in self._dataset_names:
                self._dataset_names.append(dataset)

    def _element_refs_for(self, dataset: str | None) -> torch.Tensor | None:
        if dataset is None:
            return None
        return getattr(self, f"elemrefs_{dataset}", None)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        backbone_out: dict[str, torch.Tensor],
        pos: torch.Tensor,
        atomic_numbers: torch.Tensor,
        dataset: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        backbone_out:
            Output dict from eSCNMDBackbone forward, must contain
            ``node_embedding`` [N, sph_features, sphere_channels] and
            ``batch`` [N].
        pos:
            Atomic positions [N, 3]. Must have ``requires_grad=True``
            for force computation.
        atomic_numbers:
            Integer atomic numbers [N].
        dataset:
            Dataset name used to select element references (e.g. ``"omol"``).

        Returns
        -------
        dict with keys:
            ``energy`` – per-system energy [num_systems]
            ``forces`` – per-atom forces [N, 3]  (only if pos.requires_grad)
        """
        node_emb = backbone_out["node_embedding"]   # [N, sph_features, C]
        batch = backbone_out["batch"]               # [N]

        # l=0 scalar components only
        x = node_emb[:, 0, :]                        # [N, C]

        energy_per_atom_raw = self.energy_block(x).squeeze(-1)   # [N]

        # Aggregate to per-system total energy
        num_systems = int(batch.max().item()) + 1
        energy_raw = energy_per_atom_raw.new_zeros(num_systems)
        energy_raw.index_add_(0, batch, energy_per_atom_raw)

        # Denormalise: E = E_raw * rmsd + mean
        energy = energy_raw * self.normalizer_rmsd.to(energy_raw) \
                 + self.normalizer_mean.to(energy_raw)

        # Add element references if available
        elem_refs = self._element_refs_for(dataset)
        if elem_refs is not None:
            refs_per_atom = elem_refs[atomic_numbers.long().clamp(0, len(elem_refs) - 1)]
            ref_sum = refs_per_atom.new_zeros(num_systems)
            ref_sum.index_add_(0, batch, refs_per_atom)
            energy = energy + ref_sum.to(energy)

        result: dict[str, torch.Tensor] = {"energy": energy}

        # Conservative forces: F_i = -∂E/∂r_i
        if pos.requires_grad:
            (forces,) = torch.autograd.grad(
                energy.sum(),
                pos,
                create_graph=False,
                retain_graph=False,
            )
            result["forces"] = -forces

        return result
