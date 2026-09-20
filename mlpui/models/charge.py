"""Differentiable per-structure projection onto a specified total charge."""
import torch


def constrain_charges(charges, batch, total_charge):
    if total_charge is None:
        raise ValueError("Total-charge constraint requires explicit Q for every structure")
    values = charges.reshape(-1)
    if batch.ndim != 1 or values.numel() != batch.numel() or batch.numel() == 0:
        raise ValueError("Expected one predicted charge and batch index per atom")
    if batch.dtype != torch.long or torch.any(batch < 0):
        raise ValueError("batch must contain nonnegative integer structure indices")
    total = torch.as_tensor(total_charge, device=values.device, dtype=values.dtype).reshape(-1)
    size = int(batch.max()) + 1
    if total.numel() != size or not torch.isfinite(total).all():
        raise ValueError("Q must contain one finite total charge per structure")
    counts = torch.bincount(batch, minlength=size).to(values.dtype)
    if torch.any(counts == 0):
        raise ValueError("Each structure must contain at least one atom")
    sums = values.new_zeros(size).index_add(0, batch, values)
    corrected = values + ((total - sums) / counts)[batch]
    return corrected.reshape_as(charges)
