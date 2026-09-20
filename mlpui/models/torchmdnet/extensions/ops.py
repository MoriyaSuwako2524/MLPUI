"""Neighbor search following the vendored MIT-licensed CPU kernel.

The PyTorch fallback supports force-loss double backward but uses quadratic
memory/time and does not support CUDA graph capture. Build the bundled native
extension for large GPU workloads. MLPUI_NEIGHBORS=python forces the fallback.
"""
import os
import warnings
import torch

BACKEND = "python"
if os.environ.get("MLPUI_NEIGHBORS", "auto") not in ("auto", "python", "native"):
    raise ValueError("MLPUI_NEIGHBORS must be auto, python or native")
if os.environ.get("MLPUI_NEIGHBORS") != "python":
    try:
        from . import mlpui_torchmdnet_extensions
    except ImportError as exc:
        if os.environ.get("MLPUI_NEIGHBORS") == "native":
            raise ImportError("Build the extension with scripts/build_model_extensions.py") from exc
        warnings.warn("MLPUI uses PyTorch neighbor search (quadratic memory/time). "
                      "For large GPU workloads build scripts/build_model_extensions.py --cuda.",
                      RuntimeWarning, stacklevel=2)
    else:
        BACKEND = "native"


def python_neighbor_pairs(strategy, positions, batch, box_vectors, use_periodic,
                          cutoff_lower, cutoff_upper, max_num_pairs, loop, include_transpose):
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError("positions must have shape (N, 3), N > 0")
    if cutoff_upper <= 0 or max_num_pairs <= 0:
        raise ValueError("cutoff_upper and max_num_pairs must be positive")
    if batch.shape != positions.shape[:1]:
        raise ValueError("batch must have one entry per atom")
    pairs = torch.tril_indices(len(positions), len(positions), -1, device=positions.device)
    pairs = pairs[:, batch[pairs[0]] == batch[pairs[1]]]
    vectors = positions[pairs[0]] - positions[pairs[1]]
    if use_periodic:
        boxes = box_vectors.to(device=positions.device, dtype=positions.dtype)
        count = int(batch.max()) + 1
        if boxes.ndim == 2:
            boxes = boxes.unsqueeze(0).expand(count, 3, 3)
        if boxes.shape != (count, 3, 3):
            raise ValueError("Box must have shape (3, 3) or (number of batches, 3, 3)")
        if (torch.any(boxes[:, 0, 1:] != 0) or torch.any(boxes[:, 1, 2] != 0)
                or torch.any(boxes.diagonal(dim1=1, dim2=2) < 2 * cutoff_upper)
                or torch.any(boxes[:, 0, 0] < 2 * boxes[:, 1, 0])
                or torch.any(boxes[:, 0, 0] < 2 * boxes[:, 2, 0])
                or torch.any(boxes[:, 1, 1] < 2 * boxes[:, 2, 1])):
            raise ValueError("Invalid box vectors for minimum-image neighbor search")
        boxes = boxes[batch[pairs[0]]]
        for axis in (2, 1, 0):
            shift = torch.round(vectors[:, axis] / boxes[:, axis, axis])
            vectors = vectors - shift[:, None] * boxes[:, axis, :]
    squared = vectors.square().sum(dim=1)
    positive = squared > 0
    # The native kernel suppresses all position gradients of zero-length pairs.
    vectors = torch.where(positive[:, None], vectors, vectors.detach())
    distances = torch.where(positive, torch.sqrt(torch.where(positive, squared, torch.ones_like(squared))),
                            torch.zeros_like(squared))
    mask = (distances >= cutoff_lower) & (distances < cutoff_upper)
    pairs, vectors, distances = pairs[:, mask], vectors[mask], distances[mask]
    if include_transpose:
        pairs = torch.cat((pairs, pairs.flip(0)), dim=1)
        vectors = torch.cat((vectors, -vectors))
        distances = torch.cat((distances, distances))
    if loop:
        indices = torch.arange(len(positions), device=positions.device)
        pairs = torch.cat((pairs, torch.stack((indices, indices))), dim=1)
        vectors = torch.cat((vectors, positions.new_zeros((len(positions), 3))))
        distances = torch.cat((distances, positions.new_zeros(len(positions))))
    found = torch.tensor([len(distances)], device=positions.device, dtype=torch.int32)
    padding = max(0, max_num_pairs - len(distances))
    if padding:
        pairs = torch.cat((pairs, pairs.new_full((2, padding), -1)), dim=1)
        vectors = torch.cat((vectors, positions.new_zeros((padding, 3))))
        distances = torch.cat((distances, positions.new_zeros(padding)))
    # Preserve overflow so OptimizedDistance can reject insufficient capacity.
    return pairs.to(torch.int32), vectors, distances, found


if BACKEND == "native":
    from .native_ops import get_neighbor_pairs_kernel, is_current_stream_capturing
else:
    get_neighbor_pairs_kernel = python_neighbor_pairs

    def is_current_stream_capturing():
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
