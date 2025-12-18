import math
from typing import Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchmdnet.extensions.ops import get_neighbor_pairs_kernel
import warnings

class OptimizedDistance(torch.nn.Module):
    """Compute the neighbor list for a given cutoff.

    This operation can be placed inside a CUDA graph in some cases.
    In particular, resize_to_fit and check_errors must be False.

    Note that this module returns neighbors such that :math:`r_{ij} \\ge \\text{cutoff_lower}\\quad\\text{and}\\quad r_{ij} < \\text{cutoff_upper}`.

    This function optionally supports periodic boundary conditions with
    arbitrary triclinic boxes.  The box vectors `a`, `b`, and `c` must satisfy
    certain requirements:

    .. code:: python

       a[1] = a[2] = b[2] = 0
       a[0] >= 2*cutoff, b[1] >= 2*cutoff, c[2] >= 2*cutoff
       a[0] >= 2*b[0]
       a[0] >= 2*c[0]
       b[1] >= 2*c[1]

    These requirements correspond to a particular rotation of the system and
    reduced form of the vectors, as well as the requirement that the cutoff be
    no larger than half the box width.

    Parameters
    ----------
    cutoff_lower : float
        Lower cutoff for the neighbor list.
    cutoff_upper : float
        Upper cutoff for the neighbor list.
    max_num_pairs : int
        Maximum number of pairs to store, if the number of pairs found is less than this, the list is padded with (-1,-1) pairs up to max_num_pairs unless resize_to_fit is True, in which case the list is resized to the actual number of pairs found.
        If the number of pairs found is larger than this, the pairs are randomly sampled. When check_errors is True, an exception is raised in this case.
        If negative, it is interpreted as (minus) the maximum number of neighbors per atom.
    strategy : str
        Strategy to use for computing the neighbor list. Can be one of :code:`["shared", "brute", "cell"]`.

        1. *Shared*: An O(N^2) algorithm that leverages CUDA shared memory, best for large number of particles.
        2. *Brute*: A brute force O(N^2) algorithm, best for small number of particles.
        3. *Cell*:  A cell list algorithm, best for large number of particles, low cutoffs and low batch size.
    box : torch.Tensor, optional
        The vectors defining the periodic box.  This must have shape `(3, 3)` or `(max(batch)+1, 3, 3)` if a ox per sample is desired.
        where `box_vectors[0] = a`, `box_vectors[1] = b`, and `box_vectors[2] = c`.
        If this is omitted, periodic boundary conditions are not applied.
    loop : bool, optional
        Whether to include self-interactions.
        Default: False
    include_transpose : bool, optional
        Whether to include the transpose of the neighbor list.
        Default: True
    resize_to_fit : bool, optional
        Whether to resize the neighbor list to the actual number of pairs found. When False, the list is padded with (-1,-1) pairs up to max_num_pairs
        Default: True
        If this is True the operation is not CUDA graph compatible.
    check_errors : bool, optional
        Whether to check for too many pairs. If this is True the operation is not CUDA graph compatible.
        Default: True
    return_vecs : bool, optional
        Whether to return the distance vectors.
        Default: False
    long_edge_index : bool, optional
        Whether to return edge_index as int64, otherwise int32.
        Default: True
    """

    def __init__(
        self,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_num_pairs=-32,
        return_vecs=False,
        loop=False,
        strategy="brute",
        include_transpose=True,
        resize_to_fit=True,
        check_errors=True,
        box=None,
        long_edge_index=True,
    ):
        super(OptimizedDistance, self).__init__()
        self.cutoff_upper = cutoff_upper
        self.cutoff_lower = cutoff_lower
        self.max_num_pairs = max_num_pairs
        self.strategy = strategy
        self.box: Optional[Tensor] = box
        self.loop = loop
        self.return_vecs = return_vecs
        self.include_transpose = include_transpose
        self.resize_to_fit = resize_to_fit
        self.use_periodic = True
        if self.box is None:
            self.use_periodic = False
            self.box = torch.empty((0, 0))
            if self.strategy == "cell":
                # Default the box to 3 times the cutoff, really inefficient for the cell list
                lbox = cutoff_upper * 3.0
                self.box = torch.tensor(
                    [[lbox, 0, 0], [0, lbox, 0], [0, 0, lbox]], device="cpu"
                )
        if self.strategy == "cell":
            self.box = self.box.cpu()
        self.check_errors = check_errors
        self.long_edge_index = long_edge_index

    def forward(
        self, pos: Tensor, batch: Optional[Tensor] = None, box: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Compute the neighbor list for a given cutoff.

        Parameters
        ----------
        pos : torch.Tensor
            A tensor with shape (N, 3) representing the positions.
        batch : torch.Tensor, optional
            A tensor with shape (N,). Defaults to None.
        box : torch.Tensor, optional
            The vectors defining the periodic box.  This must have shape `(3, 3)` or `(max(batch)+1, 3, 3)`,
        Returns
        -------
        edge_index : torch.Tensor
            List of neighbors for each atom in the batch.
            Shape is (2, num_found_pairs) or (2, max_num_pairs).
        edge_weight : torch.Tensor
            List of distances for each atom in the batch.
            Shape is (num_found_pairs,) or (max_num_pairs,).
        edge_vec : torch.Tensor, optional
            List of distance vectors for each atom in the batch.
            Shape is (num_found_pairs, 3) or (max_num_pairs, 3).

        Notes
        -----
        If `resize_to_fit` is True, the tensors will be trimmed to the actual number of pairs found.
        Otherwise, the tensors will have size `max_num_pairs`, with neighbor pairs (-1, -1) at the end.
        """
        use_periodic = self.use_periodic
        if not use_periodic:
            use_periodic = box is not None
        box = self.box if box is None else box
        assert box is not None, "Box must be provided"
        box = box.to(pos.dtype)
        max_pairs: int = self.max_num_pairs
        if self.max_num_pairs < 0:
            max_pairs = -self.max_num_pairs * pos.shape[0]
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        edge_index, edge_vec, edge_weight, num_pairs = get_neighbor_pairs_kernel(
            strategy=self.strategy,
            positions=pos,
            batch=batch,
            max_num_pairs=int(max_pairs),
            cutoff_lower=self.cutoff_lower,
            cutoff_upper=self.cutoff_upper,
            loop=self.loop,
            include_transpose=self.include_transpose,
            box_vectors=box,
            use_periodic=use_periodic,
        )
        if self.check_errors:
            assert (
                num_pairs[0] <= max_pairs
            ), f"Found num_pairs({num_pairs[0]}) > max_num_pairs({max_pairs})"

        # Remove (-1,-1)  pairs
        if self.resize_to_fit:
            mask = edge_index[0] != -1
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_vec = edge_vec[mask, :]
        if self.long_edge_index:
            edge_index = edge_index.to(torch.long)
        if self.return_vecs:
            return edge_index, edge_weight, edge_vec
        else:
            return edge_index, edge_weight, None
