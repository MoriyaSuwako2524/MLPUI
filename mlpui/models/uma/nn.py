from enum import Enum

class ExecutionMode(str, Enum):
    """
    Execution mode for model inference.
    """

    GENERAL = "general"



import torch
from mlpui.models.uma.inference import InferenceSettings


class ExecutionBackend:
    """
    Parameterless function dispatch for execution modes.

    Provides default PyTorch implementations for rotation and scatter
    operations. Subclass and override methods with optimized kernels
    (e.g. Triton) for specific execution modes.

    All methods are static — backends carry no instance state.

    Methods (override for optimization):
        - node_to_edge_wigner_permute: Gather node features and rotate L->M
        - permute_wigner_inv_edge_to_node: Rotate M->L and scatter to nodes
        - edge_degree_scatter: Rotate radial and scatter to nodes
        - prepare_model_for_inference: Apply backend-specific model transforms
    """

    @staticmethod
    def validate(
        model: torch.nn.Module,
        settings: InferenceSettings | None = None,
    ) -> None:
        """
        Validate that model and settings are compatible with this backend.

        Called during model construction (settings=None) and before
        first inference (settings provided).

        Args:
            model: The backbone model to validate.
            settings: Inference settings, or None at construction time.

        Raises:
            ValueError: If incompatible with this backend.
        """

    @staticmethod
    def prepare_model_for_inference(model: torch.nn.Module) -> None:
        """
        Prepare a model for inference with backend-specific transforms.

        Called once during prepare_for_inference. Override in subclasses
        to apply model transformations (e.g. SO2 block conversion).

        Args:
            model: The backbone model to prepare.
        """

    @staticmethod
    def get_layer_radial_emb(
        x_edge: torch.Tensor,
        model: torch.nn.Module,
    ) -> list[torch.Tensor]:
        """
        Get edge embeddings for each layer.

        Default implementation returns the same raw x_edge for all layers.
        SO2_Convolution will compute rad_func(x_edge) internally.

        Override in fast backends to precompute radials.

        Args:
            x_edge: Edge embeddings [E, edge_features]
            model: The backbone model

        Returns:
            List of edge embeddings, one per layer
        """
        return [x_edge] * len(model.blocks)

    @staticmethod
    def prepare_wigner(
        wigner: torch.Tensor,
        wigner_inv: torch.Tensor,
        mappingReduced,
        coefficient_index: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Transform raw Wigner matrices for this backend.

        Default: Apply coefficient selection (if mmax != lmax) and
        pre-compose with M-mapping via einsum.

        Args:
            wigner: Raw Wigner matrices [E, L, L]
            wigner_inv: Raw inverse Wigner matrices [E, L, L]
            mappingReduced: CoefficientMapping with to_m matrix
            coefficient_index: Indices for mmax != lmax selection,
                or None if mmax == lmax.

        Returns:
            Transformed (wigner, wigner_inv) ready for this backend.
        """
        if coefficient_index is not None:
            wigner = wigner.index_select(1, coefficient_index)
            wigner_inv = wigner_inv.index_select(2, coefficient_index)

        wigner = torch.einsum(
            "mk,nkj->nmj",
            mappingReduced.to_m.to(wigner.dtype),
            wigner,
        )
        wigner_inv = torch.einsum(
            "njk,mk->njm",
            wigner_inv,
            mappingReduced.to_m.to(wigner_inv.dtype),
        )
        return wigner, wigner_inv

    @staticmethod
    def node_to_edge_wigner_permute(
        x_full: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather node features and rotate L->M.

        Default: PyTorch gather + BMM.

        Args:
            x_full: Node features [N, L, C]
            edge_index: Edge indices [2, E]
            wigner: Wigner rotation matrices [E, M, L] or [E, M, 2L]

        Returns:
            Rotated edge messages [E, M, 2C]
        """
        x_source = x_full[edge_index[0]]
        x_target = x_full[edge_index[1]]
        x_message = torch.cat((x_source, x_target), dim=2)
        return torch.bmm(wigner, x_message)

    @staticmethod
    def permute_wigner_inv_edge_to_node(
        x_message: torch.Tensor,
        wigner_inv: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        node_offset: int = 0,
    ) -> torch.Tensor:
        """
        Rotate M->L and scatter edge messages to nodes.

        Default: PyTorch BMM + index_add.

        Args:
            x_message: Edge message features [E, M, C]
            wigner_inv: Inverse Wigner matrices [E, L, M]
            edge_index: Edge indices [2, E]
            num_nodes: Total number of nodes (output size)
            node_offset: Offset for node indices (for chunking)

        Returns:
            Node embeddings [N, L, C] accumulated from edge messages
        """
        # Rotate M->L
        x_rotated = torch.bmm(wigner_inv, x_message)
        # Scatter to nodes
        new_embedding = torch.zeros(
            (num_nodes,) + x_rotated.shape[1:],
            dtype=x_rotated.dtype,
            device=x_rotated.device,
        )
        new_embedding.index_add_(0, edge_index[1] - node_offset, x_rotated)
        return new_embedding

    @staticmethod
    def edge_degree_scatter(
        x: torch.Tensor,
        radial_output: torch.Tensor,
        wigner_inv: torch.Tensor,
        edge_index: torch.Tensor,
        m_0_num_coefficients: int,
        sphere_channels: int,
        rescale_factor: float,
        node_offset: int = 0,
    ) -> torch.Tensor:
        """
        Edge degree embedding: rotate radial and scatter to nodes.

        Default: PyTorch BMM + index_add.

        Args:
            x: Node features [N, L, C] to update
            radial_output: RadialMLP output [E, m0 * C]
            wigner_inv: Wigner inverse with envelope pre-fused
                [E, L, m0] or [E, L, L]
            edge_index: Edge indices [2, E]
            m_0_num_coefficients: Number of m=0 coefficients
                (3 for lmax=2)
            sphere_channels: Number of channels C
            rescale_factor: Aggregation rescale factor
            node_offset: Node offset for graph parallelism

        Returns:
            Updated node features [N, L, C]
        """
        # Reshape radial output: [E, m0*C] -> [E, m0, C]
        radial = radial_output.reshape(-1, m_0_num_coefficients, sphere_channels)

        # Slice wigner to m=0 columns and rotate:
        # [E, L, m0] @ [E, m0, C] -> [E, L, C]
        wigner_inv_m0 = wigner_inv[:, :, :m_0_num_coefficients]
        x_edge_embedding = torch.bmm(wigner_inv_m0, radial)

        # Type cast if needed
        x_edge_embedding = x_edge_embedding.to(x.dtype)

        # Scatter to destination nodes with rescaling
        return x.index_add(
            0,
            edge_index[1] - node_offset,
            x_edge_embedding / rescale_factor,
        )



def get_execution_backend(
    mode: ExecutionMode | str = ExecutionMode.GENERAL,
) -> ExecutionBackend:
    """
    Factory function to create the appropriate execution backend.

    Args:
        mode: Execution mode (enum or string). Defaults to GENERAL.

    Returns:
        Configured execution backend instance
    """
    if isinstance(mode, str):
        mode = ExecutionMode(mode)

    if mode not in _EXECUTION_BACKENDS:
        available = [m.value for m in _EXECUTION_BACKENDS]
        raise ValueError(f"Unknown execution mode: {mode}. Available: {available}")
    return _EXECUTION_BACKENDS[mode]()


_EXECUTION_BACKENDS: dict[ExecutionMode, type[ExecutionBackend]] = {
    ExecutionMode.GENERAL: ExecutionBackend,
}


class MOLEInterface:
    def set_MOLE_coefficients(
        self, atomic_numbers_full, batch_full, csd_mixed_emb
    ) -> None:
        return None

    def set_MOLE_sizes(self, nsystems, batch_full, edge_index) -> None:
        return None

    def log_MOLE_stats(self) -> None:
        return None

    def merge_MOLE_model(self, data):
        return self