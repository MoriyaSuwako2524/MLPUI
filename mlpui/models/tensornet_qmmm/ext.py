# Adapted from panxl/torchmd-net b0f7c7b42c6304c016a9e3dee78d4e2f4cbadc7d.
# Copyright Universitat Pompeu Fabra 2020-2023; MIT (LICENSE.tensornet-ext).
"""TensorNet-ext interactions with explicit ragged, nonperiodic QM/MM edges."""
import torch
from torch import Tensor, nn
from typing import Optional
from mlpui.models.torchmdnet.models.tensornet import (
    TensorNet, decompose_tensor, tensor_norm, tensor_message_passing,
    vector_to_skewtensor, vector_to_symtensor,
)
from mlpui.models.torchmdnet.models.utils import CosineCutoff, rbf_class_mapping


class Interaction_Ext(nn.Module):
    def __init__(
        self,
        num_rbf,
        rbf_type,
        trainable_rbf,
        hidden_channels,
        activation,
        cutoff_lower,
        cutoff_upper,
        dtype=torch.float32,
    ):
        super(Interaction_Ext, self).__init__()

        self.num_rbf = num_rbf
        self.hidden_channels = hidden_channels
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.ext_distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append(
            nn.Linear(num_rbf, hidden_channels, bias=True, dtype=dtype)
        )
        self.linears_scalar.append(
            nn.Linear(hidden_channels, 2 * hidden_channels, bias=True, dtype=dtype)
        )
        self.linears_scalar.append(
            nn.Linear(2 * hidden_channels, 3 * hidden_channels, bias=True, dtype=dtype)
        )
        self.act = activation()
        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.linears_scalar:
            linear.reset_parameters()

    def forward(self, pos, ext_pos, ext_charge, batch, mm_batch):
        # Build only within-structure pairs; no fixed atom counts or cross-frame edges.
        qi, mi = [], []
        for frame in torch.unique(batch):
            q = torch.where(batch == frame)[0]
            m = torch.where(mm_batch == frame)[0]
            qi.append(q.repeat_interleave(len(m)))
            mi.append(m.repeat(len(q)))
        qi, mi = torch.cat(qi), torch.cat(mi)
        vec = pos[qi] - ext_pos[mi]
        distance = torch.linalg.vector_norm(vec, dim=-1)
        if torch.any(distance < 1.e-6):
            raise ValueError('Coincident QM/MM sites are not supported')
        keep = distance < self.cutoff.cutoff_upper
        qi, mi, distance, vec = qi[keep], mi[keep], distance[keep], vec[keep]
        attr = self.ext_distance_expansion(distance)
        for linear in self.linears_scalar:
            attr = self.act(linear(attr))
        attr = (attr * self.cutoff(distance)[:, None] * ext_charge[mi, None]).reshape(
            -1, self.hidden_channels, 3)
        direction = vec / distance[:, None]
        eye = torch.eye(3, device=pos.device, dtype=pos.dtype)
        # reshape avoids upstream squeeze(0)'s one-edge special shape.
        skew = vector_to_skewtensor(direction).reshape(-1, 3, 3)
        symmetric = vector_to_symtensor(direction)
        message = (attr[..., 0, None, None] * eye
                   + attr[..., 1, None, None] * skew[:, None]
                   + attr[..., 2, None, None] * symmetric[:, None])
        # Explicit dim_size includes trailing QM sites with no external neighbors.
        result = pos.new_zeros((len(pos), self.hidden_channels, 3, 3)).index_add(0, qi, message)
        return result + 0 * (pos.sum() + ext_pos.sum() + ext_charge.sum())


class Interaction(nn.Module):
    """Interaction layer.

    :meta private:
    """

    def __init__(
        self,
        num_rbf,
        hidden_channels,
        activation,
        cutoff_lower,
        cutoff_upper,
        equivariance_invariance_group,
        dtype=torch.float32,
    ):
        super(Interaction, self).__init__()

        self.num_rbf = num_rbf
        self.hidden_channels = hidden_channels
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append(
            nn.Linear(num_rbf, hidden_channels, bias=True, dtype=dtype)
        )
        self.linears_scalar.append(
            nn.Linear(hidden_channels, 2 * hidden_channels, bias=True, dtype=dtype)
        )
        self.linears_scalar.append(
            nn.Linear(2 * hidden_channels, 3 * hidden_channels, bias=True, dtype=dtype)
        )
        self.linears_tensor = nn.ModuleList()
        for _ in range(6):
            self.linears_tensor.append(
                nn.Linear(hidden_channels, hidden_channels, bias=False)
            )
        self.act = activation()
        self.equivariance_invariance_group = equivariance_invariance_group
        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.linears_scalar:
            linear.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()

    def forward(
        self,
        X: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
        msg_ext: Optional[Tensor] = None,
    ) -> Tensor:
        C = self.cutoff(edge_weight)
        for linear_scalar in self.linears_scalar:
            edge_attr = self.act(linear_scalar(edge_attr))
        edge_attr = (edge_attr * C.view(-1, 1)).reshape(
            edge_attr.shape[0], self.hidden_channels, 3
        )
        X = X / (tensor_norm(X) + 1)[..., None, None]
        I, A, S = decompose_tensor(X)
        I = self.linears_tensor[0](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A = self.linears_tensor[1](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S = self.linears_tensor[2](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        Y = I + A + S
        Im = tensor_message_passing(
            edge_index, edge_attr[..., 0, None, None], I, X.shape[0]
        )
        Am = tensor_message_passing(
            edge_index, edge_attr[..., 1, None, None], A, X.shape[0]
        )
        Sm = tensor_message_passing(
            edge_index, edge_attr[..., 2, None, None], S, X.shape[0]
        )
        msg = Im + Am + Sm
        if msg_ext is not None:
            msg = msg + msg_ext
        if self.equivariance_invariance_group == "O(3)":
            A = torch.matmul(msg, Y)
            B = torch.matmul(Y, msg)
            I, A, S = decompose_tensor(A + B)
        if self.equivariance_invariance_group == "SO(3)":
            B = torch.matmul(Y, msg)
            I, A, S = decompose_tensor(2 * B)
        normp1 = (tensor_norm(I + A + S) + 1)[..., None, None]
        I, A, S = I / normp1, A / normp1, S / normp1
        I = self.linears_tensor[3](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A = self.linears_tensor[4](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S = self.linears_tensor[5](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        dX = I + A + S
        X = X + dX + torch.matrix_power(dX, 2)
        return X


class TensorNetExt(TensorNet):
    """Reuse the vendored QM embedding; upstream ext update has no total-charge factor."""
    def __init__(self, config, dtype):
        width = config['embedding_dimension']
        super().__init__(hidden_channels=width, num_layers=0,
                         num_rbf=config['num_rbf'], cutoff_upper=config['cutoff_upper'],
                         max_num_neighbors=config['max_num_neighbors'],
                         static_shapes=False, dtype=dtype)
        self.num_layers = config['num_layers']
        self.layers = nn.ModuleList([
            Interaction(config['num_rbf'], width, nn.SiLU, 0.,
                        config['cutoff_upper'], 'O(3)', dtype)
            for _ in range(self.num_layers)])
        self.ext_layer = Interaction_Ext(
            config['ext_num_rbf'], config['ext_rbf_type'], config['ext_trainable_rbf'],
            width, nn.SiLU, config['ext_cutoff_lower'], config['ext_cutoff_upper'], dtype)

    def forward(self, z, pos, batch, message):
        edges, distance, vec = self.distance(pos, batch)
        attr = self.distance_expansion(distance)
        direction = vec / distance.masked_fill(edges[0] == edges[1], 1)[:, None]
        x = self.tensor_embedding(z, edges, distance, direction, attr)
        for layer in self.layers:
            x = layer(x, edges, distance, attr, message)
        i, a, s = decompose_tensor(x)
        x = self.out_norm(torch.cat((tensor_norm(i), tensor_norm(a), tensor_norm(s)), -1))
        return self.act(self.linear(x))
