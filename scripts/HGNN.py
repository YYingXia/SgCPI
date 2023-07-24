import math
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import SparseTensor
from torch_geometric.nn.dense import Linear
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, ones, reset


class HGNNlayer(MessagePassing):
    def __init__(self, in_channels, out_channels, metadata):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.metadata = metadata

        self.k_lin = torch.nn.ModuleDict()
        self.q_lin = torch.nn.ModuleDict()
        self.v_lin = torch.nn.ModuleDict()
        self.o_lin = torch.nn.ModuleDict()
        for node_type in self.metadata[0]:
            self.k_lin[node_type] = Linear(in_channels, out_channels)
            self.q_lin[node_type] = Linear(in_channels, out_channels)
            self.v_lin[node_type] = Linear(in_channels, out_channels)
            self.o_lin[node_type] = Linear(out_channels, out_channels)

        self.w_edge = torch.nn.ParameterDict()
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.w_edge[edge_type] = Parameter(torch.Tensor(out_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.k_lin)
        reset(self.q_lin)
        reset(self.v_lin)
        reset(self.o_lin)
        glorot(self.w_edge)

    def forward(self, x_dict, edge_index_dict):

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}
        for node_type, x in x_dict.items():
            k_dict[node_type] = self.k_lin[node_type](x)
            q_dict[node_type] = self.q_lin[node_type](x)
            v_dict[node_type] = self.v_lin[node_type](x)
            out_dict[node_type] = []

        # Iterate over edge-types:
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            k = k_dict[src_type] @ self.w_edge[edge_type]
            out = self.propagate(edge_index, k=k, q=q_dict[dst_type], v=v_dict[src_type],size=None)
            out_dict[dst_type].append(out)

        for node_type, outs in out_dict.items():
            out = torch.stack(outs, dim=0)
            out = torch.sum(out, dim=0)
            out = self.o_lin[node_type](F.gelu(out))
            out_dict[node_type] = out

        return out_dict

    def message(self, k_j, q_i, v_j,index, ptr,size_i) -> Tensor:

        alpha = (q_i * k_j).sum(dim=-1)
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, 1)
        return out


class HGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGNNlayer(hidden_channels, hidden_channels, metadata)
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        x_dict_list = []
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict_list.append(x_dict)
        return x_dict_list