import math
import random
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader.base import DataLoaderIterator
from neighbor_loader import NeighborSampler
from torch_geometric.loader.utils import filter_data, filter_hetero_data

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch_sparse import SparseTensor

# Types for accessing data ####################################################

# Node-types are denoted by a single string, e.g.: `data['paper']`:
NodeType = str

# Edge-types are denotes by a triplet of strings, e.g.:
# `data[('author', 'writes', 'paper')]
EdgeType = Tuple[str, str, str]

# There exist some short-cuts to query edge-types (given that the full triplet
# can be uniquely reconstructed, e.g.:
# * via str: `data['writes']`
# * via Tuple[str, str]: `data[('author', 'paper')]`
QueryType = Union[NodeType, EdgeType, str, Tuple[str, str]]

Metadata = Tuple[List[NodeType], List[EdgeType]]

# A representation of a feature tensor
FeatureTensorType = Union[torch.Tensor, np.ndarray]

# Types for message passing ###################################################

Adj = Union[Tensor, SparseTensor]
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
PairOptTensor = Tuple[Optional[Tensor], Optional[Tensor]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Tensor]

# Types for sampling ##########################################################

InputNodes = Union[OptTensor, NodeType, Tuple[NodeType, OptTensor]]
InputEdges = Union[OptTensor, EdgeType, Tuple[EdgeType, OptTensor]]
NumNeighbors = Union[List[int], Dict[EdgeType, List[int]]]


class SubgraphLinkNeighborSampler(NeighborSampler):
    def __init__(self, data, subgraph_neighbors, *args, neg_sampling_ratio: float = 0.0, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.neg_sampling_ratio = neg_sampling_ratio
        self.subgraph_neighbors = subgraph_neighbors

        if issubclass(self.data_cls, Data):
            self.num_src_nodes = self.num_dst_nodes = data.num_nodes
        else:
            self.num_src_nodes = data[self.input_type[0]].num_nodes
            self.num_dst_nodes = data[self.input_type[-1]].num_nodes

    def _create_label(self, edge_label_index, edge_label):
        device = edge_label_index.device

        num_pos_edges = edge_label_index.size(1)
        num_neg_edges = int(num_pos_edges * self.neg_sampling_ratio)

        if num_neg_edges == 0:
            return edge_label_index, edge_label

        if edge_label is None:
            edge_label = torch.ones(num_pos_edges, device=device)
        else:
            assert edge_label.dtype == torch.long
            edge_label = edge_label + 1

        neg_row = torch.randint(self.num_src_nodes, (num_neg_edges, ))
        neg_col = torch.randint(self.num_dst_nodes, (num_neg_edges, ))
        neg_edge_label_index = torch.stack([neg_row, neg_col], dim=0)

        neg_edge_label = edge_label.new_zeros((num_neg_edges, ) +
                                              edge_label.size()[1:])

        edge_label_index = torch.cat([
            edge_label_index,
            neg_edge_label_index,
        ], dim=1)

        edge_label = torch.cat([edge_label, neg_edge_label], dim=0)

        return edge_label_index, edge_label

    def __call__(self, query: List[Tuple[Tensor]]):

        num_neighbors_list = []
        for (active_num, inactive_num, pred_num, hop) in self.subgraph_neighbors:
            num_neighbors = {}
            # num_neighbors['protein__sim__protein'] = [prot_sim_num] * hop

            num_neighbors['compound__active__protein'] = [active_num] * hop
            num_neighbors['compound__inactive__protein'] = [inactive_num] * hop
            num_neighbors['compound__pred__protein'] = [pred_num] * hop

            num_neighbors['protein__rev_active__compound'] = [active_num] * hop
            num_neighbors['protein__rev_inactive__compound'] = [inactive_num] * hop
            num_neighbors['protein__rev_pred__compound'] = [pred_num] * hop

            num_neighbors_list.append(num_neighbors)


        query = [torch.tensor(s) for s in zip(*query)]


        outs = []
        for num_neighbors in num_neighbors_list:
            num_hops = len(num_neighbors['compound__pred__protein'])
            if len(query) == 2:
                edge_label_index = torch.stack(query, dim=0)
                edge_label = None
            else:
                edge_label_index = torch.stack(query[:2], dim=0)
                edge_label = query[2]

            edge_label_index, edge_label = self._create_label(
                edge_label_index, edge_label)

            if issubclass(self.data_cls, Data):
                sample_fn = torch.ops.torch_sparse.neighbor_sample

                query_nodes = edge_label_index.view(-1)
                query_nodes, reverse = query_nodes.unique(return_inverse=True)
                edge_label_index = reverse.view(2, -1)

                node, row, col, edge = sample_fn(
                    self.colptr,
                    self.row,
                    query_nodes,
                    num_neighbors,
                    self.replace,
                    self.directed,
                )

                outs.append((node, row, col, edge, edge_label_index, edge_label))

            elif issubclass(self.data_cls, HeteroData):
                sample_fn = torch.ops.torch_sparse.hetero_neighbor_sample

                if self.input_type[0] != self.input_type[-1]:
                    query_src = edge_label_index[0]
                    query_src, reverse_src = query_src.unique(return_inverse=True)
                    query_dst = edge_label_index[1]
                    query_dst, reverse_dst = query_dst.unique(return_inverse=True)
                    edge_label_index = torch.stack([reverse_src, reverse_dst], 0)
                    query_node_dict = {
                        self.input_type[0]: query_src,
                        self.input_type[-1]: query_dst,
                    }
                else:  # Merge both source and destination node indices:
                    query_nodes = edge_label_index.view(-1)
                    query_nodes, reverse = query_nodes.unique(return_inverse=True)
                    edge_label_index = reverse.view(2, -1)
                    query_node_dict = {self.input_type[0]: query_nodes}

                node_dict, row_dict, col_dict, edge_dict = sample_fn(
                    self.node_types,
                    self.edge_types,
                    self.colptr_dict,
                    self.row_dict,
                    query_node_dict,
                    num_neighbors,
                    num_hops,
                    self.replace,
                    self.directed,
                )
                outs.append((node_dict, row_dict, col_dict, edge_dict, edge_label_index,
                        edge_label))
        return outs



class SubgraphLinkNeighborLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_neighbors: NumNeighbors,
        subgraph_neighbors: list,
        edge_label_index: InputEdges = None,
        edge_label: OptTensor = None,
        replace: bool = False,
        directed: bool = True,
        neg_sampling_ratio: float = 0.0,
        transform: Callable = None,
        is_sorted: bool = False,
        neighbor_sampler: Optional[SubgraphLinkNeighborSampler] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        if 'dataset' in kwargs:
            del kwargs['dataset']
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        self.data = data

        # Save for PyTorch Lightning < 1.6:
        self.num_neighbors = num_neighbors
        self.subgraph_neighbors = subgraph_neighbors
        self.edge_label_index = edge_label_index
        self.edge_label = edge_label
        self.replace = replace
        self.directed = directed
        self.transform = transform
        self.neighbor_sampler = neighbor_sampler
        self.neg_sampling_ratio = neg_sampling_ratio

        edge_type, edge_label_index = get_edge_label_index(
            data, edge_label_index)

        if neighbor_sampler is None:
            self.neighbor_sampler = SubgraphLinkNeighborSampler(
                data,
                subgraph_neighbors,
                num_neighbors,
                replace,
                directed,
                input_type=edge_type,
                is_sorted=is_sorted,
                neg_sampling_ratio=self.neg_sampling_ratio,
                share_memory=kwargs.get('num_workers', 0) > 0,
            )

        super().__init__(Dataset(edge_label_index, edge_label),
                         collate_fn=self.neighbor_sampler, **kwargs)

    def transform_fn(self, outs: Any) -> Union[Data, HeteroData]:
        data_list = []
        for out in outs:
            if isinstance(self.data, HeteroData):
                (node_dict, row_dict, col_dict, edge_dict, edge_label_index,
                 edge_label) = out
                data = filter_hetero_data(self.data, node_dict, row_dict, col_dict,
                                          edge_dict,
                                          self.neighbor_sampler.perm_dict)
                edge_type = self.neighbor_sampler.input_type
                data[edge_type].edge_label_index = edge_label_index
                data.edge_dict = edge_dict
                data.node_dict = node_dict
                if edge_label is not None:
                    data[edge_type].edge_label = edge_label

                data_list.append(data if self.transform is None else self.transform(data))

        return data_list

    def _get_iterator(self) -> Iterator:
        return DataLoaderIterator(super()._get_iterator(), self.transform_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class Dataset(torch.utils.data.Dataset):
    def __init__(self, edge_label_index: Tensor, edge_label: OptTensor = None):
        self.edge_label_index = edge_label_index
        self.edge_label = edge_label

    def __getitem__(self, idx: int) -> Tuple[int]:
        if self.edge_label is None:
            return self.edge_label_index[0, idx], self.edge_label_index[1, idx]
        else:
            return (self.edge_label_index[0, idx],
                    self.edge_label_index[1, idx], self.edge_label[idx])

    def __len__(self) -> int:
        return self.edge_label_index.size(1)


def get_edge_label_index(
    data: Union[Data, HeteroData],
    edge_label_index: InputEdges,
) -> Tuple[Optional[str], Tensor]:
    edge_type = None
    if isinstance(data, Data):
        if edge_label_index is None:
            return None, data.edge_index
        return None, edge_label_index

    assert edge_label_index is not None
    assert isinstance(edge_label_index, (list, tuple))

    if isinstance(edge_label_index[0], str):
        edge_type = edge_label_index
        edge_type = data._to_canonical(*edge_type)
        assert edge_type in data.edge_types
        return edge_type, data[edge_type].edge_index

    assert len(edge_label_index) == 2

    edge_type, edge_label_index = edge_label_index
    edge_type = data._to_canonical(*edge_type)
    assert edge_type in data.edge_types

    if edge_label_index is None:
        return edge_type, data[edge_type].edge_index

    return edge_type, edge_label_index


