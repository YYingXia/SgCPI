from collections.abc import Sequence
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader.base import DataLoaderIterator

from torch_geometric.typing import InputNodes
from typing import Dict, List, Optional, Tuple, Union


import copy
import math
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import EdgeStorage, NodeStorage
from torch_geometric.typing import EdgeType, OptTensor


def index_select(value: Tensor, index: Tensor, dim: int = 0) -> Tensor:
    out: Optional[Tensor] = None
    if torch.utils.data.get_worker_info() is not None:
        # If we are in a background process, we write directly into a shared
        # memory tensor to avoid an extra copy:
        size = list(value.size())
        size[dim] = index.numel()
        numel = math.prod(size)
        storage = value.storage()._new_shared(numel)
        out = value.new(storage).view(size)
    return torch.index_select(value, dim, index, out=out)


def edge_type_to_str(edge_type: Union[EdgeType, str]) -> str:
    # Since C++ cannot take dictionaries with tuples as key as input, edge type
    # triplets need to be converted into single strings.
    return edge_type if isinstance(edge_type, str) else '__'.join(edge_type)


def to_csc(
    data: Union[Data, EdgeStorage],
    device: Optional[torch.device] = None,
    share_memory: bool = False,
    is_sorted: bool = False,
) -> Tuple[Tensor, Tensor, OptTensor]:
    # Convert the graph data into a suitable format for sampling (CSC format).
    # Returns the `colptr` and `row` indices of the graph, as well as an
    # `perm` vector that denotes the permutation of edges.
    # Since no permutation of edges is applied when using `SparseTensor`,
    # `perm` can be of type `None`.
    perm: Optional[Tensor] = None

    if hasattr(data, 'adj_t'):
        colptr, row, _ = data.adj_t.csr()

    elif hasattr(data, 'edge_index'):
        (row, col) = data.edge_index
        if not is_sorted:
            perm = (col * data.size(0)).add_(row).argsort()
            row = row[perm]
        colptr = torch.ops.torch_sparse.ind2ptr(col[perm], data.size(1))
    else:
        raise AttributeError("Data object does not contain attributes "
                             "'adj_t' or 'edge_index'")

    colptr = colptr.to(device)
    row = row.to(device)
    perm = perm.to(device) if perm is not None else None

    if not colptr.is_cuda and share_memory:
        colptr.share_memory_()
        row.share_memory_()
        if perm is not None:
            perm.share_memory_()

    return colptr, row, perm


def to_hetero_csc(
    data: HeteroData,
    device: Optional[torch.device] = None,
    share_memory: bool = False,
    is_sorted: bool = False,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, OptTensor]]:
    # Convert the heterogeneous graph data into a suitable format for sampling
    # (CSC format).
    # Returns dictionaries holding `colptr` and `row` indices as well as edge
    # permutations for each edge type, respectively.
    # Since C++ cannot take dictionaries with tuples as key as input, edge type
    # triplets are converted into single strings.
    colptr_dict, row_dict, perm_dict = {}, {}, {}

    for store in data.edge_stores:
        key = edge_type_to_str(store._key)
        out = to_csc(store, device, share_memory, is_sorted)
        colptr_dict[key], row_dict[key], perm_dict[key] = out

    return colptr_dict, row_dict, perm_dict


def filter_node_store_(store: NodeStorage, out_store: NodeStorage,
                       index: Tensor) -> NodeStorage:
    # Filters a node storage object to only hold the nodes in `index`:
    for key, value in store.items():
        if key == 'num_nodes':
            out_store.num_nodes = index.numel()

        elif store.is_node_attr(key):
            index = index.to(value.device)
            dim = store._parent().__cat_dim__(key, value, store)
            out_store[key] = index_select(value, index, dim=dim)

    return store


def filter_edge_store_(store: EdgeStorage, out_store: EdgeStorage, row: Tensor,
                       col: Tensor, index: Tensor,
                       perm: OptTensor = None) -> EdgeStorage:
    # Filters a edge storage object to only hold the edges in `index`,
    # which represents the new graph as denoted by `(row, col)`:
    for key, value in store.items():
        if key == 'edge_index':
            edge_index = torch.stack([row, col], dim=0)
            out_store.edge_index = edge_index.to(value.device)

        elif key == 'adj_t':
            # NOTE: We expect `(row, col)` to be sorted by `col` (CSC layout).
            row = row.to(value.device())
            col = col.to(value.device())
            edge_attr = value.storage.value()
            if edge_attr is not None:
                index = index.to(edge_attr.device)
                edge_attr = edge_attr[index]
            sparse_sizes = out_store.size()[::-1]
            # TODO Currently, we set `is_sorted=False`, see:
            # https://github.com/pyg-team/pytorch_geometric/issues/4346
            out_store.adj_t = SparseTensor(row=col, col=row, value=edge_attr,
                                           sparse_sizes=sparse_sizes,
                                           is_sorted=False, trust_data=True)

        elif store.is_edge_attr(key):
            dim = store._parent().__cat_dim__(key, value, store)
            if perm is None:
                index = index.to(value.device)
                out_store[key] = index_select(value, index, dim=dim)
            else:
                perm = perm.to(value.device)
                index = index.to(value.device)
                out_store[key] = index_select(value, perm[index], dim=dim)

    return store


def filter_data(data: Data, node: Tensor, row: Tensor, col: Tensor,
                edge: Tensor, perm: OptTensor = None) -> Data:
    # Filters a data object to only hold nodes in `node` and edges in `edge`:
    out = copy.copy(data)
    filter_node_store_(data._store, out._store, node)
    filter_edge_store_(data._store, out._store, row, col, edge, perm)
    return out


def filter_hetero_data(
    data: HeteroData,
    node_dict: Dict[str, Tensor],
    row_dict: Dict[str, Tensor],
    col_dict: Dict[str, Tensor],
    edge_dict: Dict[str, Tensor],
    perm_dict: Dict[str, OptTensor],
) -> HeteroData:
    # Filters a heterogeneous data object to only hold nodes in `node` and
    # edges in `edge` for each node and edge type, respectively:
    out = copy.copy(data)

    for node_type in data.node_types:
        filter_node_store_(data[node_type], out[node_type],
                           node_dict[node_type])

    for edge_type in data.edge_types:
        edge_type_str = edge_type_to_str(edge_type)
        filter_edge_store_(data[edge_type], out[edge_type],
                           row_dict[edge_type_str], col_dict[edge_type_str],
                           edge_dict[edge_type_str], perm_dict[edge_type_str])

    return out




EdgeType = Tuple[str, str, str]
NumNeighbors = Union[List[int], Dict[EdgeType, List[int]]]




class NeighborSampler:
    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_neighbors: NumNeighbors,
        replace: bool = False,
        directed: bool = True,
        input_type: Optional[Any] = None,
        time_attr: Optional[str] = None,
        is_sorted: bool = False,
        share_memory: bool = False,
    ):
        self.data_cls = data.__class__
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.directed = directed
        self.node_time = None

        if isinstance(data, Data):
            if time_attr is not None:
                # TODO `time_attr` support for homogeneous graphs
                raise ValueError(
                    f"'time_attr' attribute not yet supported for "
                    f"'{data.__class__.__name__}' object")

            # Convert the graph data into a suitable format for sampling.
            out = to_csc(data, device='cpu', share_memory=share_memory,
                         is_sorted=is_sorted)
            self.colptr, self.row, self.perm = out
            assert isinstance(num_neighbors, (list, tuple))

        elif isinstance(data, HeteroData):
            if time_attr is not None:
                self.node_time_dict = data.collect(time_attr)
            else:
                self.node_time_dict = None

            # Convert the graph data into a suitable format for sampling.
            # NOTE: Since C++ cannot take dictionaries with tuples as key as
            # input, edge type triplets are converted into single strings.
            out = to_hetero_csc(data, device='cpu', share_memory=share_memory,
                                is_sorted=is_sorted)
            self.colptr_dict, self.row_dict, self.perm_dict = out

            self.node_types, self.edge_types = data.metadata()
            if isinstance(num_neighbors, (list, tuple)):
                num_neighbors = {key: num_neighbors for key in self.edge_types}
            assert isinstance(num_neighbors, dict)
            self.num_neighbors = {
                edge_type_to_str(key): value
                for key, value in num_neighbors.items()
            }

            self.num_hops = max([len(v) for v in self.num_neighbors.values()])

            assert input_type is not None
            self.input_type = input_type

        else:
            raise TypeError(f'NeighborLoader found invalid type: {type(data)}')

    def __call__(self, index: Union[List[int], Tensor]):
        if not isinstance(index, torch.LongTensor):
            index = torch.LongTensor(index)

        if issubclass(self.data_cls, Data):
            fn = torch.ops.torch_sparse.neighbor_sample
            node, row, col, edge = fn(
                self.colptr,
                self.row,
                index,
                self.num_neighbors,
                self.replace,
                self.directed,
            )
            return node, row, col, edge, index.numel()

        elif issubclass(self.data_cls, HeteroData):
            if self.node_time_dict is None:
                fn = torch.ops.torch_sparse.hetero_neighbor_sample
                node_dict, row_dict, col_dict, edge_dict = fn(
                    self.node_types,
                    self.edge_types,
                    self.colptr_dict,
                    self.row_dict,
                    {self.input_type: index},
                    self.num_neighbors,
                    self.num_hops,
                    self.replace,
                    self.directed,
                )
            else:
                fn = torch.ops.torch_sparse.hetero_temporal_neighbor_sample
                node_dict, row_dict, col_dict, edge_dict = fn(
                    self.node_types,
                    self.edge_types,
                    self.colptr_dict,
                    self.row_dict,
                    {self.input_type: index},
                    self.num_neighbors,
                    self.node_time_dict,
                    self.num_hops,
                    self.replace,
                    self.directed,
                )
            return node_dict, row_dict, col_dict, edge_dict, index.numel()


class NeighborLoader(torch.utils.data.DataLoader):
    r"""A data loader that performs neighbor sampling as introduced in the
    `"Inductive Representation Learning on Large Graphs"
    <https://arxiv.org/abs/1706.02216>`_ paper.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.
    More specifically, :obj:`num_neighbors` denotes how much neighbors are
    sampled for each node in each iteration.
    :class:`~torch_geometric.loader.NeighborLoader` takes in this list of
    :obj:`num_neighbors` and iteratively samples :obj:`num_neighbors[i]` for
    each node involved in iteration :obj:`i - 1`.
    Sampled nodes are sorted based on the order in which they were sampled.
    In particular, the first :obj:`batch_size` nodes represent the set of
    original mini-batch nodes.
    .. code-block:: python
        from torch_geometric.datasets import Planetoid
        from torch_geometric.loader import NeighborLoader
        data = Planetoid(path, name='Cora')[0]
        loader = NeighborLoader(
            data,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            input_nodes=data.train_mask,
        )
        sampled_data = next(iter(loader))
        print(sampled_data.batch_size)
        >>> 128
    By default, the data loader will only include the edges that were
    originally sampled (:obj:`directed = True`).
    This option should only be used in case the number of hops is equivalent to
    the number of GNN layers.
    In case the number of GNN layers is greater than the number of hops,
    consider setting :obj:`directed = False`, which will include all edges
    between all sampled nodes (but is slightly slower as a result).
    Furthermore, :class:`~torch_geometric.loader.NeighborLoader` works for both
    **homogeneous** graphs stored via :class:`~torch_geometric.data.Data` as
    well as **heterogeneous** graphs stored via
    :class:`~torch_geometric.data.HeteroData`.
    When operating in heterogeneous graphs, more fine-grained control over
    the amount of sampled neighbors of individual edge types is possible, but
    not necessary:
    .. code-block:: python
        from torch_geometric.datasets import OGB_MAG
        from torch_geometric.loader import NeighborLoader
        hetero_data = OGB_MAG(path)[0]
        loader = NeighborLoader(
            hetero_data,
            # Sample 30 neighbors for each node and edge type for 2 iterations
            num_neighbors={key: [30] * 2 for key in hetero_data.edge_types},
            # Use a batch size of 128 for sampling training nodes of type paper
            batch_size=128,
            input_nodes=('paper', hetero_data['paper'].train_mask),
        )
        sampled_hetero_data = next(iter(loader))
        print(sampled_hetero_data['paper'].batch_size)
        >>> 128
    .. note::
        For an example of using
        :class:`~torch_geometric.loader.NeighborLoader`, see
        `examples/hetero/to_hetero_mag.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py>`_.
    The :class:`~torch_geometric.loader.NeighborLoader` will return subgraphs
    where global node indices are mapped to local indices corresponding to this
    specific subgraph. However, often times it is desired to map the nodes of
    the current subgraph back to the global node indices. A simple trick to
    achieve this is to include this mapping as part of the :obj:`data` object:
    .. code-block:: python
        # Assign each node its global node index:
        data.n_id = torch.arange(data.num_nodes)
        loader = NeighborLoader(data, ...)
        sampled_data = next(iter(loader))
        print(sampled_data.n_id)
    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of neighbors to sample for each node in each iteration.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for each individual edge type.
            If an entry is set to :obj:`-1`, all neighbors will be included.
        input_nodes (torch.Tensor or str or Tuple[str, torch.Tensor]): The
            indices of nodes for which neighbors are sampled to create
            mini-batches.
            Needs to be either given as a :obj:`torch.LongTensor` or
            :obj:`torch.BoolTensor`.
            If set to :obj:`None`, all nodes will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the node type and node indices. (default: :obj:`None`)
        replace (bool, optional): If set to :obj:`True`, will sample with
            replacement. (default: :obj:`False`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)
        time_attr (str, optional): The name of the attribute that denotes
            timestamps for the nodes in the graph.
            If set, temporal sampling will be used such that neighbors are
            guaranteed to fulfill temporal constraints, *i.e.* neighbors have
            an earlier timestamp than the center node. (default: :obj:`None`)
        transform (Callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        is_sorted (bool, optional): If set to :obj:`True`, assumes that
            :obj:`edge_index` is sorted by column. This avoids internal
            re-sorting of the data and can improve runtime and memory
            efficiency. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_neighbors: NumNeighbors,
        input_nodes: InputNodes = None,
        replace: bool = False,
        directed: bool = True,
        time_attr: Optional[str] = None,
        transform: Callable = None,
        is_sorted: bool = False,
        neighbor_sampler: Optional[NeighborSampler] = None,
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
        self.input_nodes = input_nodes
        self.replace = replace
        self.directed = directed
        self.transform = transform
        self.neighbor_sampler = neighbor_sampler

        node_type, input_nodes = get_input_nodes(data, input_nodes)

        if neighbor_sampler is None:
            self.neighbor_sampler = NeighborSampler(
                data,
                num_neighbors,
                replace,
                directed,
                input_type=node_type,
                time_attr=time_attr,
                is_sorted=is_sorted,
                share_memory=kwargs.get('num_workers', 0) > 0,
            )

        super().__init__(input_nodes, collate_fn=self.neighbor_sampler,
                         **kwargs)

    def transform_fn(self, out: Any) -> Union[Data, HeteroData]:
        if isinstance(self.data, Data):
            node, row, col, edge, batch_size = out
            data = filter_data(self.data, node, row, col, edge,
                               self.neighbor_sampler.perm)
            data.batch_size = batch_size

        elif isinstance(self.data, HeteroData):
            node_dict, row_dict, col_dict, edge_dict, batch_size = out
            data = filter_hetero_data(self.data, node_dict, row_dict, col_dict,
                                      edge_dict,
                                      self.neighbor_sampler.perm_dict)
            data[self.neighbor_sampler.input_type].batch_size = batch_size

        return data if self.transform is None else self.transform(data)

    def _get_iterator(self) -> Iterator:
        return DataLoaderIterator(super()._get_iterator(), self.transform_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'





###############################################################################


def get_input_nodes(data: Union[Data, HeteroData],
                    input_nodes: InputNodes) -> Tuple[Optional[str], Sequence]:
    if isinstance(data, Data):
        if input_nodes is None:
            return None, range(data.num_nodes)
        if input_nodes.dtype == torch.bool:
            input_nodes = input_nodes.nonzero(as_tuple=False).view(-1)
        return None, input_nodes

    assert input_nodes is not None

    if isinstance(input_nodes, str):
        return input_nodes, range(data[input_nodes].num_nodes)

    assert isinstance(input_nodes, (list, tuple))
    assert len(input_nodes) == 2
    assert isinstance(input_nodes[0], str)

    if input_nodes[1] is None:
        return input_nodes[0], range(data[input_nodes[0]].num_nodes)

    node_type, input_nodes = input_nodes
    if input_nodes.dtype == torch.bool:
        input_nodes = input_nodes.nonzero(as_tuple=False).view(-1)
    return node_type, input_nodes