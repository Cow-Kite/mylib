import os.path as osp
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import EdgeAttr, GraphStore
from torch_geometric.distributed.partition import load_partition_info
from torch_geometric.typing import EdgeTensorType, EdgeType, NodeType
from torch_geometric.utils import sort_edge_index


class LocalGraphStore(GraphStore):
    r"""Implements the :class:`~torch_geometric.data.GraphStore` interface to
    act as a local graph store for distributed training.
    """
    def __init__(self):
        super().__init__()
        self._edge_index: Dict[Tuple, EdgeTensorType] = {}  # 엣지 인덱스를 저장하는 딕셔너리
        self._edge_attr: Dict[Tuple, EdgeAttr] = {} # 엣지 속성을 저장하는 딕셔너리
        self._edge_id: Dict[Tuple, Tensor] = {} # 엣지 ID를 저장하는 딕셔너리

        self.num_partitions = 1 # 파티션 수
        self.partition_idx = 0 # 현재 파티션 인덱스
        # Mapping between node ID and partition ID (노드 id와 파티션 id 간의 매핑)
        self.node_pb: Union[Tensor, Dict[NodeType, Tensor]] = None 
        # Mapping between edge ID and partition ID (엣지 id와 파티션 id간의 매핑)
        self.edge_pb: Union[Tensor, Dict[EdgeType, Tensor]] = None
        # Meta information related to partition and graph store info (파티션 및 그래프 스토어 관련 메타 정보)
        self.meta: Optional[Dict[Any, Any]] = None
        # If data is sorted based on destination nodes (CSC format):
        self.is_sorted: Optional[bool] = None

    @staticmethod
    def key(attr: EdgeAttr) -> Tuple: # 주어진 엣지 속성을 키로 변환하여 딕셔너리에서 사용하기 쉽게 함
        return (attr.edge_type, attr.layout.value)

    def get_partition_ids_from_nids( # 특정 노드 타입에 대한 노드 ID의 파티션 ID를 반환
        self,
        ids: torch.Tensor,
        node_type: Optional[NodeType] = None,
    ) -> Tensor:
        r"""Returns the partition IDs of node IDs for a specific node type."""
        if self.meta['is_hetero']:
            return self.node_pb[node_type][ids]
        else:
            return self.node_pb[ids]

    def get_partition_ids_from_eids(self, eids: torch.Tensor, # 특정 엣지 타입에 대한 엣지 ID의 파티션 ID를 반환
                                    edge_type: Optional[EdgeType] = None):
        r"""Returns the partition IDs of edge IDs for a specific edge type."""
        if self.meta['is_hetero']:
            return self.edge_pb[edge_type][eids]
        else:
            return self.edge_pb[eids]

    def put_edge_id(self, edge_id: Tensor, *args, **kwargs) -> bool: # 엣지 ID를 저장
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        self._edge_id[self.key(edge_attr)] = edge_id
        return True

    def get_edge_id(self, *args, **kwargs) -> Optional[EdgeTensorType]: # 저장된 엣지 ID를 가져옴
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        return self._edge_id.get(self.key(edge_attr))

    def remove_edge_id(self, *args, **kwargs) -> bool: # 저장된 엣지 ID를 제거함
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        return self._edge_id.pop(self.key(edge_attr), None) is not None

    def _put_edge_index(self, edge_index: EdgeTensorType, # 엣지 인덱스를 저장
                        edge_attr: EdgeAttr) -> bool:
        self._edge_index[self.key(edge_attr)] = edge_index
        self._edge_attr[self.key(edge_attr)] = edge_attr
        return True

    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[EdgeTensorType]: # 저장된 엣지 인덱스를 가져옴
        return self._edge_index.get(self.key(edge_attr), None)

    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool: # 저장된 엣지 인덱스를 제거함
        self._edge_attr.pop(self.key(edge_attr), None)
        return self._edge_index.pop(self.key(edge_attr), None) is not None

    def get_all_edge_attrs(self) -> List[EdgeAttr]: # 모든 엣지 속성을 반환
        return [self._edge_attr[key] for key in self._edge_index.keys()]

    # def get_connected_nodes(self, node_id: int) -> List[int]:
    #     r"""Returns the nodes connected to the given node ID."""
    #     connected_nodes = set()
    #     for edge_index in self._edge_index.values():
    #         src, dst = edge_index
    #         if node_id in src:
    #             indices = (src == node_id).nonzero(as_tuple=True)[0]
    #             connected_nodes.update(dst[indices].tolist())
    #         if node_id in dst:
    #             indices = (dst == node_id).nonzero(as_tuple=True)[0]
    #             connected_nodes.update(src[indices].tolist())
    #     return list(connected_nodes)   

    # Initialization ##########################################################

    @classmethod
    def from_data(
        cls,
        edge_id: Tensor, # global id
        edge_index: Tensor, # local id
        num_nodes: int, # number of local node
        is_sorted: bool = False,
    ) -> 'LocalGraphStore':
        r"""Creates a local graph store from a homogeneous or heterogenous
        :pyg:`PyG` graph.

        Args:
            edge_id (torch.Tensor): The global identifier for every local edge.
            edge_index (torch.Tensor): The local edge indices.
            num_nodes (int): The number of nodes in the local graph.
            is_sorted (bool): Whether edges are sorted by column/destination
                nodes (CSC format). (default: :obj:`False`)
        """
        graph_store = cls() # graph_store 인스턴스 생성
        graph_store.meta = {'is_hetero': False} # meta data 

        if not is_sorted:
            edge_index, edge_id = sort_edge_index(
                edge_index,
                edge_id,
                sort_by_row=False,
            )

        attr = dict(
            edge_type=None, # edge type은 없으니까 None으로
            layout='coo',
            size=(num_nodes, num_nodes),
            is_sorted=True,
        )

        # 엣지의 global id와 local id를 graph_store 인스턴스에 저장함
        graph_store.put_edge_index(edge_index, **attr)
        graph_store.put_edge_id(edge_id, **attr)

        return graph_store

    @classmethod
    def from_hetero_data(
        cls,
        edge_id_dict: Dict[EdgeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        num_nodes_dict: Dict[NodeType, int],
        is_sorted: bool = False,
    ) -> "LocalGraphStore":
        r"""Creates a local graph store from a heterogeneous :pyg:`PyG` graph.

        Args:
            edge_id_dict (Dict[EdgeType, torch.Tensor]): The global identifier
                for every local edge of every edge type.
            edge_index_dict (Dict[EdgeType, torch.Tensor]): The local edge
                indices of every edge type.
            num_nodes_dict: (Dict[str, int]): The number of nodes for every
                node type.
            is_sorted (bool): Whether edges are sorted by column/destination
                nodes (CSC format). (default: :obj:`False`)
        """
        graph_store = cls()
        graph_store.meta = {'is_hetero': True}

        for edge_type, edge_index in edge_index_dict.items():
            src, _, dst = edge_type
            attr = dict(
                edge_type=edge_type,
                layout='coo',
                size=(num_nodes_dict[src], num_nodes_dict[dst]),
                is_sorted=True,
            )
            edge_id = edge_id_dict[edge_type]
            if not is_sorted:
                edge_index, edge_id = sort_edge_index(
                    edge_index,
                    edge_id,
                    sort_by_row=False,
                )
            graph_store.put_edge_index(edge_index, **attr)
            graph_store.put_edge_id(edge_id, **attr)
        return graph_store

    # 주어진 디렉토리에서 그래프 파티션 데이터를 로드하여 LocalGraphStore 인스턴스를 생성하는 역할
    @classmethod
    def from_partition(cls, root: str, pid: int) -> 'LocalGraphStore': #pid: node rank / 파티션 번호
        part_dir = osp.join(root, f'part_{pid}') # /ogbn-products-partitions/part_0 or 1 파티션 디렉토리 경로 생성
        assert osp.exists(part_dir) # 파티션 디렉토리가 존재하는지 확인
        graph_store = cls() # 그래프 스토어 인스턴스 생성
        (
            meta,
            num_partitions,
            partition_idx,
            node_pb,
            edge_pb,
        ) = load_partition_info(root, pid) # 로드된 정보를 graph_store 인스턴스의 속성으로 설정
        graph_store.num_partitions = num_partitions
        graph_store.partition_idx = partition_idx
        graph_store.node_pb = node_pb
        graph_store.edge_pb = edge_pb
        graph_store.meta = meta
        
        # 그래프 데이터 로드
        graph_data = torch.load(osp.join(part_dir, 'graph.pt'))
        graph_store.is_sorted = meta['is_sorted']

        if not meta['is_hetero']:
            edge_index = torch.stack((graph_data['row'], graph_data['col']),
                                     dim=0)
            edge_id = graph_data['edge_id']
            if not graph_store.is_sorted:
                edge_index, edge_id = sort_edge_index(edge_index, edge_id,
                                                      sort_by_row=False)

            attr = dict(
                edge_type=None,
                layout='coo',
                size=graph_data['size'],
                is_sorted=True,
            )
            # graph_store에 edge global id, local id 저장
            graph_store.put_edge_index(edge_index, **attr)
            graph_store.put_edge_id(edge_id, **attr)

        if meta['is_hetero']:
            for edge_type, data in graph_data.items():
                attr = dict(
                    edge_type=edge_type,
                    layout='coo',
                    size=data['size'],
                    is_sorted=True,
                )
                edge_index = torch.stack((data['row'], data['col']), dim=0)
                edge_id = data['edge_id']

                if not graph_store.is_sorted:
                    edge_index, edge_id = sort_edge_index(
                        edge_index, edge_id, sort_by_row=False)
                graph_store.put_edge_index(edge_index, **attr)
                graph_store.put_edge_id(edge_id, **attr)

        return graph_store
