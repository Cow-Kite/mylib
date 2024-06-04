import argparse
import os.path as osp
import time
from contextlib import nullcontext

import torch
import torch.distributed
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from torch_geometric.data import HeteroData
from .distributed import (
    DistContext,
    DistNeighborLoader,
    LocalFeatureStore,
    LocalGraphStore,
)
from torch_geometric.nn import GraphSAGE, to_hetero



def run_proc(
    local_proc_rank: int,
    num_nodes: int,
    node_rank: int,
    dataset: str,
    dataset_root_dir: str,
    master_addr: str,

):
    is_hetero = dataset == 'ogbn-mag'

    print('--- Loading data partition files ...')
    root_dir = osp.join(osp.dirname(osp.realpath(__file__)), dataset_root_dir)
    node_label_file = osp.join(root_dir, f'{dataset}-label', 'label.pt')
    train_idx = torch.load(
        osp.join(
            root_dir,
            f'{dataset}-train-partitions',
            f'partition{node_rank}.pt',
        ))
    test_idx = torch.load(
        osp.join(
            root_dir,
            f'{dataset}-test-partitions',
            f'partition{node_rank}.pt',
        ))
    
    if is_hetero:
        train_idx = ('paper', train_idx)
        test_idx = ('paper', test_idx)

    # Load partition into local graph/feature store:
    graph = LocalGraphStore.from_partition(
        osp.join(root_dir, f'{dataset}-partitions'), node_rank)
    feature = LocalFeatureStore.from_partition(
        osp.join(root_dir, f'{dataset}-partitions'), node_rank)
    feature.labels = torch.load(node_label_file)
    partition_data = (feature, graph)
    print(f'Partition metadata: {graph.meta}')

    # Initialize distributed context:
    current_ctx = DistContext(
        world_size=num_nodes,
        rank=node_rank,
        global_world_size=num_nodes,
        global_rank=node_rank,
        group_name="distributed-ogb-sage",
    )
    current_ctx = torch.device('cpu')

    