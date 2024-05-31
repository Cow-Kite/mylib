import sys
import os.path as osp
import torch
#sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), './storage')))
from distributed import LocalGraphStore
from distributed import LocalFeatureStore

dataset = 'ogbn-products'
dataset_root_dir = './data/partitions/ogbn-products/2-parts'
print('--- Loading data partition files ...')
root_dir = osp.join(osp.dirname(osp.realpath(__file__)), dataset_root_dir)
node_label_file = osp.join(root_dir, f'{dataset}-label', 'label.pt')

node_rank = 0
for node_rank in range(0,2):
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
                        f'partition{node_rank}.pt'
            ))

    graph = LocalGraphStore.from_partition(
        osp.join(root_dir, f'{dataset}-partitions'), node_rank)

    feature = LocalFeatureStore.from_partition(
        osp.join(root_dir, f'{dataset}-partitions'), node_rank)

feature.labels = torch.load(node_label_file)
partition_data = (feature, graph)

print(f'Partition metadata: {graph.meta}')

connected_nodes = graph.get_connected_nodes(66516)
print(connected_nodes)
