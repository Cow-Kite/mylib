import sys
import os.path as osp
import torch
from multiprocessing import Process
from distributed import LocalGraphStore
from distributed import LocalFeatureStore

def process_data(node_rank, dataset, dataset_root_dir):
    print(f'--- Process {node_rank}: Loading data partition files ...')
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
                        f'partition{node_rank}.pt'
            ))
    
    graph = LocalGraphStore.from_partition(
        osp.join(root_dir, f'{dataset}-partitions'), node_rank)

    feature = LocalFeatureStore.from_partition(
        osp.join(root_dir, f'{dataset}-partitions'), node_rank)

    feature.labels = torch.load(node_label_file)
    partition_data = (feature, graph)

    print(f'Partition metadata (node_rank {node_rank}): {graph.meta}')

def main():
    dataset = 'ogbn-products'
    dataset_root_dir = './data/partitions/ogbn-products/2-parts'

    processes = []
    for node_rank in range(2):
        p = Process(target=process_data, args=(node_rank, dataset, dataset_root_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()

