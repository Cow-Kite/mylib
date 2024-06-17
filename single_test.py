import os.path as osp
import torch

# 현재 디렉토리 구조에서 distributed 모듈에서 임포트하는지 확인
from distributed import LocalGraphStore, LocalFeatureStore, DistNeighborLoader

dataset_root_dir = "/home/sykang/mylib/data/partitions/ogbn-products/2-parts"
dataset = "ogbn-products"

print('--- Loading data partition files ...')
root_dir = osp.join(osp.dirname(osp.realpath(__file__)), dataset_root_dir)
node_label_file = osp.join(root_dir, f'{dataset}-label', 'label.pt')
train_idx = torch.load(
    osp.join(
        root_dir,
        f'{dataset}-train-partitions',
        f'partition0.pt',
    ))
test_idx = torch.load(
    osp.join(
        root_dir,
        f'{dataset}-test-partitions',
        f'partition0.pt',
    ))

# Load partition into local graph store:
graph = LocalGraphStore.from_partition(
    osp.join(root_dir, f'{dataset}-partitions'), 0)
# Load partition into local feature store:
feature = LocalFeatureStore.from_partition(
    osp.join(root_dir, f'{dataset}-partitions'), 0)
feature.labels = torch.load(node_label_file)
partition_data = (feature, graph)

# 디버깅 정보 출력
print(f'Partition metadata: {graph.meta}')
print(f'feature type: {type(feature)}')  # LocalFeatureStore 타입 확인
print(f'graph type: {type(graph)}')      # LocalGraphStore 타입 확인

assert isinstance(partition_data[0], LocalFeatureStore), "partition_data[0] is not an instance of LocalFeatureStore"
assert isinstance(partition_data[1], LocalGraphStore), "partition_data[1] is not an instance of LocalGraphStore"

train_loader = DistNeighborLoader(
    data=partition_data,
    input_nodes=train_idx,
)
